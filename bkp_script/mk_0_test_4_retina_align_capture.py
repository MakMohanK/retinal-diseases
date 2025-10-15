import os
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageDraw
from picamera2 import Picamera2
from gpiozero import LED, Button

# ========================================
# FOLDER SETUP
# ========================================
temp_dir = "temp"
output_dir = "input"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ========================================
# GPIO SETUP
# ========================================
led = LED(17)
button_capture = Button(27, pull_up=False)
led.on()

# ========================================
# CAMERA THREAD CLASS
# ========================================
class CameraThread(threading.Thread):
    def __init__(self, camera_id, name, resolution=(640, 480), rotate=False):
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.name = name
        self.resolution = resolution
        self.rotate = rotate
        self.picam2 = Picamera2(camera_id)
        config = self.picam2.create_video_configuration(
            main={"format": 'RGB888', "size": resolution}
        )
        self.picam2.configure(config)
        self.frame = None
        self.running = False

    def run(self):
        self.picam2.start()
        self.running = True
        while self.running:
            frame = self.picam2.capture_array()
            if self.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            self.frame = frame

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.join()
        self.picam2.stop()

# ========================================
# CIRCULAR MASK UTILS
# ========================================
def create_circular_mask(size):
    w, h = size
    center = (w // 2, h // 2)
    radius = min(w, h) // 3.0  # Slightly larger for retina
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (center[0] - radius, center[1] - radius,
         center[0] + radius, center[1] + radius),
        fill=255
    )
    return np.array(mask)

def apply_circular_crop(image_np, mask_np):
    cropped = np.zeros_like(image_np)
    for i in range(3):
        cropped[:, :, i] = np.where(mask_np == 255, image_np[:, :, i], 0)
    return cropped

# ========================================
# ALIGNMENT (ORB + ECC)
# ========================================
def align_images_precise(fixed, moving):
    """
    Uses ORB feature-based + ECC fine-tuning for precise retinal alignment.
    """
    fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

    # ORB-based initial homography
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(fixed_gray, None)
    kp2, des2 = orb.detectAndCompute(moving_gray, None)

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            if H is not None:
                moving_warped = cv2.warpPerspective(moving, H, (fixed.shape[1], fixed.shape[0]))
            else:
                moving_warped = moving.copy()
        else:
            moving_warped = moving.copy()
    else:
        moving_warped = moving.copy()

    # ECC fine alignment
    try:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        fixed_eq = cv2.equalizeHist(fixed_gray)
        moving_eq = cv2.equalizeHist(cv2.cvtColor(moving_warped, cv2.COLOR_BGR2GRAY))
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        cc, warp_matrix = cv2.findTransformECC(fixed_eq, moving_eq, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria)
        aligned = cv2.warpPerspective(
            moving_warped, warp_matrix,
            (fixed.shape[1], fixed.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    except cv2.error:
        print("⚠️ ECC refinement failed.")
        aligned = moving_warped

    return aligned

# ========================================
# CAPTURE IMAGES FROM BOTH CAMERAS
# ========================================
def capture_frames(ir_cam, hd_cam, count=10):
    ir_frames = []
    hd_frames = []
    print("➡ Capturing frames...")

    for i in range(count):
        while True:
            ir = ir_cam.get_frame()
            hd = hd_cam.get_frame()
            if ir is not None and hd is not None:
                break
            time.sleep(0.05)

        # Resize IR to HD
        ir_resized = cv2.resize(ir, (hd.shape[1], hd.shape[0]))

        # Apply SAME circular mask for both cameras
        mask_np = create_circular_mask(hd.shape[1::-1])
        hd_cropped = apply_circular_crop(hd, mask_np)
        ir_cropped = apply_circular_crop(ir_resized, mask_np)

        ir_frames.append(ir_cropped)
        hd_frames.append(hd_cropped)

        show_image = np.hstack((hd_cropped, ir_cropped))
        cv2.imshow("HD + IR Cropped", show_image)
        cv2.waitKey(1)

        cv2.imwrite(os.path.join(temp_dir, f"ir_{i+1}.png"), ir_cropped)
        cv2.imwrite(os.path.join(temp_dir, f"hd_{i+1}.png"), hd_cropped)

    led.off()
    print("✅ Captured 10 cropped frames.")
    return ir_frames, hd_frames

# ========================================
# PROCESS FRAMES: ALIGN + OVERLAY
# ========================================
def process_frames(ir_frames, hd_frames):
    print("➡ Aligning and overlaying frames...")

    for i, (ir, hd) in enumerate(zip(ir_frames, hd_frames)):
        # Align cropped IR to cropped HD
        aligned_ir = align_images_precise(hd, ir)

        # Create yellow overlay from aligned IR intensity
        gray_ir = cv2.cvtColor(aligned_ir, cv2.COLOR_BGR2GRAY)
        overlay = np.zeros_like(hd)
        overlay[:, :, 1] = gray_ir  # Green
        overlay[:, :, 2] = gray_ir  # Red

        # Blend into HD image
        blended = cv2.addWeighted(hd, 0.7, overlay, 0.3, 0)

        filename = os.path.join(output_dir, f"aligned_overlay_{i+1}.jpg")
        cv2.imwrite(filename, blended)
        print(f"✅ Saved aligned overlay: {filename}")

# ========================================
# MAIN WRAPPER FUNCTION
# ========================================
def capture_and_process(ir_cam, hd_cam):
    ir_frames, hd_frames = capture_frames(ir_cam, hd_cam, count=10)
    process_frames(ir_frames, hd_frames)

# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == '__main__':
    ir_camera_thread = CameraThread(camera_id=0, name="IR Camera", rotate=True)
    hd_camera_thread = CameraThread(camera_id=1, name="HD Camera", rotate=False)

    ir_camera_thread.start()
    hd_camera_thread.start()

    print("📸 System ready. Press GPIO27 button to capture, crop, align, and overlay 10 images.")

    try:
        while True:
            button_capture.wait_for_press()
            led.on()
            capture_and_process(ir_camera_thread, hd_camera_thread)
    except KeyboardInterrupt:
        print("🛑 Exiting...")
    finally:
        ir_camera_thread.stop()
        hd_camera_thread.stop()
        cv2.destroyAllWindows()
