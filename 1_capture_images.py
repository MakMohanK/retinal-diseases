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
    radius = min(w, h) // 5.5
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
# ECC ALIGNMENT USING HOMOGRAPHY
# ========================================
def align_images_ecc(fixed_gray, moving_gray):
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(fixed_gray, moving_gray, warp_matrix, warp_mode, criteria)
        aligned = cv2.warpPerspective(
            moving_gray, warp_matrix,
            (fixed_gray.shape[1], fixed_gray.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        return warp_matrix, aligned
    except cv2.error:
        print("⚠️ ECC alignment failed — returning unaligned IR frame")
        return np.eye(3, 3, dtype=np.float32), moving_gray

# ========================================
# PUPIL CENTER-BASED FINE ALIGNMENT
# ========================================
def adjust_by_pupil_center(aligned_ir, hd):
    gray_ir = cv2.cvtColor(aligned_ir, cv2.COLOR_BGR2GRAY)
    gray_hd = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY)
    gray_ir = cv2.medianBlur(gray_ir, 5)
    gray_hd = cv2.medianBlur(gray_hd, 5)

    _, ir_thresh = cv2.threshold(gray_ir, 40, 255, cv2.THRESH_BINARY_INV)
    _, hd_thresh = cv2.threshold(gray_hd, 40, 255, cv2.THRESH_BINARY_INV)

    ir_m = cv2.moments(ir_thresh)
    hd_m = cv2.moments(hd_thresh)

    if ir_m["m00"] > 0 and hd_m["m00"] > 0:
        ir_cx, ir_cy = int(ir_m["m10"] / ir_m["m00"]), int(ir_m["m01"] / ir_m["m00"])
        hd_cx, hd_cy = int(hd_m["m10"] / hd_m["m00"]), int(hd_m["m01"] / hd_m["m00"])
        dx, dy = hd_cx - ir_cx, hd_cy - ir_cy
        aligned_ir = np.roll(aligned_ir, shift=(dy, dx), axis=(0, 1))
    return aligned_ir

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

        ir_resized = cv2.resize(ir, (hd.shape[1], hd.shape[0]))
        ir_frames.append(ir_resized)
        hd_frames.append(hd)

        cv2.imwrite(os.path.join(temp_dir, f"ir_{i+1}.png"), ir_resized)
        cv2.imwrite(os.path.join(temp_dir, f"hd_{i+1}.png"), hd)

    led.off()
    print("✅ Captured 10 frames.")
    return ir_frames, hd_frames

# ========================================
# PROCESS FRAMES: ALIGN + CROP + OVERLAY
# ========================================
def process_frames(ir_frames, hd_frames):
    print("➡ Processing and overlaying frames...")

    for i, (ir, hd) in enumerate(zip(ir_frames, hd_frames)):
        gray_hd = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY)
        gray_ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        # Improve contrast for ECC
        gray_hd = cv2.equalizeHist(gray_hd)
        gray_ir = cv2.equalizeHist(gray_ir)

        # ECC alignment (homography)
        warp_matrix, aligned_ir_gray = align_images_ecc(gray_hd, gray_ir)

        # Apply same warp to color IR
        aligned_ir = cv2.warpPerspective(
            ir, warp_matrix, (hd.shape[1], hd.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        # Fine adjustment by pupil centers
        aligned_ir = adjust_by_pupil_center(aligned_ir, hd)

        # Apply circular mask
        mask_np = create_circular_mask(hd.shape[1::-1])
        hd_cropped = apply_circular_crop(hd, mask_np)
        ir_cropped = apply_circular_crop(aligned_ir, mask_np)

        # Yellow overlay
        gray_aligned_ir = cv2.cvtColor(ir_cropped, cv2.COLOR_BGR2GRAY)
        overlay = np.zeros_like(hd)
        overlay[:, :, 1] = gray_aligned_ir  # Green
        overlay[:, :, 2] = gray_aligned_ir  # Red
        overlay = apply_circular_crop(overlay, mask_np)

        # Blend overlay
        blended = cv2.addWeighted(hd_cropped, 0.7, overlay, 0.3, 0)

        # Save result
        filename = os.path.join(output_dir, f"capture_{i+1}.jpg")
        cv2.imwrite(filename, blended)
        print(f"✅ Saved: {filename}")

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

    print("📸 System ready. Press GPIO27 button to capture and align 10 images.")

    try:
        while True:
            button_capture.wait_for_press()
            capture_and_process(ir_camera_thread, hd_camera_thread)
    except KeyboardInterrupt:
        print("🛑 Exiting...")
    finally:
        ir_camera_thread.stop()
        hd_camera_thread.stop()
        cv2.destroyAllWindows()
