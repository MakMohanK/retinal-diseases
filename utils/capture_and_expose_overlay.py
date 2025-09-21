import cv2
import threading
import time
import os
import numpy as np
from picamera2 import Picamera2
from gpiozero import LED, Button

# --- Camera Thread Class ---
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


# --- GPIO Setup ---
led = LED(17)
button_capture = Button(27, pull_up=False)

# --- Temporary and output folders ---
temp_dir = "temp"
output_dir = "input"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# --- Fast pixel alignment using translation only ---
def align_images_translation_fast(fixed_gray, moving_gray, max_shift=15):
    """
    Align moving_gray to fixed_gray using small translation search (fast).
    Returns x_shift, y_shift
    """
    best_score = -1
    best_dx, best_dy = 0, 0
    for dx in range(-max_shift, max_shift+1, 2):  # step=2 for speed
        for dy in range(-max_shift, max_shift+1, 2):
            M = np.float32([[1,0,dx],[0,1,dy]])
            shifted = cv2.warpAffine(moving_gray, M, (moving_gray.shape[1], moving_gray.shape[0]))
            res = cv2.matchTemplate(fixed_gray, shifted, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy
    return best_dx, best_dy


# --- Capture 10 frames simultaneously ---
def capture_frames(ir_cam, hd_cam, count=10):
    ir_frames = []
    hd_frames = []
    print("➡ Capturing frames...")
    led.on()
    for i in range(count):
        while True:
            ir = ir_cam.get_frame()
            hd = hd_cam.get_frame()
            if ir is not None and hd is not None:
                break
            time.sleep(0.05)
        # Resize IR to HD size roughly
        ir_resized = cv2.resize(ir, (hd.shape[1], hd.shape[0]))
        ir_frames.append(ir_resized)
        hd_frames.append(hd)
        # Save temporarily
        cv2.imwrite(os.path.join(temp_dir, f"ir_{i}.png"), ir_resized)
        cv2.imwrite(os.path.join(temp_dir, f"hd_{i}.png"), hd)
    led.off()
    print("➡ Frames captured")
    return ir_frames, hd_frames


# --- Process captured frames and create yellow overlay ---
def process_frames(ir_frames, hd_frames):
    for i, (ir, hd) in enumerate(zip(ir_frames, hd_frames)):
        gray_hd = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY)
        gray_ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        # Fast translation alignment
        dx, dy = align_images_translation_fast(gray_hd, gray_ir, max_shift=15)
        M = np.float32([[1,0,dx],[0,1,dy]])
        aligned_ir = cv2.warpAffine(ir, M, (hd.shape[1], hd.shape[0]))

        # Yellow overlay
        gray_aligned_ir = cv2.cvtColor(aligned_ir, cv2.COLOR_BGR2GRAY)
        overlay = np.zeros_like(hd)
        overlay[:, :, 0] = 0          # Blue
        overlay[:, :, 1] = gray_aligned_ir  # Green
        overlay[:, :, 2] = gray_aligned_ir  # Red

        # Blend with HD
        blended = cv2.addWeighted(hd, 0.7, overlay, 0.3, 0)

        # Save final blended image
        filename = os.path.join(output_dir, f"capture_{i+1}.jpg")
        cv2.imwrite(filename, blended)
        print(f"Saved {filename}")


# --- Main function ---
def capture_and_process(ir_cam, hd_cam):
    ir_frames, hd_frames = capture_frames(ir_cam, hd_cam, count=10)
    process_frames(ir_frames, hd_frames)


# --- Main Program ---
if __name__ == '__main__':
    # Rotate IR camera 180° to roughly align with HD
    ir_camera_thread = CameraThread(camera_id=0, name="IR Camera", rotate=True)
    hd_camera_thread = CameraThread(camera_id=1, name="HD Camera", rotate=False)

    ir_camera_thread.start()
    hd_camera_thread.start()

    print("System ready. Press button on GPIO27 to capture 10 yellow blended images.")

    try:
        while True:
            button_capture.wait_for_press()
            capture_and_process(ir_camera_thread, hd_camera_thread)

    except KeyboardInterrupt:
        print("Exiting program...")

    finally:
        ir_camera_thread.stop()
        hd_camera_thread.stop()
        cv2.destroyAllWindows()
