import os
import cv2
import numpy as np
import threading
import time
from picamera2 import Picamera2
from gpiozero import LED, Button
from datetime import datetime

# ========================================
# GPIO SETUP
# ========================================
led = LED(17)
button = Button(27, pull_up=False)
led.on()  # Turn on LED

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
        try:
            self.picam2.start()
            self.running = True
            while self.running:
                frame = self.picam2.capture_array()
                if self.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.frame = frame
        except Exception as e:
            print(f"[{self.name}] Error:", e)
        finally:
            try:
                self.picam2.stop()
            except Exception:
                pass

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.join(timeout=2)
        try:
            self.picam2.stop()
        except Exception:
            pass

# ========================================
# IMAGE UTILITIES
# ========================================
def crop_center_circle(frame, radius=150):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    x1, y1 = cx - radius, cy - radius
    x2, y2 = cx + radius, cy + radius
    return result[y1:y2, x1:x2]

def align_images(img1, img2):
    """Use ORB feature matching for alignment."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return None
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None
    aligned = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    return aligned

def replace_pink_with_yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Pink range
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    frame[mask > 0] = [0, 255, 255]  # Yellow in BGR
    return frame

# ========================================
# MAIN CAPTURE FUNCTION
# ========================================
def capture_images(ir_cam, hd_cam, output_dir="output", num_images=10):
    os.makedirs(f"{output_dir}/ir", exist_ok=True)
    os.makedirs(f"{output_dir}/hd", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_images):
        ir = ir_cam.get_frame()
        hd = hd_cam.get_frame()
        if ir is None or hd is None:
            continue

        ir_crop = crop_center_circle(ir)
        hd_crop = crop_center_circle(hd)

        aligned_ir = align_images(ir_crop, hd_crop)
        if aligned_ir is None:
            print(f"⚠️ Alignment failed for image {i+1}, skipping...")
            continue

        # Pink → Yellow
        ir_final = replace_pink_with_yellow(aligned_ir)
        hd_final = replace_pink_with_yellow(hd_crop)

        # Save images
        cv2.imwrite(f"{output_dir}/ir/IR_{timestamp}_{i+1}.jpg", ir_final)
        cv2.imwrite(f"{output_dir}/hd/HD_{timestamp}_{i+1}.jpg", hd_final)
        print(f"✅ Saved aligned image pair {i+1}")

        time.sleep(0.3)

# ========================================
# DISPLAY LOOP
# ========================================
def main():
    ir_camera_thread = CameraThread(camera_id=0, name="IR Camera", rotate=True)
    hd_camera_thread = CameraThread(camera_id=1, name="HD Camera", rotate=False)
    ir_camera_thread.start()
    hd_camera_thread.start()

    print("📷 Retinal Capture System Ready.")
    print("➡ Align your eye in center circle.")
    print("➡ Press button on GPIO27 to capture 10 images.")

    try:
        while True:
            ir = ir_camera_thread.get_frame()
            hd = hd_camera_thread.get_frame()
            if ir is not None and hd is not None:
                ir_crop = crop_center_circle(ir)
                hd_crop = crop_center_circle(hd)
                combined = np.hstack((ir_crop, hd_crop))
                cv2.imshow("Eye Alignment View (IR + HD)", combined)
                key = cv2.waitKey(1) & 0xFF

                # Capture on button press
                if button.is_pressed:
                    print("📸 Button pressed → capturing images...")
                    capture_images(ir_camera_thread, hd_camera_thread)
                    print("✅ Capture complete.")

                if key == ord('q'):
                    break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        ir_camera_thread.stop()
        hd_camera_thread.stop()
        led.off()
        cv2.destroyAllWindows()
        print("🛑 System exited cleanly.")

if __name__ == "__main__":
    main()
