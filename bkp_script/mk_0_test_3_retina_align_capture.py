#!/usr/bin/env python3
"""
retina_align_capture.py

- LED on GPIO17 is turned on.
- Continuous center-circle preview of IR and HD cameras shown.
- Press button on GPIO27 to capture 10 frames from each camera.
- Align IR->HD using ECC homography, ORB fallback if ECC fails.
- Fine adjust by pupil center.
- Discard frames if alignment or pupil detection fails.
- Convert pink -> yellow.
- Save aligned/processed pairs to output/ir and output/hd.
"""

import os
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageDraw
from picamera2 import Picamera2
from gpiozero import LED, Button
from datetime import datetime

# ---------------------------
# FOLDERS + GPIO
# ---------------------------
temp_dir = "temp"
output_dir = "output"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "ir"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "hd"), exist_ok=True)

led = LED(17)
button = Button(27, pull_up=False)   # use pull_up=False if you wired pull-down
led.on()                             # keep LED on by default (user requested)

# ---------------------------
# CAMERA THREAD
# ---------------------------
class CameraThread(threading.Thread):
    def __init__(self, camera_id, name, resolution=(1280, 720), rotate=False):
        super().__init__(daemon=True)
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
                f = self.picam2.capture_array()
                if self.rotate:
                    f = cv2.rotate(f, cv2.ROTATE_180)
                self.frame = f
        except Exception as e:
            print(f"[{self.name}] camera error:", e)
        finally:
            try:
                self.picam2.stop()
            except Exception:
                pass

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        try:
            self.join(timeout=2)
        except Exception:
            pass
        try:
            self.picam2.stop()
        except Exception:
            pass

# ---------------------------
# MASK / CROP / DRAW UTILITIES
# ---------------------------
def create_circular_mask_wh(w, h, radius=None):
    if radius is None:
        radius = min(w, h) // 3     # larger radius for retina
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    return np.array(mask), (cx, cy, radius)

def apply_mask_crop(img, mask):
    # Keep same HxW but zero outside circle, then return tight crop around circle
    masked = cv2.bitwise_and(img, img, mask=mask)
    # compute bounding box of mask (tight square around center)
    ys, xs = np.where(mask == 255)
    if ys.size == 0 or xs.size == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    return masked[y1:y2+1, x1:x2+1]

def draw_center_guides(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size = min(w, h) // 10
    cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), min(w, h)//6, (255, 0, 0), 2)

# ---------------------------
# ALIGNMENT: ECC + ORB fallback
# ---------------------------
def align_ecc_color(fixed_color, moving_color):
    """
    Align `moving_color` to `fixed_color` using ECC on grayscale, returns warp_matrix or None.
    """
    try:
        fixed_gray = cv2.cvtColor(fixed_color, cv2.COLOR_BGR2GRAY)
        moving_gray = cv2.cvtColor(moving_color, cv2.COLOR_BGR2GRAY)
        # histogram equalization to help ECC
        fixed_gray = cv2.equalizeHist(fixed_gray)
        moving_gray = cv2.equalizeHist(moving_gray)

        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-7)
        cc, warp_matrix = cv2.findTransformECC(fixed_gray, moving_gray, warp_matrix, warp_mode, criteria)
        # apply warp to color
        h, w = fixed_color.shape[:2]
        aligned = cv2.warpPerspective(moving_color, warp_matrix, (w, h),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return warp_matrix, aligned
    except cv2.error as e:
        # ECC often fails on low-feature or heavy illumination differences
        return None, None

def align_orb_ransac(fixed_color, moving_color, min_matches=10):
    """
    ORB + RANSAC homography fallback. Returns (H, aligned_color) or (None, None).
    """
    fixed_gray = cv2.cvtColor(fixed_color, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving_color, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(fixed_gray, None)
    kp2, des2 = orb.detectAndCompute(moving_gray, None)
    if des1 is None or des2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    # ratio test
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, None
    h, w = fixed_color.shape[:2]
    aligned = cv2.warpPerspective(moving_color, H, (w, h), flags=cv2.INTER_LINEAR)
    return H, aligned

# ---------------------------
# PUPIL / EYE PRESENCE CHECK
# ---------------------------
def detect_pupil_center(img_color):
    """
    Return (cx, cy) if pupil-like dark circle present near center, else None.
    Uses threshold + HoughCircles fallback.
    """
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    # adaptive threshold to highlight dark pupil
    _, thresh = cv2.threshold(gray, 48, 255, cv2.THRESH_BINARY_INV)
    # morphological opening to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # moments
    m = cv2.moments(thresh)
    if m["m00"] > 100:  # some area check
        cx = int(m["m10"]/m["m00"])
        cy = int(m["m01"]/m["m00"])
        return (cx, cy)
    # fallback to Hough circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=50,
                               param1=50, param2=30, minRadius=8, maxRadius=80)
    if circles is not None:
        c = circles[0][0]
        return (int(c[0]), int(c[1]))
    return None

# ---------------------------
# PINK → YELLOW
# ---------------------------
def pink_to_yellow_bgr(img_bgr):
    """
    Replace pink hues (HSV range) with yellow color. Return modified copy.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Pink ranges (approx) - tune if needed
    lower1 = np.array([140, 40, 30])   # magenta/pink lower
    upper1 = np.array([170, 255, 255]) # magenta/pink upper
    lower2 = np.array([300//2, 30, 30])  # some pinks map >180 scaled; keep safest above
    upper2 = np.array([360//2, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    # combine masks (we'll only use mask1 but code kept flexible)
    mask = mask1
    # set yellow in BGR (0,255,255)
    out = img_bgr.copy()
    out[mask > 0] = (0, 255, 255)
    return out

# ---------------------------
# CAPTURE 10 FRAMES (IR+HD)
# ---------------------------
def capture_frames_pairwise(ir_cam, hd_cam, count=10, wait_for_frames=True):
    ir_list = []
    hd_list = []
    print("➡ Starting capture of {} frames...".format(count))
    for i in range(count):
        # wait until both cameras have frames
        tries = 0
        while True:
            ir = ir_cam.get_frame()
            hd = hd_cam.get_frame()
            if ir is not None and hd is not None:
                break
            tries += 1
            if tries > 80:
                print("⚠️ Timeout waiting for frames")
                break
            time.sleep(0.03)
        if ir is None or hd is None:
            continue
        # resize IR to HD size (makes later alignment easier)
        hd_h, hd_w = hd.shape[:2]
        ir_resized = cv2.resize(ir, (hd_w, hd_h))
        ir_list.append(ir_resized)
        hd_list.append(hd)
        # show quick preview
        preview = np.hstack((cv2.resize(hd, (640, 360)), cv2.resize(ir_resized, (640, 360))))
        draw_center_guides(preview)
        cv2.imshow("Preview (HD | IR)", preview)
        cv2.waitKey(1)
        # save temp for debugging
        cv2.imwrite(os.path.join(temp_dir, f"hd_{i+1}.png"), hd)
        cv2.imwrite(os.path.join(temp_dir, f"ir_{i+1}.png"), ir_resized)
        time.sleep(0.15)
    print("✅ Captured frames: {} pairs".format(len(ir_list)))
    return ir_list, hd_list

# ---------------------------
# PROCESS: ALIGN, CHECK, SAVE
# ---------------------------
def process_and_save(ir_list, hd_list):
    saved_count = 0
    for idx, (ir, hd) in enumerate(zip(ir_list, hd_list)):
        # create circular mask for hd dims
        h, w = hd.shape[:2]
        mask_full, (cx, cy, radius) = create_circular_mask_wh(w, h, radius=min(w,h)//3)
        hd_crop = apply_mask_crop(hd, mask_full)
        ir_crop = apply_mask_crop(ir, mask_full)
        if hd_crop is None or ir_crop is None:
            print(f"⚠️ mask crop failed for pair {idx+1}, skipping")
            continue

        # Try ECC first
        warp, aligned_ir = align_ecc_color(hd_crop, ir_crop)
        if aligned_ir is None:
            # fallback to ORB + RANSAC
            H, aligned_ir = align_orb_ransac(hd_crop, ir_crop)
            if aligned_ir is None:
                print(f"⚠️ Alignment failed for pair {idx+1}, skipping")
                continue

        # fine adjust by pupil center translation
        p_ir = detect_pupil_center(aligned_ir)
        p_hd = detect_pupil_center(hd_crop)
        if (p_ir is None) or (p_hd is None):
            print(f"⚠️ Pupil not detected in pair {idx+1} (ir:{p_ir}, hd:{p_hd}), skipping")
            continue
        # compute translation
        dx = p_hd[0] - p_ir[0]
        dy = p_hd[1] - p_ir[1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_ir = cv2.warpAffine(aligned_ir, M, (aligned_ir.shape[1], aligned_ir.shape[0]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Convert pink -> yellow on both
        aligned_ir_col = pink_to_yellow_bgr(aligned_ir)
        hd_col = pink_to_yellow_bgr(hd_crop)

        # final combined check: compute normalized cross-correlation for alignment quality
        try:
            grayA = cv2.cvtColor(aligned_ir_col, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(hd_col, cv2.COLOR_BGR2GRAY)
            # crop central window to compare
            ch, cw = grayA.shape[:2]
            win = (slice(ch//4, 3*ch//4), slice(cw//4, 3*cw//4))
            corr = np.corrcoef(grayA[win].ravel(), grayB[win].ravel())[0,1]
        except Exception:
            corr = 0
        if corr < 0.25:
            print(f"⚠️ Low alignment correlation ({corr:.3f}) for pair {idx+1}, skipping")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ir_name = os.path.join(output_dir, "ir", f"IR_{timestamp}_{idx+1}.png")
        hd_name = os.path.join(output_dir, "hd", f"HD_{timestamp}_{idx+1}.png")
        cv2.imwrite(ir_name, aligned_ir_col)
        cv2.imwrite(hd_name, hd_col)
        saved_count += 1
        print(f"✅ Saved pair #{saved_count}: {ir_name} | {hd_name}")

    print(f"✅ Processing complete. {saved_count} pairs saved.")

# ---------------------------
# MAIN: preview + button handler
# ---------------------------
def main_loop(ir_cam, hd_cam):
    print("📷 Retinal capture ready. Align eye in the center circle.")
    print("Press the physical button (GPIO27) to capture 10 images.")
    mask_full, _ = None, None

    try:
        while True:
            ir = ir_cam.get_frame()
            hd = hd_cam.get_frame()
            if ir is None or hd is None:
                time.sleep(0.02)
                continue

            # ensure mask dims match current frames
            if mask_full is None or mask_full.shape[1] != hd.shape[1]:
                mask_full, _ = create_circular_mask_wh(hd.shape[1], hd.shape[0], radius=min(hd.shape[1], hd.shape[0])//3)

            # apply mask and crop for preview
            hd_preview = apply_mask_crop(hd, mask_full)
            ir_preview = apply_mask_crop(ir, mask_full)
            if hd_preview is None or ir_preview is None:
                time.sleep(0.02)
                continue

            # Show side-by-side preview (center crops)
            ph = 360
            pw = int(ph * hd_preview.shape[1] / hd_preview.shape[0])
            side = np.hstack((cv2.resize(hd_preview, (pw, ph)), cv2.resize(ir_preview, (pw, ph))))
            draw_center_guides(side)
            cv2.imshow("Eye Alignment (HD | IR) - press 'q' to quit", side)
            key = cv2.waitKey(1) & 0xFF

            # Button press triggers capture
            if button.is_pressed:
                print("🔴 Button pressed: capturing sequence...")
                led.on()        # give indication
                # short debounce
                time.sleep(0.15)
                ir_list, hd_list = capture_frames_pairwise(ir_cam, hd_cam, count=10)
                process_and_save(ir_list, hd_list)
                print("🟢 Capture sequence finished. LED remains ON.")
                # brief pause before allowing next capture
                time.sleep(0.5)

            if key == ord('q'):
                print("Exiting by user key 'q'.")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # create and start camera threads (ID ordering may need to be adjusted on your system)
    ir_camera_thread = CameraThread(camera_id=0, name="IR Camera", rotate=True, resolution=(1280,720))
    hd_camera_thread = CameraThread(camera_id=1, name="HD Camera", rotate=False, resolution=(1280,720))
    ir_camera_thread.start()
    hd_camera_thread.start()

    try:
        main_loop(ir_camera_thread, hd_camera_thread)
    finally:
        print("Cleaning up...")
        ir_camera_thread.stop()
        hd_camera_thread.stop()
        led.off()
        print("Done.")

