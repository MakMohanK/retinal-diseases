import cv2
import numpy as np
import threading
import time
from picamera2 import Picamera2

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
            print(f"[{self.name}] camera thread error:", e)
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


def draw_center_plus_on_frame(frame, color=(0,255,0), thickness=2, size=20):
    """
    Draw a + sign exactly in the center of the given frame (image).
    size = half‑length of arms.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    circle_radius=70
    circle_thickness=2
    # horizontal
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    # vertical
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)
    
    color=(255,0,0)
    cv2.circle(frame, (cx, cy), circle_radius, color, circle_thickness)

def display_side_by_side_with_centers(ir_frame, hd_frame):
    """
    Show IR and HD side by side, but draw separate + signs at
    the individual centers of each frame.
    """
    # Resize them to same height (or same size) if needed
    h1, w1 = ir_frame.shape[:2]
    h2, w2 = hd_frame.shape[:2]
    target_h = min(h1, h2)
    ir_resized = cv2.resize(ir_frame, (int(w1 * target_h / h1), target_h))
    hd_resized = cv2.resize(hd_frame, (int(w2 * target_h / h2), target_h))

    # Draw plus in each frame
    draw_center_plus_on_frame(ir_resized, color=(0,255,0), thickness=2, size=target_h//20)
    draw_center_plus_on_frame(hd_resized, color=(0,255,0), thickness=2, size=target_h//20)

    combined = np.hstack((ir_resized, hd_resized))
    cv2.imshow("IR + HD (with centers)", combined)
    cv2.waitKey(1)

# Example main loop usage
if __name__ == '__main__':
    ir_camera_thread = CameraThread(camera_id=0, name="IR Camera", rotate=True)
    hd_camera_thread = CameraThread(camera_id=1, name="HD Camera", rotate=False)
    ir_camera_thread.start()
    hd_camera_thread.start()

    try:
        while True:
            ir = ir_camera_thread.get_frame()
            hd = hd_camera_thread.get_frame()
            if ir is not None and hd is not None:
                display_side_by_side_with_centers(ir, hd)
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        ir_camera_thread.stop()
        hd_camera_thread.stop()
        cv2.destroyAllWindows()
