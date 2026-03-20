import time
import threading
import logging
from typing import Optional, Tuple
import numpy as np
import cv2

from picamera2 import Picamera2
from libcamera import controls

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Camera")

class CameraInterface:
    def __init__(self, width=640, height=480, framerate=30):
        self.width = width
        self.height = height
        self.framerate = framerate
        
        self.picam2: Optional[Picamera2] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.frame_buffer = None
        self.lock = threading.Lock()
        self.timestamp_ns = 0

    def start(self):
        logger.info("Initializing Picamera2...")
        self.picam2 = Picamera2()

        # --- FIX: Use 'YUV420' instead of 'NV12' ---
        # Picamera2 maps 'YUV420' to the correct underlying format (usually NV12 or I420)
        # depending on the platform (Pi 5 / Pi 4).
        config = self.picam2.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "YUV420", 
            },
            controls={
                "FrameRate": self.framerate
            }
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Apply Controls for NoIR Natural Look
        self.picam2.set_controls({
            "Saturation": 0.55,
            "Sharpness": 1.0,
            "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
            "AwbMode": controls.AwbModeEnum.Auto,
            "AeExposureMode": controls.AeExposureModeEnum.Short,
            "Brightness": 0.0,
            "Contrast": 1.0,
        })

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Camera thread started.")

    def stop(self):
        self.running = False
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                logger.error(f"Error closing: {e}")
            finally:
                self.picam2 = None
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Camera stopped.")

    def _capture_loop(self):
        while self.running and self.picam2:
            try:
                # Capture raw YUV data
                # On Pi 5, YUV420 usually comes back as I420 (Planar) or NV12 (Semi-Planar)
                # We check the stream configuration to know exactly what we got.
                frame = self.picam2.capture_array()
                ts = time.monotonic_ns()

                # Conversion logic
                # Picamera2 'YUV420' typically returns I420 (3 planes: Y, U, V)
                # OpenCV needs COLOR_YUV2BGR_I420 for that.
                # If it actually returned NV12, we would use COLOR_YUV2BGR_NV12.
                
                # Try I420 first (Standard for Picamera2 YUV420)
                try:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                except cv2.error:
                    # Fallback if the buffer shape suggests NV12
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)

                with self.lock:
                    self.frame_buffer = frame_bgr
                    self.timestamp_ns = ts
            
            except Exception as e:
                if self.running:
                    logger.warning(f"Capture error: {e}")
                break

    def read(self):
        with self.lock:
            if self.frame_buffer is None:
                return None, 0
            return self.frame_buffer.copy(), self.timestamp_ns

# --- MAIN ---
if __name__ == "__main__":
    cam = CameraInterface()
    
    try:
        cam.start()
        print("Press 'q' to quit.")
        while True:
            frame, ts = cam.read()
            if frame is not None:
                cv2.imshow("Natural Feed", frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
