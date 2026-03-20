import time
import threading
import logging
from typing import Optional, Tuple
import numpy as np
import cv2

from picamera2 import Picamera2
from libcamera import controls

logger = logging.getLogger(__name__)

class CameraInterface:
    """
    Picamera2-based camera interface for Raspberry Pi 5 + Camera Module 3.
    Uses NV12 format to replicate GStreamer pipeline behavior.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
        mock_mode: bool = False,
        enable_af: bool = True,
    ):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.mock_mode = mock_mode
        self.enable_af = enable_af

        self.picam2: Optional[Picamera2] = None
        self.running = False

        self.frame_buffer: Optional[np.ndarray] = None
        self.timestamp_ns: int = 0

        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.roi_bounds: Optional[Tuple[int, int, int, int]] = None

    def start(self):
        logger.info(f"Starting CameraInterface (Picamera2 | NV12 | {self.width}x{self.height}@{self.framerate})")
        if self.mock_mode:
            logger.warning("CameraInterface running in MOCK MODE")
        else:
            self._init_camera()

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
        logger.info("CameraInterface stopped")

    def _init_camera(self):
        self.picam2 = Picamera2()

        # 1. CONFIGURE FOR NV12 (YUV420)
        # This matches 'libcamerasrc ! video/x-raw,format=NV12'
        config = self.picam2.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "NV12", 
            },
            controls={
                "FrameRate": self.framerate
            }
        )
        
        self.picam2.configure(config)
        self.picam2.start()

        # 2. APPLY CONTROLS
        # We start with neutral settings. The GStreamer pipeline likely used 'Auto' defaults.
        # We only dampen Saturation slightly for NoIR.
        self.picam2.set_controls({
            "Saturation": 0.7,        # Slight dampening for NoIR
            "Sharpness": 1.0,         # Standard sharpness
            "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
            "AwbMode": controls.AwbModeEnum.Auto, # Let the ISP decide the best white balance
            "Brightness": 0.0,
            "Contrast": 1.0,
        })
        
        logger.info("Picamera2 started in NV12 mode")

        if self.enable_af:
            self.enable_continuous_autofocus()

    def _capture_loop(self):
        frame_interval = 1.0 / max(self.framerate, 1)

        while self.running:
            if self.mock_mode:
                # Mock logic...
                time.sleep(frame_interval)
            else:
                try:
                    # 3. CAPTURE NV12 BUFFER
                    # This returns a generic byte array containing Y and UV planes
                    frame_nv12 = self.picam2.capture_array() 
                    ts = time.monotonic_ns()

                    # 4. PERFORM COLOR CONVERSION (The 'videoconvert' step)
                    # Convert NV12 (YUV) directly to BGR for OpenCV display
                    frame_bgr = cv2.cvtColor(frame_nv12, cv2.COLOR_YUV2BGR_NV12)
                    
                except Exception as e:
                    logger.warning(f"Camera capture failed: {e}")
                    continue

                # Apply ROI if present
                if self.roi_bounds is not None:
                    frame_bgr = self._apply_roi(frame_bgr)

                with self.lock:
                    self.frame_buffer = frame_bgr
                    self.timestamp_ns = ts

    # ... [Rest of your Public API methods: read, set_roi, AF controls, helpers remain the same] ...
    
    def read(self) -> Tuple[Optional[np.ndarray], int]:
        with self.lock:
            if self.frame_buffer is None:
                return None, 0
            return self.frame_buffer.copy(), self.timestamp_ns

    def set_roi(self, x: int, y: int, w: int, h: int):
        if w > 0 and h > 0:
            self.roi_bounds = (x, y, w, h)

    def enable_continuous_autofocus(self):
        if self.picam2:
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

    def trigger_autofocus(self):
        if self.picam2:
            return self.picam2.autofocus_cycle()

    def set_manual_focus(self, diopters: float):
        if self.picam2:
            self.picam2.set_controls({
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": float(diopters)
            })

    def set_af_window(self, x: int, y: int, w: int, h: int):
        if self.picam2:
            self.picam2.set_controls({
                "AfMetering": controls.AfMeteringEnum.Windows,
                "AfWindows": [(x, y, w, h)]
            })

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi_bounds
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        return frame[y:y+h, x:x+w]
