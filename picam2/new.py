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
    Designed for MediaPipe / CV / IMU fusion workloads.
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

        # ROI: (x, y, w, h) in pixel coords
        self.roi_bounds: Optional[Tuple[int, int, int, int]] = None

    # ------------------------
    # Lifecycle
    # ------------------------

    def start(self):
        logger.info(
            f"Starting CameraInterface (Picamera2 | {self.width}x{self.height}@{self.framerate})"
        )

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

    # ------------------------
    # Camera setup
    # ------------------------

    def _init_camera(self):
        self.picam2 = Picamera2()

        config = self.picam2.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888",
            },
            controls={
                "FrameRate": self.framerate
            }
        )

        self.picam2.configure(config)
        self.picam2.start()

        logger.info("Picamera2 started")

        if self.enable_af:
            self.enable_continuous_autofocus()

    # ------------------------
    # Capture loop
    # ------------------------

    def _capture_loop(self):
        frame_interval = 1.0 / max(self.framerate, 1)

        while self.running:
            if self.mock_mode:
                frame = np.random.randint(
                    0, 255,
                    (self.height, self.width, 3),
                    dtype=np.uint8
                )
                ts = time.monotonic_ns()
                time.sleep(frame_interval)
            else:
                try:
                    frame = self.picam2.capture_array()  # RGB
                    ts = time.monotonic_ns()
                except Exception as e:
                    logger.warning(f"Camera capture failed: {e}")
                    continue

            # Apply ROI if present
            if self.roi_bounds is not None:
                frame = self._apply_roi(frame)

            # Convert to BGR for OpenCV / MediaPipe drawing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with self.lock:
                self.frame_buffer = frame_bgr
                self.timestamp_ns = ts

    # ------------------------
    # Public API
    # ------------------------

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """
        Non-blocking read of latest frame.
        Returns (frame, timestamp_ns)
        """
        with self.lock:
            if self.frame_buffer is None:
                return None, 0
            return self.frame_buffer.copy(), self.timestamp_ns

    def set_roi(self, x: int, y: int, w: int, h: int):
        if w > 0 and h > 0:
            self.roi_bounds = (x, y, w, h)
        else:
            logger.warning("Invalid ROI ignored")

    # ------------------------
    # Autofocus controls
    # ------------------------

    def enable_continuous_autofocus(self):
        if not self.picam2:
            return

        self.picam2.set_controls({
            "AfMode": controls.AfModeEnum.Continuous
        })
        logger.info("Continuous autofocus enabled")

    def trigger_autofocus(self) -> bool:
        if not self.picam2:
            return False
        return self.picam2.autofocus_cycle()

    def set_manual_focus(self, diopters: float):
        if not self.picam2:
            return
        self.picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": float(diopters)
        })

    def set_af_window(self, x: int, y: int, w: int, h: int):
        """
        Set autofocus window (useful for hand ROI).
        """
        if not self.picam2:
            return
        self.picam2.set_controls({
            "AfMetering": controls.AfMeteringEnum.Windows,
            "AfWindows": [(x, y, w, h)]
        })

    # ------------------------
    # Helpers
    # ------------------------

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi_bounds
        fh, fw = frame.shape[:2]

        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))

        return frame[y:y+h, x:x+w]
