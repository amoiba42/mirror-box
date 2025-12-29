import cv2
import time
import threading
import logging
from typing import Optional, Tuple, Union
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

class CameraInterface:
    """
    Abstracts camera hardware. 
    Supports:
    1. Real Mock: Reads from a video file or generic webcam (Desktop dev).
    2. Real Hardware: Placeholder for RPi5 LibCamera/Picamera2 integration.
    """

    def __init__(self, source: Union[int, str] = 0, width: int = 640, height: int = 480, mock_mode: bool = False):
        """
        Args:
            source: Camera index (0) or path to video file.
            width: Target capture width.
            height: Target capture height.
            mock_mode: If True, generates synthetic frames if source fails, or loops video.
        """
        self.source = source
        self.width = width
        self.height = height
        self.mock_mode = mock_mode
        
        self.cap = None
        self.frame_buffer = None
        self.timestamp_buffer = 0
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.fps_start_time = 0
        self.frame_count = 0

        # ROI (Region of Interest) state for future optimization
        self.roi_bounds: Optional[Tuple[int, int, int, int]] = None # x, y, w, h

    def start(self):
        """Initializes camera and starts the capture thread."""
        logger.info(f"Starting CameraInterface (Source: {self.source}, Mock: {self.mock_mode})")
        
        self.cap = cv2.VideoCapture(self.source)
        
        # Set resolution (may not work on all video files, works on webcams)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Pi 5 Note: Set FPS to 30 explicitly later for CSI cameras
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            logger.error("Failed to open camera source.")
            if not self.mock_mode:
                raise RuntimeError("Camera source could not be opened.")
            logger.info("Falling back to pure synthetic mock generation.")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Threaded worker to keep the latest frame available."""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    if isinstance(self.source, str) and self.mock_mode:
                        # Loop video file
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning("Frame capture failed.")
                        pass
                else:
                    # Update buffer with latest frame
                    with self.lock:
                        self.frame_buffer = frame
                        # Use monotonic clock for drift-free timing
                        self.timestamp_buffer = time.monotonic_ns()
            
            elif self.mock_mode:
                # Generate synthetic noise frame for testing without camera
                dummy_frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
                time.sleep(0.033) # Approx 30 FPS
                with self.lock:
                    self.frame_buffer = dummy_frame
                    self.timestamp_buffer = time.monotonic_ns()
            
            # Simple yield to prevent CPU hogging
            time.sleep(0.001)

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """
        Returns the latest available frame and its capture timestamp (ns).
        Non-blocking.
        """
        with self.lock:
            if self.frame_buffer is not None:
                return self.frame_buffer.copy(), self.timestamp_buffer
            return None, 0

    def stop(self):
        """Stops the thread and releases resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        logger.info("Camera released.")

    def set_roi(self, x, y, w, h):
        """Set a region of interest to crop subsequent frames (Performance optimization)."""
        self.roi_bounds = (x, y, w, h)