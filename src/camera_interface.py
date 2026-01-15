import cv2
import time
import threading
import logging
from typing import Optional, Tuple, Union
import numpy as np

# Try to import picamera2 for Raspberry Pi CSI support
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

class CameraInterface:
    """
    Abstracts camera hardware. 
    Supports:
    1. Real Mock: Reads from a video file or generic webcam (Desktop dev).
    2. Real Hardware: Placeholder for RPi5 LibCamera/Picamera2 integration.
    """

    def __init__(self, source: Union[int, str] = 0, width: int = 640, height: int = 480, mock_mode: bool = False, use_csi: bool = False, use_gstreamer: bool = False):
        """
        Args:
            source: Camera index (0) or path to video file.
            width: Target capture width.
            height: Target capture height.
            mock_mode: If True, generates synthetic frames if source fails, or loops video.
            use_csi: If True, use Raspberry Pi CSI camera via picamera2.
            use_gstreamer: If True, use a GStreamer pipeline for camera capture.
        """
        self.source = source
        self.width = width
        self.height = height
        self.mock_mode = mock_mode
        self.use_csi = use_csi
        self.use_gstreamer = use_gstreamer
        
        self.cap = None
        self.picam2 = None
        self.frame_buffer = None
        self.timestamp_buffer = 0
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        
        # ROI (Region of Interest) state
        self.roi_bounds: Optional[Tuple[int, int, int, int]] = None # x, y, w, h
    def start(self):
        logger.info(
            f"Starting CameraInterface "
            f"(Source: {self.source}, Mock: {self.mock_mode}, "
            f"CSI: {self.use_csi}, GStreamer: {self.use_gstreamer})"
        )

        # Enforce valid combinations
        if self.use_gstreamer:
            self.use_csi = True

            pipeline = (
                "libcamerasrc ! "
                "video/x-raw,format=NV12,"
                f"width={self.width},height={self.height},framerate=30/1 ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true"
            )

            logger.info(f"GStreamer pipeline: {pipeline}")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                logger.error("Failed to open GStreamer pipeline.")
                if not self.mock_mode:
                    raise RuntimeError("GStreamer pipeline could not be opened.")
                self.cap = None

        elif self.use_csi and PICAMERA2_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                logger.info("Picamera2 CSI camera initialized")
            except Exception as e:
                logger.error(f"Failed to initialize CSI camera: {e}")
                if not self.mock_mode:
                    raise
                self.picam2 = None

        elif not self.mock_mode:
            # Plain OpenCV fallback (USB webcam)
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not self.cap.isOpened():
                raise RuntimeError("Camera source could not be opened.")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Threaded worker to keep the latest frame available."""
        while self.running:
            frame = None
            
            # --- STRATEGY 1: Raspberry Pi CSI Camera ---
            if self.picam2:
                # FIX: Removed time.sleep(0.5) which was killing FPS
                try:
                    # Capture array (RGB because of config)
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        # Convert RGB -> BGR for OpenCV
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"CSI camera capture failed: {e}")
            
            # --- STRATEGY 2: USB Camera / Video File ---
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    if isinstance(self.source, str) and self.mock_mode:
                        # Loop video file
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning("Frame capture failed.")
                        frame = None
            
            # --- STRATEGY 3: Synthetic Mock (Fallback) ---
            elif self.mock_mode:
                dummy_frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
                time.sleep(0.033) # Limit mock generation to ~30 FPS
                frame = dummy_frame

            # --- Post-Processing & Storage ---
            if frame is not None:
                # FIX: Apply ROI cropping if set
                if self.roi_bounds:
                    x, y, w, h = self.roi_bounds
                    # Ensure ROI is within frame bounds
                    frame_h, frame_w = frame.shape[:2]
                    x = max(0, min(x, frame_w))
                    y = max(0, min(y, frame_h))
                    w = max(1, min(w, frame_w - x))
                    h = max(1, min(h, frame_h - y))
                    frame = frame[y:y+h, x:x+w]

                # Update shared buffer
                with self.lock:
                    self.frame_buffer = frame
                    self.timestamp_buffer = time.monotonic_ns()
            
            # Small yield to prevent 100% CPU usage in tight loops
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
            
        if self.picam2:
            self.picam2.stop()
            self.picam2.close() # Good practice to close resources
        
        logger.info("Camera released.")

    def set_roi(self, x, y, w, h):
        """Set a region of interest to crop subsequent frames."""
        # Sanity check to prevent crashes with negative values
        if w > 0 and h > 0 and x >= 0 and y >= 0:
            self.roi_bounds = (x, y, w, h)
        else:
            logger.warning(f"Invalid ROI parameters ignored: {x, y, w, h}")