import time
import threading
import logging
import shutil
import subprocess
from collections import deque
from typing import Optional, Tuple
import numpy as np
import cv2

# Only import Picamera2 if we are on the Pi (avoids errors on PC)
try:
    from picamera2 import Picamera2
    from libcamera import controls
    HAS_PICAM2 = True
except ImportError:
    HAS_PICAM2 = False

logger = logging.getLogger("CameraInterface")

class CameraInterface:
    def __init__(
        self, 
        width: int = 640, 
        height: int = 480, 
        framerate: int = 60,  # Default to 60 for anti-wobble
        mock_mode: bool = False,
        use_gstreamer: bool = False # If True, use OpenCV+GStreamer (no picamera2 Python dependency)
    ):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.mock_mode = mock_mode
        self.use_gstreamer = use_gstreamer
        
        # Internal state
        self.picam2 = None
        self.cap = None
        self.proc: Optional[subprocess.Popen] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._last_frame_ns = 0
        
        self.frame_buffer = None
        self.lock = threading.Lock()
        self.timestamp_ns = 0

    def start(self):
        """Starts the camera thread (or mock thread)."""
        logger.info(
            f"Starting Camera: {self.width}x{self.height} @ {self.framerate}fps | "
            f"Mock: {self.mock_mode} | GStreamer: {self.use_gstreamer}"
        )

        if self.mock_mode:
            self.running = True
            self.thread = threading.Thread(target=self._mock_capture_loop, daemon=True)
            self.thread.start()
            return

        if self.use_gstreamer:
            self.running = True
            self.thread = threading.Thread(target=self._gstreamer_capture_loop, daemon=True)
            self.thread.start()
            logger.info("Camera thread started (libcamera pipeline).")
            return

        if not HAS_PICAM2:
            logger.error("Picamera2 not found and GStreamer disabled; forcing Mock Mode.")
            self.mock_mode = True
            self.start()
            return

        # Initialize Picamera2
        logger.info("Initializing Picamera2 (YUV420 + NoIR Fixes)...")
        self.picam2 = Picamera2()

        # 1. Configure for YUV420 (High Performance / Native Format)
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

        # 2. Apply "Natural" + "Stable" Controls for NoIR Camera
        # - AeExposureMode='Short' reduces motion blur/wobble
        # - Saturation=0.55 fixes the "neon" colors from IR light
        self.picam2.set_controls({
            "AeExposureMode": controls.AeExposureModeEnum.Short,
            "Saturation": 0.55,
            "Sharpness": 1.0,
            "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
            "AwbMode": controls.AwbModeEnum.Auto,
            "Brightness": 0.0,
            "Contrast": 1.0,
        })

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Camera thread started.")

    def stop(self):
        """Stops the camera safely."""
        self.running = False
        
        # Stop Picamera2 engine
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                logger.error(f"Error closing camera: {e}")
            finally:
                self.picam2 = None

        # Stop OpenCV capture
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing VideoCapture: {e}")
            finally:
                self.cap = None

        # Stop libcamera CLI capture
        if self.proc:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except Exception as e:
                logger.error(f"Error stopping libcamera process: {e}")
            finally:
                self.proc = None
        
        # Join thread
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        logger.info("Camera stopped.")

    def _capture_loop(self):
        """Main loop for Real Camera (Picamera2)."""
        while self.running and self.picam2:
            try:
                # Capture raw YUV data (Blocking call)
                frame = self.picam2.capture_array()
                ts = time.monotonic_ns()

                # Color Conversion: YUV420 -> BGR
                # Try I420 (Standard Planar) first, fallback to NV12 if needed
                try:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                except (cv2.error, Exception):
                    try:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                    except Exception:
                        continue # Skip frame if conversion completely fails

                with self.lock:
                    self.frame_buffer = frame_bgr
                    self.timestamp_ns = ts
            
            except Exception as e:
                if self.running:
                    logger.warning(f"Capture error: {e}")
                break

    def _mock_capture_loop(self):
        """Generates fake noise frames for testing without hardware."""
        interval = 1.0 / self.framerate
        while self.running:
            # Generate random noise frame
            frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
            ts = time.monotonic_ns()
            
            with self.lock:
                self.frame_buffer = frame
                self.timestamp_ns = ts
            
            time.sleep(interval)

    def _gstreamer_capture_loop(self):
        """Main loop for Real Camera.

        Historically this used OpenCV+GStreamer (`libcamerasrc`), but many pip OpenCV builds on Pi
        are compiled with `GStreamer: NO`. In that case we fall back to `rpicam-vid`/`libcamera-vid`
        MJPEG streaming and decode frames in Python.
        """

        if self._opencv_has_gstreamer():
            self._opencv_gstreamer_capture_loop()
        else:
            self._libcamera_cli_mjpeg_capture_loop()

    def _opencv_has_gstreamer(self) -> bool:
        try:
            info = cv2.getBuildInformation()
        except Exception:
            return False
        # Look for the explicit build flag line.
        for line in info.splitlines():
            if "GStreamer:" in line:
                return "YES" in line
        return False

    def _opencv_gstreamer_capture_loop(self):
        """OpenCV+GStreamer capture loop (only if OpenCV is built with GStreamer support)."""
        def build_pipeline(fps: int) -> str:
            return (
                "libcamerasrc ! "
                f"video/x-raw,format=NV12,width={self.width},height={self.height},framerate={fps}/1 ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )

        # Try requested FPS first; if that fails, retry at 30fps (known-good default).
        requested_fps = int(self.framerate)
        for fps in (requested_fps, 30):
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            pipeline = build_pipeline(fps)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                if fps != requested_fps:
                    logger.info(f"Opened camera via GStreamer at fallback fps={fps} (requested {requested_fps}).")
                break

        if not self.cap or not self.cap.isOpened():
            logger.error("Failed to open camera via GStreamer pipeline; falling back to Mock Mode.")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.mock_mode = True
            self._mock_capture_loop()
            return

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.running:
                    logger.warning("Frame grab failed (GStreamer).")
                time.sleep(0.01)
                continue

            ts = time.monotonic_ns()
            with self.lock:
                self.frame_buffer = frame
                self.timestamp_ns = ts

    def _libcamera_cli_mjpeg_capture_loop(self):
        """Capture loop using rpicam/libcamera CLI MJPEG stream.

        This avoids the need for picamera2/libcamera Python bindings and avoids OpenCV-GStreamer.
        """
        cmd = None
        if shutil.which("rpicam-vid"):
            cmd = "rpicam-vid"
        elif shutil.which("libcamera-vid"):
            cmd = "libcamera-vid"

        if not cmd:
            logger.error("Neither rpicam-vid nor libcamera-vid found; falling back to Mock Mode.")
            self.mock_mode = True
            self._mock_capture_loop()
            return

        requested_fps = int(self.framerate) if self.framerate else 0
        if requested_fps <= 0:
            requested_fps = 30

        # If the camera process exits or stalls, restart it so the UI doesn't freeze on the last frame.
        fps_candidates = (requested_fps, 30)
        stderr_tail = deque(maxlen=50)

        def start_process(fps: int) -> bool:
            args = [
                cmd,
                "--codec", "mjpeg",
                "--width", str(self.width),
                "--height", str(self.height),
                "--framerate", str(fps),
                "--nopreview",
                "-o", "-",
            ]

            try:
                self.proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                return True
            except Exception as e:
                logger.error(f"Failed to start {cmd}: {e}")
                self.proc = None
                return False

        def read_stderr_nonblocking():
            # Best-effort: read available stderr lines to help debug exits.
            if not self.proc or self.proc.stderr is None:
                return
            try:
                while True:
                    line = self.proc.stderr.readline()
                    if not line:
                        break
                    try:
                        stderr_tail.append(line.decode(errors="replace").rstrip())
                    except Exception:
                        stderr_tail.append(str(line))
            except Exception:
                return

        # MJPEG to stdout; parse JPEG boundaries and decode.
        soi = b"\xff\xd8"  # start of image
        eoi = b"\xff\xd9"  # end of image

        while self.running:
            opened = False
            chosen_fps = requested_fps
            for fps in fps_candidates:
                if start_process(fps):
                    opened = True
                    chosen_fps = fps
                    if fps != requested_fps:
                        logger.info(f"Started {cmd} at fallback fps={fps} (requested {requested_fps}).")
                    break

            if not opened or not self.proc or self.proc.stdout is None:
                logger.error("Unable to start libcamera capture process; falling back to Mock Mode.")
                self.mock_mode = True
                self._mock_capture_loop()
                return

            buffer = bytearray()
            last_good_frame_ns = time.monotonic_ns()

            while self.running and self.proc and self.proc.poll() is None:
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    # Capture process may be stalled; give it a moment.
                    read_stderr_nonblocking()
                    now = time.monotonic_ns()
                    if now - last_good_frame_ns > 2_000_000_000:  # 2s without a decoded frame
                        logger.warning(
                            f"No camera frames decoded for >2s (fps={chosen_fps}); restarting {cmd}."
                        )
                        break
                    time.sleep(0.005)
                    continue

                buffer.extend(chunk)

                # Extract frames as long as we have complete JPEGs.
                while True:
                    start = buffer.find(soi)
                    if start == -1:
                        if len(buffer) > 2_000_000:
                            buffer.clear()
                        break

                    end = buffer.find(eoi, start + 2)
                    if end == -1:
                        if start > 0:
                            del buffer[:start]
                        break

                    jpeg = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]

                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    ts = time.monotonic_ns()
                    last_good_frame_ns = ts
                    self._last_frame_ns = ts
                    with self.lock:
                        self.frame_buffer = frame
                        self.timestamp_ns = ts

            # Process ended or stalled; capture stderr tail for diagnosis.
            read_stderr_nonblocking()
            exit_code = self.proc.poll() if self.proc else None
            unexpected = self.running
            if unexpected and exit_code is not None and exit_code != 0:
                logger.warning(f"{cmd} exited with code {exit_code}; restarting.")
                if stderr_tail:
                    logger.warning("Last camera stderr lines:\n" + "\n".join(list(stderr_tail)[-10:]))
            elif stderr_tail:
                logger.debug("Last camera stderr lines:\n" + "\n".join(list(stderr_tail)[-10:]))

            # Clean up process before restart.
            if self.proc:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=1.0)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                self.proc = None

            time.sleep(0.2)

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """Returns the latest frame (BGR) and timestamp."""
        with self.lock:
            if self.frame_buffer is None:
                return None, 0
            # Return a copy to avoid threading race conditions
            return self.frame_buffer.copy(), self.timestamp_ns
