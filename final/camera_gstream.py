import time
import threading
import subprocess
import shutil
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _opencv_has_gstreamer() -> bool:
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    for line in info.splitlines():
        if "GStreamer:" in line:
            return "YES" in line
    return False


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    framerate: int = 30
    camera_index: int = 0


class Pi5GStreamerCamera:
    """CSI camera capture via OpenCV + GStreamer using `libcamerasrc`.
    """

    def __init__(self, config: CameraConfig = CameraConfig()):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None

    def _pipeline(self) -> str:
        # Note: Broadcom GStreamer libcamerasrc uses camera-name or other props for multi-cam 
        # but the CLI frontend rpicam-vid is much simpler. 
        # For now we use the index as a hint, though many libcamerasrc builds ignore it.
        return (
            "libcamerasrc ! "
            f"video/x-raw,format=NV12,width={self.config.width},height={self.config.height},framerate={self.config.framerate}/1 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def start(self) -> None:
        if not _opencv_has_gstreamer():
            raise RuntimeError("OpenCV was built without GStreamer support.")

        requested_fps = int(self.config.framerate)
        for fps in (requested_fps, 30):
            self.config.framerate = fps
            self._cap = cv2.VideoCapture(self._pipeline(), cv2.CAP_GSTREAMER)
            if self._cap.isOpened():
                if fps != requested_fps:
                    logger.info(f"Opened camera via GStreamer at fallback fps={fps} (requested {requested_fps}).")
                return
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        raise RuntimeError(
            "Failed to open camera via OpenCV+GStreamer. "
            "Verify: OpenCV built with GStreamer support, `libcamerasrc` available, and camera not in-use."
        )


class LibcameraCliMjpegCamera:
    """CSI camera capture via `rpicam-vid`/`libcamera-vid` MJPEG to stdout.

    This works even when OpenCV is built without GStreamer.
    """

    def __init__(self, width: int = 640, height: int = 480, framerate: int = 30, camera_index: int = 0):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.camera_index = camera_index

        self.proc: Optional[subprocess.Popen] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts_ns: int = 0

    def start(self) -> None:
        cmd = None
        if shutil.which("rpicam-vid"):
            cmd = "rpicam-vid"
        elif shutil.which("libcamera-vid"):
            cmd = "libcamera-vid"

        if not cmd:
            raise RuntimeError("Neither rpicam-vid nor libcamera-vid found on PATH.")

        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(cmd,), daemon=True)
        self._thread.start()

        # Wait briefly for first frame so callers can fail fast.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame, ts = self.read()
            if frame is not None and ts > 0:
                return
            time.sleep(0.02)

        # If we didn't get a frame quickly, stop and surface stderr tail.
        stderr_tail = self._drain_stderr_tail(max_lines=10)
        self.stop()
        hint = ("\n" + stderr_tail) if stderr_tail else ""
        raise RuntimeError(f"Camera index {self.camera_index} started but no frames decoded within 5s.{hint}")

    def _drain_stderr_tail(self, max_lines: int = 10) -> str:
        if not self.proc or self.proc.stderr is None:
            return ""
        lines = deque(maxlen=max_lines)
        try:
            while True:
                line = self.proc.stderr.readline()
                if not line:
                    break
                try:
                    lines.append(line.decode(errors="replace").rstrip())
                except Exception:
                    lines.append(str(line))
        except Exception:
            return ""
        return "\n".join(lines)

    def _start_process(self, cmd: str, fps: int) -> bool:
        args = [
            cmd,
            "--camera",
            str(self.camera_index),
            "--codec",
            "mjpeg",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--framerate",
            str(int(fps)),
            "--nopreview",
            "-o",
            "-",
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
            logger.error(f"Failed to start {cmd} for camera {self.camera_index}: {e}")
            self.proc = None
            return False

    def _loop(self, cmd: str) -> None:
        requested_fps = int(self.framerate) if self.framerate else 30
        if requested_fps <= 0:
            requested_fps = 30

        fps_candidates = (requested_fps, 30)
        stderr_tail = deque(maxlen=50)

        def read_stderr_nonblocking():
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

        soi = b"\xff\xd8"
        eoi = b"\xff\xd9"

        while self._running:
            opened = False
            chosen_fps = requested_fps
            for fps in fps_candidates:
                if self._start_process(cmd, fps):
                    opened = True
                    chosen_fps = fps
                    if fps != requested_fps:
                        logger.info(f"Started {cmd} at fallback fps={fps} (requested {requested_fps}).")
                    break

            if not opened or not self.proc or self.proc.stdout is None:
                logger.error("Unable to start libcamera capture process.")
                return

            buffer = bytearray()
            last_good_frame_ns = time.monotonic_ns()

            while self._running and self.proc and self.proc.poll() is None:
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    read_stderr_nonblocking()
                    now = time.monotonic_ns()
                    if now - last_good_frame_ns > 2_000_000_000:
                        logger.warning(f"No camera frames decoded for >2s (fps={chosen_fps}); restarting {cmd}.")
                        break
                    time.sleep(0.005)
                    continue

                buffer.extend(chunk)

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

                    jpeg = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]

                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    ts = time.monotonic_ns()
                    last_good_frame_ns = ts
                    with self._lock:
                        self._frame = frame
                        self._ts_ns = ts

            read_stderr_nonblocking()
            exit_code = self.proc.poll() if self.proc else None
            if self._running and exit_code is not None and exit_code != 0:
                logger.warning(f"{cmd} exited with code {exit_code}; restarting.")
                if stderr_tail:
                    logger.warning("Last camera stderr lines:\n" + "\n".join(list(stderr_tail)[-10:]))

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
        with self._lock:
            if self._frame is None:
                return None, 0
            return self._frame.copy(), int(self._ts_ns)

    def stop(self) -> None:
        self._running = False
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

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class AutoCamera:
    """Select the best available real camera backend."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._cam = None

    def start(self) -> None:
        if _opencv_has_gstreamer():
            self._cam = Pi5GStreamerCamera(self.config)
            try:
                self._cam.start()
                return
            except Exception as e:
                logger.warning(f"GStreamer camera failed ({e}); falling back to CLI MJPEG.")

        self._cam = LibcameraCliMjpegCamera(
            width=self.config.width,
            height=self.config.height,
            framerate=self.config.framerate,
            camera_index=self.config.camera_index
        )
        self._cam.start()

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        return self._cam.read() if self._cam else (None, 0)

    def stop(self) -> None:
        if self._cam:
            self._cam.stop()
            self._cam = None


class MockCamera:
    """Simple mock camera that generates noise frames."""

    def __init__(self, width: int = 640, height: int = 480, framerate: int = 30):
        self.width = width
        self.height = height
        self.framerate = framerate
        self._running = False
        self._interval_s = 1.0 / max(1, int(framerate))

    def start(self) -> None:
        self._running = True

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        if not self._running:
            return None, 0
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        ts = time.monotonic_ns()
        time.sleep(self._interval_s)
        return frame, ts

    def stop(self) -> None:
        self._running = False


__all__ = [
    "CameraConfig",
    "Pi5GStreamerCamera",
    "LibcameraCliMjpegCamera",
    "AutoCamera",
    "MockCamera",
]
