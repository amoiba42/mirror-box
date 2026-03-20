#!/usr/bin/env python3

import logging
import os
import sys
import time
from datetime import datetime

import cv2

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from final.camera_gstream import AutoCamera, CameraConfig, MockCamera
from final.hand_tracking import HandTrackingModule
from final.mirror_display import HorizontalMirror
from final.angles import AngleProcessor, Calibration
from final.fusion_engine import FusionEngine


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FinalTrial")


CONFIG = {
    # Execution
    "mode": "real",  # "mock" or "real"
    "duration_s": 60,

    # Vision
    "mediapipe_complexity": 0,  # 0 (lite) or 1 (full)
    "invert_handedness": True,

    # Display
    "headless": False,  # if True, no cv2 window is shown
    "mirror": True,  # horizontal flip

    # Camera
    "width": 640,
    "height": 480,
    "fps": 30,

    # Angles
    "apply_calibration": True,
    "calibration_path": "calibration.json",
}
def main():
    mode = str(CONFIG["mode"]).strip().lower()
    duration_s = int(CONFIG["duration_s"])
    complexity = int(CONFIG["mediapipe_complexity"])
    headless = bool(CONFIG["headless"])
    mirror_enabled = bool(CONFIG["mirror"])
    invert_handedness = bool(CONFIG["invert_handedness"])
    width = int(CONFIG["width"])
    height = int(CONFIG["height"])
    fps = int(CONFIG["fps"])
    apply_calibration = bool(CONFIG["apply_calibration"])
    calibration_path = str(CONFIG["calibration_path"])

    if not headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        logger.warning("No display detected; forcing headless mode.")
        headless = True

    session_id = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting Trial: {session_id} | Mode: {mode}")

    is_mock = mode == "mock"

    # Camera
    if is_mock:
        camera = MockCamera(width=width, height=height, framerate=fps)
    else:
        camera = AutoCamera(CameraConfig(width=width, height=height, framerate=fps))

    # Vision + angles
    hand = HandTrackingModule(model_complexity=complexity, invert_handedness=invert_handedness)
    mirror = HorizontalMirror(enabled=mirror_enabled)

    calib = None
    if apply_calibration:
        try:
            calib = Calibration.load(calibration_path)
            logger.info(f"Loaded calibration from {calibration_path}")
        except Exception as e:
            logger.warning(f"Failed to load calibration ({calibration_path}): {e}")
            calib = None

    angle_proc = AngleProcessor(calibration=calib)

    fusion = FusionEngine(session_id=session_id)

    # Start streams
    try:
        camera.start()
    except Exception as e:
        logger.error(f"Camera failed to start: {e}")
        raise

    start_time = time.monotonic()
    last_frame_wall = time.monotonic()
    try:
        while (time.monotonic() - start_time) < duration_s:
            frame, frame_ts = camera.read()
            if frame is None:
                if (time.monotonic() - last_frame_wall) > 2.0:
                    logger.warning("No frames received from camera for >2s (check pipeline/camera access).")
                    last_frame_wall = time.monotonic()
                continue

            last_frame_wall = time.monotonic()

            # Mirror first (so the displayed + processed view matches)
            frame = mirror.apply(frame)

            detection = hand.process(frame, frame_ts)

            data = fusion.ingest(detection)

            # Optional calibrated angles (runtime use)
            if detection and apply_calibration:
                _ = angle_proc.compute_normalized(detection.landmarks)

            if not headless:
                if detection:
                    hand.draw(frame, detection)

                    idx_angles = data.get("angles", {}).get("index", {})
                    mcp = float(idx_angles.get("mcp", 0))
                    pip = float(idx_angles.get("pip", 0))
                    dip = float(idx_angles.get("dip", 0))

                    c_mcp = (0, 255, 0) if mcp > 90 else (0, 165, 255)
                    c_pip = (0, 255, 0) if pip > 90 else (0, 165, 255)
                    c_dip = (0, 255, 0) if dip > 90 else (0, 165, 255)

                    cv2.putText(
                        frame,
                        f"Hand: {detection.handedness}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Index MCP: {mcp:.0f}",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        c_mcp,
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Index PIP: {pip:.0f}",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        c_pip,
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Index DIP: {dip:.0f}",
                        (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        c_dip,
                        2,
                    )

                time_left = int(duration_s - (time.monotonic() - start_time))
                cv2.putText(
                    frame,
                    f"Time: {time_left}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Rehab Trial", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Trial interrupted by user.")
    finally:
        try:
            camera.stop()
        except Exception:
            pass
        fusion.close()
        if not headless:
            cv2.destroyAllWindows()
        logger.info("Trial Complete. Data saved.")


if __name__ == "__main__":
    main()
