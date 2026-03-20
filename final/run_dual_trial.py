#!/usr/bin/env python3

import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

# Add parent dir to sys.path if running as script
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from final.camera_gstream import AutoCamera, CameraConfig, MockCamera
from final.hand_tracking import HandTrackingModule
from final.mirror_display import HorizontalMirror
from final.angles import AngleProcessor, Calibration
from final.fusion_engine import FusionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DualTrial")

CONFIG = {
    # Execution
    "mode_good": "real", # "mock" or "real"
    "mode_bad": "mock",  # "mock" or "real"
    "duration_s": 60,

    # Cameras
    "width": 640,
    "height": 480,
    "fps": 30,
    # On Pi 5 with 1 camera, CSI is 0. 
    # USB cameras usually start at 1 or 2 (after CSI nodes).
    "cam_good_id": 0, 
    "cam_bad_id": 1,

    # Vision
    "mediapipe_complexity": 0,
    "invert_handedness": True,

    # Angles/Calibration
    "apply_calibration": True,
    "calibration_path": "calibration.json",
}

class MultiCameraController:
    def __init__(self, mode_good: str, mode_bad: str, width: int, height: int, fps: int):
        self.mode_good = mode_good
        self.mode_bad = mode_bad
        self.width = width
        self.height = height
        self.fps = fps
        self.cam_good = None
        self.cam_bad = None

    def start(self, good_id: int, bad_id: int):
        logger.info(f"Starting Good Camera (Mode: {self.mode_good}, ID: {good_id})")
        if self.mode_good == "mock":
            self.cam_good = MockCamera(self.width, self.height, self.fps)
        else:
            self.cam_good = AutoCamera(CameraConfig(self.width, self.height, self.fps, camera_index=good_id))

        logger.info(f"Starting Bad Camera (Mode: {self.mode_bad}, ID: {bad_id})")
        if self.mode_bad == "mock":
            self.cam_bad = MockCamera(self.width, self.height, self.fps)
        else:
            self.cam_bad = AutoCamera(CameraConfig(self.width, self.height, self.fps, camera_index=bad_id))

        if self.cam_good: self.cam_good.start()
        if self.cam_bad: self.cam_bad.start()

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        f_g, _ = self.cam_good.read() if self.cam_good else (None, 0)
        f_b, _ = self.cam_bad.read() if self.cam_bad else (None, 0)
        return f_g, f_b

    def stop(self):
        if self.cam_good: self.cam_good.stop()
        if self.cam_bad: self.cam_bad.stop()

def main():
    print("\n--- Mirror Box System Startup ---")
    side = input("Which side is the AFFECTED (BAD) side? (left/right): ").strip().lower()
    while side not in ["left", "right"]:
        side = input("Please enter 'left' or 'right': ").strip().lower()

    good_side = "right" if side == "left" else "left"
    logger.info(f"Configuration: Bad Side = {side}, Good Side = {good_side}")

    headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless:
        logger.warning("No display detected; hiding UI windows.")

    # Setup
    session_id = f"dual_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Controllers
    cams = MultiCameraController(CONFIG["mode_good"], CONFIG["mode_bad"], CONFIG["width"], CONFIG["height"], CONFIG["fps"])
    hand_good = HandTrackingModule(model_complexity=CONFIG["mediapipe_complexity"], invert_handedness=CONFIG["invert_handedness"])
    hand_bad = HandTrackingModule(model_complexity=CONFIG["mediapipe_complexity"], invert_handedness=CONFIG["invert_handedness"])
    
    calib = None
    if CONFIG["apply_calibration"]:
        try:
            calib = Calibration.load(CONFIG["calibration_path"])
            logger.info(f"Loaded calibration from {CONFIG['calibration_path']}")
        except Exception as e:
            logger.warning(f"Failed to load calibration ({CONFIG['calibration_path']}): {e}")
            calib = None

    angle_proc = AngleProcessor(calibration=calib)
    mirror = HorizontalMirror(enabled=True) # Flipped for "good" side display
    
    # We use two FusionEngines to log both hands, or one with custom extension.
    # Let's use two separate CSVs in same folder for simplicity.
    fusion_good = FusionEngine(f"{session_id}/good")
    fusion_bad = FusionEngine(f"{session_id}/bad")

    cams.start(CONFIG["cam_good_id"], CONFIG["cam_bad_id"])
    
    start_time = time.monotonic()
    
    # Supervisor view toggle (press 's')
    supervisor_mode = False

    try:
        while (time.monotonic() - start_time) < CONFIG["duration_s"]:
            fg, fb = cams.read()
            
            # 1. Process Bad Side (Data only, no UI)
            if fb is not None:
                det_bad = hand_bad.process(fb, time.monotonic_ns())
                fusion_bad.ingest(det_bad)

            # 2. Process Good Side (Mirror display)
            final_patient_display = None
            raw_fg_with_markers = None
            if fg is not None:
                det_good = hand_good.process(fg, time.monotonic_ns())
                
                # Apply Calibration to normalized angles
                norm_angles_good = None
                if det_good:
                    norm_angles_good = angle_proc.compute_normalized(det_good.landmarks)
                
                fusion_good.ingest(det_good)
                
                # Create the clean Patient View (Flipped, No markers)
                final_patient_display = mirror.apply(fg)
                
                # Create the Good Side Supervisor View component (Not flipped, With markers)
                raw_fg_with_markers = fg.copy()
                if det_good:
                    hand_good.draw(raw_fg_with_markers, det_good)
                    
                    # Update Supervisor display with normalized angles (0-100%)
                    if norm_angles_good:
                        idx = norm_angles_good.get("index", {})
                        mcp = idx.get("mcp", 0) * 100
                        cv2.putText(raw_fg_with_markers, f"Idx MCP: {mcp:.0f}%", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 3. Process Bad Side (Data + Supervisor display)
            raw_fb_with_markers = None
            if fb is not None:
                det_bad = hand_bad.process(fb, time.monotonic_ns())
                
                # Apply Calibration
                norm_angles_bad = None
                if det_bad:
                    norm_angles_bad = angle_proc.compute_normalized(det_bad.landmarks)

                fusion_bad.ingest(det_bad)
                
                # Create the Bad Side Supervisor View component (Not flipped, With markers)
                raw_fb_with_markers = fb.copy()
                if det_bad:
                    hand_bad.draw(raw_fb_with_markers, det_bad)
                    
                    if norm_angles_bad:
                        idx = norm_angles_bad.get("index", {})
                        mcp = idx.get("mcp", 0) * 100
                        cv2.putText(raw_fb_with_markers, f"Idx MCP: {mcp:.0f}%", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # 4. Handle UI
            if not headless:
                if supervisor_mode:
                    # Side-by-side view: Raw, No Flipped, With Markers
                    h, w = CONFIG["height"], CONFIG["width"]
                    canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
                    if raw_fg_with_markers is not None: canvas[:h, :w] = raw_fg_with_markers
                    if raw_fb_with_markers is not None: canvas[:h, w:] = raw_fb_with_markers
                    
                    cv2.putText(canvas, "GOOD SIDE (Raw+Markers)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(canvas, "BAD SIDE (Raw+Markers)", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.imshow("Supervisor View", canvas)
                
                # Patient View: Flipped, No Markers
                if final_patient_display is not None:
                    cv2.imshow("Patient View (Mirror)", final_patient_display)
                
                if not supervisor_mode and cv2.getWindowProperty("Supervisor View", cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow("Supervisor View")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                supervisor_mode = not supervisor_mode
                logger.info(f"Supervisor mode: {supervisor_mode}")

    except KeyboardInterrupt:
        pass
    finally:
        cams.stop()
        fusion_good.close()
        fusion_bad.close()
        cv2.destroyAllWindows()
        logger.info("Trial completed.")

if __name__ == "__main__":
    main()
