#!/usr/bin/env python3
import json
import logging
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from final.camera_gstream import AutoCamera, CameraConfig
from final.hand_tracking import HandTrackingModule
from final.angles import AngleEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Calibrator")

def main():
    width, height, fps = 640, 480, 30
    camera = AutoCamera(CameraConfig(width=width, height=height, framerate=fps))
    hand = HandTrackingModule(model_complexity=0)
    engine = AngleEngine()

    try:
        camera.start()
    except Exception as e:
        logger.error(f"Could not start camera: {e}")
        return

    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    joints = ["mcp", "pip", "dip"]

    # Storage for all recorded frames during a capture phase
    min_angles = {f: {j: 180.0 for j in joints} for f in fingers}
    max_angles = {f: {j: 0.0 for j in joints} for f in fingers}

    def calibrate_phase(name, target_dict, comparator):
        print(f"\n--- Phase: {name} ---")
        print("Keep hand in the TARGET POSE. Press 'c' to CAPTURE, 'q' to ABORT.")
        while True:
            frame, _ = camera.read()
            if frame is None: continue
            
            det = hand.process(frame, time.monotonic_ns())
            if det:
                hand.draw(frame, det)
                angles = engine.compute_hand_angles(det.landmarks)
                
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    print(f"Captured {name} values.")
                    for f in fingers:
                        for j in joints:
                            val = angles.get(f, {}).get(j, 0.0)
                            target_dict[f][j] = val
                    return True

            cv2.imshow("Calibrator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

    success = False
    if calibrate_phase("Extension (Flat Hand)", min_angles, min):
        if calibrate_phase("Flexion (Fist)", max_angles, max):
            success = True

    camera.stop()
    cv2.destroyAllWindows()

    if success:
        output = {
            "min_angles": min_angles,
            "max_angles": max_angles,
            "timestamp": datetime.now().isoformat()
        }
        with open("calibration.json", "w") as f:
            json.dump(output, f, indent=4)
        print("\nCalibration saved to calibration.json")
    else:
        print("\nCalibration aborted.")

if __name__ == "__main__":
    main()
