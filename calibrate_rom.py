import time
import json
import numpy as np
import cv2
import logging
from src.camera_interface import CameraInterface
from src.hand_tracker import HandTracker
from src.angle_utils import AngleEngine

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Calibrator")

def capture_phase(title, duration, cam, tracker, angle_engine):
    """
    Captures angle data for a set duration and returns the average angles.
    """
    start_time = time.monotonic()
    collected_angles = {
        'index': {'mcp': [], 'pip': [], 'dip': []}
        # Add other fingers here if needed
    }
    
    while (time.monotonic() - start_time) < duration:
        frame, ts = cam.read()
        if frame is None: continue
        
        # UI Feedback
        time_left = int(duration - (time.monotonic() - start_time))
        cv2.putText(frame, f"ACTION: {title}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Hold for: {time_left}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        res = tracker.process(frame, ts)
        if res:
            tracker.draw_custom_skeleton(frame, res)
            angles = angle_engine.compute_hand_angles(res.landmarks)
            
            # Store Index finger data for averaging
            idx = angles.get('index', {})
            if idx:
                collected_angles['index']['mcp'].append(idx.get('mcp', 0))
                collected_angles['index']['pip'].append(idx.get('pip', 0))
                collected_angles['index']['dip'].append(idx.get('dip', 0))

        cv2.imshow("Calibration Wizard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate Averages
    averages = {}
    for finger, joints in collected_angles.items():
        averages[finger] = {}
        for joint, values in joints.items():
            if values:
                # Use median to ignore outliers/jitter
                averages[finger][joint] = float(np.median(values))
            else:
                averages[finger][joint] = 0.0
                
    return averages

def main():
    cam = CameraInterface(source=0, mock_mode=False)
    tracker = HandTracker(model_complexity=0)
    angle_engine = AngleEngine()
    
    cam.start()
    
    try:
        print("Starting Calibration...")
        
        # Phase 1: Preparation
        # Loop until user presses space
        while True:
            frame, _ = cam.read()
            if frame is None: continue
            cv2.putText(frame, "Press SPACE to Calibrate OPEN HAND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration Wizard", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        
        # Phase 2: Capture Min Angles (Open Hand)
        min_vals = capture_phase("KEEP HAND OPEN", 5.0, cam, tracker, angle_engine)
        logger.info(f"Min Angles (Open): {min_vals}")
        
        # Phase 3: Wait
        while True:
            frame, _ = cam.read()
            if frame is None: continue
            cv2.putText(frame, "Press SPACE to Calibrate FIST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration Wizard", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        # Phase 4: Capture Max Angles (Fist)
        max_vals = capture_phase("SQUEEZE TIGHT", 5.0, cam, tracker, angle_engine)
        logger.info(f"Max Angles (Fist): {max_vals}")

        # Phase 5: Save
        calib_data = {
            "min_angles": min_vals,
            "max_angles": max_vals,
            "timestamp": time.time()
        }
        
        with open("calibration.json", "w") as f:
            json.dump(calib_data, f, indent=4)
        
        print("\nCalibration saved to 'calibration.json'")

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()