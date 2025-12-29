import logging
import cv2
import time
from src.camera_interface import CameraInterface
from src.hand_tracker import HandTracker

# Setup Logging
logging.basicConfig(level=logging.INFO)

def main():
    # 1. Initialize Camera (Use 0 for webcam, or path to file)
    # mock_mode=True ensures it runs even if you have no webcam connected
    cam = CameraInterface(source=0, width=640, height=480, mock_mode=True)
    
    # 2. Initialize Tracker (Complexity 0 simulates Pi performance)
    tracker = HandTracker(model_complexity=0)
    
    try:
        cam.start()
        print("Press 'q' to quit.")
        
        while True:
            # Read Frame
            frame, ts = cam.read()
            
            if frame is None:
                continue

            # Tracking
            result = tracker.process(frame, ts)

            # Visualization
            if result:
                tracker.draw_landmarks(frame, result)
                cv2.putText(frame, f"Hand: {result.handedness} ({result.confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Rehab System - Vision Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()