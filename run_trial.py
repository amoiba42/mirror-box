import time
import argparse
import logging
import cv2
from datetime import datetime

# Import our modules
from src.camera_interface import CameraInterface
from src.hand_tracker import HandTracker
from src.emg_interface import EMGInterface
from src.grip_interface import GripInterface
from src.fusion_engine import FusionEngine

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def main():
    # parser = argparse.ArgumentParser(description="Stroke Rehab System - Mock Trial Runner")
    # parser.add_argument("--mode", type=str, default="mock", choices=["mock", "real"], help="Hardware mode")
    # parser.add_argument("--duration", type=int, default=60, help="Trial duration in seconds")
    # parser.add_argument("--complexity", type=int, default=0, help="MediaPipe Complexity (0=Pi, 1=Desktop)")
    # args = parser.parse_args()
    print("=" * 60)
    print("    STROKE REHABILITATION SYSTEM - TRIAL RUNNER")
    print("=" * 60)
    print()
    
    # Interactive mode selection
    print("Hardware Mode:")
    print("  1. Mock Mode (Simulated sensors)")
    print("  2. Real Hardware Mode")
    while True:
        mode_choice = input("Select mode [1-2] (default: 1): ").strip()
        if mode_choice == "" or mode_choice == "1":
            mode = "mock"
            break
        elif mode_choice == "2":
            mode = "real"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Duration selection
    print("\nTrial Duration:")
    while True:
        duration_input = input("Enter duration in seconds (default: 60): ").strip()
        if duration_input == "":
            duration = 60
            break
        try:
            duration = int(duration_input)
            if duration > 0:
                break
            else:
                print("Duration must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    # MediaPipe complexity
    print("\nMediaPipe Model Complexity:")
    print("  0. Lite (Raspberry Pi optimized)")
    print("  1. Full (Desktop/high-performance)")
    while True:
        complexity_choice = input("Select complexity [0-1] (default: 0): ").strip()
        if complexity_choice == "" or complexity_choice == "0":
            complexity = 0
            break
        elif complexity_choice == "1":
            complexity = 1
            break
        else:
            print("Invalid choice. Please enter 0 or 1.")
    
    # Display mode for SSH/headless
    print("\nDisplay Mode:")
    print("  1. GUI Mode (show camera feed)")
    print("  2. Headless Mode (data only)")
    while True:
        display_choice = input("Select display mode [1-2] (default: 1): ").strip()
        if display_choice == "" or display_choice == "1":
            headless = False
            break
        elif display_choice == "2":
            headless = True
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print("\nConfiguration:")
    print(f"  Mode: {mode}")
    print(f"  Duration: {duration}s")
    print(f"  Complexity: {complexity}")
    print("\nPress Enter to start trial...")
    input()
    
    # Create args-like object for compatibility
    class Args:
        def __init__(self, mode, duration, complexity):
            self.mode = mode
            self.duration = duration
            self.complexity = complexity
    
    args = Args(mode, duration, complexity)

    # Session ID
    session_id = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting Trial: {session_id} | Mode: {args.mode}")

    # 1. Initialize Hardware (or Mocks)
    is_mock = (args.mode == "mock")
    
    camera = CameraInterface(mock_mode=is_mock) # Defaults to webcam even in mock if avail, or synthetic
    emg = EMGInterface(mock_mode=is_mock)
    grip = GripInterface(mock_mode=is_mock)
    
    # 2. Initialize Processing
    tracker = HandTracker(model_complexity=args.complexity)
    fusion = FusionEngine(session_id=session_id)
    
    # 3. Start Streams
    camera.start()
    emg.start()
    
    # 4. Main Loop
    start_time = time.monotonic()
    try:
        while (time.monotonic() - start_time) < args.duration:
            
            # A. Acquire Data
            frame, frame_ts = camera.read()
            emg_sample = emg.read_sample()
            grip_sample = grip.read_force()
            
            if frame is None:
                continue
                
            # B. Processing (CV Inference)
            # Resize for performance simulation if needed (optional)
            frame_small = cv2.resize(frame, (320, 240)) 
            
            hand_result = tracker.process(frame, frame_ts)
            
            # C. Fusion & Logging
            data = fusion.ingest(hand_result, emg_sample, grip_sample)
            
            # D. Visualization (Desktop Preview)
            if hand_result:
                # USE THE NEW CUSTOM DRAWING FUNCTION
                tracker.draw_custom_skeleton(frame, hand_result)
                
                # Get Index Finger Angles
                idx_angles = data['angles'].get('index', {})
                mcp = idx_angles.get('mcp', 0)
                pip = idx_angles.get('pip', 0)
                dip = idx_angles.get('dip', 0)
                
                # Visual Feedback Colors (Green if flexed > 90, else Orange)
                c_mcp = (0, 255, 0) if mcp > 90 else (0, 165, 255)
                c_pip = (0, 255, 0) if pip > 90 else (0, 165, 255)
                c_dip = (0, 255, 0) if dip > 90 else (0, 165, 255)

                # --- FIXED LAYOUT ---
                # 1. Hand Label (Top Left)
                cv2.putText(frame, f"Hand: {hand_result.handedness}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 2. Stacked Joint Data (Below Hand Label)
                cv2.putText(frame, f"Index MCP: {mcp:.0f}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_mcp, 2)
                cv2.putText(frame, f"Index PIP: {pip:.0f}", (10, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_pip, 2)
                cv2.putText(frame, f"Index DIP: {dip:.0f}", (10, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_dip, 2)

            # 3. Global Stats (Very Top) - Kept away from the rest
            time_left = int(args.duration - (time.monotonic() - start_time))
            cv2.putText(frame, f"Time: {time_left}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 4. EMG Data (Below Joints)
            cv2.putText(frame, f"EMG: {data['emg']:.1f}", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            
            cv2.imshow("Rehab Mock Trial", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        logger.info("Trial interrupted by user.")
    finally:
        # Cleanup
        camera.stop()
        emg.stop()
        fusion.close()
        cv2.destroyAllWindows()
        logger.info("Trial Complete. Data saved.")

if __name__ == "__main__":
    main()