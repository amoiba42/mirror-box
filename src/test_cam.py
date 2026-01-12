import time
from picamera2 import Picamera2
import cv2

print("Starting minimal picamera2 test...")
try:
    # 1. Initialize the camera
    picam2 = Picamera2()
    print("Camera initialized.")

    # 2. Create a configuration
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    print("Camera configured.")

    # 3. Start the camera and allow sensor to warm up
    picam2.start()
    print("Camera started, waiting for sensor to warm up...")
    time.sleep(1)

    # 4. Capture a single frame
    frame = picam2.capture_array()
    print("Frame captured.")

    # 5. Check if the frame is valid
    if frame is not None:
        print(f"Success! Frame shape: {frame.shape}")
        # Save the image to prove it worked
        cv2.imwrite("test_capture.jpg", frame)
        print("Image 'test_capture.jpg' saved successfully.")
    else:
        print("Error: capture_array() returned None. The frame is empty.")

except Exception as e:
    print(f"An exception occurred: {e}")

finally:
    # 6. Clean up
    if 'picam2' in locals() and picam2.is_open:
        picam2.stop()
        print("Camera stopped.")