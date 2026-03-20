import time
import cv2  # OpenCV for displaying the image
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

# 1. Initialize the camera
picam2 = Picamera2()

# 2. Configure for RGB888 (Pi 5 prefers this)
config = picam2.create_preview_configuration(main={"format": "NV12", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# 3. Apply Natural Look Controls (Fix for NoIR Camera)
picam2.set_controls({
    "Saturation": 0.55,      # Reduced saturation to stop "popping" neon colors
    "Sharpness": 0.8,        # Slightly softer than digital default
    "AwbMode": controls.AwbModeEnum.Auto, # Auto White Balance
    "Brightness": 0.0,
    "Contrast": 1.0
})

print("Camera running. Press 'q' to quit.")

try:
    while True:
        # 4. Capture the current frame as a numpy array
        frame = picam2.capture_array()

        # 5. Convert RGB (Camera) to BGR (OpenCV uses BGR for display)
        # Without this, your red and blue colors will be swapped!
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 6. Show the frame in a window
        cv2.imshow("Natural Camera Feed", display_frame)

        # 7. Wait 1ms for a keypress; if 'q' is pressed, break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    print("\nStopping camera...")
    picam2.stop()
    cv2.destroyAllWindows()
