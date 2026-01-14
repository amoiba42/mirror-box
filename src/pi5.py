import cv2

# This "pipeline" tells OpenCV to use libcamera instead of looking for /dev/video0
pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! "
    "videoscale ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)

print("Attempting to open camera with Libcamera pipeline...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("✅ Success! Camera opened.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Display the frame
        cv2.imshow("Pi 5 Camera Test", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("❌ Failed. Make sure 'rpicam-hello' works in the terminal first.")

cap.release()
cv2.destroyAllWindows()