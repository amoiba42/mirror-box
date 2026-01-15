import cv2

pipeline = (
    "libcamerasrc ! "
    "video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=true"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

print("Opened:", cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    # If headless, comment this out
    cv2.imshow("Pi 5 Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
