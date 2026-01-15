from new import CameraInterface
import cv2

cam = CameraInterface(enable_af=True)
cam.start()

while True:
    frame, ts = cam.read()
    if frame is None:
        continue

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()
