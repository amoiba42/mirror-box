from picamera2 import Picamera2
import cv2
import mediapipe as mp
import time

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    frame = picam2.capture_array()  # RGB
    rgb = frame
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    results = mp_hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                bgr, hand, mp.solutions.hands.HAND_CONNECTIONS
            )

    cv2.imshow("Picamera2 + MediaPipe", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
