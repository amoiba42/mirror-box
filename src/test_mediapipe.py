import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    result = hands.process(frame)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
