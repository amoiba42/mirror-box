import logging
from dataclasses import dataclass
from typing import Any, Optional
import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HandDetectionResult:
    timestamp: int
    landmarks: Any
    handedness: str
    confidence: float


class HandTracker:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

        # Basic temporal smoothing for landmarks
        self._prev_landmarks = None
        self._alpha = 0.4  # Smoothing factor (0=lock, 1=no smoothing)

    def process(self, frame: np.ndarray, timestamp: int, invert_handedness: bool = True) -> Optional[HandDetectionResult]:
        if frame is None:
            return None
        
        # Performance: ROI or downscaling could be added here if needed.
        # Stabilizing: Ensure frame is oriented correctly.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            primary_hand = results.multi_hand_landmarks[0]
            
            # Apply basic landmark smoothing
            if self._prev_landmarks is not None:
                for i in range(len(primary_hand.landmark)):
                    curr = primary_hand.landmark[i]
                    prev = self._prev_landmarks.landmark[i]
                    curr.x = prev.x + self._alpha * (curr.x - prev.x)
                    curr.y = prev.y + self._alpha * (curr.y - prev.y)
                    curr.z = prev.z + self._alpha * (curr.z - prev.z)
            self._prev_landmarks = primary_hand

            original_label = results.multi_handedness[0].classification[0].label
            score = results.multi_handedness[0].classification[0].score

            final_label = original_label
            if invert_handedness:
                final_label = "Left" if original_label == "Right" else "Right"

            return HandDetectionResult(timestamp, primary_hand, final_label, float(score))
        return None

    def draw_custom_skeleton(self, frame: np.ndarray, detection: HandDetectionResult) -> None:
        """Draw landmarks with specific colors for MCP, PIP, DIP joints."""
        if not detection or not detection.landmarks:
            return

        h, w = frame.shape[:2]
        landmarks = detection.landmarks.landmark

        self.mp_drawing.draw_landmarks(
            frame,
            detection.landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=1),
        )

        indices_mcp = [1, 5, 9, 13, 17]
        indices_pip = [2, 6, 10, 14, 18]
        indices_dip = [3, 7, 11, 15, 19]
        indices_tip = [4, 8, 12, 16, 20]
        wrist = [0]

        def draw_set(indices, color):
            for i in indices:
                lm = landmarks[i]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 6, color, -1)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)

        draw_set(wrist, (200, 200, 200))
        draw_set(indices_mcp, (0, 0, 255))
        draw_set(indices_pip, (0, 255, 0))
        draw_set(indices_dip, (255, 0, 0))
        draw_set(indices_tip, (0, 255, 255))


class HandTrackingModule:
    def __init__(
        self,
        model_complexity: int = 0,
        invert_handedness: bool = True,
        min_detection_confidence: float = 0.7,
    ):
        self.tracker = HandTracker(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
        self.invert_handedness = invert_handedness

    def process(self, frame: np.ndarray, timestamp_ns: int) -> Optional[HandDetectionResult]:
        return self.tracker.process(frame, timestamp_ns, invert_handedness=self.invert_handedness)

    def draw(self, frame: np.ndarray, detection: HandDetectionResult) -> None:
        self.tracker.draw_custom_skeleton(frame, detection)


__all__ = ["HandTrackingModule", "HandDetectionResult", "HandTracker"]
