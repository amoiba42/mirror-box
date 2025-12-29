import mediapipe as mp
import cv2
import logging
from dataclasses import dataclass
from typing import Optional, Any

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
            min_tracking_confidence=0.5
        )

    def process(self, frame, timestamp: int, invert_handedness: bool = True) -> Optional[HandDetectionResult]:
        if frame is None: return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            primary_hand = results.multi_hand_landmarks[0]
            original_label = results.multi_handedness[0].classification[0].label
            score = results.multi_handedness[0].classification[0].score
            
            # Swap label if camera is mirrored (Selfie view)
            final_label = original_label
            if invert_handedness:
                final_label = "Left" if original_label == "Right" else "Right"

            return HandDetectionResult(timestamp, primary_hand, final_label, score)
        return None

    def draw_custom_skeleton(self, frame, detection: HandDetectionResult):
        """
        Draws landmarks with specific colors for MCP, PIP, and DIP joints.
        """
        if not detection or not detection.landmarks:
            return

        h, w, _ = frame.shape
        landmarks = detection.landmarks.landmark

        # 1. Draw Connections (Lines) first - Grey
        self.mp_drawing.draw_landmarks(
            frame, detection.landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=1), # Lines
            self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=1) # Default dots (overwritten below)
        )

        # 2. Define Joint Indices
        # Thumb indices are slightly different, but fit broadly into these categories for coloring
        indices_mcp = [1, 5, 9, 13, 17]  # Knuckles (MIP)
        indices_pip = [2, 6, 10, 14, 18] # Middle Joints
        indices_dip = [3, 7, 11, 15, 19] # Distal Joints
        indices_tip = [4, 8, 12, 16, 20] # Tips
        wrist = [0]

        # 3. Helper to draw specific sets
        def draw_set(indices, color):
            for i in indices:
                lm = landmarks[i]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 6, color, -1) # Filled circle
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1) # White border

        # 4. Apply Colors (BGR format)
        draw_set(wrist, (200, 200, 200)) # Grey Wrist
        draw_set(indices_mcp, (0, 0, 255))   # Red for MCP (MIP)
        draw_set(indices_pip, (0, 255, 0))   # Green for PIP
        draw_set(indices_dip, (255, 0, 0))   # Blue for DIP
        draw_set(indices_tip, (0, 255, 255)) # Yellow for Tips