import cv2
import numpy as np
from typing import Optional

class MirrorDisplay:
    """
    Handles the visual feedback for Mirror Therapy.
    Primary function: Horizontal Flip of the healthy limb to simulate the paretic limb.
    """
    def __init__(self, title="Mirror Therapy Feed"):
        self.title = title
        self.fullscreen = False

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Flips the frame horizontally to create the mirror illusion.
        """
        if frame is None:
            return None
        # Flip around y-axis (1)
        return cv2.flip(frame, 1)

    def draw_guidelines(self, frame: np.ndarray, color=(0, 255, 0)):
        """
        Draws a central dividing line or target box overlay.
        Useful for keeping the user's hand in the camera center.
        """
        h, w = frame.shape[:2]
        center_x = w // 2
        
        # Draw dashed center line
        cv2.line(frame, (center_x, 0), (center_x, h), color, 2)
        
        # Optional: Draw a "Goal Box" in the center
        box_size = 150
        top_left = (center_x - box_size//2, h//2 - box_size//2)
        bottom_right = (center_x + box_size//2, h//2 + box_size//2)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), 1)

    def show(self, frame: np.ndarray):
        """Displays the frame (Desktop only)."""
        cv2.imshow(self.title, frame)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)