from typing import Optional

import cv2
import numpy as np


class MirrorDisplay:
    """Handles visual feedback for mirror therapy.

    Primary function: horizontal flip of the healthy limb to simulate the paretic limb.
    """

    def __init__(self, title: str = "Mirror Therapy Feed"):
        self.title = title
        self.fullscreen = False

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        return cv2.flip(frame, 1)

    def draw_guidelines(self, frame: np.ndarray, color=(0, 255, 0)) -> None:
        h, w = frame.shape[:2]
        center_x = w // 2

        cv2.line(frame, (center_x, 0), (center_x, h), color, 2)

        box_size = 150
        top_left = (center_x - box_size // 2, h // 2 - box_size // 2)
        bottom_right = (center_x + box_size // 2, h // 2 + box_size // 2)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), 1)

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.title, frame)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


class HorizontalMirror:
    """Thin wrapper: horizontal flip for mirror therapy."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._mirror = MirrorDisplay()

    def apply(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        if not self.enabled:
            return frame
        return self._mirror.process(frame)


__all__ = ["HorizontalMirror", "MirrorDisplay"]
