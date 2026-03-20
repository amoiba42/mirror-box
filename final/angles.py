import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


class AngleEngine:
    """Computes joint angles from MediaPipe landmarks.
    Focuses on flexion/extension of MCP, PIP, DIP joints.
    """

    JOINTS = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
        "wrist": 0,
    }

    @staticmethod
    def _compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Angle at p2 formed by p1-p2 and p3-p2, in degrees (0-180)."""
        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        unit_v1 = v1 / norm1
        unit_v2 = v2 / norm2

        dot_product = float(np.dot(unit_v1, unit_v2))
        dot_product = float(np.clip(dot_product, -1.0, 1.0))
        angle_rad = float(np.arccos(dot_product))
        return float(np.degrees(angle_rad))

    def compute_hand_angles(self, landmarks) -> Dict[str, Dict[str, float]]:
        """Extract MCP/PIP/DIP flexion angles for all fingers."""
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=float)
        angles: Dict[str, Dict[str, float]] = {}

        for finger, idxs in self.JOINTS.items():
            if finger == "wrist":
                continue

            if finger == "thumb":
                # MediaPipe thumb chain: CMC(1) -> MCP(2) -> IP(3) -> TIP(4)
                # To keep downstream logging uniform (mcp/pip/dip), map:
                # - mcp: flexion at MCP (CMC-MCP-IP)
                # - pip: flexion at IP  (MCP-IP-TIP)
                # - dip: not anatomically present; set to 0.0
                mcp = self._compute_angle(coords[idxs[0]], coords[idxs[1]], coords[idxs[2]])
                mcp_flex = 180 - mcp

                pip = self._compute_angle(coords[idxs[1]], coords[idxs[2]], coords[idxs[3]])
                pip_flex = 180 - pip

                angles[finger] = {"mcp": round(mcp_flex, 1), "pip": round(pip_flex, 1), "dip": 0.0}
                continue

            mcp = self._compute_angle(coords[0], coords[idxs[0]], coords[idxs[1]])
            mcp_flex = 180 - mcp

            pip = self._compute_angle(coords[idxs[0]], coords[idxs[1]], coords[idxs[2]])
            pip_flex = 180 - pip

            dip = self._compute_angle(coords[idxs[1]], coords[idxs[2]], coords[idxs[3]])
            dip_flex = 180 - dip

            angles[finger] = {
                "mcp": round(mcp_flex, 1),
                "pip": round(pip_flex, 1),
                "dip": round(dip_flex, 1),
            }

        return angles


@dataclass
class Calibration:
    min_angles: Dict[str, Dict[str, float]]
    max_angles: Dict[str, Dict[str, float]]

    @staticmethod
    def load(path: str | Path) -> "Calibration":
        with open(path, "r") as f:
            data = json.load(f)
        return Calibration(
            min_angles=data.get("min_angles", {}),
            max_angles=data.get("max_angles", {}),
        )


def normalize_angle(value: float, min_val: float, max_val: float) -> float:
    """Normalize angle into [0, 1] based on calibration."""
    if max_val <= min_val:
        return 0.0
    x = (value - min_val) / (max_val - min_val)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


class AngleProcessor:
    """Compute raw joint angles and optionally apply calibration normalization."""

    def __init__(self, calibration: Optional[Calibration] = None):
        self.engine = AngleEngine()
        self.calibration = calibration

    def compute(self, landmarks: Any) -> Dict[str, Dict[str, float]]:
        return self.engine.compute_hand_angles(landmarks)

    def compute_normalized(self, landmarks: Any) -> Dict[str, Dict[str, float]]:
        raw = self.compute(landmarks)
        calib = self.calibration
        if calib is None:
            return raw

        normalized: Dict[str, Dict[str, float]] = {}
        for finger, joints in raw.items():
            normalized[finger] = {}
            for joint_name, value in joints.items():
                min_val = calib.min_angles.get(finger, {}).get(joint_name)
                max_val = calib.max_angles.get(finger, {}).get(joint_name)
                if min_val is None or max_val is None:
                    normalized[finger][joint_name] = float(value)
                    continue
                normalized[finger][joint_name] = normalize_angle(float(value), float(min_val), float(max_val))
        return normalized
