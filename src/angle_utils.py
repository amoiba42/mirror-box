import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class AngleEngine:
    """
    Computes joint angles from MediaPipe landmarks.
    Focuses on flexion/extension of MCP, PIP, DIP joints.
    """
    
    # MediaPipe Hand Landmark Indices
    JOINTS = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20],
        'wrist': 0
    }

    @staticmethod
    def _compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculates the angle at p2 formed by lines p1-p2 and p2-p3.
        Returns degrees (0-180).
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        unit_v1 = v1 / norm1
        unit_v2 = v2 / norm2
        
        # Dot product -> angle
        dot_product = np.dot(unit_v1, unit_v2)
        # Clip to prevent numerical errors outside domain of arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)

    def compute_hand_angles(self, landmarks) -> Dict[str, Dict[str, float]]:
        """
        Extracts MCP, PIP, DIP angles for all fingers.
        Args:
            landmarks: MediaPipe normalized landmark object list.
        Returns:
            Dictionary structure: {'index': {'mcp': 45.0, 'pip': 30.0, ...}, ...}
        """
        # Convert landmarks to numpy array (N, 3)
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        
        angles = {}
        
        for finger, idxs in self.JOINTS.items():
            if finger == 'wrist':
                continue
            
            if finger == 'thumb':
                # Thumb logic is slightly different (CMC, MCP, IP)
                # Simplified here to match structure
                mcp = self._compute_angle(coords[idxs[0]], coords[idxs[1]], coords[idxs[2]])
                ip = self._compute_angle(coords[idxs[1]], coords[idxs[2]], coords[idxs[3]])
                angles[finger] = {'mcp': round(mcp, 1), 'ip': round(ip, 1)}
            else:
                # MCP: Wrist -> CMC(idx0) -> PIP(idx1) ? 
                # Standard flexion: Angle at MCP (idx0) using Wrist and PIP?
                # Simplified 3-point segment:
                # MCP angle: Vector(Index0->Wrist) vs Vector(Index0->Index1) is tricky.
                # Standard Approx:
                # MCP = Angle at idxs[0] (Knuckle) formed by Wrist and idxs[1]
                # PIP = Angle at idxs[1] formed by idxs[0] and idxs[2]
                # DIP = Angle at idxs[2] formed by idxs[1] and idxs[3]
                
                mcp = self._compute_angle(coords[0], coords[idxs[0]], coords[idxs[1]])
                # Correction: 180 - angle gives flexion displacement from straight
                mcp_flex = 180 - mcp 
                
                pip = self._compute_angle(coords[idxs[0]], coords[idxs[1]], coords[idxs[2]])
                pip_flex = 180 - pip
                
                dip = self._compute_angle(coords[idxs[1]], coords[idxs[2]], coords[idxs[3]])
                dip_flex = 180 - dip
                
                angles[finger] = {
                    'mcp': round(mcp_flex, 1),
                    'pip': round(pip_flex, 1),
                    'dip': round(dip_flex, 1)
                }
                
        return angles