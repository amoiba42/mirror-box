import time
import csv
import json
import os
import logging
from typing import Dict, Any
from angle_utils import AngleEngine

logger = logging.getLogger(__name__)

class FusionEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.angle_engine = AngleEngine()
        
        # Data Buffers
        self.frame_data = [] # Stores rows for CSV
        self.start_time = time.monotonic()
        
        # Paths
        self.output_dir = os.path.join("data", session_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "trial_log.csv")
        
        # CSV Header
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.header = [
            "timestamp_ns", "trial_time_s", 
            "hand_visible", "handedness", "confidence",
            "index_mcp", "index_pip", "index_dip",
            "emg_val", "grip_force"
        ]
        self.writer.writerow(self.header)

    def ingest(self, vision_result, emg_sample, grip_sample):
        """
        Central processing step. Called once per frame loop.
        """
        ts_now = time.monotonic_ns()
        trial_time = time.monotonic() - self.start_time
        
        # 1. Parse Vision
        hand_visible = 0
        handedness = "None"
        conf = 0.0
        angles = {'index': {'mcp': 0, 'pip': 0, 'dip': 0}} # Default
        
        if vision_result:
            hand_visible = 1
            handedness = vision_result.handedness
            conf = vision_result.confidence
            
            # Compute Angles
            full_angles = self.angle_engine.compute_hand_angles(vision_result.landmarks)
            angles = full_angles # Store all if needed, extracting index for CSV summary
            
        # 2. Parse Sensors
        emg_ts, emg_val = emg_sample
        grip_val = grip_sample
        
        # 3. Log to CSV
        row = [
            ts_now,
            round(trial_time, 4),
            hand_visible,
            handedness,
            round(conf, 2),
            angles.get('index', {}).get('mcp', 0),
            angles.get('index', {}).get('pip', 0),
            angles.get('index', {}).get('dip', 0),
            round(emg_val, 2),
            round(grip_val, 2)
        ]
        
        self.writer.writerow(row)
        
        # Buffer for runtime analysis (optional)
        self.frame_data.append(row)
        
        return {
            "angles": angles,
            "emg": emg_val,
            "grip": grip_val
        }

    def close(self):
        """Finalize logs and compute summary stats."""
        self.csv_file.close()
        
        # Generate simple JSON summary
        duration = time.monotonic() - self.start_time
        frames = len(self.frame_data)
        fps = frames / duration if duration > 0 else 0
        
        summary = {
            "session_id": self.session_id,
            "duration_sec": duration,
            "total_frames": frames,
            "avg_fps": fps,
            "file_path": self.csv_path
        }
        
        with open(os.path.join(self.output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"Session saved to {self.output_dir}")