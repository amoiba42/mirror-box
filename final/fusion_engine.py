import csv
import json
import logging
import os
import time
from typing import Any, Dict

from final.angles import AngleEngine

logger = logging.getLogger(__name__)


class FusionEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.angle_engine = AngleEngine()

        self._fingers = ["thumb", "index", "middle", "ring", "pinky"]
        self._joints = ["mcp", "pip", "dip"]

        self.frame_data = []
        self.start_time = time.monotonic()

        self.output_dir = os.path.join("data", session_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "trial_log.csv")

        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.header = [
            "timestamp_ns",
            "trial_time_s",
            "hand_visible",
            "handedness",
            "confidence",
            *[f"{finger}_{joint}" for finger in self._fingers for joint in self._joints],
            # "emg_val",    # disabled
            # "grip_force", # disabled
        ]
        self.writer.writerow(self.header)

    def ingest(self, vision_result: Any) -> Dict[str, Any]:
        ts_now = time.monotonic_ns()
        trial_time = time.monotonic() - self.start_time

        hand_visible = 0
        handedness = "None"
        conf = 0.0
        angles = {finger: {joint: 0.0 for joint in self._joints} for finger in self._fingers}

        if vision_result:
            hand_visible = 1
            handedness = getattr(vision_result, "handedness", "Unknown")
            conf = float(getattr(vision_result, "confidence", 0.0))

            landmarks = getattr(vision_result, "landmarks", None)
            if landmarks is not None:
                angles = self.angle_engine.compute_hand_angles(landmarks)

        # EMG + grip disabled
        # emg_ts, emg_val = emg_sample
        # grip_val = float(grip_sample)

        angle_cells = [
            angles.get(finger, {}).get(joint, 0.0)
            for finger in self._fingers
            for joint in self._joints
        ]

        row = [
            ts_now,
            round(trial_time, 4),
            hand_visible,
            handedness,
            round(conf, 2),
            *angle_cells,
            # round(float(emg_val), 2),
            # round(grip_val, 2),
        ]

        self.writer.writerow(row)
        self.frame_data.append(row)

        return {"angles": angles}

    def close(self) -> None:
        try:
            self.csv_file.close()
        except Exception:
            pass

        duration = time.monotonic() - self.start_time
        frames = len(self.frame_data)
        fps = frames / duration if duration > 0 else 0

        summary = {
            "session_id": self.session_id,
            "duration_sec": duration,
            "total_frames": frames,
            "avg_fps": fps,
            "file_path": self.csv_path,
        }

        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Session saved to {self.output_dir}")
