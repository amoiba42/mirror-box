#!/usr/bin/env python3
"""Test each module in isolation and print a simple PASS/FAIL report.

This script is designed to be safe over SSH/headless (no GUI required).
"""

import argparse
import time
import traceback
import os
import sys
import numpy as np

if __package__ is None:
    # Allow `python3 final/test_modules.py` from the repo root.
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from final.camera_gstream import AutoCamera, CameraConfig
from final.angles import AngleProcessor, Calibration
from final.mirror_display import HorizontalMirror
from final.hand_tracking import HandTrackingModule


def _run_test(name: str, fn):
    start = time.monotonic()
    try:
        fn()
        dt = (time.monotonic() - start) * 1000
        return True, f"PASS ({dt:.0f} ms)"
    except Exception as e:
        dt = (time.monotonic() - start) * 1000
        tb = traceback.format_exc(limit=6)
        return False, f"FAIL ({dt:.0f} ms): {e}\n{tb}"


## EMG + grip disabled in final/
# def test_emg(mock: bool):
#     ...
#
# def test_grip(mock: bool):
#     ...


def test_mirror():
    mirror = HorizontalMirror(enabled=True)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, :10] = 255
    out = mirror.apply(frame)
    assert out is not None
    assert out.shape == frame.shape
    assert int(out[:, -1].sum()) > 0


def test_angles():
    ap = AngleProcessor(calibration=None)

    class _LM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Landmarks:
        def __init__(self):
            self.landmark = [_LM(0.0, 0.0, 0.0) for _ in range(21)]

    raw = ap.compute(_Landmarks())
    assert "index" in raw


def test_calibration_load(path: str):
    if not path:
        return
    calib = Calibration.load(path)
    assert isinstance(calib.min_angles, dict)


def test_hand_tracker_init():
    _ = HandTrackingModule(model_complexity=0)


def test_camera_real(width: int, height: int, fps: int, frames: int):
    # Use AutoCamera so the test works even if OpenCV lacks GStreamer support.
    # cam = Pi5GStreamerCamera(CameraConfig(width=width, height=height, framerate=fps))
    cam = AutoCamera(CameraConfig(width=width, height=height, framerate=fps))
    cam.start()
    try:
        got = 0
        deadline = time.monotonic() + 5.0
        while got < frames and time.monotonic() < deadline:
            frame, ts = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            assert ts > 0
            got += 1
        assert got > 0
    finally:
        cam.stop()


def main():
    p = argparse.ArgumentParser(description="Module test runner for final/.")
    p.add_argument("--test-camera", action="store_true", help="Run the real CSI camera test (requires camera + GStreamer).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera-frames", type=int, default=3)
    p.add_argument("--calibration", type=str, default="calibration.json")
    args = p.parse_args()

    results = []

    # EMG + grip disabled
    # results.append(("emg(mock)", *_run_test("emg(mock)", lambda: test_emg(mock=True))))
    # results.append(("grip(mock)", *_run_test("grip(mock)", lambda: test_grip(mock=True))))
    results.append(("mirror", *_run_test("mirror", test_mirror)))
    results.append(("angles", *_run_test("angles", test_angles)))
    results.append(("calibration(load)", *_run_test("calibration(load)", lambda: test_calibration_load(args.calibration))))
    results.append(("hand_tracker(init)", *_run_test("hand_tracker(init)", test_hand_tracker_init)))

    if args.test_camera:
        results.append(
            (
                "camera(real)",
                *_run_test(
                    "camera(real)",
                    lambda: test_camera_real(args.width, args.height, args.fps, args.camera_frames),
                ),
            )
        )

    ok = True
    for name, passed, msg in results:
        status = "OK" if passed else "NO"
        print(f"[{status}] {name}: {msg.splitlines()[0]}")
        if not passed:
            ok = False

    if not ok:
        print("\n--- Details for failures ---")
        for name, passed, msg in results:
            if not passed:
                print(f"\n{name}\n{msg}")

    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
