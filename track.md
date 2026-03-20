# Change Log

This file tracks the changes made to the environment and dependencies to resolve the Python environment and `picamera2` issues.

## 2026-03-18 (Environment + Runtime)

- Identified that `picamera2`/`libcamera` imports were blocking trial execution inside Python envs.
- Confirmed `mediapipe` imports successfully in the active `python3`.
- Switched the trial camera path to OpenCV+GStreamer `libcamerasrc` instead of importing `picamera2`.

## 2026-03-18 (Code Change)

- Updated [run_trial.py](run_trial.py) to remove `from picamera2 import Picamera2`.
- Attempted a minimal `GStreamerCsiCamera` wrapper; camera open failed at runtime.
- Switched [run_trial.py](run_trial.py) to reuse [src/camera_interface.py](src/camera_interface.py) with `use_gstreamer=True` for Real mode.
	- This keeps us off `picamera2/libcamera` Python bindings.
	- It still uses `libcamerasrc` when possible, with built-in FPS fallback and a CLI fallback (`rpicam-vid`/`libcamera-vid`) if OpenCV-GStreamer isn’t available.
- “Headless Mode” still skips `cv2.imshow()`/`cv2.waitKey()` and `cv2.destroyAllWindows()`.
- Added an auto-detect safety: if GUI mode is chosen but no `DISPLAY`/`WAYLAND_DISPLAY` is present, force Headless Mode to avoid OpenCV/Qt aborting.

## 2026-03-19 (Modular Refactor & Feature Expansion)

### Phase 1: Self-Contained "final/" Package
- Created a new [final/](final/) directory containing a complete, modular rewrite of the system.
- Established a **No-Import-from-src** rule to ensure the "final" production code is independent and portable.
- Files added/rewritten:
    - [final/camera_gstream.py](final/camera_gstream.py): Robust camera backends (GStreamer-OpenCV + CLI-MJPEG fallback + Mock). Now supports index-based Multi-CSI selection.
    - [final/hand_tracking.py](final/hand_tracking.py): MediaPipe wrapper with **landmark smoothing (EMA)** for stable angle computation.
    - [final/angles.py](final/angles.py): Flexion/Extension angle engine for all 5 fingers + calibration normalization.
    - [final/fusion_engine.py](final/fusion_engine.py): Unified CSV/JSON logger supporting 15 joint angles (MCP/PIP/DIP) per frame.
    - [final/mirror_display.py](final/mirror_display.py): Horizontal flip logic for patient feedback.

### Phase 2: Dual-Camera Mirror Therapy
- Created [final/run_dual_trial.py](final/run_dual_trial.py) to support split-screen mirror therapy:
    - **Patient View**: Mirrored footage of the "Good" hand (no markers/skeletons) to create the movement illusion.
    - **Supervisor View**: Toggled with 's' key; shows raw footage of **both** hands (with markers/skeletons) side-by-side for clinical monitoring.
    - **Mixed-Mode Support**: Allows running one real camera alongside one mock (simulated) camera for testing.
    - **Startup Prompt**: Asks the supervisor to select the "Affected (Bad)" side to correctly route data logging and mirroring.

### Phase 3: Performance & Calibration
- **Stability**: Implemented 0.4-factor EMA smoothing on MediaPipe landmarks to eliminate jitter in angle logs.
- **Custom Calibration**: Built [final/calibrate.py](final/calibrate.py) to capture a patient's specific Range of Motion (Extension/Flexion) and generate `calibration.json`.
- **UI Integration**: Displays real-time closure percentages (0-100%) in the Supervisor View based on the current calibration.
