import time
import logging
import signal
from typing import Optional
import cv2
# Import components
from camera_interface import CameraInterface
from hand_tracker import HandTracker
from emg_interface import EMGInterface
from grip_interface import GripInterface
from fusion_engine import FusionEngine
from mirror_display import MirrorDisplay

logger = logging.getLogger(__name__)

class TrialController:
    """
    Orchestrates the entire session:
    1. Manages sensor lifecycle.
    2. Runs the main sync loop.
    3. Handles safe shutdown.
    """
    def __init__(self, config: dict):
        self.config = config
        self.running = False
        self.duration = config.get('duration', 60)
        
        # Components
        self.cam = CameraInterface(
            source=0, 
            mock_mode=config.get('mock', False)
        )
        self.tracker = HandTracker(model_complexity=config.get('complexity', 0))
        self.emg = EMGInterface(mock_mode=config.get('mock', False))
        self.grip = GripInterface(mock_mode=config.get('mock', False))
        self.display = MirrorDisplay()
        self.fusion = FusionEngine(session_id=config.get('session_id', 'default_session'))

        # Safety: Catch Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.warning("Interrupt signal received. Stopping trial...")
        self.running = False

    def run(self):
        logger.info("Initializing sensors...")
        self.cam.start()
        self.emg.start()
        
        logger.info("Starting Trial Loop...")
        self.running = True
        start_time = time.monotonic()
        
        try:
            while self.running:
                # Check Duration
                elapsed = time.monotonic() - start_time
                if elapsed >= self.duration:
                    logger.info("Trial duration reached.")
                    break

                # 1. Capture
                frame, ts = self.cam.read()
                if frame is None:
                    continue
                
                emg_data = self.emg.read_sample()
                grip_data = self.grip.read_force()

                # 2. Process (Tracking)
                hand_res = self.tracker.process(frame, ts)

                # 3. Mirror Logic
                mirror_frame = self.display.process(frame)
                self.display.draw_guidelines(mirror_frame)

                # 4. Visualization (Skeleton on Mirror)
                if hand_res:
                    # Note: We draw on the mirror frame. 
                    # Ideally, we should flip landmarks, but for simple feedback
                    # drawing on the *original* then flipping, OR tracking the flipped image works.
                    # Simpler approach: Draw on original, then flip.
                    # Current approach: Draw on Mirror (Requires re-tracking or coordinate math).
                    # FAST PATH: Track Original -> Draw on Original -> Flip.
                    pass 
                    
                # 5. Data Fusion (Log Original Data)
                self.fusion.ingest(hand_res, emg_data, grip_data)

                # 6. Show
                self.display.show(mirror_frame)
                
                # Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False

        except Exception as e:
            logger.error(f"Runtime Error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        logger.info("Shutting down...")
        self.cam.stop()
        self.emg.stop()
        self.fusion.close()
        cv2.destroyAllWindows()