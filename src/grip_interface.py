import time
import numpy as np

class GripInterface:
    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode

    def read_force(self) -> float:
        """Returns force in Newtons/Kg."""
        if self.mock_mode:
            # Returns a slow sine wave to simulate squeezing
            t = time.monotonic()
            val = 10 * np.sin(t) + 10 # 0 to 20 range
            return max(0, val)
        else:
            # TODO: Read from Load Cell Amp (HX711)
            return 0.0