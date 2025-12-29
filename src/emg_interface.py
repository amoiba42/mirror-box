import time
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class EMGInterface:
    def __init__(self, channel_count=1, sample_rate=500, mock_mode=False):
        self.mock_mode = mock_mode
        self.fs = sample_rate
        self.channels = channel_count
        self.running = False
        
        # Mock State
        self.t_start = time.monotonic()
        self.burst_active = False
        self.last_burst_time = 0
        
        # Hardware Stub (e.g., SPI Handle)
        self.spi = None 

    def start(self):
        logger.info(f"EMG Interface Started (Mock: {self.mock_mode})")
        if not self.mock_mode:
            # TODO: Initialize spidev for MCP3008 ADC
            pass
        self.running = True

    def read_sample(self) -> Tuple[float, float]:
        """
        Returns (timestamp_ns, voltage_value).
        """
        ts = time.monotonic_ns()
        
        if self.mock_mode:
            val = self._generate_synthetic_emg()
        else:
            # TODO: Read from ADC
            # val = self.adc.read(0)
            val = 0.0
            
        return ts, val

    def _generate_synthetic_emg(self):
        """Generates baseline noise with periodic high-amplitude bursts."""
        current_time = time.monotonic()
        
        # Baseline noise (0-50 units)
        noise = np.random.normal(20, 5)
        
        # Create a burst every 5 seconds lasting 1 second
        cycle_time = current_time % 5.0
        if 2.0 < cycle_time < 3.0:
            # Muscle contraction (higher amplitude + frequency)
            signal = np.random.normal(500, 100)
            return max(0, signal)
        
        return max(0, noise)

    def stop(self):
        self.running = False