"""Concept drift detector using ADWIN (Adaptive Windowing).

Detects when the workload distribution changes (e.g., model switches
from CNN layers to attention layers) so the RL agent knows it should
re-evaluate its policy rather than relying on stale rewards.
"""

from __future__ import annotations

import math
from collections import deque


class DriftDetector:
    """ADWIN-inspired drift detector.

    Maintains a sliding window of reward values. When the difference
    between the mean of the first half and the second half of the window
    exceeds a statistical threshold, drift is detected.

    This is a simplified version of ADWIN — the full algorithm uses
    variable-length windows and compression, but the core idea (detect
    distribution change via window-mean comparison) is the same.
    """

    def __init__(
        self,
        window_size: int = 30,
        threshold: float = 2.0,
        min_samples: int = 10,
    ) -> None:
        """Initialize drift detector.

        Args:
            window_size: Size of the sliding window.
            threshold: Number of standard deviations for the drift threshold.
            min_samples: Minimum samples before drift detection activates.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples
        self.window: deque[float] = deque(maxlen=window_size)
        self.drift_detected: bool = False
        self.drift_count: int = 0

    def update(self, reward: float) -> bool:
        """Add a reward to the window and check for drift.

        Returns True if drift is detected on this update.
        """
        self.window.append(reward)
        self.drift_detected = False

        if len(self.window) < self.min_samples:
            return False

        values = list(self.window)
        mid = len(values) // 2

        if mid < 2:
            return False

        first_half = values[:mid]
        second_half = values[mid:]

        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)

        # Standard deviation of the full window
        full_mean = sum(values) / len(values)
        variance = sum((v - full_mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.001

        # Drift if the difference between halves exceeds threshold * std
        diff = abs(mean_first - mean_second)
        if diff > self.threshold * std:
            self.drift_detected = True
            self.drift_count += 1
            # Reset window to adapt to new distribution
            self.window.clear()
            self.window.extend(second_half)
            return True

        return False

    def reset(self) -> None:
        self.window.clear()
        self.drift_detected = False
        self.drift_count = 0
