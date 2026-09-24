"""Shared data types for the rectified stereo perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoFrame:
    """One synchronized, rectified stereo pair in the left camera frame."""

    left: np.ndarray
    right: np.ndarray
    left_projection: np.ndarray
    right_projection: np.ndarray
    capture_time_ns: int = 0
