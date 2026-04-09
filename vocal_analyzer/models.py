from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioBundle:
    path: str
    signal: np.ndarray
    original_sr: int
    processed_sr: int
    original_channels: int
    original_frames: int
    original_duration_sec: float
    processed_samples: int
    processed_duration_sec: float


@dataclass
class PitchTrack:
    f0_raw: np.ndarray
    f0_interp: np.ndarray
    voiced_mask: np.ndarray


@dataclass
class AlignmentResult:
    idx_user: np.ndarray
    idx_ref: np.ndarray
    path_length: int
    distance_total: float
    distance_normalized: float


@dataclass
class TimingMetrics:
    timing_score: float
    mean_deviation: float
    rms_deviation: float
    max_deviation: float
