from __future__ import annotations

import logging

import librosa
import numpy as np

from .models import PitchTrack


def interpolate_nan_values(values: np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values_array)
    if not np.any(finite_mask):
        raise ValueError("All pitch values are NaN.")

    interpolation_indices = np.arange(values_array.shape[0])
    interpolated = values_array.copy()
    interpolated[~finite_mask] = np.interp(
        interpolation_indices[~finite_mask],
        interpolation_indices[finite_mask],
        values_array[finite_mask],
    )
    return interpolated


def convert_hz_to_cents(pitch_hz: np.ndarray, base_hz: float = 55.0) -> np.ndarray:
    pitch_array = np.asarray(pitch_hz, dtype=np.float64)
    if np.any(pitch_array <= 0.0):
        raise ValueError("Pitch values must be > 0 for cents conversion.")
    return 1200.0 * np.log2(pitch_array / base_hz)


def extract_pitch_track(
    signal: np.ndarray,
    sample_rate: int,
    fmin_hz: float,
    fmax_hz: float,
    hop_length: int,
    label: str,
    logger: logging.Logger,
) -> PitchTrack:
    f0_values, _, _ = librosa.pyin(
        signal,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sample_rate,
        hop_length=hop_length,
    )

    voiced_mask = np.isfinite(f0_values)
    voiced_count = int(np.sum(voiced_mask))
    total_frame_count = int(f0_values.shape[0])
    logger.info("%s pYIN voiced frames: %s/%s", label, voiced_count, total_frame_count)

    if voiced_count == 0:
        raise ValueError(f"{label} track has no voiced frames after pYIN.")

    interpolated_pitch = interpolate_nan_values(f0_values)
    return PitchTrack(
        f0_raw=f0_values,
        f0_interp=interpolated_pitch,
        voiced_mask=voiced_mask,
    )
