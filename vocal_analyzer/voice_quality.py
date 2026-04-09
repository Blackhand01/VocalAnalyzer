from __future__ import annotations

from typing import Any

import librosa
import numpy as np


def _resample_feature_to_frame_count(feature_values: np.ndarray, target_frame_count: int) -> np.ndarray:
    values = np.asarray(feature_values, dtype=np.float64)
    if target_frame_count <= 0:
        return np.array([], dtype=np.float64)
    if values.size == target_frame_count:
        return values
    if values.size == 0:
        return np.zeros(target_frame_count, dtype=np.float64)
    if values.size == 1:
        return np.full(target_frame_count, float(values[0]), dtype=np.float64)

    x_old = np.linspace(0.0, 1.0, num=values.size)
    x_new = np.linspace(0.0, 1.0, num=target_frame_count)
    return np.interp(x_new, x_old, values)


def _safe_correlation(series_a: np.ndarray, series_b: np.ndarray) -> float:
    if series_a.size <= 1 or series_b.size <= 1:
        return 0.0
    std_a = float(np.std(series_a))
    std_b = float(np.std(series_b))
    if std_a <= 1e-8 or std_b <= 1e-8:
        return 0.0
    return float(np.corrcoef(series_a, series_b)[0, 1])


def compute_voice_quality_frames(
    signal: np.ndarray,
    sample_rate: int,
    hop_length: int,
    frame_length: int = 2048,
    harmonic_margin: float = 3.0,
) -> dict[str, np.ndarray]:
    eps = 1e-10

    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))

    spectral_centroid_hz = librosa.feature.spectral_centroid(
        y=signal,
        sr=sample_rate,
        hop_length=hop_length,
    )[0]

    harmonic_signal = librosa.effects.harmonic(y=signal, margin=harmonic_margin)
    residual_noise = signal - harmonic_signal

    harmonic_rms = librosa.feature.rms(
        y=harmonic_signal,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    noise_rms = librosa.feature.rms(
        y=residual_noise,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    hnr_db = 20.0 * np.log10((harmonic_rms + eps) / (noise_rms + eps))

    return {
        "rms_db": rms_db,
        "spectral_centroid_hz": spectral_centroid_hz,
        "hnr_db": hnr_db,
    }


def summarize_aligned_voice_quality(
    user_quality_frames: dict[str, np.ndarray],
    reference_quality_frames: dict[str, np.ndarray],
    idx_user: np.ndarray,
    idx_reference: np.ndarray,
    user_pitch_frame_count: int,
    reference_pitch_frame_count: int,
) -> dict[str, Any]:
    user_rms_db = _resample_feature_to_frame_count(
        user_quality_frames["rms_db"],
        target_frame_count=user_pitch_frame_count,
    )
    ref_rms_db = _resample_feature_to_frame_count(
        reference_quality_frames["rms_db"],
        target_frame_count=reference_pitch_frame_count,
    )

    user_centroid_hz = _resample_feature_to_frame_count(
        user_quality_frames["spectral_centroid_hz"],
        target_frame_count=user_pitch_frame_count,
    )
    ref_centroid_hz = _resample_feature_to_frame_count(
        reference_quality_frames["spectral_centroid_hz"],
        target_frame_count=reference_pitch_frame_count,
    )

    user_hnr_db = _resample_feature_to_frame_count(
        user_quality_frames["hnr_db"],
        target_frame_count=user_pitch_frame_count,
    )
    ref_hnr_db = _resample_feature_to_frame_count(
        reference_quality_frames["hnr_db"],
        target_frame_count=reference_pitch_frame_count,
    )

    aligned_user_rms_db = user_rms_db[idx_user]
    aligned_ref_rms_db = ref_rms_db[idx_reference]
    aligned_user_centroid_hz = user_centroid_hz[idx_user]
    aligned_ref_centroid_hz = ref_centroid_hz[idx_reference]
    aligned_user_hnr_db = user_hnr_db[idx_user]
    aligned_ref_hnr_db = ref_hnr_db[idx_reference]

    dynamics_rms_corr = _safe_correlation(aligned_user_rms_db, aligned_ref_rms_db)

    return {
        "dynamics_rms_user_mean_db": float(np.mean(aligned_user_rms_db)),
        "dynamics_rms_reference_mean_db": float(np.mean(aligned_ref_rms_db)),
        "dynamics_rms_mae_db": float(np.mean(np.abs(aligned_user_rms_db - aligned_ref_rms_db))),
        "dynamics_rms_corr": dynamics_rms_corr,
        "spectral_centroid_user_mean_hz": float(np.mean(aligned_user_centroid_hz)),
        "spectral_centroid_reference_mean_hz": float(np.mean(aligned_ref_centroid_hz)),
        "spectral_centroid_gap_hz": float(
            np.mean(np.abs(aligned_user_centroid_hz - aligned_ref_centroid_hz))
        ),
        "hnr_user_mean_db": float(np.mean(aligned_user_hnr_db)),
        "hnr_reference_mean_db": float(np.mean(aligned_ref_hnr_db)),
        "hnr_gap_db": float(np.mean(np.abs(aligned_user_hnr_db - aligned_ref_hnr_db))),
    }
