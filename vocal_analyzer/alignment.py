from __future__ import annotations

import logging

import numpy as np
from dtw import dtw
from scipy.spatial.distance import cdist

from .models import AlignmentResult, TimingMetrics


def compute_pitch_score_from_cents(avg_abs_cents: float) -> float:
    if avg_abs_cents <= 10.0:
        return 100.0
    if avg_abs_cents >= 100.0:
        return 0.0
    return float(np.clip(100.0 * ((100.0 - avg_abs_cents) / 90.0), 0.0, 100.0))


def align_pitch_contours_by_dtw(
    user_cents_contour: np.ndarray,
    reference_cents_contour: np.ndarray,
    logger: logging.Logger,
) -> AlignmentResult:
    if user_cents_contour.size == 0 or reference_cents_contour.size == 0:
        raise ValueError("Contours cannot be empty for DTW alignment.")

    local_cost_matrix = cdist(
        user_cents_contour.reshape(-1, 1),
        reference_cents_contour.reshape(-1, 1),
        metric="cityblock",
    )
    dtw_alignment = dtw(local_cost_matrix, keep_internals=False)

    idx_user = np.asarray(dtw_alignment.index1, dtype=np.int64)
    idx_ref = np.asarray(dtw_alignment.index2, dtype=np.int64)
    path_length = int(idx_user.shape[0])
    if path_length == 0:
        raise ValueError("DTW alignment returned an empty path.")

    distance_total = float(dtw_alignment.distance)
    distance_normalized = float(distance_total / path_length)

    logger.info(
        "DTW completed | total_distance=%.6f path_length=%s normalized=%.6f",
        distance_total,
        path_length,
        distance_normalized,
    )

    return AlignmentResult(
        idx_user=idx_user,
        idx_ref=idx_ref,
        path_length=path_length,
        distance_total=distance_total,
        distance_normalized=distance_normalized,
    )


def compute_timing_path_metrics(
    idx_user: np.ndarray,
    idx_ref: np.ndarray,
    user_frame_count: int,
    reference_frame_count: int,
    logger: logging.Logger,
) -> TimingMetrics:
    if idx_user.size == 0 or idx_ref.size == 0:
        raise ValueError("Warping path cannot be empty for timing metrics.")

    user_denominator = max(user_frame_count - 1, 1)
    reference_denominator = max(reference_frame_count - 1, 1)
    normalized_user_path = idx_user.astype(np.float64) / user_denominator
    normalized_reference_path = idx_ref.astype(np.float64) / reference_denominator
    path_deviations = np.abs(normalized_user_path - normalized_reference_path)

    mean_deviation = float(np.mean(path_deviations))
    rms_deviation = float(np.sqrt(np.mean(np.square(path_deviations))))
    max_deviation = float(np.max(path_deviations))
    timing_score = float(np.clip(100.0 * (1.0 - rms_deviation), 0.0, 100.0))

    logger.info(
        "Timing path metrics | score=%.3f mean_dev=%.6f rms_dev=%.6f max_dev=%.6f",
        timing_score,
        mean_deviation,
        rms_deviation,
        max_deviation,
    )

    return TimingMetrics(
        timing_score=timing_score,
        mean_deviation=mean_deviation,
        rms_deviation=rms_deviation,
        max_deviation=max_deviation,
    )
