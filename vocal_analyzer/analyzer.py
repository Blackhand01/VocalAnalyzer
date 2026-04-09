from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import (
    align_pitch_contours_by_dtw,
    compute_pitch_score_from_cents,
    compute_timing_path_metrics,
)
from .audio_processing import (
    audio_bundle_to_metadata,
    build_sync_feature,
    load_audio_bundle,
    normalize_peak_amplitude,
    select_user_analysis_segment,
)
from .logging_utils import build_default_logger
from .models import AudioBundle
from .pitch_processing import convert_hz_to_cents, extract_pitch_track, interpolate_nan_values
from .timeline import (
    attach_metric_deltas,
    build_take_output_dir_name,
    extract_take_datetime_from_filename,
    make_plot_times_unique,
    resolve_take_datetime,
    sort_user_paths_chronologically,
)
from .visualization import (
    build_performance_trend_explanation_html,
    write_performance_trend_plot,
    write_pitch_alignment_plot,
)


class VocalAnalyzer:
    def __init__(
        self,
        sr: int = 22050,
        fmin: float = 60.0,
        fmax: float = 500.0,
        hop_length: int = 256,
        logger: logging.Logger | None = None,
    ) -> None:
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
        self.logger = logger or self._build_default_logger()

    def analyze(self, user_path: str, reference_path: str, output_dir: str) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / "summary.json"
        chart_path = output_path / "pitch_alignment.html"

        summary: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "inputs": {"user_path": user_path, "reference_path": reference_path},
            "output_dir": str(output_path),
            "config": {
                "sample_rate": self.sr,
                "fmin_hz": self.fmin,
                "fmax_hz": self.fmax,
                "hop_length": self.hop_length,
            },
        }

        try:
            user_audio = self._load_audio_file(Path(user_path))
            reference_audio = self._load_audio_file(Path(reference_path))

            if user_audio.original_sr != reference_audio.original_sr:
                self.logger.warning(
                    "Original sample-rate mismatch | user=%sHz reference=%sHz",
                    user_audio.original_sr,
                    reference_audio.original_sr,
                )

            duration_gap_sec = abs(
                user_audio.processed_duration_sec - reference_audio.processed_duration_sec
            )
            if duration_gap_sec > 0.050:
                self.logger.warning(
                    "Processed duration mismatch may affect alignment | user=%.3fs reference=%.3fs gap=%.3fs",
                    user_audio.processed_duration_sec,
                    reference_audio.processed_duration_sec,
                    duration_gap_sec,
                )

            user_analysis_segment = self._select_user_analysis_segment(
                user_signal=user_audio.signal,
                reference_signal=reference_audio.signal,
            )
            user_analysis_signal = user_analysis_segment["signal"]

            user_pitch_track = self._extract_pitch(user_analysis_signal, label="user")
            reference_pitch_track = self._extract_pitch(reference_audio.signal, label="reference")

            user_cents_interpolated = self._hz_to_cents(user_pitch_track["f0_interp"])
            reference_cents_interpolated = self._hz_to_cents(reference_pitch_track["f0_interp"])

            alignment = self._align_contours(user_cents_interpolated, reference_cents_interpolated)
            idx_user = alignment["idx_user"]
            idx_reference = alignment["idx_ref"]

            aligned_user_pitch_hz = user_pitch_track["f0_interp"][idx_user]
            aligned_reference_pitch_hz = reference_pitch_track["f0_interp"][idx_reference]
            aligned_user_cents = user_cents_interpolated[idx_user]

            voiced_valid_mask = (
                user_pitch_track["voiced_mask"][idx_user]
                & reference_pitch_track["voiced_mask"][idx_reference]
                & np.isfinite(aligned_user_pitch_hz)
                & np.isfinite(aligned_reference_pitch_hz)
                & (aligned_user_pitch_hz > 0.0)
                & (aligned_reference_pitch_hz > 0.0)
            )
            if not np.any(voiced_valid_mask):
                raise ValueError("No voiced aligned frames available for scoring.")

            cents_error = 1200.0 * np.log2(
                aligned_user_pitch_hz[voiced_valid_mask]
                / aligned_reference_pitch_hz[voiced_valid_mask]
            )
            avg_abs_cents = float(np.mean(np.abs(cents_error)))
            pitch_score = self._pitch_score_from_cents(avg_abs_cents)

            timing_metrics = self._compute_timing_metrics(
                idx_user=idx_user,
                idx_ref=idx_reference,
                user_frame_count=user_pitch_track["f0_raw"].shape[0],
                ref_frame_count=reference_pitch_track["f0_raw"].shape[0],
            )

            user_cents_for_stability = aligned_user_cents[voiced_valid_mask]
            if user_cents_for_stability.size <= 1:
                stability_std = 0.0
            else:
                stability_std = float(np.std(np.diff(user_cents_for_stability)))

            self._write_pitch_plot(
                output_html_path=chart_path,
                aligned_ref_hz=aligned_reference_pitch_hz,
                aligned_user_hz=aligned_user_pitch_hz,
                idx_user=idx_user,
                idx_ref=idx_reference,
                user_frame_count=user_pitch_track["f0_raw"].shape[0],
                ref_frame_count=reference_pitch_track["f0_raw"].shape[0],
            )

            summary.update(
                {
                    "status": "success",
                    "metrics": {
                        "pitch_score": pitch_score,
                        "avg_abs_cents": avg_abs_cents,
                        "timing_score": timing_metrics["timing_score"],
                        "warping_path_mean_deviation": timing_metrics["mean_deviation"],
                        "warping_path_rms_deviation": timing_metrics["rms_deviation"],
                        "warping_path_max_deviation": timing_metrics["max_deviation"],
                        "pitch_dtw_distance_normalized": alignment["distance_normalized"],
                        "pitch_dtw_distance_total": alignment["distance_total"],
                        "dtw_warp_path_length": alignment["path_length"],
                        "vocal_stability_std": stability_std,
                        "voiced_aligned_frames": int(np.sum(voiced_valid_mask)),
                    },
                    "files": {
                        "summary_json": str(summary_path),
                        "pitch_alignment_html": str(chart_path),
                    },
                    "audio_metadata": {
                        "user": self._audio_bundle_to_metadata(user_audio),
                        "user_analysis_segment": user_analysis_segment["metadata"],
                        "reference": self._audio_bundle_to_metadata(reference_audio),
                        "user_pitch_frames": int(user_pitch_track["f0_raw"].shape[0]),
                        "reference_pitch_frames": int(reference_pitch_track["f0_raw"].shape[0]),
                    },
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Analysis failed")
            summary["error"] = str(exc)
        finally:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            self.logger.info("Wrote summary: %s", summary_path)

        return summary

    def analyze_batch(
        self,
        user_paths: list[str],
        reference_path: str,
        output_dir: str,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        takes_root = output_path / "takes"
        takes_root.mkdir(parents=True, exist_ok=True)

        history_json_path = output_path / "performance_history.json"
        trend_chart_path = output_path / "performance_trend.html"
        ordered_paths = self._sort_user_paths_chronologically(user_paths)

        history: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "reference_path": reference_path,
            "output_dir": str(output_path),
            "attempts": [],
            "files": {
                "performance_history_json": str(history_json_path),
                "performance_trend_html": str(trend_chart_path),
            },
        }

        try:
            for attempt_index, user_file in enumerate(ordered_paths, start=1):
                stat = user_file.stat()
                attempt_time_local, attempt_time_source = self._resolve_take_datetime(user_file)
                take_dir = takes_root / self._build_take_output_dir_name(
                    take_path=user_file,
                    attempt_index=attempt_index,
                    mtime_epoch=attempt_time_local.timestamp(),
                )

                result = self.analyze(
                    user_path=str(user_file),
                    reference_path=reference_path,
                    output_dir=str(take_dir),
                )

                attempt_record: dict[str, Any] = {
                    "attempt_index": attempt_index,
                    "take_name": user_file.name,
                    "user_path": str(user_file),
                    "attempt_time_local": attempt_time_local.isoformat(timespec="seconds"),
                    "attempt_time_utc": attempt_time_local.astimezone(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "attempt_time_source": attempt_time_source,
                    "file_mtime_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                    "file_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "status": result.get("status", "error"),
                    "output_dir": str(take_dir),
                }

                if "metrics" in result:
                    attempt_record["metrics"] = result["metrics"]
                if "files" in result:
                    attempt_record["files"] = result["files"]
                if "error" in result:
                    attempt_record["error"] = result["error"]

                history["attempts"].append(attempt_record)

            self._attach_metric_deltas(history["attempts"])

            successful_attempts = [
                attempt
                for attempt in history["attempts"]
                if attempt.get("status") == "success" and "metrics" in attempt
            ]
            if successful_attempts:
                self._write_performance_trend_plot(
                    output_html_path=trend_chart_path,
                    attempts=successful_attempts,
                )

            success_count = len(successful_attempts)
            total_count = len(history["attempts"])
            history["successful_attempts"] = success_count
            history["failed_attempts"] = total_count - success_count
            if success_count == 0:
                history["status"] = "error"
            elif success_count == total_count:
                history["status"] = "success"
            else:
                history["status"] = "partial_success"

            if success_count >= 2:
                first_metrics = successful_attempts[0]["metrics"]
                last_metrics = successful_attempts[-1]["metrics"]
                history["trend_overview"] = {
                    "pitch_score_delta_first_to_last": (
                        last_metrics["pitch_score"] - first_metrics["pitch_score"]
                    ),
                    "timing_score_delta_first_to_last": (
                        last_metrics["timing_score"] - first_metrics["timing_score"]
                    ),
                    "avg_abs_cents_delta_first_to_last": (
                        last_metrics["avg_abs_cents"] - first_metrics["avg_abs_cents"]
                    ),
                    "vocal_stability_std_delta_first_to_last": (
                        last_metrics["vocal_stability_std"]
                        - first_metrics["vocal_stability_std"]
                    ),
                }
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Batch analysis failed")
            history["error"] = str(exc)
        finally:
            history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            self.logger.info("Wrote performance history: %s", history_json_path)

        return history

    def _load_audio_file(self, path: Path) -> AudioBundle:
        return load_audio_bundle(path=path, target_sample_rate=self.sr, logger=self.logger)

    def _extract_pitch(self, signal: np.ndarray, label: str) -> dict[str, np.ndarray]:
        pitch_track = extract_pitch_track(
            signal=signal,
            sample_rate=self.sr,
            fmin_hz=self.fmin,
            fmax_hz=self.fmax,
            hop_length=self.hop_length,
            label=label,
            logger=self.logger,
        )
        return {
            "f0_raw": pitch_track.f0_raw,
            "f0_interp": pitch_track.f0_interp,
            "voiced_mask": pitch_track.voiced_mask,
        }

    def _select_user_analysis_segment(
        self,
        user_signal: np.ndarray,
        reference_signal: np.ndarray,
        mismatch_ratio_threshold: float = 1.25,
        search_hop_length: int = 1024,
    ) -> dict[str, Any]:
        return select_user_analysis_segment(
            user_signal=user_signal,
            reference_signal=reference_signal,
            sample_rate=self.sr,
            logger=self.logger,
            mismatch_ratio_threshold=mismatch_ratio_threshold,
            search_hop_length=search_hop_length,
        )

    def _build_sync_feature(self, signal: np.ndarray, hop_length: int) -> np.ndarray:
        return build_sync_feature(signal=signal, sample_rate=self.sr, hop_length=hop_length)

    @staticmethod
    def _sort_user_paths_chronologically(user_paths: list[str]) -> list[Path]:
        return sort_user_paths_chronologically(user_paths)

    @staticmethod
    def _resolve_take_datetime(path: Path) -> tuple[datetime, str]:
        return resolve_take_datetime(path)

    @staticmethod
    def _extract_take_datetime_from_filename(path: Path) -> datetime | None:
        return extract_take_datetime_from_filename(path)

    @staticmethod
    def _build_take_output_dir_name(take_path: Path, attempt_index: int, mtime_epoch: float) -> str:
        return build_take_output_dir_name(
            take_path=take_path,
            attempt_index=attempt_index,
            mtime_epoch=mtime_epoch,
        )

    @staticmethod
    def _attach_metric_deltas(attempts: list[dict[str, Any]]) -> None:
        attach_metric_deltas(attempts)

    @staticmethod
    def _normalize_audio(signal: np.ndarray) -> np.ndarray:
        return normalize_peak_amplitude(signal)

    @staticmethod
    def _interpolate_nans(values: np.ndarray) -> np.ndarray:
        return interpolate_nan_values(values)

    @staticmethod
    def _hz_to_cents(f0_hz: np.ndarray, base_hz: float = 55.0) -> np.ndarray:
        return convert_hz_to_cents(pitch_hz=f0_hz, base_hz=base_hz)

    @staticmethod
    def _pitch_score_from_cents(avg_abs_cents: float) -> float:
        return compute_pitch_score_from_cents(avg_abs_cents)

    def _align_contours(self, user_cents: np.ndarray, ref_cents: np.ndarray) -> dict[str, Any]:
        alignment = align_pitch_contours_by_dtw(
            user_cents_contour=user_cents,
            reference_cents_contour=ref_cents,
            logger=self.logger,
        )
        return {
            "idx_user": alignment.idx_user,
            "idx_ref": alignment.idx_ref,
            "path_length": alignment.path_length,
            "distance_total": alignment.distance_total,
            "distance_normalized": alignment.distance_normalized,
        }

    def _compute_timing_metrics(
        self,
        idx_user: np.ndarray,
        idx_ref: np.ndarray,
        user_frame_count: int,
        ref_frame_count: int,
    ) -> dict[str, float]:
        metrics = compute_timing_path_metrics(
            idx_user=idx_user,
            idx_ref=idx_ref,
            user_frame_count=user_frame_count,
            reference_frame_count=ref_frame_count,
            logger=self.logger,
        )
        return {
            "timing_score": metrics.timing_score,
            "mean_deviation": metrics.mean_deviation,
            "rms_deviation": metrics.rms_deviation,
            "max_deviation": metrics.max_deviation,
        }

    def _write_pitch_plot(
        self,
        output_html_path: Path,
        aligned_ref_hz: np.ndarray,
        aligned_user_hz: np.ndarray,
        idx_user: np.ndarray,
        idx_ref: np.ndarray,
        user_frame_count: int,
        ref_frame_count: int,
    ) -> None:
        write_pitch_alignment_plot(
            output_html_path=output_html_path,
            aligned_reference_pitch_hz=aligned_ref_hz,
            aligned_user_pitch_hz=aligned_user_hz,
            user_alignment_indices=idx_user,
            reference_alignment_indices=idx_ref,
            user_frame_count=user_frame_count,
            reference_frame_count=ref_frame_count,
            sample_rate=self.sr,
            hop_length=self.hop_length,
        )
        self.logger.info("Wrote chart: %s", output_html_path)

    def _write_performance_trend_plot(
        self,
        output_html_path: Path,
        attempts: list[dict[str, Any]],
    ) -> None:
        write_performance_trend_plot(output_html_path=output_html_path, attempts=attempts)
        self.logger.info("Wrote performance trend chart: %s", output_html_path)

    @staticmethod
    def _build_trend_explanation_html(attempts: list[dict[str, Any]]) -> str:
        return build_performance_trend_explanation_html(attempts)

    @staticmethod
    def _make_plot_times_unique(datetimes: list[datetime]) -> list[datetime]:
        return make_plot_times_unique(datetimes)

    @staticmethod
    def _audio_bundle_to_metadata(bundle: AudioBundle) -> dict[str, Any]:
        return audio_bundle_to_metadata(bundle)

    @staticmethod
    def _build_default_logger() -> logging.Logger:
        return build_default_logger()
