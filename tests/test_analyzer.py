from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from vocal_analyzer.analyzer import VocalAnalyzer


class TestVocalAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = VocalAnalyzer()

    def test_load_audio_resample_mono_normalize(self) -> None:
        sr = 44100
        seconds = 1.0
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        left = 0.2 * np.sin(2 * np.pi * 220 * t)
        right = 0.4 * np.sin(2 * np.pi * 220 * t)
        stereo = np.vstack([left, right]).T

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "stereo.wav"
            sf.write(str(wav_path), stereo, sr)
            bundle = self.analyzer._load_audio_file(wav_path)

        self.assertEqual(bundle.original_sr, 44100)
        self.assertEqual(bundle.processed_sr, 22050)
        self.assertEqual(bundle.original_channels, 2)
        self.assertTrue(np.max(np.abs(bundle.signal)) <= 1.0 + 1e-8)
        self.assertGreater(np.max(np.abs(bundle.signal)), 0.99)
        self.assertAlmostEqual(bundle.processed_duration_sec, seconds, places=2)

    def test_interpolate_nans(self) -> None:
        values = np.array([np.nan, 100.0, np.nan, 200.0, np.nan], dtype=np.float64)
        interp = self.analyzer._interpolate_nans(values)
        expected = np.array([100.0, 100.0, 150.0, 200.0, 200.0], dtype=np.float64)
        np.testing.assert_allclose(interp, expected, rtol=1e-6, atol=1e-6)

    def test_interpolate_nans_all_nan_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.analyzer._interpolate_nans(np.array([np.nan, np.nan], dtype=np.float64))

    def test_pitch_score_policy_breakpoints(self) -> None:
        self.assertEqual(self.analyzer._pitch_score_from_cents(10.0), 100.0)
        self.assertEqual(self.analyzer._pitch_score_from_cents(100.0), 0.0)
        self.assertAlmostEqual(self.analyzer._pitch_score_from_cents(55.0), 50.0, places=6)

    def test_dtw_normalization_deterministic(self) -> None:
        contour = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float64)
        alignment = self.analyzer._align_contours(contour, contour)
        self.assertEqual(alignment["distance_total"], 0.0)
        self.assertEqual(alignment["distance_normalized"], 0.0)
        self.assertGreater(alignment["path_length"], 0)

    def test_timing_metrics_perfect_diagonal(self) -> None:
        metrics = self.analyzer._compute_timing_metrics(
            idx_user=np.array([0, 1, 2, 3], dtype=np.int64),
            idx_ref=np.array([0, 1, 2, 3], dtype=np.int64),
            user_frame_count=4,
            ref_frame_count=4,
        )
        self.assertEqual(metrics["timing_score"], 100.0)
        self.assertEqual(metrics["mean_deviation"], 0.0)
        self.assertEqual(metrics["rms_deviation"], 0.0)

    def test_select_user_analysis_segment_crops_long_track(self) -> None:
        reference = np.zeros(self.analyzer.sr * 2, dtype=np.float32)
        reference[2000:5000] = 1.0
        reference[20000:23000] = 0.8

        user = np.zeros(self.analyzer.sr * 6, dtype=np.float32)
        start_sample = int(1.75 * self.analyzer.sr)
        user[start_sample : start_sample + reference.shape[0]] = reference

        segment = self.analyzer._select_user_analysis_segment(user, reference)

        self.assertEqual(segment["metadata"]["selection_method"], "auto_reference_window")
        self.assertAlmostEqual(segment["metadata"]["start_sec"], 1.75, places=1)
        self.assertAlmostEqual(segment["metadata"]["duration_sec"], 2.0, places=2)
        self.assertEqual(segment["signal"].shape[0], reference.shape[0])

    def test_analyze_batch_orders_attempts_chronologically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            user_files = [
                tmp_path / "take_b.wav",
                tmp_path / "take_a.wav",
                tmp_path / "take_c.wav",
            ]
            for path in user_files:
                path.write_bytes(b"")

            now = 1_700_000_000
            os.utime(user_files[0], (now + 20, now + 20))
            os.utime(user_files[1], (now + 10, now + 10))
            os.utime(user_files[2], (now + 30, now + 30))

            def fake_analyze(user_path: str, reference_path: str, output_dir: str) -> dict[str, object]:
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                summary = output / "summary.json"
                chart = output / "pitch_alignment.html"
                summary.write_text("{}", encoding="utf-8")
                chart.write_text("<html></html>", encoding="utf-8")
                suffix = Path(user_path).stem.split("_")[-1]
                base_value = float(ord(suffix[0]) - ord("a") + 1)
                return {
                    "status": "success",
                    "metrics": {
                        "pitch_score": 10.0 * base_value,
                        "avg_abs_cents": 100.0 - base_value,
                        "timing_score": 20.0 * base_value,
                        "warping_path_rms_deviation": 0.1 * base_value,
                        "vocal_stability_std": 5.0 * base_value,
                    },
                    "files": {
                        "summary_json": str(summary),
                        "pitch_alignment_html": str(chart),
                    },
                }

            self.analyzer.analyze = fake_analyze  # type: ignore[method-assign]
            result = self.analyzer.analyze_batch(
                user_paths=[str(user_files[0]), str(user_files[1]), str(user_files[2])],
                reference_path=str(tmp_path / "reference.wav"),
                output_dir=str(tmp_path / "reports"),
            )

            ordered_names = [attempt["take_name"] for attempt in result["attempts"]]
            self.assertEqual(ordered_names, ["take_a.wav", "take_b.wav", "take_c.wav"])
            self.assertEqual(result["status"], "success")
            self.assertTrue((tmp_path / "reports" / "performance_history.json").exists())
            self.assertTrue((tmp_path / "reports" / "performance_trend.html").exists())

    def test_sort_user_paths_uses_filename_time_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            take_09 = tmp_path / "09:04:26, 00.49.wav"
            take_10 = tmp_path / "10:04:26, 00.49.wav"
            take_09.write_bytes(b"")
            take_10.write_bytes(b"")

            # Force opposite mtime ordering to verify filename timestamp priority.
            base = 1_700_000_000
            os.utime(take_09, (base + 100, base + 100))
            os.utime(take_10, (base + 50, base + 50))

            ordered = self.analyzer._sort_user_paths_chronologically([str(take_10), str(take_09)])
            self.assertEqual([path.name for path in ordered], [take_09.name, take_10.name])

    def test_make_plot_times_unique_for_duplicates(self) -> None:
        dt = datetime(2026, 4, 9, 0, 49, 0, tzinfo=timezone.utc)
        result = self.analyzer._make_plot_times_unique([dt, dt, dt + timedelta(minutes=1), dt])
        self.assertEqual(result[0], dt)
        self.assertEqual(result[1], dt + timedelta(seconds=1))
        self.assertEqual(result[2], dt + timedelta(minutes=1))
        self.assertEqual(result[3], dt + timedelta(seconds=2))


if __name__ == "__main__":
    unittest.main()
