from __future__ import annotations

import unittest

import numpy as np

from vocal_analyzer.voice_quality import (
    compute_voice_quality_frames,
    summarize_aligned_voice_quality,
)


class TestVoiceQuality(unittest.TestCase):
    def test_compute_voice_quality_frames_outputs_expected_keys(self) -> None:
        sr = 22050
        t = np.linspace(0.0, 1.0, int(sr), endpoint=False)
        signal = 0.6 * np.sin(2 * np.pi * 220 * t)

        metrics = compute_voice_quality_frames(signal=signal, sample_rate=sr, hop_length=256)
        self.assertIn("rms_db", metrics)
        self.assertIn("spectral_centroid_hz", metrics)
        self.assertIn("hnr_db", metrics)
        self.assertGreater(metrics["rms_db"].size, 0)
        self.assertEqual(metrics["rms_db"].size, metrics["spectral_centroid_hz"].size)

    def test_summarize_aligned_voice_quality_returns_finite_values(self) -> None:
        sr = 22050
        t = np.linspace(0.0, 1.0, int(sr), endpoint=False)
        ref_signal = 0.6 * np.sin(2 * np.pi * 220 * t)
        user_signal = 0.6 * np.sin(2 * np.pi * 220 * t + 0.1) + 0.03 * np.random.default_rng(0).normal(
            size=t.shape[0]
        )

        user_frames = compute_voice_quality_frames(signal=user_signal, sample_rate=sr, hop_length=256)
        ref_frames = compute_voice_quality_frames(signal=ref_signal, sample_rate=sr, hop_length=256)

        user_frame_count = 80
        ref_frame_count = 80
        idx_user = np.arange(0, 80, dtype=np.int64)
        idx_ref = np.arange(0, 80, dtype=np.int64)

        summary = summarize_aligned_voice_quality(
            user_quality_frames=user_frames,
            reference_quality_frames=ref_frames,
            idx_user=idx_user,
            idx_reference=idx_ref,
            user_pitch_frame_count=user_frame_count,
            reference_pitch_frame_count=ref_frame_count,
        )

        expected_keys = {
            "dynamics_rms_user_mean_db",
            "dynamics_rms_reference_mean_db",
            "dynamics_rms_mae_db",
            "dynamics_rms_corr",
            "spectral_centroid_user_mean_hz",
            "spectral_centroid_reference_mean_hz",
            "spectral_centroid_gap_hz",
            "hnr_user_mean_db",
            "hnr_reference_mean_db",
            "hnr_gap_db",
        }
        self.assertTrue(expected_keys.issubset(summary.keys()))
        for value in summary.values():
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
