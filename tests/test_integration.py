from __future__ import annotations

import json
import unittest
from pathlib import Path

from vocal_analyzer.analyzer import VocalAnalyzer


USER_AUDIO = Path(
    "/Users/stefanoroybisignano/Music/GarageBand/Eyes Without A Face - Verse/MyVerse01/08:04:26, 00.49.wav"
)
USER_AUDIO_BATCH = [
    Path(
        "/Users/stefanoroybisignano/Music/GarageBand/Eyes Without A Face - Verse/MyVerse01/08:04:26, 00.49.wav"
    ),
    Path(
        "/Users/stefanoroybisignano/Music/GarageBand/Eyes Without A Face - Verse/MyVerse01/09:04:26, 00.49.wav"
    ),
    Path(
        "/Users/stefanoroybisignano/Music/GarageBand/Eyes Without A Face - Verse/MyVerse01/09:04:26, 00.49-2.wav"
    ),
]
REFERENCE_AUDIO = Path(
    "/Users/stefanoroybisignano/Music/GarageBand/Eyes Without A Face - Verse/Verse01.wav"
)
OUTPUT_DIR = Path(
    "/Users/stefanoroybisignano/Desktop/Singing/VocalAnalyzer/outputs/Eyes Without A Face/Verse01/"
)


@unittest.skipUnless(USER_AUDIO.exists() and REFERENCE_AUDIO.exists(), "Integration audio files not found.")
class TestIntegration(unittest.TestCase):
    def test_real_audio_run(self) -> None:
        analyzer = VocalAnalyzer()
        result = analyzer.analyze(
            user_path=str(USER_AUDIO),
            reference_path=str(REFERENCE_AUDIO),
            output_dir=str(OUTPUT_DIR),
        )
        self.assertEqual(result.get("status"), "success", msg=result.get("error"))

        summary_path = OUTPUT_DIR / "summary.json"
        chart_path = OUTPUT_DIR / "pitch_alignment.html"
        self.assertTrue(summary_path.exists())
        self.assertTrue(chart_path.exists())

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "success")
        metrics = payload.get("metrics", {})
        for key in [
            "pitch_score",
            "avg_abs_cents",
            "timing_score",
            "warping_path_rms_deviation",
            "pitch_dtw_distance_normalized",
            "vocal_stability_std",
        ]:
            self.assertIn(key, metrics)
            self.assertIsNotNone(metrics[key])

    @unittest.skipUnless(all(path.exists() for path in USER_AUDIO_BATCH), "Batch audio files not found.")
    def test_real_audio_batch_run(self) -> None:
        analyzer = VocalAnalyzer()
        result = analyzer.analyze_batch(
            user_paths=[str(path) for path in USER_AUDIO_BATCH],
            reference_path=str(REFERENCE_AUDIO),
            output_dir=str(OUTPUT_DIR),
        )
        self.assertIn(result.get("status"), {"success", "partial_success"})
        self.assertEqual(result.get("successful_attempts"), 3)

        history_path = OUTPUT_DIR / "performance_history.json"
        trend_path = OUTPUT_DIR / "performance_trend.html"
        self.assertTrue(history_path.exists())
        self.assertTrue(trend_path.exists())

        payload = json.loads(history_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("successful_attempts"), 3)
        self.assertEqual(len(payload.get("attempts", [])), 3)
        self.assertIn("trend_overview", payload)


if __name__ == "__main__":
    unittest.main()
