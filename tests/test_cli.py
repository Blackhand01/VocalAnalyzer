from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from vocal_analyzer.cli import _normalize_verse_name, _resolve_paths


class TestCliHelpers(unittest.TestCase):
    def test_normalize_verse_name(self) -> None:
        self.assertEqual(_normalize_verse_name("Verse01"), "Verse01")
        self.assertEqual(_normalize_verse_name("verse02"), "Verse02")
        self.assertEqual(_normalize_verse_name("03"), "Verse03")

    def test_resolve_paths_verse_mode_with_session_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir) / "Song - Verse"
            user_dir = session_dir / "MyVerse01"
            user_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "Verse01.wav").write_bytes(b"ref")
            (user_dir / "09_take.wav").write_bytes(b"a")
            (user_dir / "08_take.wav").write_bytes(b"b")

            args = argparse.Namespace(
                user=None,
                reference=None,
                output=None,
                verse="Verse01",
                session_dir=str(session_dir),
                search_root="~/Music/GarageBand",
            )
            resolved = _resolve_paths(args)

            self.assertEqual(resolved["mode"], "verse_auto")
            self.assertEqual(resolved["verse_name"], "Verse01")
            self.assertEqual(resolved["reference_path"], str((session_dir / "Verse01.wav").resolve()))
            self.assertEqual(
                resolved["user_paths"],
                [
                    str((user_dir / "08_take.wav").resolve()),
                    str((user_dir / "09_take.wav").resolve()),
                ],
            )
            self.assertIn(str(Path.cwd() / "outputs"), resolved["output_dir"])

    def test_resolve_paths_explicit_mode(self) -> None:
        args = argparse.Namespace(
            user=["/tmp/a.wav", "/tmp/b.wav"],
            reference="/tmp/ref.wav",
            output="/tmp/out",
            verse=None,
            session_dir=None,
            search_root="~/Music/GarageBand",
        )
        resolved = _resolve_paths(args)
        self.assertEqual(resolved["mode"], "explicit")
        self.assertEqual(resolved["user_paths"], ["/tmp/a.wav", "/tmp/b.wav"])
        self.assertEqual(resolved["reference_path"], "/tmp/ref.wav")
        self.assertEqual(resolved["output_dir"], "/tmp/out")


if __name__ == "__main__":
    unittest.main()
