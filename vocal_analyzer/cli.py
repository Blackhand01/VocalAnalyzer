from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from .analyzer import VocalAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vocal-analyzer",
        description="Analyze user vocal performance against a reference track.",
    )
    parser.add_argument(
        "--user",
        nargs="+",
        help="One or more user vocal audio files.",
    )
    parser.add_argument("--reference", help="Path to reference vocal audio file.")
    parser.add_argument(
        "--verse",
        help="Verse label (e.g. Verse01 or 01). Auto mode compares VerseXX.wav vs all MyVerseXX/*.wav.",
    )
    parser.add_argument(
        "--session-dir",
        help="Directory containing VerseXX.wav and MyVerseXX/. If omitted, it is auto-discovered.",
    )
    parser.add_argument(
        "--search-root",
        default="~/Music/GarageBand",
        help="Root folder used to auto-discover the session when --session-dir is omitted.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for summary, plots, and batch reports.",
    )
    parser.add_argument("--sr", type=int, default=22050, help="Processing sample rate.")
    parser.add_argument("--fmin", type=float, default=60.0, help="Minimum pitch frequency in Hz.")
    parser.add_argument("--fmax", type=float, default=500.0, help="Maximum pitch frequency in Hz.")
    parser.add_argument("--hop-length", type=int, default=256, help="Hop length for pYIN.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logger verbosity.",
    )
    return parser


def _normalize_verse_name(raw: str) -> str:
    text = raw.strip()
    lower = text.lower()
    if lower.startswith("verse"):
        suffix = text[5:]
        return f"Verse{suffix}"
    return f"Verse{text}"


def _discover_session_dir(verse_name: str, search_root: Path) -> Path:
    if not search_root.exists():
        raise FileNotFoundError(f"Search root does not exist: {search_root}")

    reference_name = f"{verse_name}.wav"
    user_dir_name = f"My{verse_name}"
    candidates: list[Path] = []
    for reference_file in search_root.rglob(reference_name):
        session_dir = reference_file.parent
        if (session_dir / user_dir_name).is_dir():
            candidates.append(session_dir)

    unique_candidates = sorted(set(candidates), key=lambda path: str(path))
    if not unique_candidates:
        raise FileNotFoundError(
            f"Unable to find {reference_name} with sibling folder {user_dir_name} under {search_root}"
        )
    if len(unique_candidates) > 1:
        options = "\n".join(f"- {path}" for path in unique_candidates[:10])
        raise ValueError(
            "Multiple sessions match this verse. Use --session-dir to disambiguate.\n"
            f"{options}"
        )
    return unique_candidates[0]


def _resolve_paths(args: argparse.Namespace) -> dict[str, Any]:
    if args.verse:
        verse_name = _normalize_verse_name(args.verse)
        if args.session_dir:
            session_dir = Path(args.session_dir).expanduser().resolve()
        else:
            search_root = Path(args.search_root).expanduser().resolve()
            session_dir = _discover_session_dir(verse_name=verse_name, search_root=search_root)

        reference_path = session_dir / f"{verse_name}.wav"
        user_dir = session_dir / f"My{verse_name}"
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference file not found: {reference_path}")
        if not user_dir.is_dir():
            raise FileNotFoundError(f"User takes folder not found: {user_dir}")

        user_paths = sorted((str(path) for path in user_dir.glob("*.wav")), key=lambda path: path.lower())
        if not user_paths:
            raise FileNotFoundError(f"No .wav files found in {user_dir}")

        if args.output:
            output_dir = args.output
        else:
            output_dir = str((Path.cwd() / "outputs" / session_dir.name / verse_name).resolve())

        return {
            "mode": "verse_auto",
            "verse_name": verse_name,
            "session_dir": str(session_dir),
            "user_paths": user_paths,
            "reference_path": str(reference_path),
            "output_dir": output_dir,
        }

    if not args.user or not args.reference:
        raise ValueError("Provide --verse, or provide --user + --reference (and optional --output).")

    if args.output:
        output_dir = args.output
    elif len(args.user) == 1:
        output_dir = str((Path.cwd() / "outputs" / "single_run").resolve())
    else:
        output_dir = str((Path.cwd() / "outputs" / "batch_run").resolve())

    return {
        "mode": "explicit",
        "user_paths": args.user,
        "reference_path": args.reference,
        "output_dir": output_dir,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        resolved = _resolve_paths(args)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}")
        return 1

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    logger = logging.getLogger("vocal_analyzer")
    logger.setLevel(getattr(logging, args.log_level))

    analyzer = VocalAnalyzer(
        sr=args.sr,
        fmin=args.fmin,
        fmax=args.fmax,
        hop_length=args.hop_length,
        logger=logger,
    )
    user_paths = resolved["user_paths"]
    reference_path = resolved["reference_path"]
    output_dir = resolved["output_dir"]

    if resolved["mode"] == "verse_auto":
        print(f"Mode: verse_auto ({resolved['verse_name']})")
        print(f"Session: {resolved['session_dir']}")
        print(f"Auto-discovered takes: {len(user_paths)}")
        print(f"Reference: {reference_path}")
        print(f"Output: {output_dir}")

    if len(user_paths) == 1:
        result = analyzer.analyze(
            user_path=user_paths[0],
            reference_path=reference_path,
            output_dir=output_dir,
        )
    else:
        result = analyzer.analyze_batch(
            user_paths=user_paths,
            reference_path=reference_path,
            output_dir=output_dir,
        )

    status = result.get("status", "error")
    print(f"Status: {status}")
    if len(user_paths) == 1 and status == "success":
        metrics = result.get("metrics", {})
        files = result.get("files", {})
        print(
            "Metrics | "
            f"Pitch Score: {metrics.get('pitch_score'):.2f}, "
            f"Avg Abs Cents: {metrics.get('avg_abs_cents'):.2f}, "
            f"Timing Score: {metrics.get('timing_score'):.2f}, "
            f"Warp Path RMS Dev: {metrics.get('warping_path_rms_deviation'):.4f}, "
            f"Vocal Stability Std: {metrics.get('vocal_stability_std'):.2f}"
        )
        print(f"Summary JSON: {files.get('summary_json')}")
        print(f"Pitch Plot: {files.get('pitch_alignment_html')}")
        return 0

    if len(user_paths) > 1 and status in {"success", "partial_success"}:
        print(
            f"Batch | Attempts: {result.get('successful_attempts', 0)}/"
            f"{len(result.get('attempts', []))} successful"
        )
        for attempt in result.get("attempts", []):
            if attempt.get("status") != "success":
                print(f"{attempt.get('attempt_index'):02d} | {attempt.get('take_name')} | ERROR")
                continue
            metrics = attempt["metrics"]
            print(
                f"{attempt.get('attempt_index'):02d} | {attempt.get('take_name')} | "
                f"Pitch: {metrics.get('pitch_score'):.2f} | "
                f"Timing: {metrics.get('timing_score'):.2f} | "
                f"Cents: {metrics.get('avg_abs_cents'):.2f}"
            )
        files = result.get("files", {})
        print(f"Performance History: {files.get('performance_history_json')}")
        print(f"Performance Trend: {files.get('performance_trend_html')}")
        return 0

    print(f"Error: {result.get('error', 'Unknown error')}")
    if len(user_paths) == 1:
        print(f"Summary JSON: {output_dir.rstrip('/')}/summary.json")
    else:
        print(f"Performance History: {output_dir.rstrip('/')}/performance_history.json")
    return 1


if __name__ == "__main__":
    sys.exit(main())
