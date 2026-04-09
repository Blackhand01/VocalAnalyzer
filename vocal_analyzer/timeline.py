from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def sort_user_paths_chronologically(user_paths: list[str]) -> list[Path]:
    sortable_entries: list[tuple[datetime, str, Path]] = []
    for raw_path in user_paths:
        path = Path(raw_path)
        take_time, _ = resolve_take_datetime(path)
        sortable_entries.append((take_time, path.name, path))

    sortable_entries.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in sortable_entries]


def resolve_take_datetime(path: Path) -> tuple[datetime, str]:
    parsed_datetime = extract_take_datetime_from_filename(path)
    if parsed_datetime is not None:
        return parsed_datetime, "filename"
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone(), "file_mtime"


def extract_take_datetime_from_filename(path: Path) -> datetime | None:
    match = re.match(
        r"^(?P<day>\d{2}):(?P<month>\d{2}):(?P<year>\d{2}),\s*"
        r"(?P<hour>\d{2})[.:](?P<minute>\d{2})(?:[.:](?P<second>\d{2}))?",
        path.stem,
    )
    if match is None:
        return None

    day = int(match.group("day"))
    month = int(match.group("month"))
    year_two_digits = int(match.group("year"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second")) if match.group("second") else 0
    year = 2000 + year_two_digits if year_two_digits < 70 else 1900 + year_two_digits

    local_timezone = datetime.now().astimezone().tzinfo
    if local_timezone is None:
        return None

    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=local_timezone)
    except ValueError:
        return None


def build_take_output_dir_name(take_path: Path, attempt_index: int, mtime_epoch: float) -> str:
    timestamp = datetime.fromtimestamp(mtime_epoch).strftime("%Y%m%d_%H%M%S")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", take_path.stem).strip("._")
    if not safe_stem:
        safe_stem = "take"
    return f"{attempt_index:02d}_{timestamp}_{safe_stem}"


def attach_metric_deltas(attempts: list[dict[str, Any]]) -> None:
    previous_metrics: dict[str, float] | None = None
    for attempt in attempts:
        metrics = attempt.get("metrics")
        if attempt.get("status") != "success" or metrics is None:
            continue

        if previous_metrics is not None:
            attempt["delta_vs_previous"] = {
                "pitch_score": metrics["pitch_score"] - previous_metrics["pitch_score"],
                "timing_score": metrics["timing_score"] - previous_metrics["timing_score"],
                "avg_abs_cents": metrics["avg_abs_cents"] - previous_metrics["avg_abs_cents"],
                "vocal_stability_std": (
                    metrics["vocal_stability_std"] - previous_metrics["vocal_stability_std"]
                ),
            }
        previous_metrics = metrics


def make_plot_times_unique(datetimes: list[datetime]) -> list[datetime]:
    seen: dict[datetime, int] = {}
    unique_datetimes: list[datetime] = []
    for dt in datetimes:
        duplicate_count = seen.get(dt, 0)
        seen[dt] = duplicate_count + 1
        unique_datetimes.append(dt + timedelta(seconds=duplicate_count))
    return unique_datetimes
