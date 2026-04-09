from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .timeline import make_plot_times_unique


def write_pitch_alignment_plot(
    output_html_path: Path,
    aligned_reference_pitch_hz: np.ndarray,
    aligned_user_pitch_hz: np.ndarray,
    user_alignment_indices: np.ndarray,
    reference_alignment_indices: np.ndarray,
    user_frame_count: int,
    reference_frame_count: int,
    sample_rate: int,
    hop_length: int,
) -> None:
    alignment_length = int(aligned_reference_pitch_hz.shape[0])
    x_time_seconds = (np.arange(alignment_length) * hop_length) / sample_rate

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=("Aligned Pitch Overlay", "DTW Warping Path"),
    )

    figure.add_trace(
        go.Scatter(
            x=x_time_seconds,
            y=aligned_reference_pitch_hz,
            mode="lines",
            name="Reference Pitch (Hz)",
            line={"width": 2},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x_time_seconds,
            y=aligned_user_pitch_hz,
            mode="lines",
            name="Aligned User Pitch (Hz)",
            line={"width": 2},
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=user_alignment_indices,
            y=reference_alignment_indices,
            mode="lines",
            name="Warping Path",
            line={"width": 2},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[0, max(user_frame_count - 1, 1)],
            y=[0, max(reference_frame_count - 1, 1)],
            mode="lines",
            name="Perfect Timing (Identity)",
            line={"width": 2, "dash": "dash"},
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        title="Vocal Alignment Audit",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=900,
    )
    figure.update_xaxes(title_text="Aligned Time (s)", row=1, col=1)
    figure.update_yaxes(title_text="Pitch (Hz)", row=1, col=1)
    figure.update_xaxes(title_text="User Frame", row=2, col=1)
    figure.update_yaxes(title_text="Reference Frame", row=2, col=1)
    figure.write_html(str(output_html_path), include_plotlyjs="cdn")


def build_performance_trend_explanation_html(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return (
            "<section class='card'>"
            "<h2>Lettura Performance</h2>"
            "<p>Nessun dato disponibile.</p>"
            "</section>"
        )

    first_metrics = attempts[0]["metrics"]
    last_metrics = attempts[-1]["metrics"]

    def format_delta(value: float) -> str:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}"

    pitch_delta = last_metrics["pitch_score"] - first_metrics["pitch_score"]
    timing_delta = last_metrics["timing_score"] - first_metrics["timing_score"]
    cents_delta = last_metrics["avg_abs_cents"] - first_metrics["avg_abs_cents"]
    rms_delta = (
        last_metrics["warping_path_rms_deviation"] - first_metrics["warping_path_rms_deviation"]
    )
    stability_delta = last_metrics["vocal_stability_std"] - first_metrics["vocal_stability_std"]

    improving_pitch = pitch_delta > 0
    improving_timing = timing_delta > 0
    improving_cents = cents_delta < 0
    improving_count = sum([improving_pitch, improving_timing, improving_cents])

    if improving_count == 3:
        verdict_class = "ok"
        verdict_text = "Miglioramento netto"
    elif improving_count == 2:
        verdict_class = "warn"
        verdict_text = "Miglioramento parziale"
    else:
        verdict_class = "bad"
        verdict_text = "Peggioramento o andamento misto"

    return (
        "<section class='card'>"
        "<h2>Lettura Miglioramento/Peggioramento</h2>"
        f"<p><span class='{verdict_class}'>Verdetto: {verdict_text}</span></p>"
        f"<p>Pitch Score: <span class='mono'>{first_metrics['pitch_score']:.2f} → {last_metrics['pitch_score']:.2f}</span> "
        f"(delta <span class='mono'>{format_delta(pitch_delta)}</span>) | Valore perfetto atteso: "
        "<span class='mono'>100</span></p>"
        f"<p>Timing Score: <span class='mono'>{first_metrics['timing_score']:.2f} → {last_metrics['timing_score']:.2f}</span> "
        f"(delta <span class='mono'>{format_delta(timing_delta)}</span>) | Valore perfetto atteso: "
        "<span class='mono'>100</span></p>"
        f"<p>Avg Abs Cents: <span class='mono'>{first_metrics['avg_abs_cents']:.2f} → {last_metrics['avg_abs_cents']:.2f}</span> "
        f"(delta <span class='mono'>{format_delta(cents_delta)}</span>) | Valore perfetto atteso: "
        "<span class='mono'>0</span> (ottimo &lt; 10)</p>"
        f"<p>Warp Path RMS Deviation: <span class='mono'>{first_metrics['warping_path_rms_deviation']:.4f} → "
        f"{last_metrics['warping_path_rms_deviation']:.4f}</span> (delta <span class='mono'>{format_delta(rms_delta)}</span>) "
        "| Valore perfetto atteso: <span class='mono'>0</span></p>"
        f"<p>Vocal Stability Std: <span class='mono'>{first_metrics['vocal_stability_std']:.2f} → "
        f"{last_metrics['vocal_stability_std']:.2f}</span> (delta <span class='mono'>{format_delta(stability_delta)}</span>) "
        "| Valore atteso: <span class='mono'>più basso e stabile possibile</span></p>"
        "<p>Nota: il giudizio principale usa Pitch Score, Timing Score e Avg Abs Cents.</p>"
        "</section>"
    )


def write_performance_trend_plot(output_html_path: Path, attempts: list[dict[str, Any]]) -> None:
    attempt_datetimes = [
        datetime.fromisoformat(attempt["attempt_time_local"])
        if "attempt_time_local" in attempt
        else datetime.fromisoformat(attempt["file_mtime_local"])
        for attempt in attempts
    ]
    unique_plot_datetimes = make_plot_times_unique(attempt_datetimes)
    x_plot_values = [dt.isoformat(timespec="seconds") for dt in unique_plot_datetimes]

    take_names = [attempt["take_name"] for attempt in attempts]
    time_sources = [attempt.get("attempt_time_source", "file_mtime") for attempt in attempts]
    hover_payload = list(zip(take_names, time_sources))
    x_tick_labels = [
        f"{attempt.get('attempt_index', index + 1):02d} | {take_names[index]}"
        for index, attempt in enumerate(attempts)
    ]

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Performance Scores",
            "Pitch Error",
            "Timing Path Deviation",
            "Vocal Stability",
        ),
    )

    def add_series(row_number: int, metric_label: str, metric_values: list[float]) -> None:
        figure.add_trace(
            go.Scatter(
                x=x_plot_values,
                y=metric_values,
                mode="lines+markers",
                name=metric_label,
                customdata=hover_payload,
                marker={"size": 8},
                hovertemplate=(
                    "Take: %{customdata[0]}<br>"
                    "Time: %{x}<br>"
                    "Time Source: %{customdata[1]}<br>"
                    f"{metric_label}: "
                    "%{y:.4f}<extra></extra>"
                ),
            ),
            row=row_number,
            col=1,
        )

    add_series(1, "Pitch Score", [a["metrics"]["pitch_score"] for a in attempts])
    add_series(1, "Timing Score", [a["metrics"]["timing_score"] for a in attempts])
    add_series(2, "Avg Abs Cents", [a["metrics"]["avg_abs_cents"] for a in attempts])
    add_series(
        3,
        "Warp Path RMS Deviation",
        [a["metrics"]["warping_path_rms_deviation"] for a in attempts],
    )
    add_series(4, "Vocal Stability Std", [a["metrics"]["vocal_stability_std"] for a in attempts])

    figure.update_layout(
        title="Chronological Vocal Performance Trend",
        template="plotly_white",
        height=1200,
        showlegend=True,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.85)",
        },
    )

    for row_number in [1, 2, 3, 4]:
        figure.update_xaxes(
            categoryorder="array",
            categoryarray=x_plot_values,
            tickmode="array",
            tickvals=x_plot_values,
            ticktext=x_tick_labels,
            type="category",
            row=row_number,
            col=1,
        )

    figure.update_yaxes(title_text="Score", row=1, col=1)
    figure.update_yaxes(title_text="Cents", row=2, col=1)
    figure.update_yaxes(title_text="Deviation", row=3, col=1)
    figure.update_yaxes(title_text="Std Dev", row=4, col=1)
    figure.update_xaxes(title_text="Attempt (Chronological)", row=4, col=1)

    explanation_html = build_performance_trend_explanation_html(attempts)
    figure_html = figure.to_html(full_html=False, include_plotlyjs="cdn")

    final_html = (
        "<!doctype html>"
        "<html lang='it'>"
        "<head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<title>Vocal Performance Trend</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;background:#f7f9fc;color:#13233a;margin:0;padding:18px;}"
        ".card{background:#fff;border:1px solid #dfe6f2;border-radius:12px;padding:14px 16px;margin-bottom:14px;}"
        ".card h2{margin:0 0 8px 0;font-size:18px;}"
        ".card p{margin:5px 0;font-size:14px;line-height:1.45;}"
        ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}"
        ".ok{color:#0a7f42;font-weight:600;}"
        ".warn{color:#9a6b00;font-weight:600;}"
        ".bad{color:#b3261e;font-weight:600;}"
        "</style>"
        "</head>"
        "<body>"
        f"{explanation_html}"
        f"{figure_html}"
        "</body></html>"
    )

    output_html_path.write_text(final_html, encoding="utf-8")
