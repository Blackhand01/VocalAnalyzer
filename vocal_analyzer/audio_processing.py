from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .models import AudioBundle


def normalize_peak_amplitude(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak <= 0.0:
        return signal
    return signal / peak


def load_audio_bundle(path: Path, target_sample_rate: int, logger: logging.Logger) -> AudioBundle:
    info = sf.info(str(path))
    signal, _ = librosa.load(str(path), sr=target_sample_rate, mono=True)
    signal = normalize_peak_amplitude(signal)

    bundle = AudioBundle(
        path=str(path),
        signal=signal.astype(np.float32),
        original_sr=int(info.samplerate),
        processed_sr=target_sample_rate,
        original_channels=int(info.channels),
        original_frames=int(info.frames),
        original_duration_sec=float(info.duration),
        processed_samples=int(signal.shape[0]),
        processed_duration_sec=float(signal.shape[0] / target_sample_rate),
    )

    logger.info(
        "Loaded %s | original_sr=%s processed_sr=%s channels=%s original_dur=%.3fs processed_dur=%.3fs",
        bundle.path,
        bundle.original_sr,
        bundle.processed_sr,
        bundle.original_channels,
        bundle.original_duration_sec,
        bundle.processed_duration_sec,
    )
    return bundle


def build_sync_feature(signal: np.ndarray, sample_rate: int, hop_length: int) -> np.ndarray:
    rms_energy = librosa.feature.rms(y=signal, frame_length=2048, hop_length=hop_length)[0]
    onset_strength = librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=hop_length)

    feature_length = min(rms_energy.shape[0], onset_strength.shape[0])
    combined_feature = 0.6 * rms_energy[:feature_length] + 0.4 * onset_strength[:feature_length]
    combined_feature = combined_feature.astype(np.float64)

    feature_mean = float(np.mean(combined_feature))
    feature_std = float(np.std(combined_feature))
    if feature_std <= 1e-8:
        return combined_feature - feature_mean
    return (combined_feature - feature_mean) / feature_std


def select_user_analysis_segment(
    user_signal: np.ndarray,
    reference_signal: np.ndarray,
    sample_rate: int,
    logger: logging.Logger,
    mismatch_ratio_threshold: float = 1.25,
    search_hop_length: int = 1024,
) -> dict[str, Any]:
    user_samples = int(user_signal.shape[0])
    reference_samples = int(reference_signal.shape[0])

    full_track_metadata = {
        "selection_method": "full_track",
        "start_sample": 0,
        "end_sample": user_samples,
        "start_sec": 0.0,
        "end_sec": float(user_samples / sample_rate),
        "duration_sec": float(user_samples / sample_rate),
    }

    if reference_samples <= 0 or user_samples <= 0:
        return {"signal": user_signal, "metadata": full_track_metadata}

    if user_samples <= int(reference_samples * mismatch_ratio_threshold):
        return {"signal": user_signal, "metadata": full_track_metadata}

    reference_feature = build_sync_feature(
        reference_signal,
        sample_rate=sample_rate,
        hop_length=search_hop_length,
    )
    user_feature = build_sync_feature(
        user_signal,
        sample_rate=sample_rate,
        hop_length=search_hop_length,
    )

    if user_feature.size < reference_feature.size:
        return {"signal": user_signal, "metadata": full_track_metadata}

    correlation = np.correlate(user_feature, reference_feature, mode="valid")
    best_frame_index = int(np.argmax(correlation))
    start_sample = min(best_frame_index * search_hop_length, max(user_samples - reference_samples, 0))
    end_sample = min(start_sample + reference_samples, user_samples)
    if end_sample - start_sample < reference_samples:
        start_sample = max(0, end_sample - reference_samples)

    cropped_signal = user_signal[start_sample:end_sample]
    match_strength = float(correlation[best_frame_index] / max(reference_feature.size, 1))

    metadata = {
        "selection_method": "auto_reference_window",
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "start_sec": float(start_sample / sample_rate),
        "end_sec": float(end_sample / sample_rate),
        "duration_sec": float((end_sample - start_sample) / sample_rate),
        "search_hop_length": search_hop_length,
        "match_strength": match_strength,
    }
    logger.info(
        "Selected user analysis window | start=%.3fs end=%.3fs duration=%.3fs match=%.4f",
        metadata["start_sec"],
        metadata["end_sec"],
        metadata["duration_sec"],
        match_strength,
    )
    return {"signal": cropped_signal, "metadata": metadata}


def audio_bundle_to_metadata(bundle: AudioBundle) -> dict[str, Any]:
    return {
        "path": bundle.path,
        "original_sample_rate": bundle.original_sr,
        "processed_sample_rate": bundle.processed_sr,
        "original_channels": bundle.original_channels,
        "original_frames": bundle.original_frames,
        "original_duration_sec": bundle.original_duration_sec,
        "processed_samples": bundle.processed_samples,
        "processed_duration_sec": bundle.processed_duration_sec,
    }
