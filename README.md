# VocalAnalyzer

Python-based Vocal Accountability Platform to compare a user vocal track against a reference vocal using pYIN pitch extraction, DTW alignment, and quantitative scoring.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If audio loading fails, install ffmpeg:

```bash
brew install ffmpeg
```

## Daily Command Sequence

Start project + run `Verse01` in chronological batch:

```bash
cd "/Users/stefanoroybisignano/Desktop/Singing/VocalAnalyzer"
source .venv/bin/activate

python -m vocal_analyzer.cli \
  --verse Verse01 \
  --session-dir "/Users/stefanoroybisignano/Music/GarageBand/Eyes Without A Face - Verse" \
  --output "/Users/stefanoroybisignano/Desktop/Singing/VocalAnalyzer/outputs/Eyes Without A Face/Verse01/"

open "/Users/stefanoroybisignano/Desktop/Singing/VocalAnalyzer/outputs/Eyes Without A Face/Verse01/performance_trend.html"
```

Open `pitch_alignment`:

```bash
# ultimo take analizzato
latest_take="$(ls -1dt "/Users/stefanoroybisignano/Desktop/Singing/VocalAnalyzer/outputs/Eyes Without A Face/Verse01/takes/"*/ | head -n 1)"
open "${latest_take}pitch_alignment.html"

# tutti i pitch_alignment dei take
find "/Users/stefanoroybisignano/Desktop/Singing/VocalAnalyzer/outputs/Eyes Without A Face/Verse01/takes/" -type f -name "pitch_alignment.html" -exec open "{}" \;
```

## Voice Quality Metrics (v1.2)

`summary.json` now includes 3 additional sensor groups to evaluate *how* you sing:

- `dynamics_rms_*`:
  - `dynamics_rms_corr` (target `1.0`): similarity of loudness envelope vs reference.
  - `dynamics_rms_mae_db` (target `0`): average loudness gap in dB.
- `spectral_centroid_*`:
  - `spectral_centroid_gap_hz` (target `0`): timbre brightness distance from reference.
- `hnr_*`:
  - `hnr_gap_db` (target `0`): harmonic/noise quality distance from reference.

In `performance_trend.html`, these appear as extra optional traces when present.

## Human Note (recommended)

Track a manual `RPE` (Rate of Perceived Exertion, 1-10) after each take in your notes:

- High scores + low RPE = healthy progress.
- High scores + high RPE = likely over-effort; review technique/recovery.
