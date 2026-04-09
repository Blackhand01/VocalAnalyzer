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
