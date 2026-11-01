"""Export a voice profile as an LJSpeech dataset for fine-tuning (e.g. gokhaneraslan/chatterbox-finetuning).

Usage: .venv/bin/python finetune_prep.py VOICE_NAME OUT_DIR
Reads voices/NAME/full.wav + segments.tsv (made by add_voice.py); writes OUT_DIR/wavs/*.wav + metadata.csv
(id|raw text|normalized text). Keeps 1-15 s utterances: shorter teach little, longer get truncated in training.
"""
import csv
import sys
from pathlib import Path

import soundfile as sf

voice, out = Path("voices") / sys.argv[1], Path(sys.argv[2])
(out / "wavs").mkdir(parents=True, exist_ok=True)
y, sr = sf.read(voice / "full.wav", dtype="float32")
rows, total = [], 0.0
for r in csv.DictReader(open(voice / "segments.tsv", encoding="utf-8"), delimiter="\t"):
    a, b, text = float(r["start"]), float(r["end"]), r["text"].replace("|", " ").strip()
    if not 1.0 <= b - a <= 15.0 or not text:
        continue
    uid = f"{voice.name}_{len(rows):05d}"
    sf.write(out / "wavs" / f"{uid}.wav", y[int(a * sr):int(b * sr)], sr)
    rows.append(f"{uid}|{text}|{text}")
    total += b - a
(out / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"{len(rows)} clips, {total / 60:.1f} min -> {out}")
