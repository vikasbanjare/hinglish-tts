"""Synthesize a script (one chunk per line) with one model load, then join.

Usage: PYTHONPATH=. python synth_script.py SCRIPT.txt OUT_DIR --voice NAME [--engine best] [--speed 1.0] [--seed 0]

  --voice NAME  a profile made by add_voice.py (voices/NAME/); sets --ref and --pace-from
  --engine      indicf5    Hindi/Hinglish specialist, follows the speaker's pace (default)
                chatterbox strongest on plain English; Hindi too (paced to match the speaker)
                best       render each line with both, keep whichever sounds more like the speaker
                           with every word right (slower: ~2x render + a Whisper check per line)

Voice settings (ElevenLabs-style):
  --speed       global tempo multiplier (ElevenLabs range 0.7-1.2); per-line syllable pacing still applies
  --similarity  how hard IndicF5 follows the reference voice (cfg strength; 2-3.5 sensible)
  --seed        take variation; change it for a different read of every line
  --gap         seconds of silence between lines
  --pace-from   longer recording (+ same-name .txt) whose average tempo sets the base pace;
                the short reference alone may be read faster/slower than the speaker usually talks
Inline, inside the script:
  [speed=0.85]           slow down / speed up just this line
  [seed=7]               re-roll just this line
  [engine=chatterbox]    force an engine for just this line
  <break time="0.8s" />  exact pause inside a line (same syntax as ElevenLabs)

English lines can be plain English ("Let us know in the comments"); Hindi/Hinglish lines in Devanagari.
Pieces are cached by text + settings, so editing a line or tag re-renders only what changed.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import inference
from pacing import count_syllables, rate, speed_for

SR = 24000
ROOT = Path(__file__).resolve().parent
CB_PY = ROOT / "engines/chatterbox/.venv/bin/python"
TAG = re.compile(r"\[(speed|seed|engine)=([\w.]+)\]")
BREAK = re.compile(r'<break\s+time="([\d.]+)s"\s*/>')

ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("out_dir"); ap.add_argument("--device", default="mps")
ap.add_argument("--gap", type=float, default=0.12, help="Seconds of silence between lines (she barely pauses).")
ap.add_argument("--ref", default="data/my_voice/clean_b.wav", help="Reference WAV; transcript in same-name .txt.")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--similarity", "--cfg", dest="cfg", type=float, default=3.0)  # A/B winner
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pace-from", default="data/my_voice/edelwise_vocals.wav")
ap.add_argument("--nfe", type=int, default=32, help="Quality steps (64 = slower, slightly cleaner).")
ap.add_argument("--voice", help="Profile from add_voice.py (voices/NAME); overrides --ref and --pace-from.")
ap.add_argument("--engine", default="indicf5", choices=["indicf5", "chatterbox", "best"])
args = ap.parse_args()
if args.voice:
    v = ROOT / "voices" / args.voice
    if not (v / "ref.wav").exists():
        ap.error(f"no voice profile at {v}; run add_voice.py first")
    args.ref, args.pace_from = str(v / "ref.wav"), str(v / "full.wav")
if args.engine != "indicf5" and not CB_PY.exists():
    ap.error("Chatterbox isn't installed; run ./setup_mac.sh --with-chatterbox")
REF = Path(args.ref)

lines = [l.strip() for l in Path(args.script).read_text(encoding="utf-8").splitlines() if l.strip()]
out = Path(args.out_dir).resolve(); cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)
ref_text = REF.with_suffix(".txt").read_text(encoding="utf-8").strip()
pace = Path(args.pace_from) if args.pace_from else None
tempo = rate(pace, pace.with_suffix(".txt").read_text(encoding="utf-8")) / rate(REF, ref_text) if pace and pace.exists() else 1.0
print(f"base tempo x{tempo:.3f} (from {pace}), engine: {args.engine}")
model = worker = judge = None


def lang_of(text: str) -> str:
    return "hi" if re.search("[ऀ-ॿ]", text) else "en"


def cached(engine: str, text: str, *extra) -> Path:
    key = hashlib.sha1("|".join(map(str, (engine, text, *extra, REF.resolve()))).encode()).hexdigest()[:16]
    return cache / f"{key}.wav"


def render_indicf5(text: str, speed: float, seed: int) -> np.ndarray:
    global model
    if lang_of(text) == "en":  # IndicF5 reads Devanagari only
        from scoring.scripts.lib_normalize import to_unified_devanagari
        text = to_unified_devanagari(text)
    speed *= speed_for(text, ref_text) * tempo
    wav = cached("indicf5", text, f"{speed:.4f}", seed, args.cfg, args.nfe)
    if not wav.exists():
        model = model or inference.load_model(device=args.device)
        torch.manual_seed(seed)
        # Script is already hand-written Devanagari; skip IndicXlit (it mangles e.g. के → केई).
        sf.write(wav, inference.synthesize(model, text, REF, ref_text, preprocess=False, speed=speed,
                                           nfe_step=args.nfe, cfg_strength=args.cfg), SR)
    return sf.read(wav, dtype="float32")[0]


def render_chatterbox(text: str, speed: float, seed: int) -> np.ndarray:
    global worker
    raw = cached("chatterbox", text, seed)
    if not raw.exists():
        if worker is None:
            log = open(cache / "chatterbox.log", "a")
            worker = subprocess.Popen([str(CB_PY), str(ROOT / "engines/chatterbox_worker.py")], text=True,
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                      env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
            if worker.stdout.readline().strip() != "ready":
                raise RuntimeError(f"Chatterbox failed to start; see {cache / 'chatterbox.log'}")
        worker.stdin.write(json.dumps(dict(text=text, lang=lang_of(text), ref=str(REF.resolve()), seed=seed,
                                           out=str(raw))) + "\n"); worker.stdin.flush()
        reply = worker.stdout.readline().strip()
        if reply != "done":
            raise RuntimeError(f"Chatterbox: {reply}")
    y, sr = sf.read(raw, dtype="float32")
    y = librosa.effects.trim(librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y, top_db=40)[0]
    # Chatterbox has no speed control and reads Hindi ~30% slower than our speaker. Pace it to the same
    # syllable rate IndicF5 targets. English syllable counts are too rough for this, so English keeps its
    # natural pace (it already matched IndicF5 in the bake-off) and only gets the --speed multiplier.
    target = count_syllables(text) / (rate(REF, ref_text) * tempo * speed) if lang_of(text) == "hi" else len(y) / SR / speed
    ratio = float(np.clip(len(y) / SR / target, 0.75, 1.5))  # ponytail: atempo beyond ±50% sounds processed
    if abs(ratio - 1) < 0.03:
        return y
    with tempfile.TemporaryDirectory() as tmp:
        a, b = f"{tmp}/a.wav", f"{tmp}/b.wav"
        sf.write(a, y, SR)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a, "-filter:a", f"atempo={ratio:.4f}", b], check=True)
        return sf.read(b, dtype="float32")[0]


def render(text: str, speed: float, seed: int, engine: str) -> np.ndarray:
    global judge
    if engine == "indicf5":
        return render_indicf5(text, speed, seed)
    if engine == "chatterbox":
        return render_chatterbox(text, speed, seed)
    takes = {"indicf5": render_indicf5(text, speed, seed)}
    try:
        takes["chatterbox"] = render_chatterbox(text, speed, seed)
    except RuntimeError as e:
        print(f"    chatterbox failed ({e}); using indicf5")
        return takes["indicf5"]
    scores_file = cached("scores", text, f"{speed:.4f}", seed, args.cfg, args.nfe).with_suffix(".json")
    if scores_file.exists():
        scores = json.loads(scores_file.read_text())
    else:
        from quality import Judge
        judge = judge or Judge(args.pace_from if pace and pace.exists() else REF)
        scores = {k: judge.score(y, SR, text, lang_of(text)) for k, y in takes.items()}
        scores_file.write_text(json.dumps(scores))
    win = max(scores, key=lambda k: scores[k]["total"])
    print("    " + "  ".join(f"{k}: voice {s['similarity']:.3f} errors {s['errors']:.0%}" for k, s in scores.items())
          + f"  -> {win}", flush=True)
    return takes[win]


silence = lambda s: np.zeros(int(s * SR), dtype=np.float32)
full = []
for i, line in enumerate(lines, 1):
    opts = dict(TAG.findall(line))
    pieces = BREAK.split(TAG.sub("", line))  # text, secs, text, secs, ...
    audio = []
    print(f"[{i}/{len(lines)}] {line[:50]}", flush=True)
    for j, piece in enumerate(pieces):
        if j % 2:
            audio.append(silence(float(piece)))
        elif piece.strip():
            audio.append(render(piece.strip(), args.speed * float(opts.get("speed", 1)),
                                int(float(opts.get("seed", args.seed + i))), opts.get("engine", args.engine)))
    line_audio = np.concatenate(audio)
    sf.write(out / f"{i:02d}.wav", line_audio, SR)
    full += [line_audio, silence(args.gap)]

sf.write(out / "full.wav", np.concatenate(full[:-1]), SR)
if worker:
    worker.stdin.close(); worker.wait(timeout=30)
print(f"Wrote {out / 'full.wav'}")
