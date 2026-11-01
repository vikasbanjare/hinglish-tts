"""Turn any recording of a person into a voice profile synth_script.py can clone.

Usage: .venv/bin/python add_voice.py RECORDING.(mp3|wav|m4a|...) --name NAME [--no-clean] [--lang hi]

Writes voices/NAME/:
  full.wav + full.txt   cleaned speech + transcript (sets the speaker's natural pace)
  ref.wav  + ref.txt    the best ~10 s clip + its exact words (the voice the model copies)

Any length works. 30 s - 3 min of one person talking is plenty for zero-shot cloning;
only the first --max-minutes are used, a longer upload doesn't make the clone better.
ALWAYS listen to ref.wav and check ref.txt word for word: a wrong word there hurts every output.
"""
import argparse
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import soundfile as sf

SR = 24000
ROOT = Path(__file__).resolve().parent
VOCAB = set((ROOT / "models/IndicF5/checkpoints/vocab.txt").read_text(encoding="utf-8").split("\n"))

ap = argparse.ArgumentParser()
ap.add_argument("recording"); ap.add_argument("--name", required=True)
ap.add_argument("--no-clean", action="store_true", help="Skip music/noise removal (already a clean voice recording).")
ap.add_argument("--lang", default="hi", help="Whisper language: hi for Hindi/Hinglish (writes Devanagari), en for English.")
ap.add_argument("--max-minutes", type=float, default=10)
args = ap.parse_args()
if not Path(args.recording).is_file():
    ap.error(f"recording not found: {args.recording}")
out = ROOT / "voices" / args.name; out.mkdir(parents=True, exist_ok=True)


def run(*cmd):
    subprocess.run(cmd, check=True)


with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "in.wav"
    run("ffmpeg", "-loglevel", "error", "-y", "-i", args.recording, "-t", str(args.max_minutes * 60), "-ar", "44100", str(src))
    if not args.no_clean:
        print("Removing background music/noise (Demucs, CPU; ~1 min per minute of audio)...", flush=True)
        # CPU: htdemucs hits an MPS conv limit on Apple GPUs.
        run(sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", "cpu", "-o", tmp, str(src))
        src = Path(tmp) / "htdemucs/in/vocals.wav"
    run("ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(out / "full.wav"))

print("Transcribing (Whisper large-v3; first run downloads ~3 GB)...", flush=True)
from faster_whisper import WhisperModel  # noqa: E402  (slow import, only after audio prep succeeds)

asr = WhisperModel("large-v3", device="cpu", compute_type="int8")
segs = [s for s in asr.transcribe(str(out / "full.wav"), language=args.lang, beam_size=5, vad_filter=True)[0]]
if not segs:
    sys.exit("No speech found in the recording.")
nfc = lambda t: unicodedata.normalize("NFC", t.strip())  # vocab has only the combining nukta
# IndicF5 reads Devanagari best; Whisper types English words in Roman letters. IndicXlit converts
# only those Roman words (Devanagari passes through), e.g. "cash flow" -> "कैश फ्लो".
from scoring.scripts.lib_normalize import to_unified_devanagari  # noqa: E402

(out / "full.txt").write_text(to_unified_devanagari(" ".join(nfc(s.text) for s in segs)) + "\n", encoding="utf-8")

# Candidate reference clips: runs of consecutive Whisper segments, 5-14 s long.
clips = []
for i in range(len(segs)):
    for j in range(i, len(segs)):
        dur = segs[j].end - segs[i].start
        if dur > 14:
            break
        if dur >= 5:
            part = segs[i:j + 1]
            text = " ".join(nfc(s.text) for s in part)
            clips.append(dict(start=part[0].start, end=part[-1].end, dur=dur, text=text,
                              logprob=min(s.avg_logprob for s in part),           # worst segment's ASR confidence
                              no_speech=max(s.no_speech_prob for s in part),
                              latin=sum(c.isascii() and c.isalpha() for c in text),  # English typed in Roman letters
                              oov=sum(c not in VOCAB for c in text if not c.isspace())))  # chars the model can't read
if not clips:
    sys.exit("No 5-14 s stretch of continuous speech found; record at least ~10 s without long pauses.")


def score(c: dict) -> float:
    """Higher = better reference clip. Keys: dur, text, logprob (<=0), no_speech (0-1), latin, oov."""
    # TODO(human): rank candidate reference clips (see the Learn by Doing note).
    return -abs(c["dur"] - 10) + 5 * c["logprob"] - 100 * c["oov"]


best = max(clips, key=score)
y, _ = sf.read(out / "full.wav", dtype="float32")
pad = 0.15  # Whisper ends segments right on the last phoneme; keep the tail
sf.write(out / "ref.wav", y[max(0, int((best["start"] - pad) * SR)):int((best["end"] + pad) * SR)], SR)
ref_text = to_unified_devanagari(best["text"])
(out / "ref.txt").write_text(ref_text + "\n", encoding="utf-8")

print(f"\nVoice '{args.name}' saved to {out}")
print(f"  reference clip: {best['start']:.1f}-{best['end']:.1f} s of the recording ({best['dur']:.1f} s)")
print(f"  heard:   {best['text']}")
print(f"  ref.txt: {ref_text}")
if best["latin"]:
    print("  English words were auto-converted to Devanagari; check they match how the speaker said them.")
print("Listen to ref.wav and fix any wrong word in ref.txt before generating.")
