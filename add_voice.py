"""Turn any recording into voice profile(s) synth_script.py can clone.

Usage: .venv/bin/python add_voice.py RECORDING.(mp3|wav|m4a|...) --name NAME [--speakers 2] [--no-clean] [--lang hi]

Writes voices/NAME/ (or NAME_1, NAME_2, ... with --speakers N, ordered by talk time):
  full.wav + full.txt   only this speaker's speech + transcript (sets their natural pace, judges similarity)
  segments.tsv          start/end (s, inside full.wav) + text of every utterance: training data for fine-tuning
  ref.wav  + ref.txt    the best ~10 s clip + its exact words (the voice the model copies)
  refs/01..05.wav/.txt  the top 5 clips (01 = ref.wav), for --refs N

Any length works; 30 s - 3 min is enough to clone. Longer recordings give more reference clips to choose
from (--refs) and more fine-tuning data. --speakers N splits a podcast/interview into one profile per person.
ALWAYS listen to ref.wav and check ref.txt word for word: a wrong word there hurts every output.
"""
import argparse
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
GAP = 0.25  # seconds of silence between utterances when joining one speaker's speech
ROOT = Path(__file__).resolve().parent
VOCAB = set((ROOT / "models/IndicF5/checkpoints/vocab.txt").read_text(encoding="utf-8").split("\n"))

ap = argparse.ArgumentParser()
ap.add_argument("recording"); ap.add_argument("--name", required=True)
ap.add_argument("--speakers", type=int, default=1, help="People talking in the recording (podcast = 2).")
ap.add_argument("--no-clean", action="store_true", help="Skip music/noise removal (already a clean voice recording).")
ap.add_argument("--lang", default=None, help="hi (Hindi/Hinglish, writes Devanagari) or en; default: auto-detect.")
ap.add_argument("--max-minutes", type=float, default=120)
args = ap.parse_args()
if not Path(args.recording).is_file():
    ap.error(f"recording not found: {args.recording}")
nfc = lambda t: unicodedata.normalize("NFC", t.strip())  # vocab has only the combining nukta


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
    run("ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(Path(tmp) / "mix.wav"))
    y = sf.read(Path(tmp) / "mix.wav", dtype="float32")[0]

print(f"Transcribing {len(y) / SR / 60:.0f} min with Whisper (first run downloads the model)...", flush=True)
from faster_whisper import WhisperModel  # noqa: E402  (slow import, only after audio prep succeeds)

from faster_whisper import BatchedInferencePipeline  # noqa: E402

# Long recordings: turbo + batching is ~6-8x faster on CPU, near-identical for English; short clips keep
# full large-v3 (better on Hindi). ponytail: switch to mlx-whisper (Apple GPU) if even this is too slow.
long_rec = len(y) / SR > 15 * 60
asr = BatchedInferencePipeline(WhisperModel("large-v3-turbo" if long_rec else "large-v3", device="cpu", compute_type="int8"))
segs, mark = [], 0
import librosa  # noqa: E402

# faster-whisper assumes 16 kHz for raw arrays; passing 24 kHz silently stretches every timestamp 1.5x
it, info = asr.transcribe(librosa.resample(y, orig_sr=SR, target_sr=16000), language=args.lang, beam_size=5, batch_size=8, word_timestamps=True)
LANG = args.lang or info.language
if LANG not in ("hi", "en"):
    print(f"  note: detected '{LANG}'; treating as Hindi script. Pass --lang hi/en to override.")
    LANG = "hi"
print(f"  language: {LANG}", flush=True)
def utterances(s):
    """Split a Whisper segment at pauses and sentence ends: batched mode returns ~30 s chunks that can hold
    both speakers, and diarization/reference clips need short single-speaker pieces."""
    cur = []
    for w in s.words or []:
        if cur and (w.start - cur[-1].end > 0.5 or cur[-1].word.rstrip()[-1:] in ".?!" or w.end - cur[0].start > 12):
            yield cur; cur = []
        cur.append(w)
    if cur:
        yield cur


for s in it:
    for ws in utterances(s):
        text = nfc("".join(w.word for w in ws))
        if ws[-1].end - ws[0].start < 0.3 or not text or (segs and text == segs[-1]["text"]):
            continue  # Whisper hallucinations: zero-length or the same line repeated
        segs.append(dict(i=len(segs), start=ws[0].start, end=ws[-1].end, text=text, logprob=s.avg_logprob,
                         no_speech=s.no_speech_prob))
    if s.end - mark > 300:
        print(f"  ... {s.end / 60:.0f} min", flush=True); mark = s.end
if not segs:
    sys.exit("No speech found in the recording.")


def diarize(segs: list, k: int) -> list:
    """Split utterances into k speakers by voice fingerprint (Resemblyzer + k-means). Largest talker first."""
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import KMeans
    enc = VoiceEncoder("cpu", verbose=False)
    keep = [s for s in segs if s["end"] - s["start"] >= 1.5]  # shorter = unreliable fingerprint ("haan", "right")
    E = np.array([enc.embed_utterance(preprocess_wav(y[int(s["start"] * SR):int(s["end"] * SR)], SR)) for s in keep])
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(E)
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    sims = np.sort(E @ C.T, axis=1)
    groups = [[] for _ in range(k)]
    for s, lab, (second, top) in zip(keep, km.labels_, sims[:, -2:]):
        if top - second >= 0.05:  # ponytail: drop ambiguous utterances (crosstalk, laughter); pyannote if this misfires
            groups[lab].append(s)
    dropped = len(segs) - sum(map(len, groups))
    mins = ", ".join(f"{sum(s['end'] - s['start'] for s in g) / 60:.1f} min" for g in groups)
    print(f"Speakers: {mins} ({dropped} short/ambiguous utterances skipped)")
    return sorted(groups, key=lambda g: -sum(s["end"] - s["start"] for s in g))


def score(c: dict) -> float:
    """Higher = better reference clip. Keys: dur, text, logprob (<=0), no_speech (0-1), latin, oov."""
    # TODO(human): rank candidate reference clips (see the Learn by Doing note).
    return -abs(c["dur"] - 10) + 5 * c["logprob"] - 100 * c["oov"]


# IndicF5 reads Devanagari best; Whisper types English words in Roman letters. IndicXlit converts
# only those Roman words (Devanagari passes through), e.g. "cash flow" -> "कैश फ्लो".
from scoring.scripts.lib_normalize import to_unified_devanagari as _xlit  # noqa: E402

to_unified_devanagari = _xlit if LANG == "hi" else (lambda t: t)  # English voices keep English transcripts


def build_profile(out: Path, g: list):
    out.mkdir(parents=True, exist_ok=True)
    # full.wav = this speaker only, utterances joined with short gaps; segments.tsv indexes into it.
    parts, rows, t = [], [], 0.0
    for s in g:
        a = y[int(s["start"] * SR):int(s["end"] * SR)]
        rows.append(f"{t:.2f}\t{t + len(a) / SR:.2f}\t{to_unified_devanagari(s['text'])}")
        parts += [a, np.zeros(int(GAP * SR), np.float32)]; t += len(a) / SR + GAP
    sf.write(out / "full.wav", np.concatenate(parts), SR)
    (out / "full.txt").write_text(" ".join(r.split("\t")[2] for r in rows) + "\n", encoding="utf-8")
    (out / "lang.txt").write_text(LANG + "\n")
    (out / "segments.tsv").write_text("start\tend\ttext\n" + "\n".join(rows) + "\n", encoding="utf-8")

    # Candidate reference clips: runs of this speaker's utterances that were back-to-back in the recording
    # (no one else talked in between), 5-14 s long, cut from the original audio.
    clips = []
    for a in range(len(g)):
        for b in range(a, len(g)):
            if b > a and (g[b]["i"] != g[b - 1]["i"] + 1 or g[b]["start"] - g[b - 1]["end"] > 1.0):
                break
            dur = g[b]["end"] - g[a]["start"]
            if dur > 14:
                break
            if dur >= 5:
                part = g[a:b + 1]
                text = " ".join(s["text"] for s in part)
                clips.append(dict(start=part[0]["start"], end=part[-1]["end"], dur=dur, text=text,
                                  logprob=min(s["logprob"] for s in part),             # worst segment's ASR confidence
                                  no_speech=max(s["no_speech"] for s in part),
                                  latin=sum(c.isascii() and c.isalpha() for c in text),  # English typed in Roman letters
                                  oov=sum(c not in VOCAB for c in text if not c.isspace())))  # chars the model can't read
    if not clips:
        print(f"  {out.name}: no 5-14 s stretch of continuous speech; skipped reference clips")
        return

    # Keep the top 5 non-overlapping clips: synth_script.py --refs N tries several per line and keeps
    # the closest-sounding take, so a longer recording gives more (and more varied) candidates.
    picked = []
    for c in sorted(clips, key=score, reverse=True):
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            picked.append(c)
        if len(picked) == 5:
            break
    pad = 0.15  # Whisper ends segments right on the last phoneme; keep the tail
    (out / "refs").mkdir(exist_ok=True)
    for old in (out / "refs").glob("*"):
        old.unlink()
    for k, c in enumerate(picked, 1):
        clip = y[max(0, int((c["start"] - pad) * SR)):int((c["end"] + pad) * SR)]
        text = to_unified_devanagari(c["text"])
        for wav in (out / f"refs/{k:02d}.wav",) + ((out / "ref.wav",) if k == 1 else ()):
            sf.write(wav, clip, SR)
            wav.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")

    best = picked[0]
    talk = sum(s["end"] - s["start"] for s in g)
    print(f"\nVoice '{out.name}' saved to {out}  ({talk / 60:.1f} min of speech, {len(g)} utterances)")
    print(f"  reference clip: {best['start']:.1f}-{best['end']:.1f} s of the recording ({best['dur']:.1f} s)")
    print(f"  heard:   {best['text']}")
    print(f"  ref.txt: {to_unified_devanagari(best['text'])}")
    if best["latin"] and LANG == "hi":
        print("  English words were auto-converted to Devanagari; check they match how the speaker said them.")
    print(f"  + {len(picked) - 1} backup clips in refs/ (used by synth_script.py --refs N)")


groups = diarize(segs, args.speakers) if args.speakers > 1 else [segs]
for k, g in enumerate(groups, 1):
    build_profile(ROOT / "voices" / (args.name if len(groups) == 1 else f"{args.name}_{k}"), g)
print("\nListen to each ref.wav and fix any wrong word in ref.txt before generating (same for refs/*.txt if you use --refs).")
