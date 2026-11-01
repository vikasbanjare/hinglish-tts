"""Score bake-off outputs: voice similarity, intelligibility, pace. Run with .venv/bin/python, PYTHONPATH=.
Usage: bench/score.py VOICE_DIR OUT_ROOT engine1 engine2 ...   (each engine = OUT_ROOT/engine/*.wav)"""
import csv, sys
from pathlib import Path
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder, preprocess_wav
from pacing import count_syllables, rate
from quality import norm, wer

voice, root, engines = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
rows = list(csv.DictReader(open(Path(__file__).parent / "sentences.tsv", encoding="utf-8"), delimiter="\t"))
enc = VoiceEncoder("cpu"); target = enc.embed_utterance(preprocess_wav(str(voice / "full.wav")))
asr = WhisperModel("large-v3", device="cpu", compute_type="int8")
her_rate = rate(voice / "full.wav", (voice / "full.txt").read_text(encoding="utf-8"))


print(f"her pace: {her_rate:.2f} syl/s\n")
print(f"{'engine':<16}{'lang':<5}{'similarity':>11}{'WER':>7}{'pace':>7}   (pace = syl/s ÷ hers; 1.00 = her speed)")
for eng in engines:
    for lang in ("en", "hi"):
        sims, wers, paces = [], [], []
        for r in (r for r in rows if r["lang"] == lang):
            f = root / eng / f"{r['id']}.wav"
            if not f.exists():
                continue
            y, sr = sf.read(f, dtype="float32")
            sims.append(float(enc.embed_utterance(preprocess_wav(y, sr)) @ target))
            hyp = " ".join(s.text for s in asr.transcribe(str(f), language=lang, beam_size=5, vad_filter=True)[0])
            wers.append(wer(norm(r["text"], lang), norm(hyp, lang)))
            paces.append(count_syllables(r["text"]) / (len(y) / sr) / her_rate)
            print(f"    {eng}/{r['id']}: {hyp.strip()}", file=sys.stderr)
        if sims:
            print(f"{eng:<16}{lang:<5}{np.mean(sims):>11.3f}{np.mean(wers):>7.2f}{np.mean(paces):>7.2f}")
