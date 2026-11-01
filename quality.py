"""Judge a take: how much it sounds like the speaker, and whether every word came out.

Used by synth_script.py --engine best (picks the better engine per line) and bench/score.py.
"""
import re
import unicodedata

import numpy as np


def norm(text: str, lang: str) -> list:
    from scoring.scripts.lib_normalize import to_unified_devanagari
    t = unicodedata.normalize("NFC", to_unified_devanagari(text) if lang == "hi" else text.lower())
    return re.sub(r"[^\w\s]", "", t.replace("़", "")).split()  # nukta dropped: ASR uses it inconsistently


def wer(ref: list, hyp: list) -> float:
    """Word-level edit distance / len(ref)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


class Judge:
    def __init__(self, speaker_wav):
        from faster_whisper import WhisperModel
        from resemblyzer import VoiceEncoder, preprocess_wav
        self._prep = preprocess_wav
        self.enc = VoiceEncoder("cpu", verbose=False)
        self.target = self.enc.embed_utterance(preprocess_wav(str(speaker_wav)))
        self.asr = WhisperModel("large-v3", device="cpu", compute_type="int8")

    def similarity(self, y: np.ndarray, sr: int) -> float:
        return float(self.enc.embed_utterance(self._prep(y, sr)) @ self.target)

    def word_errors(self, y: np.ndarray, sr: int, text: str, lang: str) -> float:
        import librosa
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        hyp = " ".join(s.text for s in self.asr.transcribe(y16, language=lang, beam_size=5)[0])
        return wer(norm(text, lang), norm(hyp, lang))

    def score(self, y, sr, text, lang) -> dict:
        # ponytail: digits vs spelled-out numbers count as errors for both engines alike, so ranking stays fair
        s, e = self.similarity(y, sr), self.word_errors(y, sr, text, lang)
        return dict(similarity=s, errors=e, total=s - 0.5 * e)  # a dropped/wrong word costs more than a voice nuance


if __name__ == "__main__":
    assert wer("a b c".split(), "a x c".split()) == 1 / 3
    assert wer("a b".split(), "a b".split()) == 0
    print("quality ok")
