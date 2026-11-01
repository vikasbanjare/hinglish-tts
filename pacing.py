"""Syllable-based pacing for IndicF5.

IndicF5 sizes each output as ref_duration / ref_chars * gen_chars / speed.
Devanagari character counts track speaking time poorly ("इंफ्रास्ट्रक्चर" is 15 chars
but 4 syllables), so speed_for() picks the per-line speed that turns that
character formula into a syllable-proportional one, no model patch needed.
"""
import re

VIRAMA = "्"          # ् joins consonants into a cluster (no vowel between)
MATRAS = set("ािीुूृेैोौॅॉ")  # dependent vowel signs (they colour a consonant, not a new syllable)


def count_syllables(text: str, drop_final_schwa: bool = False) -> int:  # False fit her timing best (0.130 vs 0.144)
    """Approximate spoken syllables in Devanagari text (Latin letters count per vowel group)."""
    n = 0
    for word in text.split():
        w = [c for c in word if "ऀ" <= c <= "ॿ" and c != "़"]  # Devanagari, minus nukta
        syl = sum(1 for i, c in enumerate(w)
                  if "ऄ" <= c <= "औ"                                       # independent vowel
                  or ("क" <= c <= "ह" and w[i + 1:i + 2] != [VIRAMA]))     # consonant with a vowel
        # Hindi drops the inherent "a" after a word-final bare consonant: कमल = ka-mal, not ka-ma-la.
        if drop_final_schwa and syl > 1 and "क" <= w[-1] <= "ह":
            syl -= 1
        n += syl + len(re.findall(r"[aeiouy]+", word.lower()))  # stray Latin words
        n += 2 * sum(c.isdigit() for c in word)  # ponytail: ~2 syllables/digit spoken ("670" = six-hun-dred-se-ven-ty); ASR writes numbers as digits
    return max(n, 1)


def chars(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def rate(wav_path, text: str) -> float:
    """Syllables per second of a recording (whole file, so pass speech without long silences)."""
    import soundfile as sf
    return count_syllables(text) / sf.info(str(wav_path)).duration


def speed_for(gen_text: str, ref_text: str) -> float:
    """Speed making F5's char-based duration equal a syllable-based one."""
    return (chars(gen_text) / chars(ref_text)) * (count_syllables(ref_text) / count_syllables(gen_text))


if __name__ == "__main__":
    # Check against her real reading: per-line seconds-per-unit should vary LESS with syllables than chars.
    from pathlib import Path
    import numpy as np
    real = [3.40, 7.60, 6.54, 5.50, 6.26, 3.16, 7.48, 3.02, 1.80, 8.05, 8.02,
            0.64, 9.52, 6.84, 6.56, 1.60, 3.56, 4.46, 5.86, 8.30, 4.20]  # her line durations (s)
    script = Path("data/script/edelweiss_ipo.txt")  # the speaker's real reading; local only (not in the repo)
    if not script.exists():
        raise SystemExit("skip: needs data/script/edelweiss_ipo.txt + her timings")
    lines = [l for l in script.read_text(encoding="utf-8").splitlines() if l.strip()]
    cv = lambda f: (lambda r: r.std() / r.mean())(np.array(real) / np.array([f(l) for l in lines]))
    print(f"spread of sec/unit  chars: {cv(chars):.3f}   syllables: {cv(count_syllables):.3f}  (lower = better)")
    assert cv(count_syllables) < cv(chars), "syllable count should predict her timing better than chars"
