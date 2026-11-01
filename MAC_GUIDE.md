# Clone your voice on a Mac (Apple Silicon)

Three commands: set up once, add your voice once, generate as often as you like.
Runs fully offline after setup. Needs an M1/M2/M3/M4 Mac, ~10 GB free disk, Homebrew.

## 1. Set up (once, ~20–30 min)

1. Make a free account at [huggingface.co](https://huggingface.co), open
   [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) and accept the terms.
2. Create a **read** token at huggingface.co/settings/tokens.
3. Run:

```bash
git clone <this repo> && cd hinglish-tts && ./setup_mac.sh --with-chatterbox
```

`--with-chatterbox` adds a second engine that's stronger on plain English (~4 GB more). Leave it off for Hindi-only use.

When it asks, paste the token into the terminal prompt. Never paste it into a chat or a file.
The script is safe to re-run; it skips finished steps.

## 2. Add your voice (once per person)

Any recording of **one person** speaking: mp3, wav, m4a, a video's audio track. Background music is removed automatically.

```bash
.venv/bin/python add_voice.py ~/Downloads/my_recording.mp3 --name me
```

- **Length:** 30 s – 3 min is plenty. Longer is fine; only the first 10 min are used.
- **Best input:** your normal speaking style, the way you want the voiceovers to sound, no other voices.
- **Check the result** (important): it picks the best ~10 s clip as `voices/me/ref.wav` and writes
  what you said in `voices/me/ref.txt`. Listen to the clip and fix any wrong word in the text.
  A mismatch between the clip and its text is the #1 cause of mumbled or skipped words.
- Already-clean studio recording? Add `--no-clean` (faster).

## 3. Generate

Write the script one sentence per line in `data/script/my_script.txt`, then:

```bash
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. .venv/bin/python synth_script.py data/script/my_script.txt outputs/my_script --voice me
```

### Pick an engine

| `--engine` | Use for | Notes |
|---|---|---|
| `indicf5` (default) | Hindi / Hinglish | Follows the speaker's own pace most closely |
| `chatterbox` | English | Reads plain English directly; Hindi paced to match the speaker |
| `best` | Mixed scripts, or when quality matters most | Renders every line with both, keeps the one that sounds more like the speaker with every word right. ~2× slower. |

Force one line with `[engine=chatterbox]` at its start.

Output: `outputs/my_script/full.wav` (plus `01.wav`, `02.wav`… per line). Re-running only renders lines you changed.

### Writing the script (this decides pronunciation)

- **Hindi/Hinglish lines:** write in **Devanagari**, English words too, the way they're spoken: `आईफ़ोन`, `कैश फ्लो`.
- **Pure English lines:** write normal English (with `--engine chatterbox` or `best`).
- **Numbers as words** in the language you'd say them: `नाइंटी फाइव थाउज़ंड`, not `95,000`.
- **Acronyms joined**: `एयूएम`, not `ए यू एम`.
- One sentence or short clause per line; very long lines rush.

### Controls (ElevenLabs-style)

| Want | Do |
|---|---|
| Slower / faster overall | `--speed 0.9` / `--speed 1.1` |
| Closer to the voice | `--similarity 3.5` (default 3.0; too high can sound strained) |
| A different take of everything | `--seed 1` |
| Re-do one line | put `[seed=7]` at the start of that line |
| Slow one line | `[speed=0.85]` at the start of that line |
| A pause inside a line | `<break time="0.6s" />` |
| More/less gap between lines | `--gap 0.3` |

Pace already follows the speaker: it's measured from the whole recording you gave `add_voice.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `403` while downloading | Accept the terms on the IndicF5 page, then re-run `./setup_mac.sh` |
| A word is mispronounced | Respell it in Devanagari the way it sounds; re-run (only that line re-renders) |
| A line sounds off | Add `[seed=N]` to that line and try a few values |
| Voice doesn't sound like you | Check `ref.txt` matches `ref.wav` exactly; try a cleaner recording |
| Slow | ~1–2× real time on M-series. Normal. |

Only clone voices of people who have agreed to it.
