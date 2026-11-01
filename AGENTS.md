# Instructions for AI assistants (Claude Code, Codex, Cursor, Gemini CLI, …)

You are helping a user clone a voice and generate voiceovers with this repo, locally on their Mac.
Read this file fully before acting. The user may not be technical: do the work, explain in plain words.

## What the user can ask you for

| User says | You do |
|---|---|
| "Set this up" | Run `./setup_mac.sh --with-chatterbox` (see Setup) |
| "Add my voice" / gives a recording | `add_voice.py` → then **play/show them `ref.txt` and ask them to confirm it** |
| "Make a voiceover of this text/blog/script" | Convert text to a script file (rules below) → `synth_script.py` |
| "Slower / faster / redo line 3 / pause here" | Tags and flags (Controls below); re-run, only changed lines re-render |
| "Which sounds better?" | `--engine best`, or A/B a line with `[engine=...]` |

## Setup (once)

1. Check: Apple Silicon Mac (`uname -m` = `arm64`), Homebrew, ~10 GB free disk (`df -h ~`).
2. Tell the user to: make a Hugging Face account, **accept the terms at https://huggingface.co/ai4bharat/IndicF5**,
   create a *read* token at huggingface.co/settings/tokens.
3. Run `./setup_mac.sh --with-chatterbox`. When it reaches the Hugging Face login, the **user** types the token
   into the terminal prompt themselves (in an interactive terminal: `.venv/bin/hf auth login`).
   - **Never ask the user to paste a token into chat, and never write a token to a file.**
   - Use `.venv/bin/hf`, not `hf` (another CLI named `hf` may be on PATH).
4. It's idempotent: on failure, fix the cause and re-run. Do **not** run `setup.sh` (Ubuntu only).

## Add a voice

```bash
.venv/bin/python add_voice.py RECORDING --name NAME        # add --no-clean for a clean studio recording
```

- Input: any audio/video file of **one person**, ideally 30 s – 3 min of normal speech. Music is removed automatically.
- Output `voices/NAME/`: `ref.wav` + `ref.txt` (the clip the model copies), `full.wav` + `full.txt` (sets the pace).
- **Then show the user `voices/NAME/ref.txt` and ask them to listen to `ref.wav`.** Every word in `ref.txt` must
  match the audio exactly (Devanagari for Hindi/Hinglish). Fix wrong words in `ref.txt`; this matters more than anything else.
- Only clone a voice the user owns or has the speaker's permission for. If that's unclear, ask.

## Write the script (this is your most important job)

Put one sentence or short clause per line in `data/script/NAME.txt`. The models read text literally, so you
must write it **the way it is spoken**:

- **Hindi / Hinglish lines → Devanagari only**, English words included, as pronounced:
  `Apple का upcoming iPhone` → `ऐपल का अपकमिंग आईफ़ोन`. Don't use the automatic transliterator for whole
  English sentences: it maps word by word ("the" → थे, "could" → कोल्ड).
- **Pure English lines → normal English** (they go to the Chatterbox engine).
- **Numbers → words**, in the language the speaker would say them: `₹95,345 crore` → `नाइंटी फाइव थाउज़ंड थ्री हंड्रेड फोर्टी फाइव करोड़`,
  `FY25` → `एफ़वाई ट्वेंटी फाइव`, `2026` → `ट्वेंटी ट्वेंटी सिक्स`.
- **Acronyms → one joined word**: `एयूएम`, `आईपीओ` (spaced letters get merged wrongly).
- Use `फ़`/`ज़` freely (files are NFC-normalized). Replace `…` with a comma. Drop emojis, URLs, markdown.
- From a blog/article: keep the spoken content, drop headings/links/captions, split long sentences.
- Show the user the script before a long render if you changed wording, not just spelling.

## Generate

```bash
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. .venv/bin/python synth_script.py \
  data/script/NAME.txt outputs/NAME --voice VOICE --engine best
```

- Output: `outputs/NAME/full.wav` (+ `01.wav`… per line). Give the user the path to `full.wav`.
- `--engine indicf5` (Hindi/Hinglish, fastest), `chatterbox` (English), `best` (renders both per line, keeps
  the take that sounds more like the speaker with fewer word errors; ~2–3× slower).
- Speed: roughly 1–3× real time on M-series. For scripts over ~2 minutes run it in the background
  (`nohup … > outputs/NAME.log 2>&1 &`) and check the log; don't let a tool timeout kill it.

## Controls

| Want | Do |
|---|---|
| Slower / faster overall | `--speed 0.9` / `--speed 1.1` |
| Closer to the voice (IndicF5) | `--similarity 3.5` (default 3.0) |
| A different take of everything | `--seed 1` |
| Redo one line | start that line with `[seed=7]` |
| One line slower | start that line with `[speed=0.85]` |
| Force an engine on one line | `[engine=chatterbox]` or `[engine=indicf5]` |
| Pause inside a line | `<break time="0.6s" />` |
| Gap between lines | `--gap 0.3` |

## When something's wrong

| Symptom | Fix |
|---|---|
| A word mispronounced | Respell it in Devanagari as it sounds; re-run |
| Mumbled / skipped words everywhere | `ref.txt` doesn't match `ref.wav`: fix it, or pick a cleaner recording |
| Doesn't sound like the speaker | Cleaner/longer recording; `--similarity 3.5`; try `--engine best` |
| One bad line | `[seed=N]` on that line, try 2–3 values |
| `403` on download | User must accept terms on the IndicF5 Hugging Face page |
| MPS / "not implemented" errors | Make sure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set |
| `Chatterbox isn't installed` | `./setup_mac.sh --with-chatterbox` |

Check a render yourself before reporting success: e.g. transcribe `full.wav` with `faster_whisper` (already
installed) and compare against the script.

## Files

`setup_mac.sh` install · `add_voice.py` voice profiles · `synth_script.py` generate · `pacing.py` speed per line ·
`quality.py` judge takes · `engines/` Chatterbox (own venv) · `bench/` engine bake-off · `inference.py` IndicF5 wrapper.
Don't commit `voices/`, `models/`, `outputs/` (personal or large; already in `.gitignore`).
