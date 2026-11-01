#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac (no CUDA). Safe to re-run: finished steps are skipped.
#   ./setup_mac.sh                    IndicF5 (Hindi/Hinglish)
#   ./setup_mac.sh --with-chatterbox  + Chatterbox (best for English; enables --engine chatterbox/best)
# Needs: Homebrew. Before running, accept the terms at https://huggingface.co/ai4bharat/IndicF5
# (setup.sh in this repo is Ubuntu-only; use this one on a Mac.)
set -euo pipefail
cd "$(dirname "$0")"
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
INDICF5_COMMIT=13f7c4d627cc10111aea8fe9c0039462cacacdc7

step "1/7 System tools (uv, ffmpeg)"
[[ "$(uname -m)" == arm64 ]] || { echo "This script is for Apple Silicon Macs."; exit 1; }
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
for t in uv ffmpeg; do command -v "$t" >/dev/null || brew install "$t"; done
xcode-select -p >/dev/null 2>&1 || { echo "Run: xcode-select --install   (then re-run this script)"; exit 1; }

step "2/7 Python 3.10 environment"
[[ -x .venv/bin/python ]] || uv venv --python 3.10 .venv
export VIRTUAL_ENV="$PWD/.venv"
uv pip install torch==2.5.1 torchaudio==2.5.1 numpy==1.26.4 "Cython<3" wheel "setuptools<69" pip

step "3/7 fairseq 0.12.2 (built from source; the PyPI wheel doesn't build on macOS clang)"
if ! .venv/bin/python -c "import fairseq" 2>/dev/null; then
  curl -fsSL -o "$TMP/fs.tgz" https://github.com/facebookresearch/fairseq/archive/refs/tags/v0.12.2.tar.gz
  tar xzf "$TMP/fs.tgz" -C "$TMP" && printf "0.12.2\n" > "$TMP/fairseq-0.12.2/fairseq/version.txt"
  CFLAGS=-Wno-invalid-specialization CXXFLAGS=-Wno-invalid-specialization \
    uv pip install --no-build-isolation "$TMP/fairseq-0.12.2/"
fi

step "4/7 Everything else (exact tested versions) + IndicF5"
uv pip install -r requirements-mac.txt
if [[ ! -d IndicF5 ]]; then
  git clone -q https://github.com/AI4Bharat/IndicF5 && git -C IndicF5 checkout -q "$INDICF5_COMMIT"
fi
uv pip install --no-deps ./IndicF5   # --no-deps: its hydra pin conflicts with fairseq

step "5/7 IndicXlit transliteration models (~150 MB)"
XL="$(.venv/bin/python -c 'import ai4bharat.transliteration as t,os;print(os.path.dirname(t.__file__))')/transformer/models/en2indic/v1.0"
mkdir -p "$XL"
for f in indicxlit-en-indic-v1.0.zip word_prob_dicts.zip; do
  [[ -e "$XL/.$f.done" ]] && continue
  curl -fsSL --retry 10 -o "$TMP/$f" "https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/$f"
  unzip -q -o "$TMP/$f" -d "$XL" && touch "$XL/.$f.done"
done

step "6/7 Model weights (IndicF5 ~1.4 GB + vocos)"
# .venv/bin/hf, not plain `hf`: another CLI named hf may be on PATH.
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo "Log in to Hugging Face (paste your token into the prompt below, never into a chat):"
  .venv/bin/hf auth login
fi
.venv/bin/hf download charactr/vocos-mel-24khz --local-dir models/vocos-mel-24khz >/dev/null
.venv/bin/hf download ai4bharat/IndicF5 --local-dir models/IndicF5 >/dev/null \
  || { echo "403? Accept the terms at https://huggingface.co/ai4bharat/IndicF5 and re-run."; exit 1; }
cp patches/indicf5_model.py models/IndicF5/model.py   # load vocos/vocab from disk + speed/quality knobs

if [[ "${1:-}" == --with-chatterbox ]]; then
  step "Chatterbox engine (own environment, ~1 GB + 3 GB weights)"
  # Separate venv: it needs torch 2.6 / newer transformers than IndicF5's pinned stack.
  [[ -x engines/chatterbox/.venv/bin/python ]] || uv venv -q --python 3.11 engines/chatterbox/.venv
  VIRTUAL_ENV="$PWD/engines/chatterbox/.venv" uv pip install chatterbox-tts==0.1.7 "setuptools<81"  # perth watermarker needs pkg_resources
  engines/chatterbox/.venv/bin/python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M; M.from_pretrained(device='cpu'); print('Chatterbox OK')"
fi

step "7/7 Smoke test"
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. .venv/bin/python -c \
  "import inference; inference.load_model(device='mps'); print('Model loads OK')" 2>&1 | grep -E "OK|Error" || true

cat <<'EOF'

Setup done. Next:
  1. Add your voice:   .venv/bin/python add_voice.py ~/path/to/recording.mp3 --name me
  2. Write a script:   data/script/my_script.txt  (one sentence per line, Devanagari)
  3. Generate:         HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. \
                         .venv/bin/python synth_script.py data/script/my_script.txt outputs/my_script --voice me
     (with Chatterbox installed, add --engine best to pick the better engine per line)
EOF
