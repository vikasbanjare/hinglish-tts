#!/usr/bin/env bash
# Fine-tune Chatterbox (English) on one voice profile with LoRA, on an Apple Silicon Mac.
#   engines/finetune_chatterbox.sh VOICE_NAME        e.g. podcast_1
# Uses github.com/gokhaneraslan/chatterbox-finetuning (Apache-2.0, third-party code: review before running).
# Needs ~4 GB disk (base model) + checkpoints, ~1-3 h on an M3 Pro for ~30 min of speech.
set -euo pipefail
cd "$(dirname "$0")/.."
VOICE="${1:?usage: engines/finetune_chatterbox.sh VOICE_NAME}"
FT=engines/chatterbox-ft
KIT_COMMIT=HEAD   # pin a commit here once one is known to work

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "1/5 Dataset (voices/$VOICE -> datasets/$VOICE)"
[[ -f "datasets/$VOICE/metadata.csv" ]] || .venv/bin/python finetune_prep.py "$VOICE" "datasets/$VOICE"

step "2/5 Fine-tuning kit + its own environment"
[[ -d $FT ]] || { git clone -q https://github.com/gokhaneraslan/chatterbox-finetuning $FT && git -C $FT checkout -q $KIT_COMMIT; }
[[ -x $FT/.venv/bin/python ]] || uv venv -q --python 3.11 $FT/.venv
VIRTUAL_ENV="$PWD/$FT/.venv" uv pip install -q -r $FT/requirements.txt "setuptools<81" transformers

step "3/5 Base model (standard English Chatterbox, ~3 GB)"
(cd $FT && .venv/bin/python setup.py)

step "4/5 Mac settings (Apple GPU, fp32, small batches)"
# The kit is written for NVIDIA GPUs: bf16 + batch 32 + 8 loader workers. On a Mac: fp32, batch 4 x 8 accumulation
# (same effective batch), no worker processes, and point it at our dataset + the speaker's reference clip.
ROOT="$PWD" VOICE="$VOICE" $FT/.venv/bin/python - <<'PY'
import os, re
from pathlib import Path
root, voice = Path(os.environ["ROOT"]), os.environ["VOICE"]
cfg = root / "engines/chatterbox-ft/src/config.py"; s = cfg.read_text()
sets = {
    "csv_path: str": f'"{root}/datasets/{voice}/metadata.csv"',
    "wav_dir: str": f'"{root}/datasets/{voice}/wavs"',
    "output_dir: str": f'"{root}/engines/chatterbox-ft/output_{voice}"',
    "inference_prompt_path: str": f'"{root}/voices/{voice}/ref.wav"',
    "inference_test_text: str": '"If you invest in knowledge early, it compounds faster than any stock in your portfolio."',
    "batch_size: int": "4", "grad_accum: int": "8", "dataloader_num_workers: int": "0",
}
for k, v in sets.items():
    s, n = re.subn(rf"^(\s*{re.escape(k)}\s*=\s*)[^#\n]+", rf"\g<1>{v} ", s, flags=re.M)
    assert n == 1, f"config.py changed upstream; can't set {k}"
s = re.sub(r"^(\s*preprocessed_dir\s*=\s*)[^#\n]+", rf'\g<1>"{root}/datasets/{voice}/preprocess" ', s, flags=re.M)
cfg.write_text(s)
t = root / "engines/chatterbox-ft/train.py"; s = t.read_text()
s = s.replace('torch.device("cuda" if torch.cuda.is_available() else "cpu")',
              'torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")')
s = s.replace("bf16=True", "bf16=torch.cuda.is_available()")
t.write_text(s)
print("config patched for", voice)
PY

step "5/5 Train (LoRA). Progress + sample audio land in $FT/output_$VOICE"
(cd $FT && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train.py)
echo "Done. Test it: (cd $FT && .venv/bin/python inference.py) — then tell Claude to wire it into synth_script.py."
