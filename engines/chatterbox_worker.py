"""Chatterbox multilingual, kept loaded. Runs inside engines/chatterbox/.venv (its own torch/transformers).

Protocol: one JSON request per stdin line {"text","lang","ref","seed","out"}; replies "done" or "error <msg>".
Started by synth_script.py; you don't run this by hand.
"""
import json
import sys

reply = sys.stdout  # protocol channel; libraries print chatter ("loaded PerthNet...") that must not mix in
sys.stdout = sys.stderr

import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="mps" if torch.backends.mps.is_available() else "cpu")
print("ready", file=reply, flush=True)
for line in sys.stdin:
    req = json.loads(line)
    try:
        torch.manual_seed(req["seed"])
        wav = model.generate(req["text"], language_id=req["lang"], audio_prompt_path=req["ref"])
        torchaudio.save(req["out"], wav.cpu(), model.sr)
        print("done", file=reply, flush=True)
    except Exception as e:  # keep the worker alive; the caller falls back to the other engine
        print(f"error {e!r}".replace("\n", " "), file=reply, flush=True)
