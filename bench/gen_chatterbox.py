"""Bake-off generator for Chatterbox multilingual. Run with engines/chatterbox/.venv/bin/python."""
import csv, sys
from pathlib import Path
import torch, torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

voice, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
for row in csv.DictReader(open(Path(__file__).parent / "sentences.tsv", encoding="utf-8"), delimiter="\t"):
    torch.manual_seed(0)
    wav = model.generate(row["text"], language_id=row["lang"], audio_prompt_path=str(voice / "ref.wav"))
    torchaudio.save(str(out / f"{row['id']}.wav"), wav.cpu(), model.sr)
    print(row["id"], flush=True)
