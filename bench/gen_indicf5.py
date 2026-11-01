"""Bake-off generator for IndicF5 with our current best settings. Run with .venv/bin/python, PYTHONPATH=."""
import csv, sys
from pathlib import Path
import soundfile as sf, torch
import inference
from pacing import rate, speed_for

voice, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
# Optional 3rd arg: TSV (id, text) of hand-written spoken-Devanagari overrides, used as-is.
over = dict(csv.reader(open(sys.argv[3], encoding="utf-8"), delimiter="\t")) if len(sys.argv) > 3 else {}
ref_text = (voice / "ref.txt").read_text(encoding="utf-8").strip()
tempo = rate(voice / "full.wav", (voice / "full.txt").read_text(encoding="utf-8")) / rate(voice / "ref.wav", ref_text)
model = inference.load_model(device="mps")
from scoring.scripts.lib_normalize import to_unified_devanagari
for row in csv.DictReader(open(Path(__file__).parent / "sentences.tsv", encoding="utf-8"), delimiter="\t"):
    if over and row["id"] not in over:
        continue
    text = over.get(row["id"]) or to_unified_devanagari(row["text"])  # English -> Devanagari; Devanagari passes through
    torch.manual_seed(0)
    sf.write(out / f"{row['id']}.wav", inference.synthesize(model, text, voice / "ref.wav", ref_text, preprocess=False,
             speed=speed_for(text, ref_text) * tempo, nfe_step=32, cfg_strength=3.0), 24000)
    print(row["id"], text, flush=True)
