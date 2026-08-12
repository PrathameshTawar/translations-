import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
import torch
from app.services.model_registry import ModelRegistry
SENTS = [("short(5w)", "The weather is pleasant today."), ("med(12w)", "The bank offers a low interest rate on education loans."), ("long(25w)", "Students must submit their assignments before the end of this week to their class teacher."), ("xlong(40w)", "The committee formed last year to study the impact of climate change on agriculture in the region has published its detailed report, which recommends several new measures to protect small farmers from unpredictable weather.")]
def bench(label, num_beams, threads):
    torch.set_num_threads(threads)
    m = ModelRegistry.get_instance().get_model("nllb-200-distilled-600M")
    print(f"=== num_beams={num_beams} threads={threads} ===")
    for tag, s in SENTS:
        m.translate(s, "english", "hindi", num_beams=num_beams)
        tt=[]
        for _ in range(2):
            t0=time.time(); m.translate(s, "english", "hindi", num_beams=num_beams); tt.append((time.time()-t0)*1000)
        print(f"  {tag:<12} {statistics.mean(tt):8.0f} ms")
bench("beam4", 4, 12)
bench("beam1", 1, 12)
