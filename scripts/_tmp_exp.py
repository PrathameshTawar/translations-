import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import torch
from app.services.model_registry import ModelRegistry

SENTS = [
    ('short(5w)', 'The weather is pleasant today.'),
    ('med(12w)', 'The bank offers a low interest rate on education loans.'),
    ('long(25w)', 'Students must submit their assignments before the end of this week to their class teacher.'),
    ('xlong(40w)', 'The committee formed last year to study the impact of climate change on agriculture in the region has published its detailed report, which recommends several new measures to protect small farmers from unpredictable weather.'),
]

def bench(label, num_beams, threads, sents=SENTS, reps=2):
    torch.set_num_threads(threads)
    reg = ModelRegistry.get_instance()
    m = reg.get_model('nllb-200-distilled-600M')
    print(f'\n=== num_beams={num_beams} threads={threads} ===')
    times = {}
    for tag, s in sents:
        # skip first call (oneDNN warmup) once per process/beam config
        m.translate(s, 'english', 'hindi', num_beams=num_beams)
        tt = []
        for _ in range(reps):
            t0 = time.time()
            r = m.translate(s, 'english', 'hindi', num_beams=num_beams)
            tt.append((time.time() - t0) * 1000)
        avg = statistics.mean(tt)
        times[tag] = avg
        print(f'  {tag:<12} {avg:8.0f} ms   gen={r.timings.get("generate_ms",0):.0f} ms')

print('torch default threads:', torch.get_num_threads())
# beams=4 (current prod) at default threads
bench('beam4-threads-default', 4, torch.get_num_threads())
# beams=1 (greedy) — the cheap ablation
bench('beam1-threads-default', 1, torch.get_num_threads())
