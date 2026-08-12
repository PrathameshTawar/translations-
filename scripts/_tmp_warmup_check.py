import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
from app.services.model_registry import ModelRegistry
reg = ModelRegistry.get_instance()
m = reg.get_model('nllb-200-distilled-600M')
sents = [
    'The weather is pleasant today.',
    'The hospital is near the station.',
    'Please send me the report tomorrow.',
    'The bank offers a low interest rate on education loans.',
    'Students must submit their assignments before Friday.',
]
for i, s in enumerate(sents):
    t0 = time.time()
    r = m.translate(s, 'english', 'hindi')
    el = (time.time() - t0) * 1000
    g = r.timings.get('generate_ms', 0)
    print(f'#{i+1} gen={g:.0f}ms tok={r.timings.get("tokenize_ms",0):.0f}ms total={el:.0f}ms | {r.translated[:40]}')
