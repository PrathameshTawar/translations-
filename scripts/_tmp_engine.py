import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
from app.services.model_registry import ModelRegistry
m = ModelRegistry.get_instance().get_model("nllb-200-distilled-600M")
from app.services.translator import TranslationEngine
eng = TranslationEngine()
S = "Please tell me where I can find the nearest hospital."
CTX = "hospital = अस्पताल (Healthcare)"

def t(fn, label, n=2):
    fn()
    ts=[]
    for _ in range(n):
        t0=time.time(); fn(); ts.append((time.time()-t0)*1000)
    print(f"{label:<48} {ts[0]:6.0f} / {ts[1]:6.0f} ms")

t(lambda: m.translate(S, "english", "hindi", context=CTX), "wrapper, context")
t(lambda: eng.translate(S, "english", "hindi", context=CTX, model_name="nllb-200-distilled-600M"), "engine->registry, context")
t(lambda: eng.translate(S, "english", "hindi", model_name="nllb-200-distilled-600M"), "engine->registry, no context")
