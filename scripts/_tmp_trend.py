import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
from app.services.model_registry import ModelRegistry
m = ModelRegistry.get_instance().get_model("nllb-200-distilled-600M")
S = "Please tell me where I can find the nearest hospital."
m.translate(S, "english", "hindi")
for i in range(12):
    t0=time.time(); m.translate(S, "english", "hindi"); print(f"#{i+1:>2}: {(time.time()-t0)*1000:6.0f} ms")
