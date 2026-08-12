import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
from app.services.model_registry import ModelRegistry
m = ModelRegistry.get_instance().get_model("nllb-200-distilled-600M")
S = "Please tell me where I can find the nearest hospital."
CTX = "hospital = अस्पताल (Healthcare)"

def t(fn, label, n=2):
    fn()
    ts=[]
    for _ in range(n):
        t0=time.time(); fn(); ts.append((time.time()-t0)*1000)
    print(f"{label:<45} {ts[0]:6.0f} / {ts[1]:6.0f} ms")

t(lambda: m.translate(S, "english", "hindi"), "direct, no context")
t(lambda: m.translate(S, "english", "hindi", context=CTX), "direct, WITH context")
# does running MiniLM + FAISS right before affect NLLB?
from app.services.rag_service import IndicRAGPipeline
rag = IndicRAGPipeline.get_instance()
def with_rag():
    rag.retrieve_context(S, source_lang="english", target_lang="hindi")
    m.translate(S, "english", "hindi")
t(with_rag, "MiniLM+FAISS run, then direct")
