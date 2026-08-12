import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
from app.services.rag_service import IndicRAGPipeline
rag = IndicRAGPipeline.get_instance()
S = "Please tell me where I can find the nearest hospital."
r = rag.retrieve_context(S, source_lang="english", target_lang="hindi", top_k=5)
ctx = r.get("context")
print("CONTEXT REPR:", repr(ctx))
print("CONTEXT LEN:", len(ctx), "chars")
print("---")
from app.services.model_registry import ModelRegistry
m = ModelRegistry.get_instance().get_model("nllb-200-distilled-600M")
def t(fn, label):
    fn()
    t0=time.time(); fn(); print(f"{label:<30} {(time.time()-t0)*1000:6.0f} ms")
t(lambda: m.translate(S, "english", "hindi"), "no context")
t(lambda: m.translate(S, "english", "hindi", context=ctx), "WITH RAG context")
