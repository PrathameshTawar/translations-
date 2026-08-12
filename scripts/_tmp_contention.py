import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
from app.services.model_registry import ModelRegistry
S = "Please tell me where I can find the nearest hospital."
m = ModelRegistry.get_instance().get_model("nllb-200-distilled-600M")
def time_it(label):
    m.translate(S, "english", "hindi")
    tt=[]
    for _ in range(3):
        t0=time.time(); m.translate(S, "english", "hindi"); tt.append((time.time()-t0)*1000)
    print(f"{label}: {statistics.mean(tt):.0f} ms")
time_it("before-MiniLM")
from app.services.rag_service import IndicRAGPipeline
rag = IndicRAGPipeline.get_instance()
time_it("after-RAG-MiniLM-load")
from app.services.translation_memory import TranslationMemory
tm = TranslationMemory.get_instance()
_ = tm.encoder_available
time_it("after-TM-MiniLM-load")
print("free RAM after all:", round(__import__('psutil', importlib_mode='x') if False else 0,0))
import ctypes, os
kernel32 = ctypes.windll.kernel32
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64)]
s = MS(); s.dwLength = ctypes.sizeof(MS); kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
print(f"free RAM now: {s.ullAvailPhys/2**30:.1f} GB of {s.ullTotalPhys/2**30:.1f} GB")
