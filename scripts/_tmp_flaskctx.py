import sys, os, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
from app import create_app
from app.extensions import db
app = create_app("development")
fd, db_path = tempfile.mkstemp(suffix=".db"); os.close(fd)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + db_path.replace("\\", "/")
app.config["TESTING"] = True
with app.app_context(): db.drop_all(); db.create_all()
from app.services.model_registry import ModelRegistry
m = ModelRegistry.get_instance().get_model("nllb-200-distilled-600M")
from app.services.translator import TranslationEngine
eng = TranslationEngine()
S = "Please tell me where I can find the nearest hospital."
CTX = "hospital = अस्पताल (Healthcare)"
def t(fn, label):
    fn()
    t0=time.time(); fn(); print(f"{label:<42} {(time.time()-t0)*1000:7.0f} ms")
t(lambda: m.translate(S, "english", "hindi", context=CTX), "direct wrapper, ctx")
t(lambda: eng.translate(S, "english", "hindi", context=CTX, model_name="nllb-200-distilled-600M"), "engine, ctx")
os.remove(db_path)
