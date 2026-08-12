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
client = app.test_client()
from app.services.model_registry import ModelRegistry
m = ModelRegistry.get_instance().get_model("nllb-200-distilled-600M")
from app.services.translator import TranslationEngine
eng = TranslationEngine()

S = "Please tell me where I can find the nearest hospital."
body = {"text": S, "source_lang":"english","target_lang":"hindi","model":"nllb-200-distilled-600M"}

def t(fn, label):
    fn()
    t0=time.time(); fn(); dt=(time.time()-t0)*1000
    print(f"{label:<40} {dt:7.0f} ms")

t(lambda: client.post("/api/v1/translate/text", json={**body,"include_context":False}), "ROUTE no-rag")
t(lambda: client.post("/api/v1/translate/text", json={**body,"include_context":True}), "ROUTE with-rag")
t(lambda: m.translate(S, "english", "hindi"), "direct wrapper")
t(lambda: eng.translate(S, "english", "hindi", model_name="nllb-200-distilled-600M"), "engine")
t(lambda: client.post("/api/v1/translate/text", json={**body,"include_context":True}), "ROUTE with-rag #2")
os.remove(db_path)
