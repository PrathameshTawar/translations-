import sys, os, time, statistics, tempfile
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

S = "Please tell me where I can find the nearest hospital."
# warmup both
client.post("/api/v1/translate/text", json={"text": S, "source_lang":"english","target_lang":"hindi","model":"nllb-200-distilled-600M"})
m.translate(S, "english", "hindi")

t_route=[]
for i in range(4):
    t0=time.time()
    r = client.post("/api/v1/translate/text", json={"text": S, "source_lang":"english","target_lang":"hindi","model":"nllb-200-distilled-600M"})
    b = r.get_json()
    t_route.append((time.time()-t0)*1000)
    print(f"route #{i+1}: {(time.time()-t0)*1000:6.0f} ms   timings={b['timings']}")

t_direct=[]
for i in range(4):
    t0=time.time(); m.translate(S, "english", "hindi"); t_direct.append((time.time()-t0)*1000)
print("direct mean: %.0f ms" % statistics.mean(t_direct))
print("route  mean: %.0f ms" % statistics.mean(t_route))
os.remove(db_path)
