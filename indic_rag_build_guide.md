# Project: Reproducing "Efficient RAG for Indic-Language QA"
Target: hit Recall@5 ≥ 70%, F1 ≥ 70, hallucination ≤ 15%, latency < 2s on IndicSQuAD-Hindi.

## 0. Scope decision
Full paper = 11 languages + 2 backbones + 2 LLMs + reranker + full deployment.
For a solo build in reasonable time, cut to:
- **1 backbone**: IndicBERT (fewer params, paper shows it wins anyway)
- **2-3 languages**: Hindi (primary, has most data) + Marathi (cross-lingual check)
- **1 LLM**: LLaMA-3-8B via Groq API (free tier, fast, matches paper's low-latency config)
- All 4 datasets from paper, but only Hindi split for training

This subset still reproduces every headline number in Table II.

---

## 1. Data acquisition

```bash
pip install datasets --break-system-packages
```

- **IndicSQuAD**: `datasets.load_dataset("ai4bharat/IndicSQuAD" ...)` — check exact HF path; if unavailable, source is arXiv:2505.03688, GitHub repo linked there. This is your fine-tuning + primary eval set.
- **AI4Bharat IndicQA**: `datasets.load_dataset("ai4bharat/IndicQA", "indicqa.hi")` — cross-lingual eval only, no training.
- **FIRE VATIKA**: CEUR-WS Vol. 4173 (Kumar et al., FIRE'25) — small Hindi tourism set, request/download from task page.
- **Custom FAQ**: build your own — scrape ~500 Hindi/Marathi/English FAQ pairs from 2-3 public bank/university FAQ pages. This demonstrates the "realistic deployment" claim in the paper.

Store each as JSONL: `{"question": ..., "context": ..., "answer": ..., "lang": "hi"}`.

---

## 2. Preprocessing pipeline (Section V-A of paper)

Build `preprocess.py` with 5 stages, in order:

1. **NFC Unicode normalization** — `unicodedata.normalize("NFC", text)` on every field. Critical for Devanagari where combining marks have multiple encodings.
2. **Script-aware tokenization** — `pip install indic-nlp-library`, use `indicnlp.tokenize.indic_tokenize`.
3. **Chunking** — passages > 400 tokens → sliding windows of 200-400 tokens, 20% overlap. Paper's ablation (Table IV) shows 200 tokens is the sweet spot — build this as the default but keep chunk size as a config parameter so you can reproduce the full ablation curve (100/200/400/600).
4. **Metadata tagging** — attach `lang` tag to every chunk (this drives per-language FAISS routing later).
5. **Noise stripping** — drop doc IDs, scrape metadata, HTML artifacts before it touches training or indexing.

```python
def preprocess(text, lang):
    text = unicodedata.normalize("NFC", text)
    tokens = indic_tokenize.trivial_tokenize(text, lang)
    return tokens
```

---

## 3. Fine-tune the embedding model (Section V-B) — the core contribution

```bash
pip install sentence-transformers==2.6.1 transformers==4.38 torch --break-system-packages
```

Use **Multiple Negatives Ranking (MNR) loss** — in-batch negatives, no hard-negative mining needed:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("ai4bharat/indic-bert")  # or "distilbert-base-multilingual-cased"

train_examples = [InputExample(texts=[q, positive_passage]) for q, positive_passage in qa_pairs]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model, scale=1/0.05)  # temperature τ=0.05

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,                      # paper: 3-5
    warmup_steps=int(0.1 * len(train_dataloader) * 4),
    optimizer_params={"lr": 5e-5},
    weight_decay=0.01,
    output_path="indicbert-finetuned-hindi",
)
```

Loss being implemented (paper eq. i):
L = −log[ exp(sim(q,p⁺)/τ) / Σᵢ exp(sim(q,pᵢ)/τ) ]

Early stopping: hold out 10% of IndicSQuAD, evaluate Recall@5 every N steps, stop when it plateaus.
This step alone should take you from ~64% → ~74% Recall@5 per the paper's ablation — **this is the single biggest lever (+10.3 pts)**, so get it right before anything else.

---

## 4. Build the FAISS index (Section V-C)

```bash
pip install faiss-cpu --break-system-packages
```

```python
import faiss, numpy as np

embeddings = model.encode(all_passage_chunks, convert_to_numpy=True)  # 768-dim
d = embeddings.shape[1]

if len(all_passage_chunks) < 5000:
    index = faiss.IndexFlatL2(d)          # exact search for small corpora (VATIKA, custom FAQ)
else:
    nlist = int(np.sqrt(len(all_passage_chunks)))   # paper's heuristic
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(embeddings)

index.add(embeddings)
faiss.write_index(index, "hi_index.faiss")
```

Build **one index per language** — no cross-lingual leakage, as the paper specifies. Store a `lang → index_path` map.

---

## 5. LangChain retrieval-generation pipeline (Section V-D)

```bash
pip install langchain langchain-community --break-system-packages
```

Steps to wire, in order:
1. Language detection → `pip install fasttext`, load `lid.176.bin`, route query to correct FAISS index.
2. Encode query with your fine-tuned model.
3. Retrieve top-K (default K=5, per ablation sweet spot) via FAISS cosine/L2.
4. **Rerank** with cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` — rescore the K passages, keep top-K′.
5. Concatenate reranked passages + query into a structured prompt template that explicitly instructs: *"Only answer using the provided context; say so explicitly if the answer isn't present."*
6. Send to LLM (LLaMA-3-8B via Groq API — free, fast, matches paper's 620ms/200-token figure).

```python
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=your_reranking_retriever)
```

---

## 6. Hallucination detection layer (Section V-E)

Use an off-the-shelf NLI model as a factual-consistency scorer:

```python
from transformers import pipeline
nli = pipeline("text-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

def consistency_score(answer, passages):
    premise = " ".join(passages)
    result = nli(f"{premise} [SEP] {answer}")
    return result[0]["score"]  # treat as confidence; flag if < 0.5
```

Attach this score to every generated answer. If < 0.5, surface a low-confidence warning in the UI — don't block generation.

---

## 7. Evaluation harness (Section VI-B) — reproduce Tables II, III, IV exactly

Implement these metrics as one script (`evaluate.py`):

| Metric | Formula/tool |
|---|---|
| Recall@K | fraction of queries where gold passage ∈ top-K retrieved |
| MRR | mean of 1/rank of first correct passage |
| EM / F1 | standard SQuAD scoring (`evaluate` library, `squad` metric) |
| ROUGE-L | `rouge-score` package |
| BERTScore-F1 | `bert-score` package (use a multilingual model) |
| Hallucination rate | % of answers with NLI consistency score ≤ 0.5 |
| Latency | median wall-clock over 200 queries, broken into encode/retrieve/rerank/generate |

Run this against **4 baselines** to reproduce Table II:
1. TF-IDF + LLM (sklearn `TfidfVectorizer`, unigrams+bigrams)
2. BM25 + LLM (`rank_bm25` package)
3. Untuned SBERT (`msmarco-distilbert-base-v4`) + LLM
4. LLM-alone (no retrieval)
5. Your fine-tuned RAG system

Expected result pattern (from paper, your target):

| System | Recall@5 | F1 | Hallucination |
|---|---|---|---|
| TF-IDF | ~50% | ~58 | ~31% |
| BM25 | ~57% | ~61 | ~29% |
| Untuned SBERT | ~62% | ~66 | ~24% |
| LLM-alone | N/A | ~52 | ~42% |
| **Yours** | **~72%** | **~71** | **~13%** |

---

## 8. Ablations (Section VIII) — run these as config sweeps, log to a CSV

- Chunk size ∈ {100, 200, 400, 600} tokens
- Top-K ∈ {1, 5, 10, 20}
- Backbone ∈ {IndicBERT, mDistilBERT} × {fine-tuned, not fine-tuned}
- Reranker on/off

Write one `run_ablation.py` that takes these as CLI flags and appends a row to `ablation_results.csv`. This directly reproduces Table IV and gives you the paper's exact narrative (200 tokens optimal, fine-tuning is the biggest lever, reranker +3.9 Recall@5 for +0.4s latency).

---

## 9. Deployment (Section IX)

Minimal viable version of the paper's 6-module architecture:

```
docker-compose.yml
├── embedding-service     # loads fine-tuned model, exposes /encode
├── faiss-service         # loads per-language indexes, exposes /search
├── reranker-service      # cross-encoder, exposes /rerank
├── langchain-orchestrator # ties encode→search→rerank→generate
├── fastapi-gateway       # public REST API
└── react-frontend        # query box, Devanagari input, shows answer + confidence score + sources
```

FastAPI endpoint sketch:

```python
@app.post("/ask")
async def ask(query: str, lang: str = None):
    lang = lang or detect_language(query)
    passages = retrieve(query, lang, k=5)
    reranked = rerank(query, passages, k=3)
    answer = generate(query, reranked)
    score = consistency_score(answer, reranked)
    return {"answer": answer, "confidence": score, "sources": reranked}
```

Add a "Download PDF report" button (query + passages + scores + timestamp) — matches paper's Section IX feature.

---

## 10. Suggested build order (so you always have something running)

1. Data + preprocessing → 2. Untuned SBERT baseline + FAISS + evaluate.py (gives you TF-IDF/BM25/SBERT rows immediately) → 3. Fine-tune IndicBERT → re-run evaluate.py (this is your headline result) → 4. Add reranker → 5. Add hallucination scorer → 6. Run ablations → 7. Wrap in FastAPI+React → 8. Write up results against Tables II/III/IV.

This order means by step 3 you already have a defensible "improvement over baselines" result even if you stop there — everything after is polish that matches the paper's remaining claims (reranking, deployment, hallucination transparency).
