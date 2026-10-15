==============================================================
              INDIC TRANSLATOR — MASTER BENCHMARK
==============================================================

## Scorecard (acceptance criteria)

| Category | Target | Measured | Status |
| --- | ---: | ---: | --- |
| Term accuracy with enforcement (term-bearing FLORES subset) | >= 90% | n/a | NOT MEASURED |
| chrF++ change from enforcement vs no enforcement | >= -0.5 chrF++ | n/a | NOT MEASURED |
| Term-bearing test sentences (smallest language) | >= 200 sents | n/a | NOT MEASURED |
| Retrieval Recall@5, paraphrase queries | >= 80% | 0.0% (keyword-fallback) | FAIL |
| Distractor sentences with a term enforced | <= 5% | 0.0% | PASS |
| Retrieval index size | >= 1000 entries | 15 | FAIL |
| P95 translation latency, CPU (worst pair) | <= 3000ms | n/a | NOT MEASURED |
| Degenerate/hallucinated outputs | <= 5% | n/a | NOT MEASURED |
| Language detection macro-F1 (in-repo test split) | >= 0.95 | 0.98 | PASS |

Summary: 2 PASS / 2 FAIL / 5 not measured

Targets are fixed in `app/evaluation/report.py` (DEFAULT_TARGETS) before any run:

- **Term accuracy with enforcement (term-bearing FLORES subset)**: A glossary the system was told to use should appear in at least 9 of 10 outputs (exact-match usage rate, Dinu et al. 2019).
- **chrF++ change from enforcement vs no enforcement**: Enforcing terms must not cost more than 0.5 chrF++ on the same sentences.
- **Term-bearing test sentences (smallest language)**: Below ~200 sentences the chrF++ CI is too wide to support a claim.
- **Retrieval Recall@5, paraphrase queries**: Exact matches are handled lexically; retrieval earns its place only on mentions that do not repeat the term.
- **Distractor sentences with a term enforced**: Injecting a term into a sentence that does not need it corrupts the translation.
- **Retrieval index size**: Recall on a tiny index is not evidence; the shipped glossaries have 15 entries.
- **P95 translation latency, CPU (worst pair)**: Interactive use on a CPU-only host.
- **Degenerate/hallucinated outputs**: Repetition loops, empty or wrong-script outputs.
- **Language detection macro-F1 (in-repo test split)**: Wrong source-language detection breaks every later step.

Test suites (CI checks, not scored): functional 26/26, unicode 20/20, security 40/40, e2e 10/10

## Retrieval (terminology RAG)

Backend: **keyword-fallback** · 735 queries · index size **15** entries (too small to be evidence)

| Query type | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| sentence | 95.6% | 97.8% | 0.9667 |
| paraphrase | 0.0% | 0.0% | 0 |

| Enforcement (what the translate route would inject) | Value |
| --- | ---: |
| Sentence queries: correct term enforced | 100.0% |
| Distractor queries: any term enforced (false positive) | 0.0% |
| Precision of enforced terms | 91.8% |

*Index size 15 entries per language (< 1000: a sanity check, not evidence of retrieval quality; pad the index with --pad-index). Paraphrase queries do not contain the term, so lexical backends cannot retrieve them.*

## Translation memory (with vs without)

*A cache hit skips the model, so the latency reduction is by construction; the useful numbers are the hit rates.*

Pair: **english->hindi** · sentences: 8 · encoder: False

| Metric | Value |
| --- | ---: |
| Model latency mean | 5 ms |
| TM-hit latency mean | 0 ms |
| Exact hit rate | 100.0% |
| Semantic hit rate | 0.0% |
| Latency reduction | 100.0% |

## Quality estimation (good vs bad detection)

*Synthetic set: known-good outputs vs hand-made degenerate ones. Separating these is a sanity check for the heuristic, not a quality-estimation benchmark (that needs human DA scores, e.g. WMT QE data).*

| Metric | Value |
| --- | ---: |
| Scorer | heuristic |
| Samples | 129 |
| Accuracy | 0.9767 |
| Precision | 0.9333 |
| Recall | 1.0 |
| F1 | 0.9655 |
| AUROC | n/a |
| Word-order violations flagged | 24 |

## Language detection (Unicode-script detector)

Detector: unicode-script + devanagari word/2-5-suffix model (char n-gram fallback) · samples: 361 · test: 180

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9833 |
| Macro F1 | 0.9837 |
| Weighted F1 | 0.9833 |
| Devanagari (hi/mr) F1 — learned part | 0.9347 |

Per-class (P / R / F1):

| Language | P | R | F1 | n |
| --- | ---: | ---: | ---: | ---: |
| english | 1.0 | 1.0 | 1.0 | 21 |
| hindi | 0.9545 | 0.913 | 0.9333 | 23 |
| marathi | 0.9167 | 0.9565 | 0.9362 | 23 |
| bengali | 1.0 | 1.0 | 1.0 | 23 |
| gujarati | 1.0 | 1.0 | 1.0 | 23 |
| tamil | 1.0 | 1.0 | 1.0 | 22 |
| telugu | 1.0 | 1.0 | 1.0 | 23 |
| kannada | 1.0 | 1.0 | 1.0 | 22 |

Confusion matrix (rows=actual, cols=predicted):

```
             english     hindi   marathi   bengali  gujarati     tamil    telugu   kannada
english           21         0         0         0         0         0         0         0
hindi              0        21         2         0         0         0         0         0
marathi            0         1        22         0         0         0         0         0
bengali            0         0         0        23         0         0         0         0
gujarati           0         0         0         0        23         0         0         0
tamil              0         0         0         0         0        22         0         0
telugu             0         0         0         0         0         0        23         0
kannada            0         0         0         0         0         0         0        22
```

*Note:* english/bengali/gujarati/tamil/telugu/kannada are deterministic script-range checks (hence F1 1.0 on clean text); only the Devanagari hi/mr disambiguation is learned. The benchmark trains the word/suffix model on a held-out half of the data (honest generalization); the production detector trains on the full curated set.

## Security & robustness (hostile inputs)

Mode: **mock** · 40/40 cases (100.0%)

| Category | Passed | Total | Rate |
| --- | ---: | ---: | ---: |
| input_validation | 15 | 15 | 100.0% |
| content_robustness | 7 | 7 | 100.0% |
| injection_abuse | 6 | 6 | 100.0% |
| auth | 12 | 12 | 100.0% |

*Hostile/malformed inputs must degrade to clean 4xx/2xx — a 500 (unhandled exception) is a security finding.*

## End-to-end pipeline lifecycle

Mode: **mock** · 10/10 stages (100.0%)

| Stage | Status | Latency | Detail |
| --- | --- | ---: | --- |
| register | PASS | 140.19 ms | registered + token issued |
| login | PASS | 109.52 ms | logged in + token issued |
| me | PASS | 2.79 ms | profile fetched |
| translate_with_rag | PASS | 9.64 ms | translation_id=745718e4-d62e-4a2e-9e4a-75996059691f context_used=False model=mock-echo |
| feedback | PASS | 5.89 ms | feedback accepted |
| history_list | PASS | 8.99 ms | 1 history entries |
| history_detail | PASS | 4.0 ms | translation detail fetched |
| ask_legacy | PASS | 5.59 ms | legacy /ask worked |
| detect | PASS | 0.91 ms | language detected |
| languages | PASS | 0.5 ms | 8 languages |

## Throughput & concurrency

Mode: **mock**

Sequential latency profile:

| Metric | Value |
| --- | ---: |
| Calls | 21 |
| Mean | 2.16 ms |
| Median (P50) | 2.15 ms |
| P95 | 2.21 ms |
| P99 | 2.44 ms |
| Min / Max | 2.11 / 2.44 ms |
| Throughput | 462.0 sents/s |

By length bucket (mean / P95):
- medium: n=15 mean=2.17 ms p95=2.44 ms
- short: n=6 mean=2.15 ms p95=2.21 ms

Concurrency stress test:

| Concurrency | Requests | Wall (s) | req/s | mean (ms) | P95 (ms) | err% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.019 | 428.34 | 2.2 | 2.59 | 0.0 |
| 2 | 16 | 0.018 | 872.64 | 2.14 | 2.25 | 0.0 |
| 4 | 32 | 0.019 | 1685.09 | 2.14 | 2.19 | 0.0 |
| 8 | 64 | 0.02 | 3275.32 | 2.12 | 2.2 | 0.0 |
| 16 | 128 | 0.022 | 5765.55 | 2.12 | 2.22 | 0.0 |

Peak throughput: **5765.55 req/s**

*Mock mode uses a deterministic 2ms stub — it measures framework/harness overhead, NOT model inference latency. Run with --live for real inference numbers.*

## Hallucination & degenerate outputs

Mode: **mock** · detector: deterministic heuristics (entity/leakage/repetition)

| Metric | Value |
| --- | ---: |
| Cases | 11 |
| Entity preservation | 1.0 |
| Mean source leakage | 0.1853 |
| Mean repetition | 0.0 |
| Degenerate output rate | 0.0% |

Detector calibration: **100.0%** correct on known-good vs known-degenerate outputs (probe: `Dr. Meera Patel will charge $1,250 for the surgery on 12/08/2026.`)

*Heuristic (model-free) hallucination detection: entity preservation, source leakage, and repetition ratios. Calibration run confirms the detector separates good vs degenerate outputs. Use an LLM judge or human review for semantic hallucination.*

## Notes

- Translation quality is reported, not targeted; compare FLORES-200 numbers with published baselines for the same model.
- Curated references are machine-generated and lightly edited; they are supplementary only.