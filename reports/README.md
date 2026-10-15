# Reports

Only curated runs are committed here; new runs write to `backend/reports/`
(gitignored).

| Run | What it is | Status |
| --- | --- | --- |
| [`2026-10-15_offline/`](2026-10-15_offline/benchmark.md) | `python scripts/run_benchmark.py` (no models): test suites, retrieval on the new query set, language ID. Reproducible on any machine and in CI. | **Current.** 2 PASS / 2 FAIL / 5 not measured |
| [`2026-08-12_curated_set/`](2026-08-12_curated_set/benchmark.md) | First full live run (NLLB-600M and M2M100-418M, beam 2) on the 42-sentence curated set. | **Superseded.** Kept for history; see the caveats below |

## Not yet run

These need the translation models and FLORES-200 (gated on Hugging Face),
which the environment that produced the reports above could not download:

```bash
cd backend
# Main quality table: FLORES-200 devtest, all 1,012 sentences
python scripts/run_benchmark.py --live --flores 1012 --flores-dir /path/to/flores200_dataset
# Terminology on/off ablation on the term-bearing FLORES subset
python scripts/run_ablation.py --lang hindi,tamil --dataset flores \
    --flores-dir /path/to/flores200_dataset --glossary bigger_glossary.json
# Fold the ablation into the scorecard
python scripts/run_benchmark.py --live --flores 1012 --flores-dir ... \
    --ablation-results reports/ablation/terminology_ablation_<stamp>.json
```

## Why the 2026-08-12 numbers should not be quoted

- **Translation quality** was measured on 42 hand-picked sentences whose
  references were machine-generated and lightly edited. BLEU 67 for
  English→Hindi with NLLB-600M is far above published FLORES-200 scores for
  that model, which is what you would expect when references resemble model
  output.
- **Retrieval 100%** came from querying each 15-entry glossary with its own
  15 entries. The replacement benchmark (`data/eval/terminology_queries.jsonl`)
  uses sentences, paraphrases and distractors; the keyword backend scores
  0% Recall@5 on paraphrases.
- **The scorecard was 10/10 PASS** because several targets were tautological
  (end-to-end tests ≥ 100%, cache lookup ≤ 300 ms) or tuned to the result
  (QE F1 ≥ 0.95 on a synthetic good-vs-degenerate set). Targets now live in
  `backend/app/evaluation/report.py` with a rationale each, and CI-style
  suites are reported as counts, not scored.
- **The RAG ablation** (`ablation_results.csv`, removed) had one row: Hindi,
  3 sentences.
