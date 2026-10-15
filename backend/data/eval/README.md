# Evaluation data

`terminology_queries.jsonl` — English queries for the terminology retrieval
benchmark (`app/evaluation/retrieval.py`). Written by the maintainers; every
glossary language shares the same 15 English terms, so one query file serves
all seven glossaries.

| type | count | relevant | what it tests |
| --- | ---: | --- | --- |
| `sentence` | 45 | the term | a natural sentence that contains the term (3 per term) |
| `paraphrase` | 30 | the term | the concept without the term's surface form (2 per term); lexical matching cannot solve these |
| `distractor` | 30 | none | no glossary term, including near-misses (`riverbank`, `machinery`, `healthy`); measures false positives |

Some `sentence` queries contain a second glossary term (e.g. "The bank uses
machine learning…"); only the term the query was written for is labelled
relevant, so precision@K is a lower bound.

The glossaries have 15 entries, which makes any retrieval score on them weak
evidence. Pad the index with unrelated entries (`--pad-index FILE`, a JSON
glossary or a plain list of English terms, one per line) to measure retrieval
against a realistic index size. The benchmark reports the index size it ran
with, and the scorecard marks retrieval as not meaningful below 1,000 entries.
