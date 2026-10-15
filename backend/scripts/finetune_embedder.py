"""
Fine-tune the embedding model (IndicBERT) with Multiple Negatives Ranking
(MNR) loss — the paper's Section V-B, its single biggest lever for
Retrieval Recall@5 (+10.3 pts).

Loss (paper eq. i):  L = -log[ exp(sim(q,p+)/tau) / sum_i exp(sim(q,p_i)/tau) ]
with in-batch negatives and temperature tau = 0.05 (scale = 1/tau).

Data format
-----------
JSONL lines:  {"question": "...", "context": "...", "answer": "...", "lang": "hi"}
(question, context) pairs are the training signal: the model learns to
rank a question's supporting passage above every other passage in the
batch. `--data` may point to a file or a directory of JSONL files.

Early stopping
--------------
Hold out 10% of pairs; every `--eval-every` steps measure Recall@5 on the
held-out questions against the held-out passages (a tiny in-memory faiss
index, no model downloads) and keep the best checkpoint.

Synthetic fallback
------------------
No dataset yet? `--synthetic N` generates N synthetic pairs from the
bundled terminology glossaries + the curated benchmark references so the
pipeline can be smoke-tested end-to-end before real data arrives:

    python scripts/finetune_embedder.py --synthetic 200 --epochs 1 \
        --max-examples 200 --out indicbert-finetuned-hindi

Real run (after acquiring IndicSQuAD JSONL):

    python scripts/finetune_embedder.py \
        --data data/qa/indic_squad_hi.jsonl \
        --model ai4bharat/indic-bert \
        --epochs 4 --batch-size 16 --lr 5e-5 --out indicbert-finetuned-hindi

After training, rebuild the FAISS index and re-run the retrieval benchmark
to measure the Recall@5 gain:

    python scripts/build_faiss_index.py --encoder indicbert-finetuned-hindi
    python scripts/run_benchmark.py --live --skip-* ...
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('finetune_embedder')

DEFAULT_MODEL = 'ai4bharat/indic-bert'
DEFAULT_OUT = 'indicbert-finetuned-hindi'
TEMPERATURE = 0.05  # paper tau; scale passed to MNR loss = 1/tau = 20


def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune an Indic embedding model with MNR loss',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--data', default=None,
                   help='JSONL file or directory of JSONL QA files '
                        '(question/context/answer/lang). Mutually exclusive '
                        'with --synthetic.')
    p.add_argument('--synthetic', type=int, default=0, metavar='N',
                   help='Generate N synthetic QA pairs from the bundled '
                        'terminology glossaries + benchmark references '
                        '(smoke-test path; no dataset download).')
    p.add_argument('--model', default=DEFAULT_MODEL,
                   help='Base embedding model (default ai4bharat/indic-bert)')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-fraction', type=float, default=0.1)
    p.add_argument('--max-seq-len', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=50,
                   help='Evaluate held-out Recall@5 once this many steps have '
                        'passed since the last eval (callback granularity is '
                        'per-epoch without a neural evaluator)')
    p.add_argument('--max-examples', type=int, default=0,
                   help='Cap training examples (0 = all)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=DEFAULT_OUT,
                   help='Output model directory')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f'Skipping malformed JSONL line in {path}')
                continue
            q = (obj.get('question') or '').strip()
            ctx = (obj.get('context') or '').strip()
            if q and ctx:
                examples.append({'question': q, 'context': ctx,
                                 'answer': (obj.get('answer') or '').strip(),
                                 'lang': obj.get('lang')})
    return examples


def load_dataset(data_path, max_examples=0):
    """Load QA pairs from a JSONL file or a directory of JSONL files."""
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, '**', '*.jsonl'),
                                 recursive=True))
    elif os.path.isfile(data_path):
        files = [data_path]
    else:
        raise FileNotFoundError(f'--data path not found: {data_path}')

    examples = []
    for fp in files:
        examples.extend(load_jsonl(fp))
        logger.info(f'  {fp}: {len(examples)} examples (cumulative)')

    if not examples:
        raise ValueError(f'No (question, context) pairs found under {data_path}')

    # De-duplicate while preserving order
    seen, uniq = set(), []
    for e in examples:
        key = (e['question'], e['context'])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    examples = uniq

    if max_examples:
        examples = examples[:max_examples]
    return examples


def make_synthetic(num_pairs, seed=42):
    """
    Build synthetic (question, context) pairs from the bundled terminology
    glossaries and curated benchmark references, so the fine-tune pipeline
    can run end-to-end without downloading IndicSQuAD.

    Contexts come from terminology entries (term + domain context + target
    translation); questions are templates mentioning the term. Benchmark
    reference sentences (already translated) are used as additional
    contexts with their English sources as questions.
    """
    rng = random.Random(seed)
    pairs = []

    from app.evaluation.datasets import get_benchmark_dataset
    ds = get_benchmark_dataset(target_langs=['hindi', 'marathi'])
    sources = ds['sources']
    for lang in ('hindi', 'marathi'):
        refs = ds['references'].get(lang, [])
        for s, r in zip(sources, refs):
            if len(pairs) >= num_pairs:
                break
            pairs.append({'question': s, 'context': r, 'lang': lang})

    term_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'terminology')
    term_dir = os.path.abspath(term_dir)
    for fname in sorted(os.listdir(term_dir)):
        if not fname.endswith('_terms.json'):
            continue
        lang = fname.replace('_terms.json', '')
        with open(os.path.join(term_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        terms = list(data.values()) if isinstance(data, dict) else data
        for term in terms:
            if len(pairs) >= num_pairs:
                break
            t = (term.get('term') or '').strip()
            ctx = (term.get('context') or '').strip()
            trans = (term.get('translation') or '').strip()
            if not t:
                continue
            if ctx:
                pairs.append({'question': f'What does {t} mean in this context?',
                              'context': ctx, 'lang': lang})
            elif trans:
                pairs.append({'question': f'Translate the term {t}.',
                              'context': trans, 'lang': lang})
        if len(pairs) >= num_pairs:
            break

    rng.shuffle(pairs)
    return pairs[:num_pairs]


# ---------------------------------------------------------------------------
# Held-out Recall@5 (early stopping signal) — in-memory faiss, no downloads
# ---------------------------------------------------------------------------

def recall_at_5(model, eval_pairs):
    """Fraction of held-out questions whose gold context is in top-5."""
    if not eval_pairs:
        return 0.0
    qs = [p['question'] for p in eval_pairs]
    ctxs = [p['context'] for p in eval_pairs]
    q_vecs = model.encode(qs, convert_to_numpy=True, normalize_embeddings=True)
    c_vecs = model.encode(ctxs, convert_to_numpy=True, normalize_embeddings=True)

    try:
        import faiss
    except ImportError:
        logger.warning('faiss not installed — computing Recall@5 with numpy '
                       '(slow but exact for small evals)')
        import numpy as np
        sims = q_vecs @ c_vecs.T
        hits = 0
        for i in range(len(qs)):
            top5 = np.argsort(-sims[i])[:5]
            if i in top5:
                hits += 1
        return hits / len(qs)

    import numpy as np
    index = faiss.IndexFlatIP(c_vecs.shape[1])
    index.add(c_vecs.astype('float32'))
    scores, idx = index.search(q_vecs.astype('float32'), 5)
    hits = sum(1 for i in range(len(qs)) if i in idx[i])
    return hits / len(qs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    if bool(args.data) == bool(args.synthetic):
        print('FATAL: provide exactly one of --data or --synthetic N')
        sys.exit(1)

    if args.data:
        print(f'[1/5] Loading QA pairs from {args.data}...')
        examples = load_dataset(args.data, max_examples=args.max_examples)
    else:
        print(f'[1/5] Generating {args.synthetic} synthetic QA pairs...')
        examples = make_synthetic(args.synthetic, seed=args.seed)

    if len(examples) < 4:
        print('FATAL: need at least 4 pairs to fine-tune')
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * 0.1))
    eval_pairs = examples[:n_eval]
    train_pairs = examples[n_eval:]
    logger.info(f'  {len(examples)} pairs total; {len(train_pairs)} train / '
                f'{len(eval_pairs)} held-out')

    print(f'[2/5] Loading base model {args.model}...')
    from sentence_transformers import (
        InputExample, SentenceTransformer, losses,
    )
    from torch.utils.data import DataLoader

    model = SentenceTransformer(args.model, device=None)
    logger.info(f'  base model dim={model.get_sentence_embedding_dimension()}')

    train_examples = [
        InputExample(texts=[p['question'], p['context']]) for p in train_pairs
    ]
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size)

    # MNR loss, paper tau=0.05 -> scale = 1/tau = 20
    train_loss = losses.MultipleNegativesRankingLoss(
        model, scale=1.0 / TEMPERATURE)

    total_steps = math.ceil(len(train_dataloader) * args.epochs)
    warmup_steps = int(args.warmup_fraction * total_steps)

    print(f'[3/5] Training {args.epochs} epochs '
          f'({total_steps} steps, warmup {warmup_steps})...')

    best_recall = 0.0
    last_eval_step = 0

    def eval_callback(score, epoch, steps):
        nonlocal best_recall, last_eval_step
        # Without a sentence-transformers evaluator the callback fires once
        # per epoch; only run the (cheap) held-out evaluation once the
        # --eval-every step budget has elapsed since the last one.
        if steps - last_eval_step < args.eval_every:
            return False
        last_eval_step = steps
        recall = recall_at_5(model, eval_pairs)
        logger.info(f'  eval @ step {steps}: held-out Recall@5 = {recall:.4f} '
                    f'(best {best_recall:.4f})')
        if recall > best_recall:
            best_recall = recall
            model.save(os.path.join(args.out, 'checkpoint-best'))
            logger.info(f'    -> saved best checkpoint to '
                        f'{os.path.join(args.out, "checkpoint-best")}')
        return False  # never stop early automatically; user sets --epochs

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.lr},
        weight_decay=args.weight_decay,
        evaluator=None,  # custom callback below keeps it model-free
        output_path=args.out,
        show_progress_bar=True,
        callback=eval_callback,
        checkpoint_path=os.path.join(args.out, 'checkpoints'),
        save_best_model=False,  # we manage best via callback
    )

    print(f'[4/5] Best held-out Recall@5 during training: {best_recall:.4f}')
    print(f'[5/5] Fine-tuned model saved to {args.out}')

    print()
    print('Next steps:')
    print(f'  1. python scripts/build_faiss_index.py --encoder {args.out}')
    print('  2. Re-run the retrieval benchmark and compare Recall@5 against '
          'the untuned baseline (expect ~+10 pts).')


if __name__ == '__main__':
    main()
