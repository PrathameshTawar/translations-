"""
Terminology ablation — RAG/terminology on vs off, per strategy.

Runs every terminology strategy (none / prompt / inline / placeholder /
constrained) with the same model on the term-bearing subset of a parallel
test set and reports term accuracy, chrF++ (with a paired-bootstrap CI vs
'none'), BLEU, fallback rate and latency. See
app/evaluation/terminology_ablation.py for the protocol.

Test sets:
  --dataset flores        FLORES-200 devtest (HF Hub, gated: huggingface-cli
                          login), or local files via --flores-dir / FLORES_DIR
  --dataset tsv:PATH      tab-separated "source<TAB>reference" lines

Glossary: data/terminology/{lang}_terms.json plus any --glossary files (JSON
list of {"term", "translation"}). The 15-term glossaries match few FLORES
sentences; the run is flagged "underpowered" below --min-sentences (200).

Usage (from backend/):
    python scripts/run_ablation.py --lang hindi --dataset flores \\
        --glossary my_hindi_glossary.json --out ../reports/ablation
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.evaluation.terminology_ablation import (  # noqa: E402
    MIN_SENTENCES, render_markdown, run_ablation,
)
from app.services.term_index import load_terms  # noqa: E402
from app.services.terminology import STRATEGIES  # noqa: E402

logger = logging.getLogger('run_ablation')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lang', default='hindi', help='Comma-separated target languages')
    p.add_argument('--dataset', default='flores', help="'flores' or 'tsv:PATH'")
    p.add_argument('--flores-dir', default=None, help='Local FLORES-200 directory')
    p.add_argument('--glossary', action='append', default=[],
                   help='Extra glossary JSON (repeatable)')
    p.add_argument('--strategies', default=','.join(STRATEGIES))
    p.add_argument('--model', default='nllb-200-distilled-600M')
    p.add_argument('--num-beams', type=int, default=None)
    p.add_argument('--min-sentences', type=int, default=MIN_SENTENCES)
    p.add_argument('--max-sentences', type=int, default=None)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--out', default=os.path.join('..', 'reports', 'ablation'))
    return p.parse_args()


def load_pairs(dataset, lang, flores_dir):
    if dataset == 'flores':
        from app.evaluation.datasets import load_flores_dataset
        ds = load_flores_dataset(target_langs=[lang], split='devtest',
                                 local_dir=flores_dir)
        if 'error' in ds:
            raise SystemExit(f"FLORES load failed: {ds['error']}")
        return ds['sources'], ds['references'][lang], ds['metadata']['source']
    if dataset.startswith('tsv:'):
        path = dataset[4:]
        sources, refs = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 2:
                    sources.append(parts[0])
                    refs.append(parts[1])
        return sources, refs, f'tsv:{os.path.basename(path)}'
    raise SystemExit(f'Unknown --dataset {dataset!r}')


def load_glossary(lang, extra_paths):
    glossary = load_terms(lang)
    for path in extra_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        glossary += list(data.values()) if isinstance(data, dict) else data
    return glossary


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args = parse_args()
    strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f'Unknown strategies: {sorted(unknown)}')

    from app.services.model_registry import ModelRegistry
    registry = ModelRegistry.get_instance()

    results = {}
    for lang in [l.strip() for l in args.lang.split(',') if l.strip()]:
        sources, refs, source_name = load_pairs(args.dataset, lang, args.flores_dir)
        glossary = load_glossary(lang, args.glossary)

        def translate_fn(text, _lang=lang, **kw):
            return registry.translate(text=text, source_lang='english',
                                      target_lang=_lang, model=args.model, **kw)

        gen = {'num_beams': args.num_beams} if args.num_beams else {}
        logger.info(f'en->{lang}: {len(sources)} pairs from {source_name}, '
                    f'glossary {len(glossary)} entries')
        res = run_ablation(translate_fn, sources, refs, glossary, strategies,
                           min_sentences=args.min_sentences,
                           max_sentences=args.max_sentences,
                           n_bootstrap=args.bootstrap, **gen)
        res['dataset'] = source_name
        res['model'] = args.model
        results[lang] = res

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    base = os.path.join(args.out, f'terminology_ablation_{stamp}')
    with open(base + '.json', 'w', encoding='utf-8') as f:
        json.dump({'generated_at': stamp, 'args': vars(args), 'results': results},
                  f, ensure_ascii=False, indent=2)
    md = render_markdown(results)
    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write(md + '\n')
    print(md)
    print(f'\nSaved {base}.json and {base}.md')


if __name__ == '__main__':
    main()
