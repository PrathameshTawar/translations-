"""
Run evaluation benchmarks from the command line.

Usage:
    # Evaluate NLLB-200 on Hindi
    python scripts/run_evaluation.py --langs hindi

    # Evaluate with specific metrics
    python scripts/run_evaluation.py --langs hindi,bengali --metrics bleu,chrf,bertscore

    # Full benchmark with latency
    python scripts/run_evaluation.py --langs all --metrics all --run-latency

    # List available languages
    python scripts/run_evaluation.py --list-langs

    # Generate HTML report
    python scripts/run_evaluation.py --langs hindi --output report.html
"""

import argparse
import json
import sys
import os
import logging

# Add parent to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('run_evaluation')


def parse_args():
    parser = argparse.ArgumentParser(
        description='IndicAI Evaluation Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_evaluation.py --langs hindi
  python scripts/run_evaluation.py --langs hindi,bengali --metrics bleu,chrf
  python scripts/run_evaluation.py --langs all --metrics all --run-latency
  python scripts/run_evaluation.py --list-langs
  python scripts/run_evaluation.py --langs hindi --output report.json
        """,
    )
    parser.add_argument(
        '--langs',
        type=str,
        default='hindi',
        help='Comma-separated target languages or "all" (default: hindi)',
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default='bleu,chrf,bertscore',
        help='Comma-separated metrics or "all" (default: bleu,chrf,bertscore)',
    )
    parser.add_argument(
        '--run-latency',
        action='store_true',
        help='Also run latency benchmarks (takes longer)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (JSON format). Prints to stdout if not set.',
    )
    parser.add_argument(
        '--list-langs',
        action='store_true',
        help='List available benchmark languages and exit',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='Number of warmup runs for latency benchmark (default: 2)',
    )
    return parser.parse_args()


def load_model_fns(target_lang: str):
    """
    Load translation model functions for evaluation.
    Uses ModelRegistry with lazy-loading — only the specified models are loaded.
    """
    try:
        from app.services.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()

        def make_translate_fn(model_name):
            def translate_fn(text, source_lang='english', target_lang=target_lang):
                result = registry.translate(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    model=model_name,
                )
                return result.to_dict()
            return translate_fn

        model_fns = {}
        # Only load NLLB by default (avoid downloading M2M100/IndicTrans2)
        nllb = registry.get_model('nllb-200-distilled-600M')
        if nllb and nllb.available:
            model_fns['nllb-200-distilled-600M'] = make_translate_fn('nllb-200-distilled-600M')

        # Optionally try loading other models if they're already cached
        for name in ['indictrans2', 'm2m100_418M']:
            model = registry.get_model(name)
            if model and model.available:
                model_fns[name] = make_translate_fn(name)

        if not model_fns:
            logger.error("No translation models could be loaded")
            return {}

        return model_fns
    except Exception as e:
        logger.error(f"Failed to load translation models: {e}")
        return {}


def main():
    args = parse_args()

    # Import datasets
    try:
        from app.evaluation.datasets import (
            get_benchmark_dataset, REFERENCES, LANGUAGE_META,
        )
    except ImportError as e:
        logger.error(f"Failed to import evaluation datasets: {e}")
        logger.error("Make sure you're running from the project root.")
        sys.exit(1)

    # List languages mode
    if args.list_langs:
        print("\n=== Available Benchmark Languages ===\n")
        for code, meta in LANGUAGE_META.items():
            print(f"  {code:<12} {meta['name']:<12} {meta['native']}")
        print(f"\n  Total: {len(LANGUAGE_META)} languages\n")
        return

    # Parse languages
    if args.langs.lower() == 'all':
        target_langs = list(REFERENCES.keys())
    else:
        target_langs = [l.strip().lower() for l in args.langs.split(',')]

    # Validate languages
    available_langs = set(REFERENCES.keys())
    invalid_langs = [l for l in target_langs if l not in available_langs]
    if invalid_langs:
        logger.error(f"Invalid languages: {invalid_langs}")
        logger.info(f"Available languages: {list(available_langs)}")
        sys.exit(1)

    # Parse metrics
    if args.metrics.lower() == 'all':
        metrics = None  # Evaluator will use all available
    else:
        metrics = [m.strip().lower() for m in args.metrics.split(',')]

    logger.info(f"Languages: {target_langs}")
    logger.info(f"Metrics: {metrics or 'all available'}")

    # Get benchmark dataset
    dataset = get_benchmark_dataset(target_langs=target_langs)
    sources = dataset['sources']
    references = dataset['references']

    logger.info(f"Loaded {len(sources)} test sentences")
    logger.info(f"Evaluating across {len(target_langs)} languages...")

    # Run benchmark for each language
    from app.evaluation.benchmark import BenchmarkRunner
    runner = BenchmarkRunner(metrics=metrics)

    all_results = {}
    for lang in target_langs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: English → {lang}")
        logger.info(f"{'='*60}")

        model_fns = load_model_fns(lang)
        if not model_fns:
            logger.warning(f"No models available for {lang}, skipping...")
            continue

        result = runner.run(
            model_fns=model_fns,
            sources=sources,
            references=references,
            source_lang='english',
            target_lang=lang,
            run_latency=args.run_latency,
            warmup_runs=args.warmup,
        )
        all_results[lang] = result

        # Print summary
        if 'models' in result:
            for model_name, model_result in result['models'].items():
                summary = model_result.get('summary', {})
                print(f"\n  [{model_name}]")
                for metric, score in summary.items():
                    if score is not None:
                        print(f"    {metric:<14}: {score}")
                if 'latency' in model_result and 'mean_ms' in model_result['latency']:
                    print(f"    {'latency':<14}: {model_result['latency']['mean_ms']} ms")
                if 'latency' in model_result and 'throughput_sentences_per_sec' in model_result['latency']:
                    print(f"    {'throughput':<14}: {model_result['latency']['throughput_sentences_per_sec']} sentences/sec")

    # Print overall comparison
    print(f"\n\n{'='*60}")
    print("OVERALL COMPARISON")
    print(f"{'='*60}")

    for lang, result in all_results.items():
        if 'models' in result:
            for model_name, model_result in result['models'].items():
                summary = model_result.get('summary', {})
                scores = ' | '.join(
                    f"{k}={v}" for k, v in summary.items() if v is not None
                )
                print(f"  en->{lang:<8} [{model_name:<20}] {scores}")

    # Save output if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\n[OK] Evaluation complete!\n")


if __name__ == '__main__':
    main()
