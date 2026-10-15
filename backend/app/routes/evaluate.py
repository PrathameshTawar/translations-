"""
Evaluation API routes.

Provides endpoints for running evaluations and viewing results.
Supports both on-demand evaluation and saved historical results.
"""

import time
import logging
import threading
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from app.extensions import db

logger = logging.getLogger(__name__)

evaluate_bp = Blueprint('evaluate', __name__)


# In-memory cache for evaluation results (replace with DB later)
_evaluation_cache = {}
_evaluation_cache_lock = threading.Lock()
_EVALUATION_CACHE_MAX = 50


@evaluate_bp.route('/languages', methods=['GET'])
def get_evaluation_languages():
    """List available evaluation languages."""
    from app.evaluation.datasets import LANGUAGE_META, REFERENCES
    return jsonify({
        'languages': [
            {
                'code': code,
                'name': meta['name'],
                'native': meta['native'],
                'has_references': code in REFERENCES,
                'num_references': len(REFERENCES.get(code, [])),
            }
            for code, meta in LANGUAGE_META.items()
        ],
    })


@evaluate_bp.route('/metrics', methods=['GET'])
def get_available_metrics():
    """List available evaluation metrics and their status."""
    from app.evaluation import metrics as metric_modules

    available = {}
    for name, cls in [
        ('bleu', metric_modules.BLEUScorer),
        ('chrf', metric_modules.ChrFScorer),
        ('bertscore', metric_modules.BERTScoreScorer),
        ('comet', metric_modules.COMETScorer),
        ('quality', metric_modules.QualityScorerWrapper),
        ('latency', metric_modules.LatencyBenchmark),
    ]:
        try:
            instance = cls()
            available[name] = {
                'available': instance.available,
                'description': instance.__doc__.strip().split('.')[0] if instance.__doc__ else '',
            }
        except Exception as e:
            available[name] = {'available': False, 'error': str(e)}

    return jsonify({'metrics': available})


@evaluate_bp.route('/run', methods=['POST'])
def run_evaluation():
    """
    Run evaluation across all available models.

    Body (optional):
    {
        "target_langs": ["hindi", "bengali"],  // default: ["hindi"]
        "metrics": ["bleu", "chrf", "bertscore"],  // default: all available
        "models": ["nllb-200-distilled-600M", "m2m100_418M"],  // default: all available
        "run_latency": false  // default: false
    }
    """
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400
    target_langs = data.get('target_langs', ['hindi'])
    metrics = data.get('metrics')
    requested_models = data.get('models')  # None means all available
    run_latency = data.get('run_latency', False)

    # Validate
    from app.evaluation.datasets import REFERENCES, get_benchmark_dataset
    valid_langs = set(REFERENCES.keys())
    invalid = [l for l in target_langs if l not in valid_langs]
    if invalid:
        return jsonify({
            'error': f'Invalid languages: {invalid}',
            'valid_languages': list(valid_langs),
        }), 400

    # Get dataset
    dataset = get_benchmark_dataset(target_langs=target_langs)
    sources = dataset['sources']
    references = dataset['references']

    # Load model registry and get available models
    from app.services.model_registry import ModelRegistry
    registry = ModelRegistry.get_instance()

    if not registry.model_names:
        return jsonify({'error': 'No translation models available'}), 500

    # Determine which models to use
    if requested_models:
        model_names = [m for m in requested_models if m in registry.model_names]
        if not model_names:
            return jsonify({
                'error': f'None of the requested models are available: {requested_models}',
                'available_models': registry.model_names,
            }), 400
    else:
        model_names = registry.model_names

    # Build model_fns for benchmark runner
    model_fns = {}
    for name in model_names:
        def make_translate_fn(mname):
            def translate_fn(text, source_lang='english', target_lang='hindi'):
                result = registry.translate(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    model=mname,
                )
                return result.to_dict()
            return translate_fn
        model_fns[name] = make_translate_fn(name)

    results = {}
    for lang in target_langs:
        from app.evaluation.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(metrics=metrics)

        eval_result = runner.run(
            model_fns=model_fns,
            sources=sources,
            references=references,
            source_lang='english',
            target_lang=lang,
            run_latency=run_latency,
            warmup_runs=2,
        )

        results[lang] = eval_result

    # Cache the latest results (bounded, thread-safe)
    now = datetime.now(timezone.utc).isoformat()
    eval_id = f'eval_{int(time.time())}'
    entry = {
        'id': eval_id,
        'timestamp': now,
        'target_langs': target_langs,
        'models_used': model_names,
        'results': results,
    }
    with _evaluation_cache_lock:
        _evaluation_cache[eval_id] = entry
        # Cap the cache size to prevent unbounded memory growth
        while len(_evaluation_cache) > _EVALUATION_CACHE_MAX:
            _evaluation_cache.pop(next(iter(_evaluation_cache)))

    return jsonify({
        'evaluation_id': eval_id,
        'timestamp': now,
        'models_used': model_names,
        'results': results,
        'summary': _build_summary(results),
    })


@evaluate_bp.route('/results', methods=['GET'])
def get_evaluation_results():
    """Get historical evaluation results."""
    eval_id = request.args.get('id')
    if eval_id:
        result = _evaluation_cache.get(eval_id)
        if not result:
            return jsonify({'error': 'Evaluation not found'}), 404
        return jsonify(result)

    # Return all cached results (newest first)
    with _evaluation_cache_lock:
        cached = list(_evaluation_cache.items())
    return jsonify({
        'evaluations': [
            {
                'id': eid,
                'timestamp': data['timestamp'],
                'target_langs': data['target_langs'],
            }
            for eid, data in reversed(cached)
        ]
    })


@evaluate_bp.route('/datasets', methods=['GET'])
def get_datasets_info():
    """Get info about available benchmark datasets."""
    from app.evaluation.datasets import get_benchmark_dataset, DOMAIN_LABELS, ENGLISH_SENTENCES

    # Count by domain
    domain_counts = {}
    for d in DOMAIN_LABELS:
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # Languages that have reference translations available
    with _evaluation_cache_lock:
        last_run = next(reversed(list(_evaluation_cache.values())), None)
    langs_with_refs = list(last_run['results'].keys()) if last_run else []

    return jsonify({
        'total_sentences': len(ENGLISH_SENTENCES),
        'domains': domain_counts,
        'languages_with_references': langs_with_refs,
    })


def _build_summary(results: dict) -> dict:
    """Build a concise summary from evaluation results."""
    summary = {}
    for lang, result in results.items():
        if 'models' not in result:
            continue
        for model_name, model_result in result['models'].items():
            s = model_result.get('summary', {})
            summary[f'en→{lang} ({model_name})'] = {
                k: v for k, v in s.items() if v is not None
            }
            if 'latency' in model_result and 'mean_ms' in model_result['latency']:
                summary[f'en→{lang} ({model_name})']['latency_ms'] = model_result['latency']['mean_ms']
    return summary
