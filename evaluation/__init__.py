"""
IndicAI Evaluation Framework

Provides standardized metrics and benchmarks for evaluating
translation quality across Indic languages.

Metrics:
    - BLEU        (lexical overlap via sacrebleu)
    - chrF++      (character n-gram F-score via sacrebleu)
    - BERTScore   (semantic similarity via bert-score)
    - COMET       (neural quality estimation, optional)
    - Latency     (inference time, throughput, peak memory)
    - Quality     (our built-in QualityScorer wrapper)

Usage:
    from evaluation.evaluator import Evaluator
    results = Evaluator().evaluate(references, hypotheses, metrics=['bleu', 'chrf', 'bertscore'])
"""

__version__ = '0.1.0'
