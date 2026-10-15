"""
Shared utility functions for evaluation metrics.
"""


def std(values):
    """Compute standard deviation of a list of numbers."""
    if not values:
        return 0.0
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return variance ** 0.5


def release_torch_memory():
    """
    Collect Python garbage and free any CUDA caching-allocator blocks.

    Used by the neural metric scorers (COMET/BERTScore) when unloading their
    models, so a long matrix run does not accumulate multiple GB of resident
    model weights on CPU machines (--metrics all on 16GB laptops).
    """
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
