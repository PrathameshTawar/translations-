"""
Terminology FAISS index — build, persist and load per-language indexes.

The indexes are *derived* artifacts: they are rebuilt from
``data/terminology/{lang}_terms.json`` whenever they are missing, so they are
never committed to git. ``scripts/build_faiss_index.py`` pre-builds them for
large glossaries; the RAG pipeline and the faiss-service fall back to an
on-demand build otherwise.

Index location: ``RAG_INDEX_DIR`` env var, default ``backend/data/indexes``.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TERMINOLOGY_DIR = os.getenv(
    'RAG_TERMINOLOGY_DIR', os.path.join(BACKEND_ROOT, 'data', 'terminology'))
INDEX_DIR = os.getenv('RAG_INDEX_DIR', os.path.join(BACKEND_ROOT, 'data', 'indexes'))


def index_paths(lang: str, index_dir: str = None):
    """Return (index_path, mapping_path) for a language."""
    index_dir = index_dir or INDEX_DIR
    return (os.path.join(index_dir, f'{lang}_index.faiss'),
            os.path.join(index_dir, f'{lang}_mapping.json'))


def load_terms(lang: str, terminology_dir: str = None) -> list:
    """Load a glossary file (dict-of-dicts or list-of-dicts) as a list."""
    path = os.path.join(terminology_dir or TERMINOLOGY_DIR, f'{lang}_terms.json')
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return list(data.values()) if isinstance(data, dict) else list(data)


def term_text(term: dict) -> str:
    """Text that is embedded for a glossary entry."""
    return f"{term.get('term', '')} {term.get('context', '')}".strip()


def build_index(terms: list, encoder):
    """Encode glossary entries and return an in-memory FAISS index."""
    import faiss

    embeddings = encoder.encode([term_text(t) for t in terms],
                                convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index


def save_index(lang: str, index, terms: list, index_dir: str = None):
    import faiss

    index_dir = index_dir or INDEX_DIR
    os.makedirs(index_dir, exist_ok=True)
    index_path, mapping_path = index_paths(lang, index_dir)
    faiss.write_index(index, index_path)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    return index_path, mapping_path


def load_or_build(lang: str, encoder, index_dir: str = None,
                  terminology_dir: str = None, persist: bool = True):
    """
    Return (index, mapping) for ``lang``, or (None, None) when no glossary.

    Loads a persisted index when present; otherwise builds one from the
    glossary and (best effort) writes it to ``index_dir``.
    """
    import faiss

    index_path, mapping_path = index_paths(lang, index_dir)
    if os.path.exists(index_path) and os.path.exists(mapping_path):
        index = faiss.read_index(index_path)
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        return index, mapping

    terms = load_terms(lang, terminology_dir)
    if not terms or encoder is None:
        return None, None
    index = build_index(terms, encoder)
    if persist:
        try:
            save_index(lang, index, terms, index_dir)
        except OSError as e:  # read-only mount etc. — the in-memory index still works
            logger.info(f'Could not persist FAISS index for {lang}: {e}')
    return index, terms
