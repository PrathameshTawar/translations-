"""
Pre-build the per-language terminology FAISS indexes.

Optional: the RAG pipeline builds a missing index on demand. Pre-building is
useful for large glossaries or read-only deployments.

Usage (from backend/):
    python scripts/build_faiss_index.py [--index-dir data/indexes]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.term_index import (  # noqa: E402
    INDEX_DIR, TERMINOLOGY_DIR, build_index, load_terms, save_index,
)

ENCODER = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', default=INDEX_DIR)
    parser.add_argument('--encoder', default=ENCODER)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    print(f'Loading encoder: {args.encoder}')
    encoder = SentenceTransformer(args.encoder)

    for fname in sorted(os.listdir(TERMINOLOGY_DIR)):
        if not fname.endswith('_terms.json'):
            continue
        lang = fname[:-len('_terms.json')]
        terms = load_terms(lang)
        index = build_index(terms, encoder)
        index_path, _ = save_index(lang, index, terms, args.index_dir)
        print(f'  [OK] {lang}: {len(terms)} terms -> {index_path}')


if __name__ == '__main__':
    main()
