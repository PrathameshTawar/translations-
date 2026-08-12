"""
Build FAISS vector index from terminology JSON files.
Run once before starting the server for semantic search support.

Usage:
    cd translator-backend
    python scripts/build_faiss_index.py
"""
import os
import json
import sys

def build_index(lang: str, terms: list, encoder, output_dir: str = '.'):
    try:
        import faiss
        import numpy as np
    except ImportError:
        print("faiss-cpu not installed. Run: pip install faiss-cpu")
        return

    texts = [f"{t.get('term','')} {t.get('context','')}" for t in terms]
    print(f"  Encoding {len(texts)} terms for '{lang}'…")
    embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    
    index_path = os.path.join(output_dir, f'{lang}_index.faiss')
    mapping_path = os.path.join(output_dir, f'{lang}_mapping.json')
    
    faiss.write_index(index, index_path)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    
    print(f"  [OK] Saved {index_path} and {mapping_path}")


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed.")
        sys.exit(1)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'terminology')
    data_dir = os.path.abspath(data_dir)

    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

    print(f"Loading encoder: {model_name}…")
    encoder = SentenceTransformer(model_name)

    for fname in os.listdir(data_dir):
        if not fname.endswith('_terms.json'):
            continue
        lang = fname.replace('_terms.json', '')
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        terms = list(data.values()) if isinstance(data, dict) else data
        print(f"\nBuilding index for: {lang} ({len(terms)} terms)")
        build_index(lang, terms, encoder, output_dir=os.path.dirname(__file__) + '/..')

    print("\nAll indexes built successfully!")


if __name__ == '__main__':
    main()
