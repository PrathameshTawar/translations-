"""
RAG service with graceful fallbacks.
- Works without FAISS index (returns empty terms)
- Works without fasttext (skips language detection)
- Works without Groq LLM (skips generative Q&A)
- Optional cross-encoder reranker (skipped when the model is unavailable)
- Singleton pattern to avoid reloading models on every request
"""
import os
import json
import logging

from app.services.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

_instance = None


class IndicRAGPipeline:
    """
    Terminology-based RAG pipeline for context-aware translation.
    Loads terminology JSONs from data/terminology/ and optionally
    uses a FAISS vector index for semantic search.
    """

    @classmethod
    def get_instance(cls):
        global _instance
        if _instance is None:
            _instance = cls()
        return _instance

    def __init__(self):
        self.encoder = None
        self.indices = {}
        self.mappings = {}
        self.terminology_db = {}  # lang -> list of term dicts (fallback keyword search)
        self._reranker = None     # lazy CrossEncoderReranker (created on first use)

        self._load_terminology_db()
        self._try_load_encoder()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_terminology_db(self):
        """Load terminology JSONs from data/terminology/ for keyword fallback."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'terminology')
        data_dir = os.path.abspath(data_dir)
        if not os.path.exists(data_dir):
            logger.warning(f"Terminology data directory not found: {data_dir}")
            return

        for fname in os.listdir(data_dir):
            if not fname.endswith('.json'):
                continue
            lang = fname.replace('_terms.json', '')
            fpath = os.path.join(data_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Accept both dict-of-dicts and list-of-dicts
                if isinstance(data, dict):
                    self.terminology_db[lang] = list(data.values())
                elif isinstance(data, list):
                    self.terminology_db[lang] = data
                logger.info(f"Loaded {len(self.terminology_db[lang])} terms for lang={lang}")
            except Exception as e:
                logger.error(f"Failed to load terminology file {fname}: {e}")

    def _try_load_encoder(self):
        """Try to load sentence-transformer encoder. Non-fatal on failure."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"Loaded encoder: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load encoder (RAG will use keyword search): {e}")
            self.encoder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_context(self, query: str, source_lang: str = 'english',
                         target_lang: str = None, top_k: int = 5,
                         use_reranker: bool = False) -> dict:
        """
        Retrieve relevant terminology for a query.

        The terminology glossaries are English->Indic per language
        (e.g. data/terminology/hindi_terms.json maps an English term to its
        Hindi translation), so the glossary is selected by *target_lang*:
        translating into Hindi should retrieve the Hindi glossary.

        Args:
            query: text to find terminology for.
            source_lang: language of the query text.
            target_lang: language being translated into (selects the
                         glossary). Falls back to source_lang when None.
            top_k: max terms to return.
            use_reranker: rescore the retrieved terms with the cross-encoder
                         reranker before building the context. Off by default
                         (the ~430MB model only loads when explicitly
                         requested, and it adds latency per query).

        Returns: { context: str | None, terms: list[dict],
                   reranked: bool, reranker_available: bool }
        """
        # A glossary exists per target language; source_lang has no glossary
        # of its own (English terms live inside each {target}_terms.json).
        lang_key = (target_lang or source_lang or 'english').lower()

        # Try FAISS semantic search first
        terms = self._faiss_search(query, lang_key, top_k)

        # Fall back to keyword search if FAISS unavailable
        if not terms:
            terms = self._keyword_search(query, lang_key, top_k)

        if not terms:
            return {'context': None, 'terms': [], 'reranked': False,
                    'reranker_available': False}

        reranked = False
        if use_reranker:
            terms, reranked = self._rerank_terms(query, terms, top_k)

        context_lines = []
        for t in terms:
            context_lines.append(f"{t.get('term', '')} = {t.get('translation', '')}")
        context = '\n'.join(context_lines)

        reranker = getattr(self, '_reranker', None)
        return {'context': context, 'terms': terms, 'reranked': reranked,
                'reranker_available': reranker.available
                if reranker is not None else False}

    # ------------------------------------------------------------------
    # Reranking (optional cross-encoder second stage)
    # ------------------------------------------------------------------

    def _rerank_terms(self, query: str, terms: list, top_k: int):
        """
        Rescore retrieved terms with the cross-encoder reranker.

        Returns (terms, reranked): the (possibly reordered) term list and
        whether the reranker actually ran. Falls back to the input order
        when the model is unavailable or scoring fails.
        """
        if getattr(self, '_reranker', None) is None:
            self._reranker = CrossEncoderReranker()
        if not self._reranker.available:
            return terms, False
        try:
            out = self._reranker.rerank(query, terms, top_k=top_k,
                                        score_key='term')
            return out.get('reranked', terms), out.get('available', False)
        except Exception as e:
            logger.debug(f'Rerank fallback for lang: {e}')
            return terms, False

    # ------------------------------------------------------------------
    # Search backends
    # ------------------------------------------------------------------

    def _faiss_search(self, query: str, lang: str, top_k: int) -> list:
        if self.encoder is None:
            return []
        try:
            import faiss
            import numpy as np
            if not self._load_faiss_index(lang):
                return []
            emb = self.encoder.encode([query], convert_to_numpy=True)
            dists, idxs = self.indices[lang].search(emb, top_k)
            results = []
            mapping = self.mappings.get(lang, [])
            for idx in idxs[0]:
                if idx != -1 and idx < len(mapping):
                    results.append(mapping[idx])
            return results
        except Exception as e:
            logger.debug(f"FAISS search failed for lang={lang}: {e}")
            return []

    def _load_faiss_index(self, lang: str) -> bool:
        if lang in self.indices:
            return True
        try:
            import faiss
            index_path = f'{lang}_index.faiss'
            mapping_path = f'{lang}_mapping.json'
            if os.path.exists(index_path) and os.path.exists(mapping_path):
                self.indices[lang] = faiss.read_index(index_path)
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.mappings[lang] = json.load(f)
                return True
        except Exception as e:
            logger.debug(f"Could not load FAISS index for {lang}: {e}")
        return False

    def _keyword_search(self, query: str, lang: str, top_k: int) -> list:
        """Simple keyword overlap search against terminology DB."""
        terms = self.terminology_db.get(lang, [])
        if not terms:
            # Try 'english' as fallback
            terms = self.terminology_db.get('english', [])

        query_words = set(query.lower().split())
        scored = []
        for term in terms:
            term_text = (term.get('term') or '').lower()
            term_words = set(term_text.split())
            overlap = len(query_words & term_words)
            if overlap > 0:
                scored.append((overlap, term))

        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:top_k]]
