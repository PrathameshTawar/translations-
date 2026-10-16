"""
Translation Memory — embedding-based cache for reusing past translations.

Uses sentence-transformers to find semantically similar past translations
and returns cached results instead of re-running the translation model.

This dramatically improves latency for repeated or similar content while
ensuring terminological consistency across translations.
"""

import os
import json
import time
import logging
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np

from app.services.translators.base import TranslationResult

logger = logging.getLogger(__name__)

# Default similarity threshold (0.92 = very high, only cache near-identical matches)
# Lower values (0.85) capture more paraphrases but risk semantic drift
DEFAULT_SIMILARITY_THRESHOLD = 0.92

# Persistence file for cross-restart cache
TM_DATA_FILE = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'translation_memory.json'
)


@dataclass
class TMEntry:
    """A single translation memory entry."""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model: str = 'unknown'
    created_at: float = 0.0
    last_accessed: float = 0.0
    hit_count: int = 1
    original_length: int = 0
    similarity_score: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TMEntry':
        return cls(**d)


class TranslationMemory:
    """
    Embedding-based translation memory with cosine similarity search.

    Singleton — shares the same cache across all requests.
    Thread-safe for concurrent access.

    Usage:
        tm = TranslationMemory.get_instance()
        result = tm.lookup('Hello world', 'english', 'hindi')
        if result:
            return result
        # ... translate ...
        tm.store('Hello world', translated_text, 'english', 'hindi', model='nllb')
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self._encoder = None
        self._encoder_failed = False
        self._entries: List[TMEntry] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, D) matrix
        self._embed_rows: List[Optional[int]] = []  # entry index -> embedding row (or None)
        self._similarity_threshold = similarity_threshold
        self._stats = {
            'total_entries': 0,
            'hits': 0,
            'misses': 0,
            'size_bytes': 0,
            'last_saved': None,
            'last_loaded': None,
        }
        self._load_persistence()

    # ------------------------------------------------------------------
    # Encoder lazy-loading
    # ------------------------------------------------------------------

    @property
    def encoder(self):
        """Lazy-load the sentence-transformer encoder."""
        if self._encoder is None and not self._encoder_failed:
            self._try_load_encoder()
        return self._encoder

    @property
    def encoder_available(self) -> bool:
        return self._encoder is not None or self._try_load_encoder()

    def _try_load_encoder(self) -> bool:
        """Try once to load encoder, return True if successful.
        Caches failure to avoid retrying on every access."""
        if self._encoder_failed:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            return True
        except Exception:
            self._encoder_failed = True
            return False

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        similarity_threshold: Optional[float] = None,
    ) -> Optional[TMEntry]:
        """
        Look up a translation in memory.

        Returns the closest matching entry if similarity exceeds threshold,
        otherwise returns None.

        Uses cosine similarity on sentence-transformer embeddings.
        Also performs exact string match first (fast path).
        """
        if not text or not self._entries:
            return None

        threshold = similarity_threshold or self._similarity_threshold

        # Fast path: exact string + language match
        for entry in self._entries:
            if (entry.source_text == text
                    and entry.source_lang == source_lang
                    and entry.target_lang == target_lang):
                # Update stats
                entry.hit_count += 1
                entry.last_accessed = time.time()
                self._stats['hits'] += 1
                entry.similarity_score = 1.0
                return entry

        # Semantic path: embedding similarity search
        if not self.encoder_available or self._embeddings is None or len(self._entries) == 0:
            self._stats['misses'] += 1
            return None

        try:
            # Embed query
            query_emb = self.encoder.encode([text], convert_to_numpy=True)
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

            # Compute cosine similarity with all entries
            similarities = np.dot(self._embeddings, query_emb.T).flatten()

            # Find best match among same language pairs
            best_idx = -1
            best_sim = 0.0
            for i, entry in enumerate(self._entries):
                row = self._embed_rows[i] if i < len(self._embed_rows) else None
                if row is None:
                    continue  # no embedding for this entry
                if entry.source_lang == source_lang and entry.target_lang == target_lang:
                    sim = similarities[row]
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = i

            if best_idx >= 0 and best_sim >= threshold:
                entry = self._entries[best_idx]
                entry.hit_count += 1
                entry.last_accessed = time.time()
                entry.similarity_score = float(best_sim)
                self._stats['hits'] += 1
                logger.debug(f"TM hit: similarity={best_sim:.4f} threshold={threshold}")
                return entry

            self._stats['misses'] += 1
            return None

        except Exception as e:
            logger.debug(f"TM lookup error: {e}")
            self._stats['misses'] += 1
            return None

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        model: str = 'unknown',
    ) -> None:
        """
        Store a new translation in memory.

        Only stores if the text is non-empty and not already cached
        (by exact match).
        """
        if not text or not translated_text:
            return

        # Skip if exact match already exists
        for entry in self._entries:
            if (entry.source_text == text
                    and entry.source_lang == source_lang
                    and entry.target_lang == target_lang):
                # Update existing entry
                entry.hit_count += 1
                entry.last_accessed = time.time()
                return

        # Create new entry
        entry = TMEntry(
            source_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            model=model,
            created_at=time.time(),
            last_accessed=time.time(),
            original_length=len(text),
        )

        # Compute embedding BEFORE appending so the entry list and the
        # embedding matrix never drift out of alignment. If the encoder
        # fails, the entry is stored without semantic-search support but
        # the exact-match fast path still works.
        emb = None
        if self.encoder_available:
            try:
                emb = self.encoder.encode([text], convert_to_numpy=True)
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            except Exception as e:
                logger.debug(f"TM store embedding error: {e}")
                emb = None

        with self._lock:
            if emb is not None:
                if self._embeddings is None:
                    self._embeddings = emb
                else:
                    self._embeddings = np.vstack([self._embeddings, emb])
                self._embed_rows.append(self._embeddings.shape[0] - 1)
            else:
                self._embed_rows.append(None)
            self._entries.append(entry)
            self._stats['total_entries'] = len(self._entries)
        logger.debug(f"TM stored: {text[:50]}... -> {translated_text[:50]}...")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Get translation memory statistics."""
        total = self._stats['hits'] + self._stats['misses']
        return {
            'total_entries': len(self._entries),
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': round(self._stats['hits'] / max(total, 1), 4),
            'total_lookups': total,
            'similarity_threshold': self._similarity_threshold,
            'encoder_loaded': self.encoder_available,
            'embeddings_shape': list(self._embeddings.shape) if self._embeddings is not None else None,
            'size_bytes': self._stats['size_bytes'],
            'last_saved': self._stats['last_saved'],
            'language_pairs': self._get_language_pairs(),
            'top_models': self._get_top_models(5),
        }

    def _get_language_pairs(self) -> Dict[str, int]:
        """Get count of entries per language pair."""
        pairs = {}
        for entry in self._entries:
            key = f"{entry.source_lang}->{entry.target_lang}"
            pairs[key] = pairs.get(key, 0) + 1
        return pairs

    def _get_top_models(self, n: int = 5) -> List[Dict]:
        """Get most used translation models."""
        models = {}
        for entry in self._entries:
            models[entry.model] = models.get(entry.model, 0) + 1
        sorted_models = sorted(models.items(), key=lambda x: -x[1])
        return [{'model': m, 'count': c} for m, c in sorted_models[:n]]

    def clear(self) -> int:
        """Clear all entries. Returns number of entries removed."""
        with self._lock:
            count = len(self._entries)
            self._entries = []
            self._embeddings = None
            self._embed_rows = []
            self._stats['total_entries'] = 0
        logger.info(f"TM cleared: {count} entries removed")
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_persist_path(self) -> str:
        """Get the path for persisting TM data."""
        path = os.environ.get('TM_DATA_FILE', TM_DATA_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def save(self, path: Optional[str] = None) -> bool:
        """
        Save translation memory to disk as JSON.

        Note: Embeddings are NOT saved to keep the file small.
        They'll be recomputed on load.
        """
        path = path or self._get_persist_path()
        try:
            data = {
                'version': 1,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'entries': [e.to_dict() for e in self._entries],
                'stats': self._stats,
                'similarity_threshold': self._similarity_threshold,
            }
            # Remove embeddings from serialized entries
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            self._stats['last_saved'] = datetime.now(timezone.utc).isoformat()
            size = os.path.getsize(path)
            self._stats['size_bytes'] = size
            logger.info(f"TM saved: {len(self._entries)} entries, {size:,} bytes")
            return True
        except Exception as e:
            logger.error(f"Failed to save TM: {e}")
            return False

    def _load_persistence(self, path: Optional[str] = None):
        """Load translation memory from disk."""
        path = path or self._get_persist_path()
        if not os.path.exists(path):
            logger.info(f"No TM data file found at {path}, starting fresh")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entries_data = data.get('entries', [])
            self._entries = [TMEntry.from_dict(e) for e in entries_data]
            self._similarity_threshold = data.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)
            self._stats['last_loaded'] = datetime.now(timezone.utc).isoformat()
            self._stats['total_entries'] = len(self._entries)
            self._embed_rows = [None] * len(self._entries)

            logger.info(f"TM loaded: {len(self._entries)} entries from {path}")

            # Recompute embeddings (embeddings aren't stored in JSON)
            if self._try_load_encoder() and self._entries:
                try:
                    texts = [e.source_text for e in self._entries]
                    self._embeddings = self._encoder.encode(texts, convert_to_numpy=True)
                    self._embeddings = self._embeddings / np.linalg.norm(self._embeddings, axis=1, keepdims=True)
                    self._embed_rows = list(range(self._embeddings.shape[0]))
                    logger.info(f"TM embeddings recomputed: {self._embeddings.shape}")
                except Exception as e:
                    logger.warning(f"Could not recompute TM embeddings: {e}")

        except Exception as e:
            logger.error(f"Failed to load TM from {path}: {e}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def warmup(self, sentences: List[str], source_lang: str = 'english', target_lang: str = 'hindi'):
        """
        Pre-populate TM with known good translations.
        Useful for loading high-quality translations at startup.
        """
        # This is a placeholder — actual warmup data would come from
        # professionally translated sentence pairs
        logger.info(f"TM warmup: no preloaded data available")

    @property
    def total_entries(self) -> int:
        """Total number of entries in the translation memory."""
        return len(self._entries)

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get most recent translation memory entries."""
        sorted_entries = sorted(self._entries, key=lambda e: -e.created_at)
        return [e.to_dict() for e in sorted_entries[:n]]

    def get_most_accessed(self, n: int = 10) -> List[Dict]:
        """Get most frequently accessed entries."""
        sorted_entries = sorted(self._entries, key=lambda e: -e.hit_count)
        return [e.to_dict() for e in sorted_entries[:n]]

    def __repr__(self):
        return (
            f"TranslationMemory(entries={len(self._entries)}, "
            f"hits={self._stats['hits']}, misses={self._stats['misses']}, "
            f"threshold={self._similarity_threshold})"
        )
