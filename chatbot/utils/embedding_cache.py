import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict

from chatbot.utils.embeddings import DenseEmbedding


# ------------------- Embedding Cache Classes -------------------

class PrecomputedEmbedding(BaseModel):
    """
    Represents a precomputed embedding for a query.
    This model is used to store embeddings that have already been computed
    and are ready for use in similarity searches or other operations.

    Attributes:
        query (str): The original query text for which the embedding was computed.
        embedding (DenseEmbedding): The embedding vector corresponding to the query.
    """
    query: str
    embedding: DenseEmbedding

    # Pydantic model configuration
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class EmbeddingCache:
    """
    A simple, efficient local file-based cache for query embeddings.
    
    Features:
    - LRU eviction policy
    - Thread-safe operations
    - Automatic periodic saves
    - Backup and recovery
    
    Args:
        cache_path (str): Path to the cache file. Defaults to "query_embedding_cache.json".
        max_cache_size (int): Maximum number of embeddings to cache. Defaults to 1000.
        save_interval (int): Save after every N cache updates. Defaults to 10.
        auto_save_interval (int): Auto-save interval in seconds. Defaults to 300 (5 minutes).
    
    Example:
        >>> cache = EmbeddingCache(
        ...     cache_path="embeddings.json",
        ...     max_cache_size=500,
        ...     save_interval=5
        ... )
        >>> 
        >>> # Store embedding
        >>> cache.put("hello world", [0.1, 0.2, 0.3])
        >>> 
        >>> # Retrieve embedding
        >>> embedding = cache.get("hello world")
        >>> print(embedding)  # [0.1, 0.2, 0.3]
        >>> 
        >>> # Get statistics
        >>> stats = cache.get_stats()
        >>> print(f"Cache size: {stats['cache_size']}")
        >>> 
        >>> # Cleanup on shutdown
        >>> cache.cleanup()
    """
    
    def __init__(
        self,
        cache_path: str = "query_embedding_cache.json",
        max_cache_size: int = 1000,
        save_interval: int = 10,
        auto_save_interval: int = 300
    ):
        self.cache_path = Path(cache_path)
        self.max_cache_size = max_cache_size
        self.save_interval = save_interval
        self.auto_save_interval = auto_save_interval
        
        # Cache storage: hash -> (embedding, metadata)
        self._cache: Dict[str, DenseEmbedding] = {}
        self._metadata: Dict[str, Dict] = {}
        
        # Counters and locks
        self._updates_count = 0
        self._last_save_time = time.time()
        self._lock = threading.Lock()
        
        # Initialize cache
        self._load_cache()
        self._start_auto_save_worker()

    def get(self, query_text: str) -> Optional[DenseEmbedding]:
        """
        Retrieve cached embedding for a query.
        
        Args:
            query_text (str): The query text to look up.
            
        Returns:
            Optional[DenseEmbedding]: The cached embedding or None if not found.
        """
        query_hash = self._get_hash(query_text)
        
        with self._lock:
            if query_hash not in self._cache:
                return None
            
            # Update access metadata for LRU
            current_time = time.time()
            metadata = self._metadata[query_hash]
            metadata["last_accessed"] = current_time
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            
            return self._cache[query_hash]

    def put(self, query_text: str, embedding: DenseEmbedding) -> None:
        """
        Store an embedding in the cache.
        
        Args:
            query_text (str): The query text.
            embedding (DenseEmbedding): The embedding vector to cache.
        """
        query_hash = self._get_hash(query_text)
        current_time = time.time()
        
        with self._lock:
            # Evict if cache is full and this is a new entry
            if len(self._cache) >= self.max_cache_size and query_hash not in self._cache:
                self._evict_lru_entries()
            
            # Store embedding and metadata
            self._cache[query_hash] = embedding
            self._metadata[query_hash] = {
                "created": current_time,
                "last_accessed": current_time,
                "access_count": 1,
                "query_length": len(query_text)
            }
            
            self._updates_count += 1
        
        # Check if we should save
        self._maybe_save()

    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict: Statistics including cache size, hit rate, etc.
        """
        with self._lock:
            total_accesses = sum(
                meta.get("access_count", 0) 
                for meta in self._metadata.values()
            )
            
            return {
                "cache_size": len(self._cache),
                "max_cache_size": self.max_cache_size,
                "total_accesses": total_accesses,
                "updates_since_last_save": self._updates_count,
                "last_save_time": datetime.fromtimestamp(self._last_save_time).isoformat(),
                "cache_file": str(self.cache_path)
            }

    def clear(self) -> None:
        """Clear all cached data and save empty cache."""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
            self._updates_count += 1
        
        self._save_cache(force=True)
        logger.info("Cache cleared")

    def cleanup(self) -> None:
        """Save cache and cleanup resources. Call this before shutdown."""
        self._save_cache(force=True)
        logger.info("Cache cleanup completed")

    def _get_hash(self, query_text: str) -> str:
        """Generate MD5 hash for query text."""
        return hashlib.md5(query_text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.cache_path.exists():
            logger.info(f"Cache file not found: {self.cache_path}. Starting with empty cache.")
            return
        
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._cache = data.get("cache", {})
                self._metadata = data.get("metadata", {})
            
            logger.info(f"Loaded {len(self._cache)} cached embeddings from {self.cache_path}")
            
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning(f"Failed to load cache from {self.cache_path}: {e}. Starting fresh.")
            self._cache = {}
            self._metadata = {}

    def _save_cache(self, force: bool = False) -> None:
        """Save cache to disk."""
        if not force and not self._should_save():
            return
        
        try:
            # Create backup if file exists
            backup_path = None
            if self.cache_path.exists():
                backup_path = self.cache_path.with_suffix(".backup")
                self.cache_path.rename(backup_path)
            
            # Save cache data
            cache_data = {
                "cache": self._cache,
                "metadata": self._metadata,
                "saved_at": datetime.now().isoformat(),
                "cache_size": len(self._cache)
            }
            
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            # Update counters and remove backup
            with self._lock:
                self._updates_count = 0
                self._last_save_time = time.time()
            
            if backup_path and backup_path.exists():
                backup_path.unlink()
            
            logger.debug(f"Saved cache with {len(self._cache)} entries to {self.cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            
            # Restore from backup if save failed
            if backup_path and backup_path.exists():
                backup_path.rename(self.cache_path)
                logger.info("Restored cache from backup after save failure")

    def _should_save(self) -> bool:
        """Check if cache should be saved based on conditions."""
        return (
            self._updates_count >= self.save_interval or
            (time.time() - self._last_save_time) >= self.auto_save_interval
        )

    def _maybe_save(self) -> None:
        """Save cache if conditions are met."""
        if self._should_save():
            self._save_cache()

    def _evict_lru_entries(self, evict_count: int = None) -> None:
        """
        Evict least recently used entries.
        
        Args:
            evict_count (int): Number of entries to evict. Defaults to 10% of max size.
        """
        if not evict_count:
            evict_count = max(1, self.max_cache_size // 10)
        
        # Sort by last_accessed (oldest first)
        sorted_items = sorted(
            self._metadata.items(),
            key=lambda x: x[1].get("last_accessed", 0)
        )
        
        # Remove oldest entries
        for i in range(min(evict_count, len(sorted_items))):
            query_hash = sorted_items[i][0]
            self._cache.pop(query_hash, None)
            self._metadata.pop(query_hash, None)
        
        logger.debug(f"Evicted {evict_count} LRU entries from cache")

    def _start_auto_save_worker(self) -> None:
        """Start background thread for periodic auto-save."""
        def auto_save_worker():
            while True:
                time.sleep(self.auto_save_interval)
                if len(self._cache) > 0:
                    self._save_cache()
        
        worker_thread = threading.Thread(
            target=auto_save_worker, 
            daemon=True, 
            name="EmbeddingCacheAutoSave"
        )
        worker_thread.start()
        logger.debug("Started auto-save background worker")

# Example usage:
if __name__ == "__main__":
    cache = EmbeddingCache(
        cache_path="embeddings.json",
        max_cache_size=500,
        save_interval=5
    )
    
    # Store an embedding
    cache.put("hello world", [0.1, 0.2, 0.3])
    
    # Retrieve the embedding
    embedding = cache.get("hello world")
    print(f"Retrieved embedding: {embedding}")
    
    # Get cache statistics
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Cleanup on shutdown
    cache.cleanup()
