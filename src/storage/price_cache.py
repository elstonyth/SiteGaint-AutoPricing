"""
Price caching system to reduce Pokedata API calls.

Provides TTL-based caching for:
- Product prices (default: 2 hours)
- Product metadata (default: 24 hours)
- Search results (default: 15 minutes)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CachedPrice:
    """Cached price entry with metadata."""
    product_id: str
    price_usd: float
    source: str
    raw_prices: Dict[str, float]
    cached_at: float  # Unix timestamp
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.time() > (self.cached_at + self.ttl_seconds)
    
    def time_remaining(self) -> int:
        """Get seconds until expiration."""
        remaining = (self.cached_at + self.ttl_seconds) - time.time()
        return max(0, int(remaining))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedPrice":
        """Create from dictionary."""
        return cls(**data)


class PriceCache:
    """
    File-based price cache with TTL support.
    
    Stores cached prices in a JSON file for persistence across restarts.
    Supports configurable TTL for different cache types.
    """
    
    # Default TTL values in seconds
    DEFAULT_PRICE_TTL = 2 * 60 * 60  # 2 hours
    DEFAULT_PRODUCT_TTL = 24 * 60 * 60  # 24 hours
    DEFAULT_SEARCH_TTL = 15 * 60  # 15 minutes
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        price_ttl: int = None,
        product_ttl: int = None,
        search_ttl: int = None,
    ):
        """
        Initialize the price cache.
        
        Args:
            cache_dir: Directory to store cache files.
            price_ttl: TTL for price entries in seconds.
            product_ttl: TTL for product metadata in seconds.
            search_ttl: TTL for search results in seconds.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.price_ttl = price_ttl or self.DEFAULT_PRICE_TTL
        self.product_ttl = product_ttl or self.DEFAULT_PRODUCT_TTL
        self.search_ttl = search_ttl or self.DEFAULT_SEARCH_TTL
        
        self._price_cache_file = self.cache_dir / "price_cache.json"
        self._product_cache_file = self.cache_dir / "product_cache.json"
        self._search_cache_file = self.cache_dir / "search_cache.json"
        
        # In-memory cache (loaded from file)
        self._price_cache: Dict[str, CachedPrice] = {}
        self._product_cache: Dict[str, Dict[str, Any]] = {}
        self._search_cache: Dict[str, Dict[str, Any]] = {}
        
        # Stats tracking
        self._hits = 0
        self._misses = 0
        self._api_calls_saved = 0
        
        # Load existing cache from disk
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        # Load price cache
        if self._price_cache_file.exists():
            try:
                with open(self._price_cache_file, "r") as f:
                    data = json.load(f)
                    for pid, entry in data.items():
                        cached = CachedPrice.from_dict(entry)
                        if not cached.is_expired():
                            self._price_cache[pid] = cached
                logger.info(f"Loaded {len(self._price_cache)} cached prices")
            except Exception as e:
                logger.warning(f"Failed to load price cache: {e}")
        
        # Load product cache
        if self._product_cache_file.exists():
            try:
                with open(self._product_cache_file, "r") as f:
                    data = json.load(f)
                    now = time.time()
                    for pid, entry in data.items():
                        if now < entry.get("expires_at", 0):
                            self._product_cache[pid] = entry
                logger.info(f"Loaded {len(self._product_cache)} cached products")
            except Exception as e:
                logger.warning(f"Failed to load product cache: {e}")
        
        # Load search cache
        if self._search_cache_file.exists():
            try:
                with open(self._search_cache_file, "r") as f:
                    data = json.load(f)
                    now = time.time()
                    for key, entry in data.items():
                        if now < entry.get("expires_at", 0):
                            self._search_cache[key] = entry
                logger.info(f"Loaded {len(self._search_cache)} cached searches")
            except Exception as e:
                logger.warning(f"Failed to load search cache: {e}")
    
    def _save_price_cache(self) -> None:
        """Save price cache to disk."""
        try:
            data = {pid: cp.to_dict() for pid, cp in self._price_cache.items()}
            with open(self._price_cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save price cache: {e}")
    
    def _save_product_cache(self) -> None:
        """Save product cache to disk."""
        try:
            with open(self._product_cache_file, "w") as f:
                json.dump(self._product_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save product cache: {e}")
    
    def _save_search_cache(self) -> None:
        """Save search cache to disk."""
        try:
            with open(self._search_cache_file, "w") as f:
                json.dump(self._search_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save search cache: {e}")
    
    # =========================================================================
    # Price Cache Methods
    # =========================================================================
    
    def get_price(self, product_id: str) -> Optional[CachedPrice]:
        """
        Get cached price for a product.
        
        Args:
            product_id: Pokedata product ID.
            
        Returns:
            CachedPrice if found and not expired, None otherwise.
        """
        cached = self._price_cache.get(str(product_id))
        
        if cached is None:
            self._misses += 1
            return None
        
        if cached.is_expired():
            # Remove expired entry
            del self._price_cache[str(product_id)]
            self._misses += 1
            return None
        
        self._hits += 1
        self._api_calls_saved += 1
        return cached
    
    def set_price(
        self,
        product_id: str,
        price_usd: float,
        source: str = None,
        raw_prices: Dict[str, float] = None,
        ttl: int = None,
    ) -> None:
        """
        Cache a price entry.
        
        Args:
            product_id: Pokedata product ID.
            price_usd: Price in USD.
            source: Price source (e.g., "Pokedata Raw").
            raw_prices: Raw price data from all sources.
            ttl: Optional custom TTL in seconds.
        """
        entry = CachedPrice(
            product_id=str(product_id),
            price_usd=price_usd,
            source=source or "unknown",
            raw_prices=raw_prices or {},
            cached_at=time.time(),
            ttl_seconds=ttl or self.price_ttl,
        )
        self._price_cache[str(product_id)] = entry
        self._save_price_cache()
    
    def get_prices_batch(
        self,
        product_ids: List[str],
    ) -> Tuple[Dict[str, CachedPrice], List[str]]:
        """
        Get cached prices for multiple products.
        
        Args:
            product_ids: List of Pokedata product IDs.
            
        Returns:
            Tuple of (cached_prices dict, list of missing IDs that need API call).
        """
        cached = {}
        missing = []
        
        for pid in product_ids:
            price = self.get_price(pid)
            if price:
                cached[pid] = price
            else:
                missing.append(pid)
        
        return cached, missing
    
    # =========================================================================
    # Product Cache Methods
    # =========================================================================
    
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get cached product metadata."""
        entry = self._product_cache.get(str(product_id))
        
        if entry is None:
            return None
        
        if time.time() > entry.get("expires_at", 0):
            del self._product_cache[str(product_id)]
            return None
        
        self._hits += 1
        self._api_calls_saved += 1
        return entry.get("data")
    
    def set_product(
        self,
        product_id: str,
        data: Dict[str, Any],
        ttl: int = None,
    ) -> None:
        """Cache product metadata."""
        self._product_cache[str(product_id)] = {
            "data": data,
            "cached_at": time.time(),
            "expires_at": time.time() + (ttl or self.product_ttl),
        }
        self._save_product_cache()
    
    # =========================================================================
    # Search Cache Methods
    # =========================================================================
    
    def get_search(self, query: str, language: str = None) -> Optional[List[Dict]]:
        """Get cached search results."""
        cache_key = f"{query}:{language or 'all'}"
        entry = self._search_cache.get(cache_key)
        
        if entry is None:
            return None
        
        if time.time() > entry.get("expires_at", 0):
            del self._search_cache[cache_key]
            return None
        
        self._hits += 1
        self._api_calls_saved += 1
        return entry.get("results")
    
    def set_search(
        self,
        query: str,
        results: List[Dict],
        language: str = None,
        ttl: int = None,
    ) -> None:
        """Cache search results."""
        cache_key = f"{query}:{language or 'all'}"
        self._search_cache[cache_key] = {
            "results": results,
            "cached_at": time.time(),
            "expires_at": time.time() + (ttl or self.search_ttl),
        }
        self._save_search_cache()
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def clear_all(self) -> Dict[str, int]:
        """
        Clear all caches.
        
        Returns:
            Dict with counts of cleared entries.
        """
        counts = {
            "prices": len(self._price_cache),
            "products": len(self._product_cache),
            "searches": len(self._search_cache),
        }
        
        self._price_cache.clear()
        self._product_cache.clear()
        self._search_cache.clear()
        
        self._save_price_cache()
        self._save_product_cache()
        self._save_search_cache()
        
        logger.info(f"Cleared all caches: {counts}")
        return counts
    
    def clear_prices(self) -> int:
        """Clear price cache only."""
        count = len(self._price_cache)
        self._price_cache.clear()
        self._save_price_cache()
        return count
    
    def clear_expired(self) -> Dict[str, int]:
        """Remove all expired entries from caches."""
        now = time.time()
        removed = {"prices": 0, "products": 0, "searches": 0}
        
        # Clean price cache
        expired_prices = [
            pid for pid, cp in self._price_cache.items() if cp.is_expired()
        ]
        for pid in expired_prices:
            del self._price_cache[pid]
            removed["prices"] += 1
        
        # Clean product cache
        expired_products = [
            pid for pid, entry in self._product_cache.items()
            if now > entry.get("expires_at", 0)
        ]
        for pid in expired_products:
            del self._product_cache[pid]
            removed["products"] += 1
        
        # Clean search cache
        expired_searches = [
            key for key, entry in self._search_cache.items()
            if now > entry.get("expires_at", 0)
        ]
        for key in expired_searches:
            del self._search_cache[key]
            removed["searches"] += 1
        
        if any(removed.values()):
            self._save_price_cache()
            self._save_product_cache()
            self._save_search_cache()
            logger.info(f"Cleared expired entries: {removed}")
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "prices_cached": len(self._price_cache),
            "products_cached": len(self._product_cache),
            "searches_cached": len(self._search_cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
            "api_calls_saved": self._api_calls_saved,
            "price_ttl_hours": self.price_ttl / 3600,
            "product_ttl_hours": self.product_ttl / 3600,
            "search_ttl_minutes": self.search_ttl / 60,
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information for UI display."""
        stats = self.get_stats()
        
        # Calculate cache age distribution
        now = time.time()
        price_ages = []
        for cp in self._price_cache.values():
            age_minutes = (now - cp.cached_at) / 60
            price_ages.append(age_minutes)
        
        avg_age = sum(price_ages) / len(price_ages) if price_ages else 0
        
        return {
            **stats,
            "avg_price_age_minutes": round(avg_age, 1),
            "oldest_price_minutes": round(max(price_ages), 1) if price_ages else 0,
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance (lazy initialization)
_cache_instance: Optional[PriceCache] = None


def get_cache(
    cache_dir: str = "data/cache",
    price_ttl: int = None,
    **kwargs,
) -> PriceCache:
    """
    Get or create the global cache instance.
    
    Args:
        cache_dir: Directory for cache files.
        price_ttl: Optional TTL override for prices.
        
    Returns:
        PriceCache instance.
    """
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = PriceCache(
            cache_dir=cache_dir,
            price_ttl=price_ttl,
            **kwargs,
        )
    
    return _cache_instance


def clear_cache() -> Dict[str, int]:
    """Clear the global cache."""
    cache = get_cache()
    return cache.clear_all()
