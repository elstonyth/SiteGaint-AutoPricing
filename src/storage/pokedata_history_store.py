"""
Pokedata price history storage module.

Stores and retrieves historical price data from a CSV cache file.
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.pokedata_client.models import PriceHistoryPoint


logger = logging.getLogger(__name__)


# Default path for price history CSV
DEFAULT_HISTORY_PATH = "data/cache/pokedata_price_history.csv"

# CSV columns
COLUMNS = ["pokedata_id", "asset_type", "date", "price_usd"]


class PokedataHistoryStore:
    """
    Manages storage and retrieval of Pokedata price history.
    
    Stores price data in a CSV file with columns:
    pokedata_id, asset_type, date, price_usd
    """
    
    def __init__(self, history_path: Optional[str] = None) -> None:
        """
        Initialize the history store.
        
        Args:
            history_path: Path to the CSV file. Defaults to data/cache/pokedata_price_history.csv
        """
        self.history_path = Path(history_path or DEFAULT_HISTORY_PATH)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """Create the history file with headers if it doesn't exist."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.history_path.exists():
            with open(self.history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
            logger.info(f"Created price history file: {self.history_path}")
    
    def record_price(
        self,
        pokedata_id: str,
        asset_type: str,
        price_usd: float,
        date: Optional[str] = None,
    ) -> None:
        """
        Record a price entry in the history.
        
        If an entry for the same (pokedata_id, asset_type, date) exists,
        it will be updated. Otherwise, a new entry is appended.
        
        Args:
            pokedata_id: Pokedata product ID.
            asset_type: Asset type (e.g., "PRODUCT").
            price_usd: Price in USD.
            date: Date string (YYYY-MM-DD). Defaults to today.
        """
        if not pokedata_id or price_usd is None:
            return
        
        date = date or datetime.now().strftime("%Y-%m-%d")
        
        # Read existing entries
        entries = self._read_all()
        
        # Check for existing entry with same key
        key = (str(pokedata_id), asset_type, date)
        updated = False
        
        for i, entry in enumerate(entries):
            if (entry["pokedata_id"], entry["asset_type"], entry["date"]) == key:
                entries[i]["price_usd"] = str(price_usd)
                updated = True
                break
        
        if not updated:
            entries.append({
                "pokedata_id": str(pokedata_id),
                "asset_type": asset_type,
                "date": date,
                "price_usd": str(price_usd),
            })
        
        # Write back
        self._write_all(entries)
        logger.debug(f"Recorded price for {pokedata_id}: ${price_usd} on {date}")
    
    def get_history(
        self,
        pokedata_id: str,
        asset_type: str = "PRODUCT",
        limit: int = 5,
    ) -> List[PriceHistoryPoint]:
        """
        Get price history for a product.
        
        Args:
            pokedata_id: Pokedata product ID.
            asset_type: Asset type filter.
            limit: Maximum number of entries to return (most recent first).
            
        Returns:
            List of PriceHistoryPoint sorted by date (newest first).
        """
        entries = self._read_all()
        
        # Filter by product and asset type
        filtered = [
            e for e in entries
            if e["pokedata_id"] == str(pokedata_id) and e["asset_type"] == asset_type
        ]
        
        # Sort by date descending
        filtered.sort(key=lambda x: x["date"], reverse=True)
        
        # Convert to PriceHistoryPoint
        result = []
        for entry in filtered[:limit]:
            try:
                result.append(PriceHistoryPoint(
                    date=entry["date"],
                    price_usd=float(entry["price_usd"]),
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid history entry: {entry}, error: {e}")
        
        return result
    
    def _read_all(self) -> List[dict]:
        """Read all entries from the CSV file."""
        entries = []
        
        if not self.history_path.exists():
            return entries
        
        try:
            with open(self.history_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                entries = list(reader)
        except Exception as e:
            logger.error(f"Failed to read history file: {e}")
        
        return entries
    
    def _write_all(self, entries: List[dict]) -> None:
        """Write all entries to the CSV file."""
        try:
            with open(self.history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(entries)
        except Exception as e:
            logger.error(f"Failed to write history file: {e}")


# Module-level singleton instance
_store: Optional[PokedataHistoryStore] = None


def _get_store() -> PokedataHistoryStore:
    """Get or create the singleton store instance."""
    global _store
    if _store is None:
        _store = PokedataHistoryStore()
    return _store


def record_price(
    pokedata_id: str,
    asset_type: str,
    price_usd: float,
    date: Optional[str] = None,
) -> None:
    """
    Record a price entry in the history.
    
    Convenience function using the module-level store.
    
    Args:
        pokedata_id: Pokedata product ID.
        asset_type: Asset type (e.g., "PRODUCT").
        price_usd: Price in USD.
        date: Date string (YYYY-MM-DD). Defaults to today.
    """
    _get_store().record_price(pokedata_id, asset_type, price_usd, date)


def get_history(
    pokedata_id: str,
    asset_type: str = "PRODUCT",
    limit: int = 5,
) -> List[PriceHistoryPoint]:
    """
    Get price history for a product.
    
    Convenience function using the module-level store.
    
    Args:
        pokedata_id: Pokedata product ID.
        asset_type: Asset type filter.
        limit: Maximum number of entries to return.
        
    Returns:
        List of PriceHistoryPoint sorted by date (newest first).
    """
    return _get_store().get_history(pokedata_id, asset_type, limit)
