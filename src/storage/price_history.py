"""
Price history tracking service.

Logs all price exports to an Excel file for audit and analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Default history file path
HISTORY_FILE = Path("data/output/price_history.xlsx")


class PriceHistoryService:
    """
    Service for tracking price change history.
    
    Logs every export with old/new prices to an Excel file.
    """
    
    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize the price history service.
        
        Args:
            history_file: Path to the history Excel file.
        """
        self.history_file = history_file or HISTORY_FILE
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """Create the history file if it doesn't exist."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.history_file.exists():
            # Create empty history file with columns
            df = pd.DataFrame(columns=[
                'export_date',
                'export_time',
                'sku',
                'isku',
                'product_name',
                'old_price',
                'new_price',
                'change_amount',
                'change_percent',
                'pokedata_id',
                'pokedata_price_usd',
                'fx_rate',
                'source',
            ])
            df.to_excel(self.history_file, index=False, engine='openpyxl')
            logger.info(f"Created price history file: {self.history_file}")
    
    def log_export(
        self,
        exports: List[Dict[str, Any]],
        fx_rate: float = 1.0,
        source: str = "Pokedata",
    ) -> int:
        """
        Log a batch of price exports.
        
        Args:
            exports: List of dicts with keys:
                - sku: Product SKU
                - isku: Internal SKU (optional)
                - product_name: Product name
                - old_price: Previous price
                - new_price: New exported price
                - pokedata_id: Pokedata product ID (optional)
                - pokedata_price_usd: USD price from Pokedata (optional)
            fx_rate: FX rate used for conversion.
            source: Price data source.
            
        Returns:
            Number of records logged.
        """
        if not exports:
            return 0
        
        now = datetime.now()
        export_date = now.strftime("%Y-%m-%d")
        export_time = now.strftime("%H:%M:%S")
        
        # Build records
        records = []
        for item in exports:
            old_price = float(item.get('old_price', 0) or 0)
            new_price = float(item.get('new_price', 0) or 0)
            
            change_amount = new_price - old_price
            change_percent = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
            
            records.append({
                'export_date': export_date,
                'export_time': export_time,
                'sku': item.get('sku', ''),
                'isku': item.get('isku', ''),
                'product_name': item.get('product_name', item.get('name', '')),
                'old_price': old_price,
                'new_price': new_price,
                'change_amount': round(change_amount, 2),
                'change_percent': round(change_percent, 2),
                'pokedata_id': item.get('pokedata_id', ''),
                'pokedata_price_usd': item.get('pokedata_price_usd', ''),
                'fx_rate': fx_rate,
                'source': source,
            })
        
        # Load existing history
        try:
            existing_df = pd.read_excel(self.history_file, engine='openpyxl')
        except Exception:
            existing_df = pd.DataFrame()
        
        # Append new records
        new_df = pd.DataFrame(records)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Save
        combined_df.to_excel(self.history_file, index=False, engine='openpyxl')
        
        logger.info(f"Logged {len(records)} price exports to history")
        return len(records)
    
    def get_history(
        self,
        sku: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Get price history with optional filters.
        
        Args:
            sku: Filter by SKU (partial match).
            start_date: Filter from date (YYYY-MM-DD).
            end_date: Filter to date (YYYY-MM-DD).
            limit: Maximum records to return.
            
        Returns:
            List of history records.
        """
        if not self.history_file.exists():
            return []
        
        try:
            df = pd.read_excel(self.history_file, engine='openpyxl')
        except Exception as e:
            logger.error(f"Failed to read history: {e}")
            return []
        
        if df.empty:
            return []
        
        # Apply filters
        if sku:
            sku_lower = sku.lower()
            df = df[
                df['sku'].astype(str).str.lower().str.contains(sku_lower, na=False) |
                df['product_name'].astype(str).str.lower().str.contains(sku_lower, na=False)
            ]
        
        if start_date:
            df = df[df['export_date'] >= start_date]
        
        if end_date:
            df = df[df['export_date'] <= end_date]
        
        # Sort by date descending (most recent first)
        df = df.sort_values(['export_date', 'export_time'], ascending=[False, False])
        
        # Limit
        df = df.head(limit)
        
        # Convert to list of dicts
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    record[col] = ""
                elif isinstance(val, (int, float)):
                    record[col] = val
                else:
                    record[col] = str(val)
            records.append(record)
        
        return records
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the history.
        
        Returns:
            Dict with total_exports, total_products, date_range, etc.
        """
        if not self.history_file.exists():
            return {
                "total_exports": 0,
                "unique_products": 0,
                "date_range": None,
            }
        
        try:
            df = pd.read_excel(self.history_file, engine='openpyxl')
        except Exception:
            return {
                "total_exports": 0,
                "unique_products": 0,
                "date_range": None,
            }
        
        if df.empty:
            return {
                "total_exports": 0,
                "unique_products": 0,
                "date_range": None,
            }
        
        # Safely get stats
        try:
            unique_products = df['sku'].nunique() if 'sku' in df.columns else 0
        except Exception:
            unique_products = 0
        
        try:
            first_export = str(df['export_date'].min()) if 'export_date' in df.columns else None
            last_export = str(df['export_date'].max()) if 'export_date' in df.columns else None
        except Exception:
            first_export = None
            last_export = None
        
        try:
            total_sessions = df.groupby(['export_date', 'export_time']).ngroups if 'export_date' in df.columns and 'export_time' in df.columns else 0
        except Exception:
            total_sessions = 0
        
        return {
            "total_exports": len(df),
            "unique_products": unique_products,
            "first_export": first_export,
            "last_export": last_export,
            "total_sessions": total_sessions,
        }
    
    def export_history(
        self,
        sku: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Export filtered history as DataFrame (for download).
        
        Args:
            sku: Filter by SKU.
            start_date: Filter from date.
            end_date: Filter to date.
            
        Returns:
            DataFrame with filtered history.
        """
        records = self.get_history(sku=sku, start_date=start_date, end_date=end_date, limit=10000)
        return pd.DataFrame(records)
    
    def clear_history(self) -> int:
        """
        Clear all history.
        
        Returns:
            Number of records cleared.
        """
        if not self.history_file.exists():
            return 0
        
        try:
            df = pd.read_excel(self.history_file, engine='openpyxl')
            count = len(df)
        except Exception:
            count = 0
        
        # Create fresh empty file
        df = pd.DataFrame(columns=[
            'export_date', 'export_time', 'sku', 'isku', 'product_name',
            'old_price', 'new_price', 'change_amount', 'change_percent',
            'pokedata_id', 'pokedata_price_usd', 'fx_rate', 'source',
        ])
        df.to_excel(self.history_file, index=False, engine='openpyxl')
        
        logger.info(f"Cleared {count} history records")
        return count


# Singleton instance
_history_service: Optional[PriceHistoryService] = None


def get_history_service() -> PriceHistoryService:
    """Get the singleton history service instance."""
    global _history_service
    if _history_service is None:
        _history_service = PriceHistoryService()
    return _history_service


def log_price_export(
    exports: List[Dict[str, Any]],
    fx_rate: float = 1.0,
    source: str = "Pokedata",
) -> int:
    """Convenience function to log price exports."""
    return get_history_service().log_export(exports, fx_rate, source)
