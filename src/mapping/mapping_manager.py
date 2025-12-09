"""
Mapping manager for SKU to Pokedata ID relationships.

Loads and manages the mapping between SiteGiant product SKUs and
Pokedata product IDs, including language and asset type information.
"""

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import unquote

import pandas as pd

from src.utils.config_loader import AppConfig


logger = logging.getLogger(__name__)


@dataclass
class DuplicateScanResult:
    """
    Result of scanning a mapping file for duplicate SKUs.
    
    Attributes:
        has_duplicates: True if duplicates were found.
        duplicate_skus: List of SKUs that appear more than once.
        duplicate_details: Dict mapping SKU -> list of row indices where it appears.
        total_rows: Total number of rows in the file.
        unique_skus: Number of unique SKUs.
    """
    has_duplicates: bool = False
    duplicate_skus: List[str] = field(default_factory=list)
    duplicate_details: Dict[str, List[int]] = field(default_factory=dict)
    total_rows: int = 0
    unique_skus: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_duplicates": self.has_duplicates,
            "duplicate_skus": self.duplicate_skus,
            "duplicate_count": len(self.duplicate_skus),
            "duplicate_details": self.duplicate_details,
            "total_rows": self.total_rows,
            "unique_skus": self.unique_skus,
        }


def extract_pokedata_id_from_url(url: str) -> str:
    """
    Extract product slug/ID from a Pokedata URL.
    
    Handles URL formats like:
    - https://www.pokedata.io/product/Mega+Kangaskhan+ex+Box
    - https://pokedata.io/product/Scarlet+%26+Violet+Booster+Box
    
    Args:
        url: Pokedata product URL.
        
    Returns:
        Extracted product slug (decoded), or empty string if invalid.
    """
    if not url or (isinstance(url, float) and pd.isna(url)):
        return ""
    
    url = str(url).strip()
    
    # Match /product/ followed by the product slug
    match = re.search(r'/product/([^/?#]+)', url)
    if match:
        slug = match.group(1)
        # Decode URL encoding: + -> space, %XX -> character
        slug = slug.replace('+', ' ')
        slug = unquote(slug)
        return slug
    
    return ""


def scan_duplicates(file_path: Path) -> DuplicateScanResult:
    """
    Scan a mapping file for duplicate SKUs without loading it fully.
    
    Args:
        file_path: Path to the mapping file (Excel or CSV).
        
    Returns:
        DuplicateScanResult: Scan results with duplicate information.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file format is unsupported.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {file_path}")
    
    # Load the file
    suffix = file_path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(file_path, engine="openpyxl")
    elif suffix == ".xls":
        df = pd.read_excel(file_path, engine="xlrd")
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported mapping file format: {suffix}")
    
    # Normalize SKU column name
    sku_col = None
    for col in df.columns:
        if col.lower().strip() == "sku":
            sku_col = col
            break
    
    if sku_col is None:
        raise ValueError("Mapping file missing required 'sku' column")
    
    # Get SKUs as strings, stripped
    df["_sku_normalized"] = df[sku_col].astype(str).str.strip()
    
    # Find duplicates
    sku_counts = df["_sku_normalized"].value_counts()
    duplicate_skus = sku_counts[sku_counts > 1].index.tolist()
    
    # Build details: which rows contain each duplicate
    duplicate_details = {}
    for sku in duplicate_skus:
        # Get 1-indexed row numbers (add 2: 1 for 0-index, 1 for header row)
        rows = df[df["_sku_normalized"] == sku].index.tolist()
        duplicate_details[sku] = [r + 2 for r in rows]  # Excel row numbers
    
    result = DuplicateScanResult(
        has_duplicates=len(duplicate_skus) > 0,
        duplicate_skus=duplicate_skus,
        duplicate_details=duplicate_details,
        total_rows=len(df),
        unique_skus=len(sku_counts),
    )
    
    if result.has_duplicates:
        logger.warning(
            f"Duplicate SKUs detected in mapping file: {duplicate_skus}"
        )
    
    return result


class MappingEntry:
    """
    Represents a single SKU to Pokedata mapping entry.
    
    Attributes:
        sku: SiteGiant product SKU.
        language: Product language (ENGLISH or JAPANESE).
        pokedata_id: Pokedata product ID.
        asset_type: Product type/category.
        pokedata_url: Direct URL to Pokedata product page.
    """
    
    def __init__(
        self,
        sku: str,
        language: str,
        pokedata_id: str,
        asset_type: str,
        pokedata_url: Optional[str] = None,
    ) -> None:
        """
        Initialize a mapping entry.
        
        Args:
            sku: SiteGiant product SKU.
            language: Product language (ENGLISH or JAPANESE).
            pokedata_id: Pokedata product ID.
            asset_type: Product type/category.
            pokedata_url: Direct URL to Pokedata product page.
        """
        self.sku = sku
        self.language = language
        self.pokedata_id = pokedata_id
        self.asset_type = asset_type
        self.pokedata_url = pokedata_url


class MappingManager:
    """
    Manages SKU to Pokedata ID mappings.
    
    Responsible for:
    - Loading mapping files (Excel/CSV)
    - Looking up Pokedata IDs by SKU
    - Joining SiteGiant product data with mapping information
    - Identifying unmapped SKUs
    
    Attributes:
        config: Application configuration.
        mappings: Dictionary of SKU -> MappingEntry.
        mapping_df: Raw DataFrame of mapping data.
    """
    
    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the mapping manager.
        
        Args:
            config: Application configuration.
        """
        self.config = config
        self.mappings: Dict[str, MappingEntry] = {}
        self.mapping_df: Optional[pd.DataFrame] = None
    
    def load_mapping_file(
        self,
        file_path: Path,
        strategy: Optional[str] = None,
    ) -> DuplicateScanResult:
        """
        Load a mapping file into memory with duplicate handling.
        
        Supports Excel (.xlsx) and CSV (.csv) formats.
        
        Required columns:
            - sku (or SKU): SiteGiant product SKU
            - pokedata_id: Pokedata product ID
            - auto_update: Y/N flag for auto-update
        
        Optional columns:
            - isku (or ISKU): SiteGiant internal SKU
            - name/product_name: Product name from SiteGiant
            - pokedata_name: Product name from Pokedata
            - pokedata_url: URL to Pokedata product page
            - pokedata_language: ENGLISH/JAPANESE (default: ENGLISH)
            - pokedata_asset_type: PRODUCT/CARD (default: PRODUCT)
        
        Args:
            file_path: Path to the mapping file.
            strategy: How to handle duplicate SKUs:
                - None: Use config default (mapping.duplicate_handling)
                - "merge": Keep last entry for each SKU (overwrites)
                - "ignore": Keep first entry for each SKU (skips duplicates)
            
        Returns:
            DuplicateScanResult: Information about any duplicates found.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {file_path}")
        
        logger.info(f"Loading mapping file from: {file_path}")
        
        # Load based on file extension
        suffix = file_path.suffix.lower()
        if suffix == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        elif suffix == ".xls":
            df = pd.read_excel(file_path, engine="xlrd")
        elif suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported mapping file format: {suffix}")
        
        # Normalize column names (handle case variations)
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower == "sku":
                column_map[col] = "sku"
            elif col_lower == "isku":
                column_map[col] = "isku"
            elif col_lower in ["name", "product_name", "product name"]:
                column_map[col] = "name"
            elif col_lower == "pokedata_id":
                column_map[col] = "pokedata_id"
            elif col_lower == "pokedata_name":
                column_map[col] = "pokedata_name"
            elif col_lower == "pokedata_url":
                column_map[col] = "pokedata_url"
            elif col_lower == "pokedata_language":
                column_map[col] = "pokedata_language"
            elif col_lower == "pokedata_asset_type":
                column_map[col] = "pokedata_asset_type"
            elif col_lower == "auto_update":
                column_map[col] = "auto_update"
        
        if column_map:
            df = df.rename(columns=column_map)
        
        # Validate required columns
        required_cols = ["sku", "pokedata_id", "pokedata_language", "auto_update"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Mapping file missing required columns: {missing}. "
                           f"Available columns: {list(df.columns)}")
        
        # NOTE: We no longer auto-fill pokedata_id from URL extraction here.
        # The URL contains a product NAME/SLUG (e.g., "Hot Air Arena Booster Box"),
        # NOT a numeric ID. The numeric ID must be obtained via API search.
        # Use the "Refresh IDs" feature in the UI to look up numeric IDs via API.
        if "pokedata_id" not in df.columns:
            df["pokedata_id"] = ""
        
        # Auto-fill pokedata_name from URL if pokedata_name is empty (for display purposes only)
        if "pokedata_url" in df.columns:
            if "pokedata_name" not in df.columns:
                df["pokedata_name"] = ""
            
            # Find rows where pokedata_name is empty but pokedata_url exists
            mask = (
                (df["pokedata_name"].isna() | (df["pokedata_name"].astype(str).str.strip() == "")) &
                (df["pokedata_url"].notna() & (df["pokedata_url"].astype(str).str.strip() != ""))
            )
            
            if mask.any():
                # Ensure pokedata_name is string type to avoid dtype warning
                df["pokedata_name"] = df["pokedata_name"].astype(str)
                
                extracted_count = 0
                for idx in df[mask].index:
                    url = df.at[idx, "pokedata_url"]
                    extracted_name = extract_pokedata_id_from_url(url)
                    if extracted_name:
                        # Store in pokedata_name (NOT pokedata_id - that needs numeric ID from API)
                        df.at[idx, "pokedata_name"] = extracted_name
                        extracted_count += 1
                
                if extracted_count > 0:
                    logger.info(f"Auto-extracted {extracted_count} pokedata_name(s) from URLs for display")
        
        # Detect duplicate SKUs
        df["_sku_normalized"] = df["sku"].astype(str).str.strip()
        sku_counts = df["_sku_normalized"].value_counts()
        duplicate_skus = sku_counts[sku_counts > 1].index.tolist()
        
        # Build duplicate details
        duplicate_details = {}
        for sku in duplicate_skus:
            rows = df[df["_sku_normalized"] == sku].index.tolist()
            duplicate_details[sku] = [r + 2 for r in rows]  # Excel row numbers
        
        scan_result = DuplicateScanResult(
            has_duplicates=len(duplicate_skus) > 0,
            duplicate_skus=duplicate_skus,
            duplicate_details=duplicate_details,
            total_rows=len(df),
            unique_skus=len(sku_counts),
        )
        
        # Determine strategy: use parameter, or fall back to config default
        if strategy is None:
            strategy = getattr(self.config.mapping, 'duplicate_handling', 'merge')
        
        # Log warning if duplicates found
        if scan_result.has_duplicates:
            logger.warning(
                f"Duplicate SKUs detected: {duplicate_skus}. "
                f"Using '{strategy}' strategy."
            )
            
            # Handle duplicates based on strategy
            if strategy == "ignore":
                # Keep first occurrence only
                df = df.drop_duplicates(subset=["_sku_normalized"], keep="first")
                logger.info(f"Ignored {len(duplicate_skus)} duplicate SKUs (kept first entry)")
            else:  # "merge" - keep last (default behavior)
                df = df.drop_duplicates(subset=["_sku_normalized"], keep="last")
                logger.info(f"Merged {len(duplicate_skus)} duplicate SKUs (kept last entry)")
        
        # Clean up temp column
        df = df.drop(columns=["_sku_normalized"])
        
        self.mapping_df = df
        
        # Populate mappings dictionary
        for _, row in df.iterrows():
            sku = str(row.get("sku", "")).strip()
            if not sku:
                continue
            
            # Get pokedata_url
            pokedata_url = row.get("pokedata_url", "")
            if pd.isna(pokedata_url):
                pokedata_url = None
            else:
                pokedata_url = str(pokedata_url).strip() or None
            
            # Get pokedata_id (must be numeric ID from API, not extracted from URL)
            pokedata_id = row.get("pokedata_id", "")
            if pd.isna(pokedata_id) or str(pokedata_id).strip() == "":
                pokedata_id = ""
            else:
                pokedata_id = str(pokedata_id).strip()
            
            language = row.get("pokedata_language", "")
            if pd.isna(language):
                language = ""
            else:
                language = str(language).strip().upper()
            
            asset_type = row.get("pokedata_asset_type", "PRODUCT")
            if pd.isna(asset_type):
                asset_type = "PRODUCT"
            else:
                asset_type = str(asset_type).strip()
            
            self.mappings[sku] = MappingEntry(
                sku=sku,
                language=language,
                pokedata_id=pokedata_id,
                asset_type=asset_type,
                pokedata_url=pokedata_url,
            )
        
        logger.info(f"Loaded {len(self.mappings)} mappings from file")
        
        return scan_result
    
    def get_mapping(self, sku: str) -> Optional[MappingEntry]:
        """
        Get the mapping entry for a given SKU.
        
        Args:
            sku: The SiteGiant product SKU to look up.
            
        Returns:
            MappingEntry if found, None if SKU is not mapped.
        """
        return self.mappings.get(sku)
    
    def get_pokedata_id(self, sku: str) -> Optional[str]:
        """
        Get the Pokedata ID for a given SKU.
        
        Args:
            sku: The SiteGiant product SKU.
            
        Returns:
            Pokedata ID if mapped, None otherwise.
        """
        mapping = self.get_mapping(sku)
        return mapping.pokedata_id if mapping else None
    
    def get_language(self, sku: str) -> Optional[str]:
        """
        Get the language for a given SKU.
        
        Args:
            sku: The SiteGiant product SKU.
            
        Returns:
            Language string (ENGLISH/JAPANESE) if mapped, None otherwise.
        """
        mapping = self.get_mapping(sku)
        return mapping.language if mapping else None
    
    def join_with_products(
        self,
        products_df: pd.DataFrame,
        sku_column: str = "sku",
    ) -> pd.DataFrame:
        """
        Join product DataFrame with mapping information.
        
        Adds columns: pokedata_id, pokedata_language, pokedata_asset_type, pokedata_url, auto_update, is_mapped
        
        Args:
            products_df: DataFrame with product data.
            sku_column: Name of the SKU column in products_df.
            
        Returns:
            pd.DataFrame: Products with mapping columns added.
        """
        if self.mapping_df is None:
            raise ValueError("No mapping file loaded. Call load_mapping_file() first.")
        
        # Left join products with mapping
        result = products_df.merge(
            self.mapping_df[[
                "sku", "pokedata_name", "pokedata_language", "pokedata_asset_type",
                "pokedata_id", "pokedata_url", "auto_update", "notes"
            ]],
            left_on=sku_column,
            right_on="sku",
            how="left",
            suffixes=("", "_mapping")
        )
        
        # Remove duplicate sku column if created
        if "sku_mapping" in result.columns:
            result = result.drop(columns=["sku_mapping"])
        
        # Fill NaN values for mapping columns
        result["pokedata_name"] = result["pokedata_name"].fillna("")
        result["pokedata_language"] = result["pokedata_language"].fillna("")
        result["pokedata_asset_type"] = result["pokedata_asset_type"].fillna("PRODUCT")
        result["pokedata_id"] = result["pokedata_id"].fillna("")
        result["pokedata_url"] = result["pokedata_url"].fillna("")
        result["auto_update"] = result["auto_update"].fillna("N")
        result["notes"] = result["notes"].fillna("")
        
        # Convert pokedata_id to string
        result["pokedata_id"] = result["pokedata_id"].astype(str).str.strip()
        result["pokedata_id"] = result["pokedata_id"].replace("nan", "")
        
        # Add is_mapped column: True if pokedata_id exists and auto_update == "Y"
        result["is_mapped"] = (
            (result["pokedata_id"] != "") & 
            (result["pokedata_id"].notna()) &
            (result["auto_update"].str.upper() == "Y")
        )
        
        logger.info(f"Joined {len(result)} products with mapping. "
                   f"Mapped for update: {result['is_mapped'].sum()}")
        
        return result
    
    def get_unmapped_skus(self, products_df: pd.DataFrame, sku_column: str = "sku") -> List[str]:
        """
        Get list of SKUs that don't have a mapping.
        
        Args:
            products_df: DataFrame with product data.
            sku_column: Name of the SKU column.
            
        Returns:
            List[str]: SKUs without mapping entries.
        """
        product_skus = set(products_df[sku_column].astype(str).str.strip())
        mapped_skus = set(self.mappings.keys())
        unmapped = list(product_skus - mapped_skus)
        return unmapped
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded mappings.
        
        Returns:
            Dict with counts by language, asset_type, etc.
        """
        stats = {
            "total": len(self.mappings),
            "with_pokedata_id": 0,
            "english": 0,
            "japanese": 0,
            "by_asset_type": {},
        }
        
        for entry in self.mappings.values():
            if entry.pokedata_id:
                stats["with_pokedata_id"] += 1
            
            if entry.language.upper() == "ENGLISH":
                stats["english"] += 1
            elif entry.language.upper() == "JAPANESE":
                stats["japanese"] += 1
            
            asset_type = entry.asset_type or "Unknown"
            stats["by_asset_type"][asset_type] = stats["by_asset_type"].get(asset_type, 0) + 1
        
        return stats


def load_mapping(file_path: Path, config: Optional[AppConfig] = None) -> pd.DataFrame:
    """
    Convenience function to load a mapping file as DataFrame.
    
    Args:
        file_path: Path to the mapping file.
        config: Optional configuration.
        
    Returns:
        pd.DataFrame: Mapping data.
    """
    if config is None:
        config = AppConfig()
    
    manager = MappingManager(config)
    manager.load_mapping_file(file_path)
    return manager.mapping_df


def apply_mapping(
    products_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    sku_column: str = "sku",
) -> pd.DataFrame:
    """
    Convenience function to apply mapping to products.
    
    Args:
        products_df: DataFrame with product data.
        mapping_df: DataFrame with mapping data.
        sku_column: Name of the SKU column.
        
    Returns:
        pd.DataFrame: Products with mapping columns added.
    """
    config = AppConfig()
    manager = MappingManager(config)
    manager.mapping_df = mapping_df
    
    # Build mappings dict from dataframe
    for _, row in mapping_df.iterrows():
        sku = str(row.get("sku", "")).strip()
        if sku:
            manager.mappings[sku] = MappingEntry(
                sku=sku,
                language=str(row.get("pokedata_language", "")).strip(),
                pokedata_id=str(row.get("pokedata_id", "")).strip(),
                asset_type=str(row.get("pokedata_asset_type", "PRODUCT")).strip(),
                pokedata_url=str(row.get("pokedata_url", "")).strip() or None,
            )
    
    return manager.join_with_products(products_df, sku_column)
