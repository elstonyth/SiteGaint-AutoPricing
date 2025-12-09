"""
Helper utilities for the SiteGiant Pricing web application.

Common functions used across routes to reduce code duplication.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage.settings_store import get_api_key

logger = logging.getLogger(__name__)


# =============================================================================
# API Key Resolution
# =============================================================================


def resolve_api_key(override: str | None = None) -> str | None:
    """
    Resolve API key from multiple sources in priority order.

    Priority: override > stored settings > environment variable

    Args:
        override: Optional API key override from request.

    Returns:
        Resolved API key or None if not found.
    """
    if override and override.strip():
        return override.strip()

    stored_key = get_api_key()
    if stored_key:
        return stored_key

    env_key = os.environ.get("POKEDATA_API_KEY", "")
    if env_key:
        return env_key

    return None


def has_valid_api_key(override: str | None = None) -> bool:
    """Check if a valid API key is available."""
    return resolve_api_key(override) is not None


# =============================================================================
# Column Normalization
# =============================================================================

COLUMN_VARIANTS = {
    "sku": ["SKU", "sku", "Sku", "Product SKU", "product_sku"],
    "name": ["Product Name", "Name", "name", "Title", "title", "product_name"],
    "price": ["Price", "price", "Selling Price", "selling_price", "Current Price"],
}


def find_column(df: pd.DataFrame, variants: list[str]) -> str | None:
    """
    Find a column in DataFrame by checking multiple name variants.

    Args:
        df: DataFrame to search.
        variants: List of possible column names.

    Returns:
        Actual column name found, or None.
    """
    df_columns_lower = {c.lower(): c for c in df.columns}

    for variant in variants:
        if variant in df.columns:
            return variant
        if variant.lower() in df_columns_lower:
            return df_columns_lower[variant.lower()]

    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column names to standard format.

    Renames: SKU variants → 'sku', Name variants → 'name', Price variants → 'price'

    Args:
        df: DataFrame with original columns.

    Returns:
        DataFrame with normalized column names.
    """
    df = df.copy()

    for standard_name, variants in COLUMN_VARIANTS.items():
        found_col = find_column(df, variants)
        if found_col and found_col != standard_name:
            df = df.rename(columns={found_col: standard_name})

    return df


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_sitegiant_columns(df: pd.DataFrame) -> list[str]:
    """
    Validate that DataFrame has required SiteGiant export columns.

    Args:
        df: DataFrame to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    if df is None or df.empty:
        return ["File is empty or could not be read"]

    # Map internal names to user-friendly error messages
    error_messages = {
        "sku": "Missing SKU column",
        "name": "Missing Product Name column",
        "price": "Missing Price column",
    }

    for required, variants in COLUMN_VARIANTS.items():
        if find_column(df, variants) is None:
            errors.append(error_messages.get(required, f"Missing {required} column"))

    return errors


def validate_mapping_columns(df: pd.DataFrame) -> list[str]:
    """
    Validate that DataFrame has required mapping columns.

    Required: sku, pokedata_id, pokedata_language, auto_update

    Args:
        df: DataFrame to validate.

    Returns:
        List of missing column names (empty if valid).
    """
    required_cols = ["sku", "pokedata_id", "pokedata_language", "auto_update"]
    missing = []

    df_columns_lower = [c.lower() for c in df.columns]

    for col in required_cols:
        if col not in df_columns_lower:
            missing.append(col)

    return missing


def validate_mapping_file(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """
    Comprehensive validation for mapping file with user-friendly messages.

    Args:
        df: DataFrame to validate.
        filename: Original filename.

    Returns:
        Dict with 'valid', 'errors', 'warnings', and 'suggestions'.
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": [],
    }

    # Check if file is empty
    if df.empty:
        result["valid"] = False
        result["errors"].append(
            {
                "type": "empty_file",
                "message": "The uploaded file is empty",
                "suggestion": "Please upload a file with at least one product row",
            }
        )
        return result

    # Check required columns
    required_cols = {
        "sku": "Product SKU (unique identifier)",
        "pokedata_id": "Pokedata product ID (numeric)",
        "pokedata_language": "Language (ENGLISH or JAPANESE)",
        "auto_update": "Auto-update flag (Y or N)",
    }

    df_columns_lower = [c.lower().strip() for c in df.columns]
    missing = []

    for col, description in required_cols.items():
        if col not in df_columns_lower:
            missing.append(f"'{col}' - {description}")

    if missing:
        result["valid"] = False
        result["errors"].append(
            {
                "type": "missing_columns",
                "message": f"Missing required columns: {len(missing)}",
                "details": missing,
                "suggestion": "Download the template to see the correct format",
            }
        )

    # Check for empty SKUs
    if "sku" in df_columns_lower:
        sku_col = next(c for c in df.columns if c.lower().strip() == "sku")
        empty_skus = df[sku_col].isna().sum() + (df[sku_col].astype(str).str.strip() == "").sum()
        if empty_skus > 0:
            result["warnings"].append(
                {
                    "type": "empty_skus",
                    "message": f"{empty_skus} rows have empty SKUs",
                    "suggestion": "Rows without SKUs will be ignored",
                }
            )

    # Check for duplicate SKUs
    if "sku" in df_columns_lower:
        sku_col = next(c for c in df.columns if c.lower().strip() == "sku")
        duplicates = df[sku_col].dropna().duplicated().sum()
        if duplicates > 0:
            result["warnings"].append(
                {
                    "type": "duplicate_skus",
                    "message": f"{duplicates} duplicate SKUs found",
                    "suggestion": "Duplicate SKUs may cause unexpected behavior",
                }
            )

    # Check pokedata_id format
    if "pokedata_id" in df_columns_lower:
        id_col = next(c for c in df.columns if c.lower().strip() == "pokedata_id")
        empty_ids = df[id_col].isna().sum()
        if empty_ids > 0:
            result["warnings"].append(
                {
                    "type": "empty_ids",
                    "message": f"{empty_ids} products have no Pokedata ID",
                    "suggestion": "Use Auto-lookup to find IDs from URLs, or enter manually",
                }
            )

    # Check for pokedata_url (helpful for auto-lookup)
    if "pokedata_url" not in df_columns_lower:
        result["suggestions"].append(
            {
                "type": "no_url_column",
                "message": "No 'pokedata_url' column found",
                "suggestion": "Add pokedata_url column to enable auto-lookup feature",
            }
        )

    # Check language values
    if "pokedata_language" in df_columns_lower:
        lang_col = next(c for c in df.columns if c.lower().strip() == "pokedata_language")
        valid_langs = ["ENGLISH", "JAPANESE", "EN", "JP"]
        invalid_langs = (
            df[~df[lang_col].astype(str).str.upper().isin(valid_langs)][lang_col].dropna().unique()
        )
        if len(invalid_langs) > 0:
            result["warnings"].append(
                {
                    "type": "invalid_language",
                    "message": f"Invalid language values: {list(invalid_langs)[:3]}",
                    "suggestion": "Use 'ENGLISH' or 'JAPANESE'",
                }
            )

    # Check auto_update values
    if "auto_update" in df_columns_lower:
        auto_col = next(c for c in df.columns if c.lower().strip() == "auto_update")
        valid_auto = ["Y", "N", "YES", "NO", "TRUE", "FALSE", "1", "0"]
        invalid_auto = (
            df[~df[auto_col].astype(str).str.upper().isin(valid_auto)][auto_col].dropna().unique()
        )
        if len(invalid_auto) > 0:
            result["warnings"].append(
                {
                    "type": "invalid_auto_update",
                    "message": f"Invalid auto_update values: {list(invalid_auto)[:3]}",
                    "suggestion": "Use 'Y' or 'N'",
                }
            )

    return result


def get_friendly_upload_error(error: Exception, filename: str) -> str:
    """Convert technical errors to user-friendly messages."""
    error_str = str(error).lower()

    if "no such file" in error_str or "file not found" in error_str:
        return "File not found. Please try uploading again."
    elif "permission" in error_str:
        return "Cannot access the file. Please close it if open in Excel."
    elif "corrupt" in error_str or "invalid" in error_str:
        return f"The file '{filename}' appears to be corrupted. Please try re-exporting it."
    elif "unsupported" in error_str:
        return "Unsupported file format. Please upload .xlsx, .xls, or .csv files only."
    elif "encoding" in error_str:
        return "File encoding error. Try saving the file as UTF-8 CSV or .xlsx format."
    elif "memory" in error_str:
        return "File is too large. Please split into smaller files."
    else:
        return f"Failed to read file: {str(error)}"


# =============================================================================
# File Loading Helpers
# =============================================================================


def load_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    """
    Load DataFrame from file content.

    Supports CSV and Excel formats.

    Args:
        content: File content as bytes.
        filename: Original filename (for format detection).

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If file format is unsupported or file is corrupt.
    """
    from io import BytesIO

    try:
        filename_lower = filename.lower()
        if filename_lower.endswith(".csv"):
            return pd.read_csv(BytesIO(content))
        elif filename_lower.endswith(".xlsx"):
            return pd.read_excel(BytesIO(content), engine="openpyxl")
        elif filename_lower.endswith(".xls"):
            return pd.read_excel(BytesIO(content), engine="xlrd")
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    except Exception as e:
        raise ValueError(f"Failed to read file '{filename}': {str(e)}")


def get_mapping_info(mapping_path: Path) -> dict[str, Any]:
    """
    Get information about a mapping file.

    Args:
        mapping_path: Path to mapping file.

    Returns:
        Dictionary with file info (exists, row_count, last_modified, etc.)
    """
    if not mapping_path.exists():
        return {
            "exists": False,
            "path": str(mapping_path),
            "row_count": 0,
            "last_modified": None,
        }

    try:
        stat = mapping_path.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime)

        suffix = mapping_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(mapping_path)
        elif suffix == ".xlsx":
            df = pd.read_excel(mapping_path, engine="openpyxl")
        elif suffix == ".xls":
            df = pd.read_excel(mapping_path, engine="xlrd")
        else:
            df = pd.read_excel(mapping_path, engine="openpyxl")  # fallback for unknown

        return {
            "exists": True,
            "path": str(mapping_path),
            "row_count": len(df),
            "columns": list(df.columns),
            "last_modified": last_modified.strftime("%Y-%m-%d %H:%M:%S"),
            "file_size": f"{stat.st_size / 1024:.1f} KB",
        }
    except Exception as e:
        logger.warning(f"Error reading mapping file {mapping_path}: {e}")
        return {
            "exists": True,
            "path": str(mapping_path),
            "error": str(e),
            "row_count": 0,
            "last_modified": None,
        }


# =============================================================================
# Formatting Helpers
# =============================================================================


def format_price(value: Any, decimals: int = 2) -> str:
    """Format a price value for display."""
    if pd.isna(value) or value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value: Any, decimals: int = 1) -> str:
    """Format a percentage value for display."""
    if pd.isna(value) or value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)


def format_results_for_display(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert results DataFrame to list of dicts for template display.

    Args:
        df: Results DataFrame with pricing data.

    Returns:
        List of row dictionaries formatted for display.
    """
    if df is None or df.empty:
        return []

    display_cols = [
        "name",
        "sku",
        "price",
        "new_price_myr",
        "abs_change",
        "pct_change",
        "status",
        "status_reason",
        "auto_update",
        "include_in_update",
    ]

    if "pct_change_vs_last_run" in df.columns:
        display_cols.insert(6, "pct_change_vs_last_run")

    available_cols = [c for c in display_cols if c in df.columns]

    results = []
    for _, row in df[available_cols].iterrows():
        item = {}
        for col in available_cols:
            val = row[col]
            if pd.isna(val):
                item[col] = ""
            elif col in ["price", "new_price_myr", "abs_change"]:
                item[col] = format_price(val)
            elif col in ["pct_change", "pct_change_vs_last_run"]:
                item[col] = format_percentage(val)
            elif col == "include_in_update":
                item[col] = bool(val)
            else:
                item[col] = str(val)
        results.append(item)

    return results
