"""
Unified Excel/CSV file handling utilities.

Consolidates all Excel reading logic to prevent engine mismatches.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


# Engine mapping for Excel formats
EXCEL_ENGINES = {
    ".xlsx": "openpyxl",
    ".xls": "xlrd",
}


def read_excel_file(
    source: Union[bytes, Path, str],
    filename: str = None,
) -> pd.DataFrame:
    """
    Read Excel or CSV file with correct engine selection.
    
    Args:
        source: File content as bytes, or path to file.
        filename: Original filename (required if source is bytes).
        
    Returns:
        Loaded DataFrame.
        
    Raises:
        ValueError: If file format is unsupported or file is corrupt.
    """
    # Determine suffix
    if isinstance(source, (Path, str)):
        path = Path(source)
        suffix = path.suffix.lower()
        file_source = path
    else:
        if not filename:
            raise ValueError("filename is required when source is bytes")
        suffix = Path(filename).suffix.lower()
        file_source = BytesIO(source)
    
    try:
        if suffix == ".csv":
            logger.debug(f"Reading CSV file: {filename or source}")
            return pd.read_csv(file_source)
        elif suffix in EXCEL_ENGINES:
            engine = EXCEL_ENGINES[suffix]
            logger.debug(f"Reading Excel file: {filename or source} with engine={engine}")
            return pd.read_excel(file_source, engine=engine)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .xlsx, .xls, .csv"
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")


def write_excel_file(
    df: pd.DataFrame,
    path: Union[Path, str],
    sheet_name: str = "Sheet1",
) -> None:
    """
    Write DataFrame to Excel file.
    
    Args:
        df: DataFrame to write.
        path: Output file path.
        sheet_name: Excel sheet name.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(path, index=False, sheet_name=sheet_name, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
    
    logger.info(f"Wrote {len(df)} rows to {path}")


def get_supported_extensions() -> list:
    """Return list of supported file extensions."""
    return [".xlsx", ".xls", ".csv"]
