"""
I/O helper functions.

Common file and Excel operations used across the application.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd


logger = logging.getLogger(__name__)


def read_excel_file(
    file_path: Path,
    sheet_name: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read an Excel file into a DataFrame.
    
    Automatically detects file format (.xlsx, .xls) and uses appropriate engine.
    
    Args:
        file_path: Path to the Excel file.
        sheet_name: Specific sheet to read. Defaults to first sheet.
        **kwargs: Additional arguments passed to pd.read_excel.
        
    Returns:
        pd.DataFrame: Data from the Excel file.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is unsupported.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".xlsx":
        engine = "openpyxl"
    elif suffix == ".xls":
        engine = "xlrd"
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    logger.debug(f"Reading Excel file: {file_path} (engine={engine})")
    
    return pd.read_excel(
        file_path,
        sheet_name=sheet_name or 0,
        engine=engine,
        **kwargs,
    )


def write_excel_file(
    df: pd.DataFrame,
    file_path: Path,
    sheet_name: str = "Sheet1",
    index: bool = False,
    **kwargs,
) -> Path:
    """
    Write a DataFrame to an Excel file.
    
    Args:
        df: DataFrame to write.
        file_path: Output file path.
        sheet_name: Name of the Excel sheet.
        index: Whether to include DataFrame index.
        **kwargs: Additional arguments passed to df.to_excel.
        
    Returns:
        Path: Path to the created file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Writing Excel file: {file_path}")
    
    df.to_excel(
        file_path,
        sheet_name=sheet_name,
        index=index,
        engine="openpyxl",
        **kwargs,
    )
    
    return file_path


def read_csv_file(
    file_path: Path,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame.
    
    Args:
        file_path: Path to the CSV file.
        **kwargs: Additional arguments passed to pd.read_csv.
        
    Returns:
        pd.DataFrame: Data from the CSV file.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.debug(f"Reading CSV file: {file_path}")
    
    return pd.read_csv(file_path, **kwargs)


def write_csv_file(
    df: pd.DataFrame,
    file_path: Path,
    index: bool = False,
    **kwargs,
) -> Path:
    """
    Write a DataFrame to a CSV file.
    
    Args:
        df: DataFrame to write.
        file_path: Output file path.
        index: Whether to include DataFrame index.
        **kwargs: Additional arguments passed to df.to_csv.
        
    Returns:
        Path: Path to the created file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Writing CSV file: {file_path}")
    
    df.to_csv(file_path, index=index, **kwargs)
    
    return file_path


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        Path: The directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_list(
    directory: Path,
    pattern: str = "*.xlsx",
    recursive: bool = False,
) -> List[Path]:
    """
    Get list of files matching a pattern in a directory.
    
    Args:
        directory: Directory to search.
        pattern: Glob pattern for matching files.
        recursive: Whether to search subdirectories.
        
    Returns:
        List[Path]: Matching file paths.
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def validate_file_path(path: Path, must_exist: bool = True) -> bool:
    """
    Validate a file path.
    
    Args:
        path: Path to validate.
        must_exist: Whether the file must already exist.
        
    Returns:
        bool: True if path is valid.
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        return False
    
    if must_exist and not path.is_file():
        return False
    
    return True
