"""
SiteGiant Excel exporter module.

Generates Excel files formatted for re-import into SiteGiant Webstore,
preserving all non-price columns from the original export.
"""

import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Set

import pandas as pd

from src.utils.config_loader import AppConfig


logger = logging.getLogger(__name__)


class SiteGiantExporter:
    """
    Exporter for generating SiteGiant-compatible Excel files.
    
    Responsible for:
    - Preserving original column structure
    - Updating only price columns
    - Filtering to include only approved updates
    - Generating timestamped output files
    
    Attributes:
        config: Application configuration.
        output_dir: Directory for output files.
    """
    
    def __init__(self, config: AppConfig, output_dir: Optional[Path] = None) -> None:
        """
        Initialize the SiteGiant exporter.
        
        Args:
            config: Application configuration.
            output_dir: Directory for output files. Uses config default if not provided.
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("data/output")
    
    def prepare_export_data(
        self,
        original_df: pd.DataFrame,
        updated_df: pd.DataFrame,
        price_column: str = "price",
        new_price_column: str = "new_price_myr",
        include_column: str = "include",
    ) -> pd.DataFrame:
        """
        Prepare data for export by merging original and updated data.
        
        Only includes rows marked for inclusion and updates only price columns.
        
        Args:
            original_df: Original SiteGiant export data.
            updated_df: DataFrame with new prices and include flags.
            price_column: Name of price column in SiteGiant format.
            new_price_column: Name of column with calculated new prices.
            include_column: Name of column with include/exclude flags.
            
        Returns:
            pd.DataFrame: Data ready for export.
        """
        # Start with original data
        result = original_df.copy()
        
        # If include column exists, filter by it
        if include_column in updated_df.columns:
            include_mask = updated_df[include_column] == True
        else:
            # Default: include all rows with valid new prices
            include_mask = updated_df[new_price_column].notna()
        
        # Get rows to update
        rows_to_update = updated_df[include_mask]
        
        # Update prices for matching SKUs
        if "sku" in result.columns and "sku" in rows_to_update.columns:
            price_updates = rows_to_update.set_index("sku")[new_price_column].to_dict()
            
            def update_price(row):
                sku = row.get("sku")
                if sku in price_updates and pd.notna(price_updates[sku]):
                    return price_updates[sku]
                return row.get(price_column)
            
            result[price_column] = result.apply(update_price, axis=1)
        
        return result
    
    def generate_filename(self, prefix: str = "sitegiant_price_update") -> str:
        """
        Generate a timestamped filename for the export.
        
        Args:
            prefix: Filename prefix.
            
        Returns:
            str: Filename with timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.xlsx"
    
    def export_to_excel(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
        sheet_name: str = "Products",
    ) -> Path:
        """
        Export DataFrame to Excel file.
        
        Args:
            df: DataFrame to export.
            output_path: Full path for output file. Auto-generated if not provided.
            sheet_name: Name of the Excel sheet.
            
        Returns:
            Path: Path to the created file.
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if output_path is None:
            filename = self.generate_filename()
            output_path = self.output_dir / filename
        else:
            output_path = Path(output_path)
        
        logger.info(f"Exporting to Excel: {output_path}")
        
        # Write to Excel
        df.to_excel(output_path, index=False, engine="openpyxl", sheet_name=sheet_name)
        
        logger.info(f"Successfully exported {len(df)} rows to {output_path}")
        return output_path
    
    def export_with_summary(
        self,
        df: pd.DataFrame,
        summary_data: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export DataFrame with a summary sheet.
        
        Creates an Excel file with:
        - Products sheet: Data for import
        - Summary sheet: Statistics and metadata
        
        Args:
            df: Product data to export.
            summary_data: Dictionary with summary statistics.
            output_path: Full path for output file.
            
        Returns:
            Path: Path to the created file.
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_path is None:
            filename = self.generate_filename()
            output_path = self.output_dir / filename
        else:
            output_path = Path(output_path)
        
        # Create summary DataFrame
        summary_rows = []
        for key, value in summary_data.items():
            summary_rows.append({"Metric": key, "Value": str(value)})
        summary_df = pd.DataFrame(summary_rows)
        
        # Write both sheets
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Products", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        logger.info(f"Exported {len(df)} products with summary to {output_path}")
        return output_path
    
    def validate_export_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate data before export.
        
        Checks for:
        - Required columns present
        - No empty prices
        - Valid price values
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            List[str]: List of validation errors (empty if valid).
        """
        errors = []
        
        # Check required columns
        required = ["sku", "price"]
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for null SKUs
        if "sku" in df.columns:
            null_skus = df["sku"].isna().sum()
            if null_skus > 0:
                errors.append(f"Found {null_skus} rows with null SKU")
        
        # Check for negative prices
        if "price" in df.columns:
            negative_prices = (df["price"] < 0).sum()
            if negative_prices > 0:
                errors.append(f"Found {negative_prices} rows with negative prices")
        
        return errors
    
    def create_backup(self, original_file: Path) -> Path:
        """
        Create a backup of the original file before overwriting.
        
        Args:
            original_file: Path to the original file.
            
        Returns:
            Path: Path to the backup file.
        """
        original_file = Path(original_file)
        if not original_file.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {original_file}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{original_file.stem}_backup_{timestamp}{original_file.suffix}"
        backup_path = original_file.parent / backup_name
        
        shutil.copy2(original_file, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        return backup_path


def add_include_in_update_column(
    df: pd.DataFrame,
    status_column: str = "status",
    auto_update_column: str = "auto_update",
    include_statuses: Set[str] = None,
) -> pd.DataFrame:
    """
    Add include_in_update column for GUI preview.
    
    Default: True for OK/WARNING with auto_update=Y, False otherwise.
    
    Args:
        df: DataFrame with status and auto_update columns.
        status_column: Name of status column.
        auto_update_column: Name of auto_update flag column.
        include_statuses: Statuses to include by default (default: OK, WARNING).
        
    Returns:
        pd.DataFrame: DataFrame with include_in_update column added.
    """
    if include_statuses is None:
        include_statuses = {"OK", "WARNING"}
    
    df = df.copy()
    
    def should_include(row):
        status = row.get(status_column, "")
        auto_update = str(row.get(auto_update_column, "N")).upper()
        new_price = row.get("new_price_myr")
        
        return (
            status in include_statuses and 
            auto_update == "Y" and 
            pd.notna(new_price)
        )
    
    df["include_in_update"] = df.apply(should_include, axis=1)
    
    return df


def find_price_column(df: pd.DataFrame) -> str:
    """Find the price column name in the DataFrame (preserves original casing)."""
    price_variants = ["Price", "price", "PRICE", "Selling Price", "selling_price", "Current Price"]
    for col in price_variants:
        if col in df.columns:
            return col
    return "Price"  # Default to SiteGiant's standard


def find_sku_column(df: pd.DataFrame) -> str:
    """Find the SKU column name in the DataFrame (preserves original casing)."""
    sku_variants = ["SKU", "sku", "Sku", "Product SKU", "product_sku"]
    for col in sku_variants:
        if col in df.columns:
            return col
    return "SKU"  # Default to SiteGiant's standard


def build_update_dataframe(
    original_df: pd.DataFrame,
    results_df: pd.DataFrame,
    include_statuses: Set[str] = None,
    price_column: str = None,  # Auto-detect if None
    new_price_column: str = "new_price_myr",
    status_column: str = "status",
    auto_update_column: str = "auto_update",
    include_column: str = "include_in_update",
) -> pd.DataFrame:
    """
    Build a DataFrame with updated prices for SiteGiant import.
    
    Preserves the ORIGINAL column names from SiteGiant export to ensure
    compatibility when re-importing to SiteGiant.
    
    For rows where status is in include_statuses AND auto_update == "Y":
    - Overwrite price column with `new_price_myr`.
    For all other rows:
    - Keep the original price.
    
    If `include_in_update` column exists in results_df, it takes priority.
    
    Args:
        original_df: Original SiteGiant export data (preserves column names).
        results_df: DataFrame with pricing results and statuses.
        include_statuses: Set of statuses to include (default: OK, WARNING).
        price_column: Name of price column (auto-detected if None).
        new_price_column: Name of column with calculated new prices.
        status_column: Name of status column.
        auto_update_column: Name of auto_update flag column.
        include_column: Name of include/exclude column (for GUI checkbox support).
        
    Returns:
        pd.DataFrame: DataFrame ready for SiteGiant import (preserves all original columns).
    """
    if include_statuses is None:
        include_statuses = {"OK", "WARNING"}
    
    # Preserve original DataFrame structure
    result = original_df.copy()
    
    # Auto-detect column names from original DataFrame to preserve SiteGiant format
    if price_column is None:
        price_column = find_price_column(original_df)
    
    sku_column = find_sku_column(original_df)
    
    logger.debug(f"Using price column: '{price_column}', SKU column: '{sku_column}'")
    
    # Build mapping of SKU -> new price
    price_updates = {}
    
    for _, row in results_df.iterrows():
        # Try both normalized 'sku' and original column name
        sku = row.get("sku") or row.get(sku_column)
        new_price = row.get(new_price_column)
        
        if not sku or pd.isna(new_price):
            continue
        
        # Check if include_in_update column exists and use it
        if include_column in results_df.columns:
            include = row.get(include_column, False)
            if include:
                price_updates[sku] = new_price
        else:
            # Fall back to status + auto_update logic
            status = row.get(status_column, "")
            auto_update = str(row.get(auto_update_column, "N")).upper()
            
            if status in include_statuses and auto_update == "Y":
                price_updates[sku] = new_price
    
    # Apply updates to original DataFrame using original column names
    if sku_column in result.columns and price_column in result.columns:
        def update_price(row):
            sku = row.get(sku_column)
            if sku in price_updates:
                return price_updates[sku]
            return row.get(price_column)
        
        result[price_column] = result.apply(update_price, axis=1)
    
    logger.info(f"Updated {len(price_updates)} prices out of {len(result)} products")
    
    return result


def write_update_file(
    df: pd.DataFrame,
    output_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Write update DataFrame to Excel file.
    
    Args:
        df: DataFrame to export.
        output_dir: Output directory.
        filename: Optional filename (auto-generated if not provided).
        
    Returns:
        Path: Path to created file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"sitegiant_price_update_{timestamp}.xlsx"
    
    output_path = output_dir / filename
    
    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info(f"Wrote update file: {output_path}")
    
    return output_path
