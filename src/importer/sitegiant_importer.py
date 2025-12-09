"""
SiteGiant Excel importer module.

Loads and filters sealed products from SiteGiant Webstore export files.
Handles column mapping and data validation.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.config_loader import AppConfig

logger = logging.getLogger(__name__)


# Common column name variations in SiteGiant exports
COLUMN_VARIANTS = {
    "name": ["Product Name", "Name", "name", "product_name", "Product name", "title", "Title"],
    "isku": ["Internal SKU", "iSKU", "isku", "ISKU", "internal_sku", "Internal sku"],
    "sku": ["SKU", "sku", "Sku", "Product SKU", "product_sku"],
    "price": ["Price", "price", "Selling Price", "selling_price", "Current Price", "price_myr"],
}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column in the DataFrame from a list of candidate names."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


class SiteGiantImporter:
    """
    Importer for SiteGiant Webstore export Excel files.

    Responsible for:
    - Loading Excel files exported from SiteGiant
    - Mapping columns to internal field names
    - Filtering products based on configuration (active, in-stock, etc.)
    - Validating required data is present

    Attributes:
        config: Application configuration object.
        column_mapping: Dict mapping internal names to SiteGiant column names.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the SiteGiant importer.

        Args:
            config: Application configuration containing column mappings
                   and filter settings.
        """
        self.config = config
        self.column_mapping: dict[str, str] = {}

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a SiteGiant export Excel file.

        Args:
            file_path: Path to the Excel file to load.

        Returns:
            pd.DataFrame: Raw data from the Excel file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"SiteGiant export file not found: {file_path}")

        logger.info(f"Loading SiteGiant export from: {file_path}")

        # Determine engine based on file extension
        suffix = file_path.suffix.lower()
        if suffix == ".xlsx":
            engine = "openpyxl"
        elif suffix == ".xls":
            engine = "xlrd"
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Expected .xlsx or .xls")

        df = pd.read_excel(file_path, engine=engine)

        logger.info(f"Loaded {len(df)} rows from SiteGiant export")
        logger.debug(f"Columns found: {list(df.columns)}")

        return df

    def validate_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Validate that required columns are present in the DataFrame.

        Args:
            df: DataFrame to validate.

        Returns:
            List[str]: List of missing column names (empty if all present).
        """
        missing = []

        # Check for SKU column (required)
        sku_col = find_column(df, COLUMN_VARIANTS["sku"])
        if sku_col is None:
            missing.append("sku")

        # Check for name column (required)
        name_col = find_column(df, COLUMN_VARIANTS["name"])
        if name_col is None:
            missing.append("name")

        # Check for price column (required)
        price_col = find_column(df, COLUMN_VARIANTS["price"])
        if price_col is None:
            missing.append("price")

        if missing:
            logger.error(f"Missing required columns: {missing}")
            logger.debug(f"Available columns: {list(df.columns)}")

        return missing

    def filter_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter products based on configuration settings.

        Applies filters such as:
        - Active products only
        - In-stock products only
        - Specific product categories

        Args:
            df: DataFrame containing all products.

        Returns:
            pd.DataFrame: Filtered DataFrame with only relevant products.
        """
        # For now, return as-is since the export is already filtered to sealed products
        # Future: apply config.sitegiant.filters settings
        logger.debug(f"No additional filtering applied. Row count: {len(df)}")
        return df

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to internal standard names.

        Maps SiteGiant-specific column names to consistent internal names
        used throughout the application.

        Args:
            df: DataFrame with original SiteGiant column names.

        Returns:
            pd.DataFrame: DataFrame with normalized column names.
        """
        df = df.copy()

        # Find and rename columns to standard names
        rename_map = {}

        # SKU column
        sku_col = find_column(df, COLUMN_VARIANTS["sku"])
        if sku_col and sku_col != "sku":
            rename_map[sku_col] = "sku"
            self.column_mapping["sku"] = sku_col

        # Name column
        name_col = find_column(df, COLUMN_VARIANTS["name"])
        if name_col and name_col != "name":
            rename_map[name_col] = "name"
            self.column_mapping["name"] = name_col

        # iSKU column (optional)
        isku_col = find_column(df, COLUMN_VARIANTS["isku"])
        if isku_col and isku_col != "isku":
            rename_map[isku_col] = "isku"
            self.column_mapping["isku"] = isku_col

        # Price column
        price_col = find_column(df, COLUMN_VARIANTS["price"])
        if price_col and price_col != "price":
            rename_map[price_col] = "price"
            self.column_mapping["price"] = price_col

        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"Renamed columns: {rename_map}")

        # Ensure isku column exists (even if empty)
        if "isku" not in df.columns:
            df["isku"] = ""

        return df

    def import_products(self, file_path: Path) -> pd.DataFrame:
        """
        Full import pipeline: load, validate, filter, and normalize.

        This is the main entry point for importing SiteGiant data.

        Args:
            file_path: Path to the SiteGiant export Excel file.

        Returns:
            pd.DataFrame: Processed DataFrame ready for mapping.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing or data is invalid.
        """
        # 1. Load the file
        df = self.load_file(file_path)

        # 2. Validate required columns
        missing = self.validate_columns(df)
        if missing:
            raise ValueError(
                f"Missing required columns in SiteGiant export: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # 3. Filter products (no-op for pre-filtered sealed exports)
        df = self.filter_products(df)

        # 4. Normalize column names
        df = self.normalize_columns(df)

        logger.info(f"Successfully imported {len(df)} products from SiteGiant export")
        return df


def load_sitegiant_export(file_path: Path, config: AppConfig | None = None) -> pd.DataFrame:
    """
    Convenience function to load a SiteGiant export file.

    Args:
        file_path: Path to the SiteGiant export Excel file.
        config: Optional configuration (uses defaults if not provided).

    Returns:
        pd.DataFrame: Processed DataFrame ready for mapping.
    """
    if config is None:
        config = AppConfig()

    importer = SiteGiantImporter(config)
    return importer.import_products(file_path)
