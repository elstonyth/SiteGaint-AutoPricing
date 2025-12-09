"""
Generate mapping template from SiteGiant export.

CLI script to create a pre-filled mapping template from a SiteGiant export file.
The template contains SiteGiant product data with empty Pokedata columns
for manual mapping.

Usage:
    python -m src.mapping.generate_mapping_template
    python -m src.mapping.generate_mapping_template path/to/sitegiant_export.xlsx
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Template column definitions - organized by source
TEMPLATE_COLUMNS = [
    # SiteGiant columns (auto-filled from export)
    "sku",  # SiteGiant SKU - unique ID (auto-filled) [REQUIRED]
    "isku",  # SiteGiant internal SKU (auto-filled if present)
    "name",  # SiteGiant product name (auto-filled)
    "price",  # Current SiteGiant price in MYR (auto-filled)
    # Pokedata columns (manual entry)
    "pokedata_id",  # Pokedata product ID [REQUIRED]
    "pokedata_name",  # Pokedata product name (for reference)
    "pokedata_url",  # URL to Pokedata page (optional)
    "pokedata_language",  # ENGLISH or JAPANESE [REQUIRED]
    # Control columns
    "auto_update",  # Y/N flag for auto updates [REQUIRED]
    "notes",  # Free text comments
]

# SiteGiant column name variations (common names to look for)
SITEGIANT_COLUMN_MAP = {
    "name": ["Product Name", "Name", "name", "product_name", "Product name"],
    "isku": ["Internal SKU", "iSKU", "isku", "ISKU", "internal_sku", "Internal sku"],
    "sku": ["SKU", "sku", "Sku", "Product SKU"],
    "price": ["Price", "price", "Selling Price", "selling_price", "Current Price"],
}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Find a column in the DataFrame from a list of candidate names.

    Args:
        df: DataFrame to search.
        candidates: List of possible column names.

    Returns:
        The matching column name, or None if not found.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def load_sitegiant_export(file_path: Path) -> pd.DataFrame:
    """
    Load a SiteGiant export Excel file.

    Args:
        file_path: Path to the SiteGiant export file.

    Returns:
        DataFrame with the export data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
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
        raise ValueError(f"Unsupported file format: {suffix}")

    df = pd.read_excel(file_path, engine=engine)

    logger.info(f"Loaded {len(df)} rows from SiteGiant export")
    logger.debug(f"Columns found: {list(df.columns)}")

    return df


def create_mapping_template(sitegiant_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mapping template DataFrame from SiteGiant export data.

    Args:
        sitegiant_df: DataFrame with SiteGiant export data.

    Returns:
        DataFrame with template structure and pre-filled SiteGiant data.
    """
    # Find matching columns in the SiteGiant export
    col_name = find_column(sitegiant_df, SITEGIANT_COLUMN_MAP["name"])
    col_isku = find_column(sitegiant_df, SITEGIANT_COLUMN_MAP["isku"])
    col_sku = find_column(sitegiant_df, SITEGIANT_COLUMN_MAP["sku"])
    col_price = find_column(sitegiant_df, SITEGIANT_COLUMN_MAP["price"])

    # Validate required columns
    if col_sku is None:
        raise ValueError(
            f"Could not find SKU column. Available columns: {list(sitegiant_df.columns)}"
        )

    if col_name is None:
        logger.warning("Could not find product name column, will use empty values")

    if col_price is None:
        logger.warning("Could not find price column, will use empty values")

    logger.info(
        f"Column mapping: name='{col_name}', isku='{col_isku}', sku='{col_sku}', price='{col_price}'"
    )

    # Create template DataFrame
    template_df = pd.DataFrame(columns=TEMPLATE_COLUMNS)

    # Copy SiteGiant data (auto-filled)
    template_df["sku"] = sitegiant_df[col_sku]
    template_df["isku"] = sitegiant_df[col_isku] if col_isku else ""
    template_df["name"] = sitegiant_df[col_name] if col_name else ""
    template_df["price"] = sitegiant_df[col_price] if col_price else ""

    # Set defaults for Pokedata columns (to be filled manually)
    template_df["pokedata_id"] = ""
    template_df["pokedata_name"] = ""
    template_df["pokedata_url"] = ""
    template_df["pokedata_language"] = "ENGLISH"  # Default, change to JAPANESE as needed

    # Control columns
    template_df["auto_update"] = "Y"  # Default to allow updates
    template_df["notes"] = ""

    return template_df


def save_template(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the mapping template to an Excel file.

    Args:
        df: Template DataFrame to save.
        output_path: Path for the output file.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Excel
    df.to_excel(output_path, index=False, engine="openpyxl", sheet_name="Mapping")

    logger.info(f"Template saved to: {output_path}")


def get_config_paths() -> tuple[Path, Path]:
    """
    Get default input and output paths from config.

    Returns:
        Tuple of (default_input_path, default_output_path).
    """
    import yaml

    config_file = Path("config/config.yaml")
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
        paths = raw_config.get("paths", {})
        input_path = Path(paths.get("default_sitegiant_export", "data/input/sitegiant_export.xlsx"))
        output_path = Path(
            paths.get(
                "mapping_template_path", "data/mapping/sitegiant_pokedata_mapping_template.xlsx"
            )
        )
    else:
        input_path = Path("data/input/sitegiant_export.xlsx")
        output_path = Path("data/mapping/sitegiant_pokedata_mapping_template.xlsx")

    return input_path, output_path


def generate_template(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> int:
    """
    Generate a mapping template from a SiteGiant export file.

    Args:
        input_path: Path to the SiteGiant export file. Uses config default if not provided.
        output_path: Path for the output template. Uses config default if not provided.

    Returns:
        Number of rows written to the template.
    """
    # Load default paths from config
    default_input, default_output = get_config_paths()

    if input_path is None:
        input_path = default_input
        logger.info(f"Using default input path from config: {input_path}")

    if output_path is None:
        output_path = default_output
        logger.info(f"Using default output path from config: {output_path}")

    # Load SiteGiant export
    sitegiant_df = load_sitegiant_export(input_path)

    # Create template
    template_df = create_mapping_template(sitegiant_df)

    # Save template
    save_template(template_df, output_path)

    row_count = len(template_df)
    logger.info(f"Successfully created mapping template with {row_count} rows")

    return row_count


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a mapping template from a SiteGiant export file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.mapping.generate_mapping_template
    python -m src.mapping.generate_mapping_template data/input/sitegiant_export.xlsx
    python -m src.mapping.generate_mapping_template path/to/export.xlsx -o data/mapping/custom_template.xlsx
        """,
    )

    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the SiteGiant export Excel file (default: from config.yaml)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for the template (default: from config.yaml)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI script."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # Get paths from config for display
    default_input, default_output = get_config_paths()

    try:
        row_count = generate_template(
            input_path=args.input_file,
            output_path=args.output,
        )

        actual_input = args.input_file or default_input
        actual_output = args.output or default_output

        print("\n✓ Mapping template created successfully!")
        print(f"  Input file:  {actual_input}")
        print(f"  Output file: {actual_output}")
        print(f"  Rows written: {row_count}")
        print("\nNext steps:")
        print("  1. Open the template in Excel")
        print("  2. For each product, find it on https://www.pokedata.io/products")
        print("  3. Fill in these columns:")
        print("     - pokedata_id:   Enter the Pokedata product ID")
        print("     - pokedata_name: Copy the product name (for reference)")
        print("     - pokedata_url:  Paste the Pokedata product URL (optional)")
        print("  4. Set auto_update to 'N' for any products you want to skip")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n✗ Error: {e}")
        if args.input_file is None:
            print(f"  Hint: Place your SiteGiant export at {default_input}")
            print(
                "        Or provide a path: python -m src.mapping.generate_mapping_template path/to/export.xlsx"
            )
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
