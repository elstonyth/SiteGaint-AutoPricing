"""
CLI entry point for SiteGiant Pricing Automation Tool.

This module wires together all components and provides command-line
interface for running the pricing automation workflow.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict
from collections import Counter

from src.utils.logging_setup import setup_logging
from src.utils.config_loader import load_config, load_env, AppConfig
from src.importer.sitegiant_importer import SiteGiantImporter
from src.mapping.mapping_manager import MappingManager
from src.pokedata_client.api_client import PokedataClient
from src.pricing.fx_provider import FXProvider, get_fx_rate
from src.pricing.pricing_engine import PricingEngine, attach_pricing
from src.risk.threshold_engine import (
    ThresholdEngine,
    load_price_history,
    save_price_history,
    attach_pct_change_vs_last_run,
)
from src.exporter.sitegiant_exporter import (
    SiteGiantExporter,
    build_update_dataframe,
    write_update_file,
    add_include_in_update_column,
)


logger = logging.getLogger(__name__)


def get_default_paths(config: AppConfig) -> Dict[str, Path]:
    """Get default file paths from config."""
    # Use mapping_master_path from config if available
    mapping_path = getattr(config.paths, 'mapping_master_path', 'data/mapping/master_mapping.xlsx')
    
    return {
        "input": Path("data/input/sitegiant_export.xlsx"),
        "mapping": Path(mapping_path),
        "output_dir": Path("data/output"),
        "cache": Path("data/cache/previous_pokedata_prices.csv"),
        "price_history": Path("data/cache/price_history.csv"),
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="SiteGiant Pricing Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.main --input data/input/export.xlsx --mapping data/mapping/mapping.xlsx
    python -m src.main --fx-rate 4.7
    python -m src.main -v  # verbose mode
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to SiteGiant export Excel file (default: from config)",
    )
    
    parser.add_argument(
        "--mapping", "-m",
        type=Path,
        help="Path to Pokedata mapping Excel file (default: data/mapping/master_mapping.xlsx)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path for output Excel file (default: auto-generated in data/output)",
    )
    
    parser.add_argument(
        "--fx-rate", "-r",
        type=float,
        help="Manual USD to MYR exchange rate (default: from config)",
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI interface instead of CLI",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output file",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    """
    Run the CLI workflow.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    # 1. Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    default_paths = get_default_paths(config)
    
    # Resolve paths
    input_path = args.input or default_paths["input"]
    mapping_path = args.mapping or default_paths["mapping"]
    output_dir = args.output.parent if args.output else default_paths["output_dir"]
    
    logger.info(f"Input file: {input_path}")
    logger.info(f"Mapping file: {mapping_path}")
    
    # 2. Load SiteGiant export
    logger.info("Loading SiteGiant export...")
    importer = SiteGiantImporter(config)
    try:
        products_df = importer.import_products(input_path)
        logger.info(f"Loaded {len(products_df)} products")
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        print(f"\n✗ Error: Input file not found: {input_path}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input file: {e}")
        print(f"\n✗ Error: {e}")
        return 1
    
    # 3. Load mapping
    logger.info("Loading mapping file...")
    mapping_manager = MappingManager(config)
    try:
        mapping_manager.load_mapping_file(mapping_path)
        logger.info(f"Loaded {len(mapping_manager.mappings)} mappings")
    except FileNotFoundError as e:
        logger.error(f"Mapping file not found: {e}")
        print(f"\n✗ Error: Mapping file not found: {mapping_path}")
        print(f"\n  To create a mapping file:")
        print(f"  1. Run 'python -m src.mapping.generate_mapping_template' to generate a template")
        print(f"  2. Or use the web app: uvicorn src.webapp.main:app --reload")
        print(f"     Then go to Settings → Mapping to upload a mapping file")
        return 1
    
    # 4. Apply mapping
    logger.info("Applying mapping to products...")
    mapped_df = mapping_manager.join_with_products(products_df)
    mapped_count = mapped_df["is_mapped"].sum()
    logger.info(f"Mapped products for update: {mapped_count}/{len(mapped_df)}")
    
    # 5. Get unique Pokedata IDs to fetch
    unique_ids = mapped_df[mapped_df["is_mapped"]]["pokedata_id"].unique().tolist()
    unique_ids = [pid for pid in unique_ids if pid and pid != "nan"]
    logger.info(f"Unique Pokedata IDs to fetch: {len(unique_ids)}")
    
    # 6. Fetch Pokedata prices
    pokedata_prices = {}
    if unique_ids:
        api_key = os.environ.get("POKEDATA_API_KEY", "")
        if not api_key:
            logger.warning("POKEDATA_API_KEY not set - skipping API fetch")
            print("\n⚠ Warning: POKEDATA_API_KEY not set. Skipping Pokedata price fetch.")
            print("  Set the API key in your .env file to enable price fetching.")
        else:
            logger.info("Fetching Pokedata prices...")
            client = PokedataClient(config, api_key)
            pokedata_prices = client.get_prices_batch(unique_ids)
            
            # Count successful fetches
            successful = sum(1 for p in pokedata_prices.values() 
                           if p and p.primary_price_usd is not None)
            logger.info(f"Fetched prices: {successful}/{len(unique_ids)} successful")
    
    # 7. Get FX rate (returns tuple of rate and source)
    fx_rate, fx_source = get_fx_rate(config, manual_override=args.fx_rate)
    logger.info(f"FX rate (USD→MYR): {fx_rate:.4f} [source: {fx_source}]")
    
    # 8. Attach pricing (computes new_price_myr, abs_change, pct_change)
    logger.info("Calculating new prices...")
    priced_df = attach_pricing(mapped_df, pokedata_prices, fx_rate, config)
    
    # 9. Load price history and compute pct_change_vs_last_run (optional)
    price_history_path = default_paths["price_history"]
    history_df = load_price_history(price_history_path)
    if not history_df.empty:
        logger.info("Attaching pct_change_vs_last_run from price history...")
        priced_df = attach_pct_change_vs_last_run(priced_df, history_df)
    
    # 10. Apply thresholds (uses pct_change vs SiteGiant price)
    logger.info("Evaluating price thresholds...")
    threshold_engine = ThresholdEngine(config)
    results_df = threshold_engine.evaluate_batch(priced_df)
    
    # 11. Add include_in_update column (for GUI checkbox support)
    results_df = add_include_in_update_column(results_df)
    
    # 12. Print summary
    print("\n" + "="*60)
    print("PRICING SUMMARY")
    print("="*60)
    
    status_counts = Counter(results_df["status"])
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    print(f"\n  Total products: {len(results_df)}")
    print(f"  FX rate used: {fx_rate:.4f} ({fx_source})")
    
    # Count actual price updates (based on include_in_update)
    include_count = results_df["include_in_update"].sum()
    print(f"  Products to update: {include_count}")
    
    # Show pct_change stats if available
    if "pct_change" in results_df.columns:
        valid_pct = results_df["pct_change"].dropna()
        if len(valid_pct) > 0:
            print(f"  Avg pct_change: {valid_pct.mean():.2f}%")
    
    # 13. Build and write update file
    if args.dry_run:
        print("\n[DRY RUN] - No output file written")
    else:
        logger.info("Building update file...")
        update_df = build_update_dataframe(products_df, results_df)
        
        output_path = write_update_file(update_df, output_dir)
        
        print(f"\n✓ Update file written: {output_path}")
        
        # 14. Save price history for next run
        logger.info("Saving price history...")
        save_price_history(results_df, price_history_path)
    
    print("="*60 + "\n")
    
    return 0


def run_gui() -> int:
    """
    Launch the web application GUI.
    
    Returns:
        int: Exit code.
    """
    logger.info("Launching GUI...")
    print("\n" + "="*60)
    print("SiteGiant Pricing Automation - Web App")
    print("="*60)
    print("\nTo launch the web app, run:")
    print("  uvicorn src.webapp.main:app --reload")
    print("\nThen open: http://127.0.0.1:8000")
    print("="*60 + "\n")
    return 0


def main() -> int:
    """
    Main entry point.
    
    Returns:
        int: Exit code.
    """
    # Load environment variables
    load_env()
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("SiteGiant Pricing Automation Tool starting...")
    
    try:
        if args.gui:
            return run_gui()
        else:
            return run_cli(args)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n✗ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
