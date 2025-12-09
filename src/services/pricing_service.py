"""
Pricing Service for SiteGiant Pricing Automation.

Handles the core price processing logic, extracted from routes for:
- Better testability
- Separation of concerns
- Reusability
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.exporter.sitegiant_exporter import add_include_in_update_column
from src.mapping.mapping_manager import apply_mapping
from src.pokedata_client.api_client import PokedataClient
from src.pricing.fx_provider import get_fx_rate
from src.pricing.pricing_engine import attach_pricing
from src.risk.threshold_engine import ThresholdEngine
from src.utils.config_loader import AppConfig
from src.webapp.helpers import normalize_columns, resolve_api_key

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for price processing."""

    fx_rate: float | None = None
    soft_threshold: float = 20.0
    hard_threshold: float = 50.0
    margin_divisor: float = 0.8
    api_key_override: str | None = None
    force_refresh: bool = False


@dataclass
class ProcessingResult:
    """Result of price processing."""

    results_df: pd.DataFrame
    fx_rate_used: float
    fx_source: str
    demo_mode: bool
    demo_warning: str | None = None
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if processing was successful (not demo mode)."""
        return not self.demo_mode


class PricingService:
    """
    Service for processing SiteGiant prices with Pokedata data.

    Handles:
    - Column normalization
    - Mapping application
    - Pokedata price fetching (or demo mode simulation)
    - FX conversion
    - Threshold evaluation
    """

    def __init__(self, app_config: AppConfig):
        """
        Initialize pricing service.

        Args:
            app_config: Application configuration.
        """
        self.app_config = app_config
        self.logger = logging.getLogger(f"{__name__}.PricingService")

    def process(
        self,
        sitegiant_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        config: ProcessingConfig,
    ) -> ProcessingResult:
        """
        Process prices for SiteGiant products.

        Args:
            sitegiant_df: SiteGiant export DataFrame.
            mapping_df: Mapping DataFrame (SKU → Pokedata ID).
            config: Processing configuration.

        Returns:
            ProcessingResult with processed DataFrame and metadata.
        """
        self.logger.info(f"Processing {len(sitegiant_df)} products")

        # Step 1: Normalize column names
        df = normalize_columns(sitegiant_df)

        # Step 2: Apply mapping
        df = apply_mapping(df, mapping_df, sku_column="sku")

        # Step 3: Get FX rate
        fx_rate, fx_source = self._resolve_fx_rate(config.fx_rate)

        # Step 4: Fetch Pokedata prices (or use demo mode)
        pokedata_prices, demo_mode, demo_warning = self._fetch_prices(df, config, fx_rate)

        # Step 5: Attach pricing
        df = attach_pricing(df, pokedata_prices, fx_rate, self.app_config)

        # Step 6: Apply threshold engine
        df = self._apply_thresholds(df, config)

        # Step 7: Add include_in_update column
        df = add_include_in_update_column(df)

        # Store metadata in DataFrame
        df["fx_rate_used"] = fx_rate
        df["fx_source"] = fx_source
        df["demo_mode"] = demo_mode
        df["demo_mode_warning"] = demo_warning or ""

        # Calculate stats
        stats = self._calculate_stats(df)

        self.logger.info(
            f"Processing complete: {stats.get('total', 0)} products, "
            f"{stats.get('ok', 0)} OK, {stats.get('warning', 0)} warnings, "
            f"demo_mode={demo_mode}"
        )

        return ProcessingResult(
            results_df=df,
            fx_rate_used=fx_rate,
            fx_source=fx_source,
            demo_mode=demo_mode,
            demo_warning=demo_warning,
            stats=stats,
        )

    def _resolve_fx_rate(self, override: float | None) -> tuple[float, str]:
        """Resolve FX rate from override or config."""
        if override and override > 0:
            return override, "manual_override"
        return get_fx_rate(self.app_config)

    def _fetch_prices(
        self,
        df: pd.DataFrame,
        config: ProcessingConfig,
        fx_rate: float,
    ) -> tuple[dict[str, Any], bool, str | None]:
        """
        Fetch prices from Pokedata API or generate demo prices.

        Returns:
            Tuple of (prices_dict, is_demo_mode, demo_warning_message)
        """
        api_key = resolve_api_key(config.api_key_override)
        pokedata_prices = {}
        demo_mode = True
        demo_warning = None

        if api_key:
            try:
                client = PokedataClient(self.app_config, api_key)
                unique_ids = self._get_unique_pokedata_ids(df)

                if unique_ids:
                    pokedata_prices = client.get_prices_batch(
                        unique_ids, force_refresh=config.force_refresh
                    )
                    valid_prices = sum(
                        1
                        for p in pokedata_prices.values()
                        if hasattr(p, "primary_price_usd") and p.primary_price_usd is not None
                    )
                    if valid_prices > 0:
                        demo_mode = False
                        self.logger.info(f"Fetched {valid_prices} prices from Pokedata")
            except Exception as e:
                self.logger.warning(f"Pokedata API error: {e}. Using demo mode.")

        # Demo mode: generate simulated prices
        if demo_mode:
            demo_warning = (
                "DEMO MODE ACTIVE: Using simulated random prices. "
                "These are NOT real market prices. Configure your Pokedata API key "
                "in Settings to fetch real prices."
            )
            self.logger.warning(demo_warning)
            pokedata_prices = self._generate_demo_prices(df, config.margin_divisor, fx_rate)

        return pokedata_prices, demo_mode, demo_warning

    def _get_unique_pokedata_ids(self, df: pd.DataFrame) -> list[str]:
        """Extract unique Pokedata IDs from mapped rows."""
        if "is_mapped" not in df.columns or "pokedata_id" not in df.columns:
            return []

        unique_ids = df[df["is_mapped"]]["pokedata_id"].unique().tolist()
        return [pid for pid in unique_ids if pid and str(pid).strip() and str(pid).lower() != "nan"]

    def _generate_demo_prices(
        self,
        df: pd.DataFrame,
        margin_divisor: float,
        fx_rate: float,
    ) -> dict[str, Any]:
        """Generate simulated prices for demo mode."""
        demo_prices = {}

        if "is_mapped" not in df.columns:
            return demo_prices

        for _, row in df[df["is_mapped"]].iterrows():
            pid = str(row.get("pokedata_id", ""))
            if not pid or pid.lower() == "nan":
                continue

            current_price = row.get("price", 100)
            if pd.notna(current_price) and float(current_price) > 0:
                # Simulate price with +/- 10% variance
                variance = random.uniform(-0.1, 0.1)
                simulated_usd = (float(current_price) * margin_divisor / fx_rate) * (1 + variance)

                # Create a simple object with the price attribute
                demo_prices[pid] = type(
                    "DemoPrice", (), {"primary_price_usd": simulated_usd, "source": "DEMO"}
                )()

        return demo_prices

    def _apply_thresholds(
        self,
        df: pd.DataFrame,
        config: ProcessingConfig,
    ) -> pd.DataFrame:
        """Apply threshold engine to evaluate price changes."""
        threshold_engine = ThresholdEngine(self.app_config)
        threshold_engine.thresholds.soft_vs_sitegiant_pct = config.soft_threshold
        threshold_engine.thresholds.hard_vs_sitegiant_pct = config.hard_threshold
        return threshold_engine.evaluate_batch(df)

    def _calculate_stats(self, df: pd.DataFrame) -> dict[str, int]:
        """Calculate status counts from results DataFrame."""
        stats = {"total": len(df)}

        if "status" in df.columns:
            status_counts = df["status"].value_counts()
            stats["ok"] = int(status_counts.get("OK", 0))
            stats["warning"] = int(status_counts.get("WARNING", 0))
            stats["blocked"] = int(status_counts.get("BLOCKED", 0))
            stats["no_data"] = int(status_counts.get("NO_DATA", 0))
            stats["unmapped"] = int(status_counts.get("UNMAPPED", 0))

        if "include_in_update" in df.columns:
            stats["to_update"] = int(df["include_in_update"].sum())

        return stats

    def apply_edited_prices(
        self,
        df: pd.DataFrame,
        edited_prices: dict[str, float],
    ) -> tuple[pd.DataFrame, int]:
        """
        Apply user-edited prices to results DataFrame.

        Args:
            df: Results DataFrame.
            edited_prices: Dict of SKU → new price.

        Returns:
            Tuple of (updated DataFrame, count of edits applied).
        """
        if not edited_prices or "sku" not in df.columns:
            return df, 0

        df = df.copy()
        edited_count = 0

        for sku, new_price in edited_prices.items():
            mask = df["sku"] == sku
            if not mask.any():
                continue

            old_price = df.loc[mask, "new_price_myr"].iloc[0]
            if pd.isna(old_price) or abs(float(old_price) - new_price) <= 0.01:
                continue

            df.loc[mask, "new_price_myr"] = new_price

            # Recalculate abs_change and pct_change
            current_price = df.loc[mask, "price"].iloc[0]
            if pd.notna(current_price) and float(current_price) > 0:
                df.loc[mask, "abs_change"] = new_price - float(current_price)
                df.loc[mask, "pct_change"] = (
                    (new_price - float(current_price)) / float(current_price)
                ) * 100

            edited_count += 1

        if edited_count > 0:
            self.logger.info(f"Applied {edited_count} manual price edits")

        return df, edited_count
