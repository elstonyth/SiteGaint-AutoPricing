"""
Pricing engine module.

Handles USD→MYR conversion with configurable formula and rounding rules.

Formula: P_final = (P_usd × R) / margin_divisor
Where:
- P_usd = Pokedata price in USD
- R = USD to MYR exchange rate
- margin_divisor = profit margin divisor (default 0.8)
"""

import logging
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from typing import Any

import pandas as pd

from src.pricing.fx_provider import FXProvider
from src.utils.config_loader import AppConfig

logger = logging.getLogger(__name__)


class PricingEngine:
    """
    Engine for calculating MYR prices from USD prices.

    Applies:
    - FX conversion (USD → MYR)
    - Margin formula (P_myr / margin_divisor)
    - Rounding rules

    Attributes:
        config: Application configuration.
        fx_provider: FX rate provider instance.
        margin_divisor: Divisor for margin calculation (default 0.8).
    """

    def __init__(self, config: AppConfig, fx_provider: FXProvider) -> None:
        """
        Initialize the pricing engine.

        Args:
            config: Application configuration with rounding rules.
            fx_provider: FX provider for exchange rates.
        """
        self.config = config
        self.fx_provider = fx_provider
        # Get margin divisor from config (default 0.8 = 20% margin)
        margin_divisor = getattr(config.fx, "margin_divisor", 0.8)
        self.margin_divisor = Decimal(str(margin_divisor))

    def set_margin_divisor(self, divisor: float) -> None:
        """
        Set the margin divisor for price calculation.

        Args:
            divisor: New margin divisor (e.g., 0.8 = 20% margin).

        Raises:
            ValueError: If divisor is invalid.
        """
        if divisor <= 0 or divisor > 1:
            raise ValueError(f"Invalid margin divisor: {divisor}. Must be between 0 and 1.")
        self.margin_divisor = Decimal(str(divisor))

    def convert_usd_to_myr_raw(self, price_usd: float) -> Decimal:
        """
        Convert USD price to raw MYR (before margin).

        P_myr_raw = P_usd × R

        Args:
            price_usd: Price in USD.

        Returns:
            Decimal: Raw MYR price.
        """
        rate = self.fx_provider.get_rate()
        return Decimal(str(price_usd)) * rate

    def apply_margin(self, price_myr_raw: Decimal) -> Decimal:
        """
        Apply margin formula to raw MYR price.

        P_final = P_myr_raw / margin_divisor

        Args:
            price_myr_raw: Raw MYR price (after FX conversion).

        Returns:
            Decimal: Final price with margin applied.
        """
        return price_myr_raw / self.margin_divisor

    def round_price(
        self,
        price: Decimal,
        decimal_places: int = 2,
        method: str = "round",
        round_to_nearest: float | None = None,
    ) -> Decimal:
        """
        Apply rounding rules to a price.

        Args:
            price: Price to round.
            decimal_places: Number of decimal places.
            method: Rounding method (round, floor, ceil).
            round_to_nearest: Optional snap to nearest value (e.g., 0.05).

        Returns:
            Decimal: Rounded price.
        """
        if round_to_nearest:
            # Round to nearest increment (e.g., 0.05 for 5 sen)
            increment = Decimal(str(round_to_nearest))
            price = (price / increment).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * increment

        # Apply decimal places
        quantize_str = "0." + "0" * decimal_places if decimal_places > 0 else "1"

        if method == "floor":
            rounding = ROUND_DOWN
        elif method == "ceil":
            rounding = ROUND_UP
        else:
            rounding = ROUND_HALF_UP

        return price.quantize(Decimal(quantize_str), rounding=rounding)

    def calculate_final_price(
        self,
        price_usd: float,
        apply_rounding: bool = True,
    ) -> dict[str, Any]:
        """
        Calculate the final MYR price from USD.

        Full formula: P_final = round((P_usd × R) / margin_divisor)

        Args:
            price_usd: Pokedata price in USD.
            apply_rounding: Whether to apply rounding rules.

        Returns:
            Dict with:
                - price_usd: Original USD price
                - price_myr_raw: MYR price before margin
                - price_myr_final: Final MYR price
                - fx_rate: FX rate used
                - margin_divisor: Margin divisor used
        """
        fx_rate = self.fx_provider.get_rate()
        price_myr_raw = self.convert_usd_to_myr_raw(price_usd)
        price_myr_with_margin = self.apply_margin(price_myr_raw)

        if apply_rounding:
            # Get rounding settings from config
            decimal_places = getattr(self.config.rounding, "decimal_places", 2)
            method = getattr(self.config.rounding, "method", "round")
            round_to_nearest = getattr(self.config.rounding, "round_to_nearest", None)
            price_myr_final = self.round_price(
                price_myr_with_margin,
                decimal_places=decimal_places,
                method=method,
                round_to_nearest=round_to_nearest,
            )
        else:
            price_myr_final = price_myr_with_margin

        return {
            "price_usd": price_usd,
            "price_myr_raw": float(price_myr_raw),
            "price_myr_final": float(price_myr_final),
            "fx_rate": float(fx_rate),
            "margin_divisor": float(self.margin_divisor),
        }

    def calculate_prices_batch(
        self,
        df: pd.DataFrame,
        usd_price_column: str = "pokedata_price_usd",
        output_column: str = "new_price_myr",
    ) -> pd.DataFrame:
        """
        Calculate MYR prices for a batch of products.

        Args:
            df: DataFrame with USD prices.
            usd_price_column: Column name containing USD prices.
            output_column: Column name for calculated MYR prices.

        Returns:
            pd.DataFrame: DataFrame with new price column added.
        """
        df = df.copy()
        fx_rate = self.fx_provider.get_rate()

        # Get rounding settings from config
        decimal_places = getattr(self.config.rounding, "decimal_places", 2)
        rounding_method = getattr(self.config.rounding, "method", "round")
        round_to_nearest = getattr(self.config.rounding, "round_to_nearest", None)

        def calc_price(usd_price):
            if pd.isna(usd_price) or usd_price is None or usd_price == "":
                return None
            try:
                usd_price = float(usd_price)
                if usd_price <= 0:
                    return None

                # Apply formula: (USD * FX) / margin_divisor
                price_myr_raw = Decimal(str(usd_price)) * fx_rate
                price_myr_with_margin = price_myr_raw / self.margin_divisor

                # Round
                rounded = self.round_price(
                    price_myr_with_margin,
                    decimal_places=decimal_places,
                    method=rounding_method,
                    round_to_nearest=round_to_nearest,
                )
                return float(rounded)
            except (ValueError, TypeError):
                return None

        df[output_column] = df[usd_price_column].apply(calc_price)
        df["fx_rate_used"] = float(fx_rate)

        return df

    def get_pricing_summary(self, price_usd: float) -> str:
        """
        Get a human-readable summary of price calculation.

        Args:
            price_usd: USD price to summarize.

        Returns:
            str: Formatted pricing breakdown.
        """
        result = self.calculate_final_price(price_usd)
        return (
            f"USD ${result['price_usd']:.2f} × {result['fx_rate']:.4f} = "
            f"MYR {result['price_myr_raw']:.2f} ÷ {result['margin_divisor']} = "
            f"MYR {result['price_myr_final']:.2f}"
        )


def compute_new_price_myr(
    usd_price: float,
    fx_rate: float,
    margin_divisor: float = 0.8,
    decimal_places: int = 2,
) -> float:
    """
    Convenience function to compute new MYR price.

    Args:
        usd_price: Price in USD.
        fx_rate: USD to MYR exchange rate.
        margin_divisor: Margin divisor (default 0.8 = 20% margin).
        decimal_places: Decimal places for rounding.

    Returns:
        float: Final MYR price.
    """
    raw_myr = Decimal(str(usd_price)) * Decimal(str(fx_rate))
    final = raw_myr / Decimal(str(margin_divisor))

    quantize_str = "0." + "0" * decimal_places if decimal_places > 0 else "1"
    rounded = final.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    return float(rounded)


def attach_pricing(
    df: pd.DataFrame,
    pokedata_prices: dict[str, Any],
    fx_rate: float,
    config: AppConfig,
    old_price_column: str = "price",
) -> pd.DataFrame:
    """
    Attach Pokedata pricing to DataFrame and compute change metrics.

    Uses the SiteGiant export price (`price` column) as the "previous" price.
    Computes:
    - new_price_myr: Calculated new price from Pokedata USD + FX conversion
    - abs_change: new_price_myr - price (absolute difference)
    - pct_change: (abs_change / price) * 100 (percentage change)

    Args:
        df: DataFrame with product data (must have pokedata_id and price columns).
        pokedata_prices: Dict mapping pokedata_id -> PokedataPriceInfo.
        fx_rate: USD to MYR exchange rate.
        config: Application configuration.
        old_price_column: Column name for the current/old price (default: "price").

    Returns:
        pd.DataFrame: DataFrame with pricing and change columns added.
    """
    df = df.copy()

    # Get margin divisor from config
    margin_divisor = getattr(config.fx, "margin_divisor", 0.8)
    decimal_places = getattr(config.rounding, "decimal_places", 2)

    def get_usd_price(pokedata_id):
        if not pokedata_id or pokedata_id == "nan" or pd.isna(pokedata_id):
            return None
        price_info = pokedata_prices.get(str(pokedata_id))
        if price_info is None:
            return None
        # Handle both PokedataPriceInfo objects and dicts
        if hasattr(price_info, "primary_price_usd"):
            return price_info.primary_price_usd
        elif isinstance(price_info, dict):
            return price_info.get("primary_price_usd")
        return None

    # Add USD price column
    df["pokedata_price_usd"] = df["pokedata_id"].apply(get_usd_price)
    df["fx_rate_used"] = fx_rate

    # Calculate new MYR price
    def calc_myr(usd_price):
        if pd.isna(usd_price) or usd_price is None:
            return None
        return compute_new_price_myr(usd_price, fx_rate, margin_divisor, decimal_places)

    df["new_price_myr"] = df["pokedata_price_usd"].apply(calc_myr)

    # Compute change metrics: abs_change and pct_change vs SiteGiant price
    def calc_abs_change(row):
        new_price = row.get("new_price_myr")
        old_price = row.get(old_price_column)
        if pd.isna(new_price) or new_price is None:
            return None
        if pd.isna(old_price) or old_price is None:
            return None
        return round(float(new_price) - float(old_price), decimal_places)

    def calc_pct_change(row):
        new_price = row.get("new_price_myr")
        old_price = row.get(old_price_column)
        if pd.isna(new_price) or new_price is None:
            return None
        if pd.isna(old_price) or old_price is None or old_price <= 0:
            return None
        return round(((float(new_price) - float(old_price)) / float(old_price)) * 100, 2)

    df["abs_change"] = df.apply(calc_abs_change, axis=1)
    df["pct_change"] = df.apply(calc_pct_change, axis=1)

    return df
