"""
Threshold engine for price change risk assessment.

Checks price changes against soft and hard thresholds to determine
if updates should be allowed, warned, or blocked.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.risk.status_codes import ChangeDirection, PriceStatus
from src.utils.config_loader import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """
    Configuration for price change thresholds.

    Attributes:
        soft_vs_sitegiant_pct: Soft threshold % vs current SiteGiant price.
        soft_vs_previous_pct: Soft threshold % vs yesterday's Pokedata price.
        hard_vs_sitegiant_pct: Hard threshold % vs current SiteGiant price.
        hard_vs_previous_pct: Hard threshold % vs yesterday's Pokedata price.
    """

    soft_vs_sitegiant_pct: float = 20.0
    soft_vs_previous_pct: float = 15.0
    hard_vs_sitegiant_pct: float = 50.0
    hard_vs_previous_pct: float = 30.0


@dataclass
class ThresholdResult:
    """
    Result of threshold checking for a single product.

    Attributes:
        status: Overall status (OK, WARNING, BLOCKED, etc.).
        change_pct_vs_sitegiant: Percentage change vs current SiteGiant price.
        change_pct_vs_previous: Percentage change vs previous Pokedata price.
        direction: Direction of price change.
        reasons: List of reasons for the status.
    """

    status: PriceStatus
    change_pct_vs_sitegiant: float | None = None
    change_pct_vs_previous: float | None = None
    direction: ChangeDirection = ChangeDirection.UNKNOWN
    reasons: list[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class ThresholdEngine:
    """
    Engine for evaluating price changes against thresholds.

    Compares:
    1. New price vs current SiteGiant price
    2. New Pokedata price vs previous day's Pokedata price (spike detection)

    Assigns status codes based on soft/hard thresholds.

    Attributes:
        config: Application configuration.
        thresholds: Threshold configuration.
        previous_prices: Cache of previous Pokedata prices for spike detection.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the threshold engine.

        Args:
            config: Application configuration with threshold settings.
        """
        self.config = config
        # Load thresholds from config
        self.thresholds = self._load_thresholds_from_config(config)
        self.previous_prices: dict[str, float] = {}

    def _load_thresholds_from_config(self, config: AppConfig) -> ThresholdConfig:
        """
        Load threshold configuration from AppConfig.

        Args:
            config: Application configuration with threshold settings.

        Returns:
            ThresholdConfig: Parsed threshold configuration.
        """
        # Get thresholds from config, with fallback to defaults
        thresholds_config = getattr(config, "thresholds", None)
        if thresholds_config is None:
            return ThresholdConfig()

        soft = getattr(thresholds_config, "soft", None)
        hard = getattr(thresholds_config, "hard", None)

        return ThresholdConfig(
            soft_vs_sitegiant_pct=getattr(soft, "vs_sitegiant_pct", 20.0) if soft else 20.0,
            soft_vs_previous_pct=getattr(soft, "vs_previous_pct", 15.0) if soft else 15.0,
            hard_vs_sitegiant_pct=getattr(hard, "vs_sitegiant_pct", 50.0) if hard else 50.0,
            hard_vs_previous_pct=getattr(hard, "vs_previous_pct", 30.0) if hard else 30.0,
        )

    def load_previous_prices(self, cache_file: Path) -> None:
        """
        Load previous Pokedata prices from cache file.

        Used for spike detection by comparing today's prices to yesterday's.

        Args:
            cache_file: Path to CSV file with historical prices.
        """
        cache_file = Path(cache_file)
        if not cache_file.exists():
            logger.info(f"No previous price cache found at: {cache_file}")
            return

        logger.info(f"Loading previous prices from: {cache_file}")

        try:
            df = pd.read_csv(cache_file)

            # Get latest price for each pokedata_id
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date", ascending=False)
                df = df.drop_duplicates(subset=["pokedata_id"], keep="first")

            for _, row in df.iterrows():
                pokedata_id = str(row.get("pokedata_id", ""))
                price_usd = row.get("price_usd")
                if pokedata_id and pd.notna(price_usd):
                    self.previous_prices[pokedata_id] = float(price_usd)

            logger.info(f"Loaded {len(self.previous_prices)} previous prices")

        except Exception as e:
            logger.warning(f"Failed to load previous prices: {e}")

    def save_current_prices(self, prices: dict[str, float], cache_file: Path) -> None:
        """
        Save current prices to cache for future comparison.

        Args:
            prices: Dictionary of pokedata_id -> price_usd.
            cache_file: Path to save the cache file.
        """
        from datetime import datetime

        cache_file = Path(cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")

        rows = []
        for pokedata_id, price_usd in prices.items():
            if price_usd is not None:
                rows.append(
                    {
                        "date": today,
                        "pokedata_id": pokedata_id,
                        "price_usd": price_usd,
                    }
                )

        if not rows:
            logger.info("No prices to save to cache")
            return

        new_df = pd.DataFrame(rows)

        # Append to existing file or create new
        if cache_file.exists():
            existing_df = pd.read_csv(cache_file)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_csv(cache_file, index=False)
        logger.info(f"Saved {len(rows)} prices to cache: {cache_file}")

    def get_previous_price(self, pokedata_id: str) -> float | None:
        """
        Get the previous price for a Pokedata product ID.

        Args:
            pokedata_id: Pokedata product ID.

        Returns:
            Previous USD price if available, None otherwise.
        """
        return self.previous_prices.get(pokedata_id)

    def calculate_change_percent(
        self,
        old_price: float,
        new_price: float,
    ) -> float:
        """
        Calculate percentage change between two prices.

        Args:
            old_price: Original price.
            new_price: New price.

        Returns:
            Percentage change (positive = increase, negative = decrease).
        """
        if old_price == 0:
            return 0.0 if new_price == 0 else 100.0

        return ((new_price - old_price) / old_price) * 100

    def get_change_direction(self, old_price: float, new_price: float) -> ChangeDirection:
        """
        Determine the direction of price change.

        Args:
            old_price: Original price.
            new_price: New price.

        Returns:
            ChangeDirection enum value.
        """
        if new_price > old_price:
            return ChangeDirection.INCREASE
        elif new_price < old_price:
            return ChangeDirection.DECREASE
        else:
            return ChangeDirection.NO_CHANGE

    def check_thresholds(
        self,
        new_price_myr: float,
        current_sitegiant_price: float | None,
        pokedata_id: str | None = None,
        current_pokedata_price_usd: float | None = None,
    ) -> ThresholdResult:
        """
        Check a price change against all thresholds.

        Args:
            new_price_myr: Calculated new MYR price.
            current_sitegiant_price: Current price in SiteGiant (MYR).
            pokedata_id: Product ID for looking up previous price.
            current_pokedata_price_usd: Current Pokedata price for spike detection.

        Returns:
            ThresholdResult with status and details.
        """
        reasons = []
        status = PriceStatus.OK

        change_vs_sitegiant = None
        change_vs_previous = None
        direction = ChangeDirection.UNKNOWN

        # Check vs current SiteGiant price
        if current_sitegiant_price is not None and current_sitegiant_price > 0:
            change_vs_sitegiant = self.calculate_change_percent(
                current_sitegiant_price, new_price_myr
            )
            direction = self.get_change_direction(current_sitegiant_price, new_price_myr)

            abs_change = abs(change_vs_sitegiant)

            if abs_change >= self.thresholds.hard_vs_sitegiant_pct:
                status = PriceStatus.BLOCKED
                reasons.append(
                    f"Change vs SiteGiant ({change_vs_sitegiant:.1f}%) exceeds "
                    f"hard threshold ({self.thresholds.hard_vs_sitegiant_pct}%)"
                )
            elif abs_change >= self.thresholds.soft_vs_sitegiant_pct:
                if status != PriceStatus.BLOCKED:
                    status = PriceStatus.WARNING
                reasons.append(
                    f"Change vs SiteGiant ({change_vs_sitegiant:.1f}%) exceeds "
                    f"soft threshold ({self.thresholds.soft_vs_sitegiant_pct}%)"
                )

        # Check vs previous Pokedata price (spike detection)
        if pokedata_id and current_pokedata_price_usd:
            previous_price = self.get_previous_price(pokedata_id)
            if previous_price is not None:
                change_vs_previous = self.calculate_change_percent(
                    previous_price, current_pokedata_price_usd
                )

                abs_change = abs(change_vs_previous)

                if abs_change >= self.thresholds.hard_vs_previous_pct:
                    status = PriceStatus.BLOCKED
                    reasons.append(
                        f"Pokedata price spike ({change_vs_previous:.1f}%) exceeds "
                        f"hard threshold ({self.thresholds.hard_vs_previous_pct}%)"
                    )
                elif abs_change >= self.thresholds.soft_vs_previous_pct:
                    if status != PriceStatus.BLOCKED:
                        status = PriceStatus.WARNING
                    reasons.append(
                        f"Pokedata price spike ({change_vs_previous:.1f}%) exceeds "
                        f"soft threshold ({self.thresholds.soft_vs_previous_pct}%)"
                    )

        return ThresholdResult(
            status=status,
            change_pct_vs_sitegiant=change_vs_sitegiant,
            change_pct_vs_previous=change_vs_previous,
            direction=direction,
            reasons=reasons,
        )

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        new_price_col: str = "new_price_myr",
        current_price_col: str = "price",
        pokedata_id_col: str = "pokedata_id",
        pokedata_price_col: str = "pokedata_price_usd",
        is_mapped_col: str = "is_mapped",
        pct_change_col: str = "pct_change",
        abs_change_col: str = "abs_change",
    ) -> pd.DataFrame:
        """
        Evaluate thresholds for a batch of products.

        Uses the pct_change column (computed from SiteGiant price as "previous")
        to determine OK/WARNING/BLOCKED status based on soft/hard thresholds.

        Status logic:
        - UNMAPPED: Product not mapped or auto_update != Y
        - NO_DATA: No Pokedata price available
        - BLOCKED: |pct_change| >= hard_threshold
        - WARNING: soft_threshold <= |pct_change| < hard_threshold
        - OK: |pct_change| < soft_threshold

        Args:
            df: DataFrame with price data (must have pct_change, abs_change pre-computed).
            new_price_col: Column with new MYR prices.
            current_price_col: Column with current SiteGiant prices.
            pokedata_id_col: Column with Pokedata IDs.
            pokedata_price_col: Column with current Pokedata USD prices.
            is_mapped_col: Column indicating if row is mapped.
            pct_change_col: Column with pre-computed percentage change.
            abs_change_col: Column with pre-computed absolute change.

        Returns:
            pd.DataFrame: Original DataFrame with status and status_reason columns added.
        """
        df = df.copy()

        statuses = []
        reasons = []

        for _idx, row in df.iterrows():
            is_mapped = row.get(is_mapped_col, False)
            new_price = row.get(new_price_col)
            pct_change = row.get(pct_change_col)

            # Check mapping status first
            if not is_mapped:
                statuses.append(PriceStatus.UNMAPPED.value)
                reasons.append("Product not mapped or auto_update != Y")
                continue

            # Check if we have pricing data
            if pd.isna(new_price) or new_price is None:
                statuses.append(PriceStatus.NO_DATA.value)
                reasons.append("No Pokedata price available")
                continue

            # Check pct_change against thresholds
            if pd.isna(pct_change) or pct_change is None:
                # No pct_change means no current price to compare
                statuses.append(PriceStatus.OK.value)
                reasons.append("No current price for comparison")
                continue

            abs_pct = abs(pct_change)

            # Apply threshold logic: BLOCKED > WARNING > OK
            if abs_pct >= self.thresholds.hard_vs_sitegiant_pct:
                statuses.append(PriceStatus.BLOCKED.value)
                reasons.append(
                    f"Change {pct_change:.1f}% exceeds hard threshold "
                    f"({self.thresholds.hard_vs_sitegiant_pct}%)"
                )
            elif abs_pct >= self.thresholds.soft_vs_sitegiant_pct:
                statuses.append(PriceStatus.WARNING.value)
                reasons.append(
                    f"Change {pct_change:.1f}% exceeds soft threshold "
                    f"({self.thresholds.soft_vs_sitegiant_pct}%)"
                )
            else:
                statuses.append(PriceStatus.OK.value)
                reasons.append("")

        df["status"] = statuses
        df["status_reason"] = reasons

        return df

    def get_threshold_summary(self) -> dict[str, Any]:
        """
        Get current threshold configuration.

        Returns:
            Dict with all threshold values.
        """
        return {
            "soft_vs_sitegiant_pct": self.thresholds.soft_vs_sitegiant_pct,
            "soft_vs_previous_pct": self.thresholds.soft_vs_previous_pct,
            "hard_vs_sitegiant_pct": self.thresholds.hard_vs_sitegiant_pct,
            "hard_vs_previous_pct": self.thresholds.hard_vs_previous_pct,
        }


def compute_change_metrics(old_price: float, new_price: float) -> dict[str, float | None]:
    """
    Compute change metrics between two prices.

    Args:
        old_price: Original price.
        new_price: New price.

    Returns:
        Dict with delta and pct_change.
    """
    if old_price is None or new_price is None:
        return {"delta": None, "pct_change": None}

    delta = new_price - old_price
    pct_change = ((new_price - old_price) / old_price * 100) if old_price != 0 else 0.0

    return {"delta": delta, "pct_change": pct_change}


def load_price_history(cache_file: Path) -> pd.DataFrame:
    """
    Load price history from cache file.

    Args:
        cache_file: Path to price_history.csv.

    Returns:
        pd.DataFrame: DataFrame with sku, pokedata_id, last_new_price_myr, run_date.
    """
    cache_file = Path(cache_file)
    if not cache_file.exists():
        logger.info(f"No price history cache found at: {cache_file}")
        return pd.DataFrame(columns=["sku", "pokedata_id", "last_new_price_myr", "run_date"])

    try:
        df = pd.read_csv(cache_file)
        logger.info(f"Loaded price history: {len(df)} records")
        return df
    except Exception as e:
        logger.warning(f"Failed to load price history: {e}")
        return pd.DataFrame(columns=["sku", "pokedata_id", "last_new_price_myr", "run_date"])


def save_price_history(
    results_df: pd.DataFrame,
    cache_file: Path,
    sku_col: str = "sku",
    pokedata_id_col: str = "pokedata_id",
    new_price_col: str = "new_price_myr",
) -> None:
    """
    Save current prices to price history cache.

    Args:
        results_df: DataFrame with pricing results.
        cache_file: Path to save price_history.csv.
        sku_col: SKU column name.
        pokedata_id_col: Pokedata ID column name.
        new_price_col: New price column name.
    """
    from datetime import datetime

    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")

    rows = []
    for _, row in results_df.iterrows():
        sku = row.get(sku_col)
        pokedata_id = row.get(pokedata_id_col)
        new_price = row.get(new_price_col)

        if pd.notna(new_price) and new_price is not None:
            rows.append(
                {
                    "sku": sku,
                    "pokedata_id": str(pokedata_id) if pd.notna(pokedata_id) else "",
                    "last_new_price_myr": new_price,
                    "run_date": today,
                }
            )

    if not rows:
        logger.info("No prices to save to history cache")
        return

    new_df = pd.DataFrame(rows)
    new_df.to_csv(cache_file, index=False)
    logger.info(f"Saved {len(rows)} prices to history cache: {cache_file}")


def attach_pct_change_vs_last_run(
    df: pd.DataFrame,
    history_df: pd.DataFrame,
    new_price_col: str = "new_price_myr",
    sku_col: str = "sku",
) -> pd.DataFrame:
    """
    Attach pct_change_vs_last_run column by comparing to price history.

    Args:
        df: DataFrame with current pricing results.
        history_df: DataFrame with price history from last run.
        new_price_col: Column with new MYR prices.
        sku_col: SKU column for joining.

    Returns:
        pd.DataFrame: DataFrame with pct_change_vs_last_run column added.
    """
    df = df.copy()

    if history_df.empty:
        df["pct_change_vs_last_run"] = None
        df["last_new_price_myr"] = None
        return df

    # Build lookup dict from history
    history_lookup = {}
    for _, row in history_df.iterrows():
        sku = row.get("sku")
        if sku:
            history_lookup[sku] = row.get("last_new_price_myr")

    def calc_pct_vs_last(row):
        sku = row.get(sku_col)
        new_price = row.get(new_price_col)
        last_price = history_lookup.get(sku)

        if pd.isna(new_price) or new_price is None:
            return None
        if last_price is None or pd.isna(last_price) or last_price <= 0:
            return None

        return round(((float(new_price) - float(last_price)) / float(last_price)) * 100, 2)

    def get_last_price(row):
        sku = row.get(sku_col)
        return history_lookup.get(sku)

    df["last_new_price_myr"] = df.apply(get_last_price, axis=1)
    df["pct_change_vs_last_run"] = df.apply(calc_pct_vs_last, axis=1)

    return df


def evaluate_status_for_row(
    row: dict[str, Any],
    thresholds: ThresholdConfig,
    is_mapped_key: str = "is_mapped",
    new_price_key: str = "new_price_myr",
    old_price_key: str = "price",
    pokedata_price_key: str = "pokedata_price_usd",
) -> dict[str, Any]:
    """
    Evaluate status for a single row.

    Args:
        row: Dict-like row data.
        thresholds: Threshold configuration.
        is_mapped_key: Key for mapping status.
        new_price_key: Key for new MYR price.
        old_price_key: Key for current SiteGiant price.
        pokedata_price_key: Key for Pokedata USD price.

    Returns:
        Dict with status, status_reason, pct_change, abs_change.
    """
    is_mapped = row.get(is_mapped_key, False)
    new_price = row.get(new_price_key)
    old_price = row.get(old_price_key)

    # Check mapping first
    if not is_mapped:
        return {
            "status": PriceStatus.UNMAPPED.value,
            "status_reason": "Product not mapped or auto_update != Y",
            "pct_change": None,
            "abs_change": None,
        }

    # Check for data
    if new_price is None or (isinstance(new_price, float) and pd.isna(new_price)):
        return {
            "status": PriceStatus.NO_DATA.value,
            "status_reason": "No Pokedata price available",
            "pct_change": None,
            "abs_change": None,
        }

    # Calculate changes
    if old_price is not None and old_price > 0:
        metrics = compute_change_metrics(old_price, new_price)
        pct_change = metrics["pct_change"]
        abs_change = metrics["delta"]

        abs_pct = abs(pct_change) if pct_change else 0

        # Check thresholds
        if abs_pct >= thresholds.hard_vs_sitegiant_pct:
            status = PriceStatus.BLOCKED.value
            reason = f"Change {pct_change:.1f}% exceeds hard threshold {thresholds.hard_vs_sitegiant_pct}%"
        elif abs_pct >= thresholds.soft_vs_sitegiant_pct:
            status = PriceStatus.WARNING.value
            reason = f"Change {pct_change:.1f}% exceeds soft threshold {thresholds.soft_vs_sitegiant_pct}%"
        else:
            status = PriceStatus.OK.value
            reason = ""

        return {
            "status": status,
            "status_reason": reason,
            "pct_change": pct_change,
            "abs_change": abs_change,
        }

    return {
        "status": PriceStatus.OK.value,
        "status_reason": "No current price for comparison",
        "pct_change": None,
        "abs_change": None,
    }
