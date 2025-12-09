"""
Configuration loader module.

Loads application configuration from YAML files and environment variables.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """File path configuration."""

    input_dir: str = "data/input"
    output_dir: str = "data/output"
    mapping_dir: str = "data/mapping"
    cache_dir: str = "data/cache"
    logs_dir: str = "logs"
    default_mapping_file: str = "pokedata_mapping.xlsx"
    price_cache_file: str = "previous_pokedata_prices.csv"


@dataclass
class GoogleFXConfig:
    """Google Finance FX configuration."""

    base_currency: str = "USD"
    quote_currency: str = "MYR"
    pair_symbol: str = "USD-MYR"
    timeout_seconds: int = 10


@dataclass
class FXConfig:
    """FX conversion configuration."""

    mode: str = "google"  # "google" or "manual"
    default_rate: float = 4.70
    margin_divisor: float = 0.8
    google: GoogleFXConfig = field(default_factory=GoogleFXConfig)


@dataclass
class RoundingConfig:
    """Price rounding configuration."""

    decimal_places: int = 2
    method: str = "round"
    round_to_nearest: float | None = None


@dataclass
class ThresholdLevelConfig:
    """Single threshold level configuration."""

    vs_sitegiant_pct: float = 20.0
    vs_previous_pct: float = 15.0


@dataclass
class ThresholdsConfig:
    """Threshold configuration for soft and hard limits."""

    soft: ThresholdLevelConfig = field(default_factory=lambda: ThresholdLevelConfig(20.0, 15.0))
    hard: ThresholdLevelConfig = field(default_factory=lambda: ThresholdLevelConfig(50.0, 30.0))


@dataclass
class PokedataConfig:
    """Pokedata API configuration."""

    api_key_env: str = "POKEDATA_API_KEY"
    base_url: str = "https://www.pokedata.io"
    search_endpoint: str = "/v0/search"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 5


@dataclass
class SiteGiantColumnsConfig:
    """SiteGiant column name configuration."""

    sku: str = "SKU"
    product_name: str = "Product Name"
    current_price: str = "Price"
    stock: str = "Stock"
    status: str = "Status"


@dataclass
class SiteGiantFiltersConfig:
    """SiteGiant filter configuration."""

    in_stock_only: bool = False
    active_only: bool = True


@dataclass
class SiteGiantConfig:
    """SiteGiant configuration."""

    columns: SiteGiantColumnsConfig = field(default_factory=SiteGiantColumnsConfig)
    filters: SiteGiantFiltersConfig = field(default_factory=SiteGiantFiltersConfig)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class GUIConfig:
    """GUI configuration."""

    title: str = "SiteGiant Pricing Automation"
    page_size: int = 50
    theme: str = "light"


@dataclass
class MappingConfig:
    """Mapping file configuration."""

    # How to handle duplicate SKUs: "merge" (keep last) or "ignore" (keep first)
    duplicate_handling: str = "merge"


@dataclass
class AppConfig:
    """
    Main application configuration.

    Aggregates all configuration sections into a single object.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    fx: FXConfig = field(default_factory=FXConfig)
    rounding: RoundingConfig = field(default_factory=RoundingConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    pokedata: PokedataConfig = field(default_factory=PokedataConfig)
    sitegiant: SiteGiantConfig = field(default_factory=SiteGiantConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)


def load_env(env_file: Path = Path(".env")) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file.
    """
    if env_file.exists():
        load_dotenv(env_file)
        logger.debug(f"Loaded environment from: {env_file}")
    else:
        logger.debug(f"No .env file found at: {env_file}")


def load_config(config_file: Path = Path("config/config.yaml")) -> AppConfig:
    """
    Load application configuration from YAML file.

    Args:
        config_file: Path to configuration YAML file.

    Returns:
        AppConfig: Loaded configuration object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}. Using defaults.")
        return AppConfig()

    with open(config_file, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        return AppConfig()

    # Parse raw_config into AppConfig dataclass
    config = _parse_config(raw_config)
    logger.info(f"Loaded configuration from: {config_file}")
    return config


def _parse_config(raw: dict[str, Any]) -> AppConfig:
    """
    Parse raw YAML dict into AppConfig dataclass.

    Args:
        raw: Raw dictionary from YAML file.

    Returns:
        AppConfig: Parsed configuration object.
    """
    # Parse paths
    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        input_dir=paths_raw.get("sitegiant_export_dir", "data/input"),
        output_dir=paths_raw.get("output_dir", "data/output"),
        mapping_dir=paths_raw.get("mapping_dir", "data/mapping"),
        cache_dir=paths_raw.get("cache_dir", "data/cache"),
        logs_dir=paths_raw.get("logs_dir", "logs"),
        default_mapping_file=paths_raw.get("default_mapping_file", "pokedata_mapping.xlsx"),
        price_cache_file=paths_raw.get("price_cache_file", "previous_pokedata_prices.csv"),
    )
    # Add mapping_master_path as extra attribute
    if "mapping_master_path" in paths_raw:
        paths.mapping_master_path = paths_raw["mapping_master_path"]

    # Parse FX config
    fx_raw = raw.get("fx", {})
    google_raw = fx_raw.get("google", {})
    google_config = GoogleFXConfig(
        base_currency=google_raw.get("base_currency", "USD"),
        quote_currency=google_raw.get("quote_currency", "MYR"),
        pair_symbol=google_raw.get("pair_symbol", "USD-MYR"),
        timeout_seconds=google_raw.get("timeout_seconds", 10),
    )
    fx = FXConfig(
        mode=fx_raw.get("mode", "google"),
        default_rate=fx_raw.get("default_rate", 4.70),
        margin_divisor=fx_raw.get("margin_divisor", 0.8),
        google=google_config,
    )

    # Parse rounding config
    rounding_raw = raw.get("rounding", {})
    rounding = RoundingConfig(
        decimal_places=rounding_raw.get("decimal_places", 2),
        method=rounding_raw.get("method", "round"),
        round_to_nearest=rounding_raw.get("round_to_nearest"),
    )

    # Parse thresholds config
    thresholds_raw = raw.get("thresholds", {})
    soft_raw = thresholds_raw.get("soft", {})
    hard_raw = thresholds_raw.get("hard", {})
    thresholds = ThresholdsConfig(
        soft=ThresholdLevelConfig(
            vs_sitegiant_pct=soft_raw.get("vs_sitegiant_pct", 20.0),
            vs_previous_pct=soft_raw.get("vs_previous_pct", 15.0),
        ),
        hard=ThresholdLevelConfig(
            vs_sitegiant_pct=hard_raw.get("vs_sitegiant_pct", 50.0),
            vs_previous_pct=hard_raw.get("vs_previous_pct", 30.0),
        ),
    )

    # Parse pokedata config
    pokedata_raw = raw.get("pokedata", {})
    pokedata = PokedataConfig(
        api_key_env=pokedata_raw.get("api_key_env", "POKEDATA_API_KEY"),
        base_url=pokedata_raw.get("base_url", "https://www.pokedata.io"),
        search_endpoint=pokedata_raw.get("search_endpoint", "/v0/search"),
        timeout=pokedata_raw.get("timeout", 30),
        max_retries=pokedata_raw.get("max_retries", 3),
        retry_delay=pokedata_raw.get("retry_delay", 1.0),
        rate_limit=pokedata_raw.get("rate_limit", 5),
    )

    # Parse sitegiant config
    sitegiant_raw = raw.get("sitegiant", {})
    columns_raw = sitegiant_raw.get("columns", {})
    filters_raw = sitegiant_raw.get("filters", {})
    sitegiant = SiteGiantConfig(
        columns=SiteGiantColumnsConfig(
            sku=columns_raw.get("sku", "SKU"),
            product_name=columns_raw.get("product_name", "Product Name"),
            current_price=columns_raw.get("current_price", "Price"),
            stock=columns_raw.get("stock", "Stock"),
            status=columns_raw.get("status", "Status"),
        ),
        filters=SiteGiantFiltersConfig(
            in_stock_only=filters_raw.get("in_stock_only", False),
            active_only=filters_raw.get("active_only", True),
        ),
    )

    # Parse logging config
    logging_raw = raw.get("logging", {})
    logging_config = LoggingConfig(
        level=logging_raw.get("level", "INFO"),
        format=logging_raw.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )

    # Parse GUI config
    gui_raw = raw.get("gui", {})
    gui = GUIConfig(
        title=gui_raw.get("title", "SiteGiant Pricing Automation"),
        page_size=gui_raw.get("page_size", 50),
        theme=gui_raw.get("theme", "light"),
    )

    # Parse mapping config
    mapping_raw = raw.get("mapping", {})
    mapping = MappingConfig(
        duplicate_handling=mapping_raw.get("duplicate_handling", "merge"),
    )

    return AppConfig(
        paths=paths,
        fx=fx,
        rounding=rounding,
        thresholds=thresholds,
        pokedata=pokedata,
        sitegiant=sitegiant,
        logging=logging_config,
        gui=gui,
        mapping=mapping,
    )


def get_env_var(key: str, default: str | None = None) -> str | None:
    """
    Get an environment variable with optional default.

    Args:
        key: Environment variable name.
        default: Default value if not set.

    Returns:
        Environment variable value or default.
    """
    return os.environ.get(key, default)


def get_api_key() -> str | None:
    """
    Get the Pokedata API key from environment.

    Returns:
        API key if set, None otherwise.
    """
    return get_env_var("POKEDATA_API_KEY")
