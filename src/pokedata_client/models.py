"""
Data models for Pokedata API responses.

Contains typed dataclasses/models for API response parsing and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ProductLanguage(str, Enum):
    """Product language options."""

    ENGLISH = "ENGLISH"
    JAPANESE = "JAPANESE"


class AssetType(str, Enum):
    """Product asset type categories."""

    BOOSTER_BOX = "Booster Box"
    ELITE_TRAINER_BOX = "Elite Trainer Box"
    BOOSTER_PACK = "Booster Pack"
    COLLECTION_BOX = "Collection Box"
    TIN = "Tin"
    BLISTER = "Blister"
    OTHER = "Other"


@dataclass
class PokedataPriceInfo:
    """
    Simplified price info returned by the client.

    Attributes:
        product_id: Pokedata product ID.
        primary_price_usd: Primary/best price in USD (may be None if unavailable).
        source: Price data source (e.g., "Pokedata Raw", "TCGPlayer", "eBay Raw").
        raw_prices: Dict of all available prices by source.
        error: Error message if fetch failed.
        cached: True if returned from cache (no API call made).
    """

    product_id: str
    primary_price_usd: float | None = None
    source: str | None = None
    raw_prices: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    cached: bool = False


@dataclass
class PokedataPriceData:
    """
    Price data from Pokedata API.

    Attributes:
        product_id: Pokedata product ID.
        price_usd: Current market price in USD.
        price_low: Low market price in USD.
        price_high: High market price in USD.
        price_average: Average market price in USD.
        last_updated: Timestamp of last price update.
        source: Price data source/marketplace.
    """

    product_id: str
    price_usd: float
    price_low: float | None = None
    price_high: float | None = None
    price_average: float | None = None
    last_updated: datetime | None = None
    source: str | None = None

    @classmethod
    def from_api_response(cls, product_id: str, data: dict[str, Any]) -> "PokedataPriceData":
        """
        Create PokedataPriceData from API response.

        Args:
            product_id: The product ID.
            data: Raw API response dictionary.

        Returns:
            PokedataPriceData: Parsed price data object.
        """
        # Parse common price fields
        price_usd = data.get("price", data.get("market_price", 0.0))

        return cls(
            product_id=product_id,
            price_usd=float(price_usd) if price_usd else 0.0,
            price_low=data.get("price_low"),
            price_high=data.get("price_high"),
            price_average=data.get("price_average"),
            source=data.get("source", "unknown"),
        )


@dataclass
class PokedataProduct:
    """
    Product information from Pokedata API.

    Attributes:
        product_id: Unique Pokedata product ID.
        name: Product name/title.
        language: Product language (ENGLISH/JAPANESE).
        asset_type: Product category/type.
        set_name: TCG set name.
        release_date: Product release date.
        image_url: Product image URL.
        pokedata_url: URL to Pokedata product page.
    """

    product_id: str
    name: str
    language: str
    asset_type: str
    set_name: str | None = None
    release_date: datetime | None = None
    image_url: str | None = None
    pokedata_url: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PokedataProduct":
        """
        Create PokedataProduct from API response.

        Args:
            data: Raw API response dictionary.

        Returns:
            PokedataProduct: Parsed product object.
        """
        return cls(
            product_id=str(data.get("id", data.get("product_id", ""))),
            name=data.get("name", ""),
            language=data.get("language", "ENGLISH"),
            asset_type=data.get("asset_type", data.get("type", "PRODUCT")),
            set_name=data.get("set_name", data.get("set")),
            image_url=data.get("image_url", data.get("image")),
            pokedata_url=data.get("url", data.get("pokedata_url")),
        )


@dataclass
class PokedataSearchResult:
    """
    Search result from Pokedata product search.

    Attributes:
        product_id: Pokedata product ID.
        name: Product name.
        language: Product language.
        asset_type: Product type.
        url: URL to product page.
        relevance_score: Search relevance score.
    """

    product_id: str
    name: str
    language: str | None = None
    asset_type: str = "PRODUCT"
    url: str | None = None
    relevance_score: float | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PokedataSearchResult":
        """
        Create PokedataSearchResult from API response.

        Args:
            data: Raw API response dictionary.

        Returns:
            PokedataSearchResult: Parsed search result.
        """
        return cls(
            product_id=str(data.get("id", data.get("product_id", ""))),
            name=data.get("name", ""),
            language=data.get("language"),
            asset_type=data.get("asset_type", data.get("type", "PRODUCT")),
            url=data.get("url", data.get("pokedata_url")),
            relevance_score=data.get("score"),
        )


@dataclass
class PriceHistoryEntry:
    """
    Historical price entry for trend analysis.

    Attributes:
        date: Date of the price record.
        price_usd: Price in USD on that date.
        product_id: Pokedata product ID.
    """

    date: datetime
    price_usd: float
    product_id: str


@dataclass
class PriceHistoryPoint:
    """
    Simple price history point for display.

    Attributes:
        date: Date of the price record (as string YYYY-MM-DD).
        price_usd: Price in USD on that date.
    """

    date: str
    price_usd: float
