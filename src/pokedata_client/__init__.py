"""
Pokedata API client module.

Provides a client for interacting with the Pokedata API
to fetch pricing data for TCG products.
"""

from src.pokedata_client.api_client import (
    PokedataApiKeyError,
    PokedataAuthError,
    PokedataClient,
    PokedataClientError,
    PokedataRateLimitError,
)
from src.pokedata_client.models import (
    PokedataPriceData,
    PokedataPriceInfo,
    PokedataProduct,
    PokedataSearchResult,
    PriceHistoryEntry,
    PriceHistoryPoint,
)

__all__ = [
    "PokedataClient",
    "PokedataClientError",
    "PokedataAuthError",
    "PokedataApiKeyError",
    "PokedataRateLimitError",
    "PokedataProduct",
    "PokedataPriceData",
    "PokedataSearchResult",
    "PokedataPriceInfo",
    "PriceHistoryPoint",
    "PriceHistoryEntry",
]
