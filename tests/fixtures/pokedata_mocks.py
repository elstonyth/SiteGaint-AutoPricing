"""
Mock responses for Pokedata API calls.

Use with the `responses` library to mock HTTP requests in tests.
"""

import json
from typing import Any, Dict, List, Optional

import responses

# Pokedata API base URL
POKEDATA_BASE_URL = "https://www.pokedata.io"
POKEDATA_SEARCH_ENDPOINT = f"{POKEDATA_BASE_URL}/v0/search"


# Sample product data
SAMPLE_PRODUCTS = {
    "66": {
        "id": 66,
        "name": "Scarlet & Violet Booster Box",
        "type": "PRODUCT",
        "language": "ENGLISH",
        "prices": {
            "tcgplayer": {"market": 89.65, "low": 85.00, "mid": 90.00, "high": 95.00},
        },
        "primary_price_usd": 89.65,
    },
    "89": {
        "id": 89,
        "name": "Pokemon 151 Japanese Booster Box",
        "type": "PRODUCT",
        "language": "JAPANESE",
        "prices": {
            "tcgplayer": {"market": 75.50, "low": 70.00, "mid": 76.00, "high": 80.00},
        },
        "primary_price_usd": 75.50,
    },
    "123": {
        "id": 123,
        "name": "Obsidian Flames Booster Box",
        "type": "PRODUCT",
        "language": "ENGLISH",
        "prices": {
            "tcgplayer": {"market": 95.00, "low": 90.00, "mid": 95.00, "high": 100.00},
        },
        "primary_price_usd": 95.00,
    },
}


def get_sample_product(product_id: str) -> Optional[Dict[str, Any]]:
    """Get sample product data by ID."""
    return SAMPLE_PRODUCTS.get(str(product_id))


def mock_pokedata_search_response(product_id: str) -> Dict[str, Any]:
    """Build a mock Pokedata search API response."""
    product = get_sample_product(product_id)
    if product:
        return {
            "success": True,
            "data": product,
        }
    else:
        return {
            "success": False,
            "error": f"Product not found: {product_id}",
        }


@responses.activate
def setup_pokedata_mocks(product_ids: Optional[List[str]] = None):
    """
    Set up mock responses for Pokedata API.
    
    Use as a decorator or context manager with @responses.activate
    
    Args:
        product_ids: List of product IDs to mock. If None, mocks all sample products.
    """
    ids_to_mock = product_ids or list(SAMPLE_PRODUCTS.keys())
    
    for pid in ids_to_mock:
        response_data = mock_pokedata_search_response(pid)
        
        # Mock the search endpoint for this product
        responses.add(
            responses.GET,
            POKEDATA_SEARCH_ENDPOINT,
            json=response_data,
            status=200 if response_data.get("success") else 404,
            match=[responses.matchers.query_param_matcher({"id": pid})],
        )


def add_pokedata_mock(product_id: str, price_usd: Optional[float] = None):
    """
    Add a single Pokedata mock response.
    
    Call this within a @responses.activate block.
    
    Args:
        product_id: Product ID to mock.
        price_usd: Custom price to return. If None, uses sample data.
    """
    product = get_sample_product(product_id)
    
    if product and price_usd is not None:
        product = product.copy()
        product["primary_price_usd"] = price_usd
    
    if product:
        response_data = {"success": True, "data": product}
        status = 200
    else:
        response_data = {"success": False, "error": f"Product not found: {product_id}"}
        status = 404
    
    responses.add(
        responses.GET,
        POKEDATA_SEARCH_ENDPOINT,
        json=response_data,
        status=status,
    )


def add_pokedata_error_mock(error_message: str = "Internal server error", status_code: int = 500):
    """Add a mock that returns an error response."""
    responses.add(
        responses.GET,
        POKEDATA_SEARCH_ENDPOINT,
        json={"success": False, "error": error_message},
        status=status_code,
    )


def add_pokedata_timeout_mock():
    """Add a mock that simulates a timeout."""
    responses.add(
        responses.GET,
        POKEDATA_SEARCH_ENDPOINT,
        body=responses.ConnectionError("Connection timed out"),
    )


# Batch response helper
def mock_batch_prices(product_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Generate mock batch price data.
    
    Returns a dict mapping product_id -> price data.
    """
    result = {}
    for pid in product_ids:
        product = get_sample_product(pid)
        if product:
            result[pid] = {
                "product_id": pid,
                "primary_price_usd": product["primary_price_usd"],
                "source": "tcgplayer",
            }
    return result
