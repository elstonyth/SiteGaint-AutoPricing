"""
Pokedata API client implementation.

Wrapper for Pokedata API calls including:
- Authentication
- Product search
- Pricing data retrieval
- Rate limiting and retry logic
"""

import logging
import os
import time
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.pokedata_client.models import (
    PokedataPriceData,
    PokedataPriceInfo,
    PokedataProduct,
    PokedataSearchResult,
)
from src.storage.price_cache import PriceCache, get_cache
from src.utils.config_loader import AppConfig

logger = logging.getLogger(__name__)


class PokedataClientError(Exception):
    """Base exception for Pokedata API errors."""

    pass


class PokedataAuthError(PokedataClientError):
    """Authentication error with Pokedata API."""

    pass


class PokedataRateLimitError(PokedataClientError):
    """Rate limit exceeded error."""

    pass


class PokedataApiKeyError(PokedataClientError):
    """API key is missing or invalid."""

    pass


class PokedataClient:
    """
    Client for interacting with the Pokedata API.

    Handles:
    - API authentication via API key
    - Product search by ID or name
    - Fetching current market prices
    - Rate limiting and request retries

    Attributes:
        config: Application configuration.
        api_key: Pokedata API key.
        base_url: Base URL for API requests.
        session: Requests session with retry logic.
    """

    def __init__(self, config: AppConfig, api_key: str | None = None) -> None:
        """
        Initialize the Pokedata API client.

        Args:
            config: Application configuration with API settings.
            api_key: Pokedata API key for authentication.
                     If None, reads from environment variable specified in config.

        Raises:
            PokedataApiKeyError: If no API key is provided or found in environment.
        """
        self.config = config
        self._history_store = None

        # Resolve API key: explicit > env variable from config > default env var
        if api_key:
            self.api_key = api_key
        else:
            # Try to get env var name from config, default to POKEDATA_API_KEY
            env_var_name = getattr(config, "pokedata", {})
            if hasattr(env_var_name, "api_key_env"):
                env_var_name = env_var_name.api_key_env
            elif isinstance(env_var_name, dict):
                env_var_name = env_var_name.get("api_key_env", "POKEDATA_API_KEY")
            else:
                env_var_name = "POKEDATA_API_KEY"

            self.api_key = os.environ.get(env_var_name, "")

        # Get base URL from config or env
        pokedata_config = getattr(config, "pokedata", None)
        if pokedata_config:
            # Handle both dict and object config
            if isinstance(pokedata_config, dict):
                self.base_url = pokedata_config.get("base_url", "")
            elif hasattr(pokedata_config, "base_url"):
                self.base_url = pokedata_config.base_url
            else:
                self.base_url = ""
        else:
            self.base_url = ""

        # Fallback to env var or default if not set
        if not self.base_url:
            self.base_url = os.environ.get("POKEDATA_BASE_URL", "https://www.pokedata.io")

        self.session = self._create_session()

        # Rate limiting state
        self._last_request_time: float = 0
        self._rate_limit_delay: float = 0.2  # 5 requests per second = 200ms between
        self._backoff_until: float = 0  # Timestamp until which we should wait
        self._consecutive_429s: int = 0  # Track consecutive rate limit hits
        self._max_backoff: float = 60.0  # Maximum backoff time in seconds

        # Initialize cache
        self._cache: PriceCache = get_cache()
        self._use_cache: bool = True  # Can be disabled for force refresh

    def has_api_key(self) -> bool:
        """Check if an API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def set_use_cache(self, use_cache: bool) -> None:
        """Enable or disable cache usage."""
        self._use_cache = use_cache
        logger.info(f"Cache {'enabled' if use_cache else 'disabled'}")

    def clear_cache(self) -> dict[str, int]:
        """Clear all cached data."""
        return self._cache.clear_all()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    def require_api_key(self) -> None:
        """
        Raise an error if no API key is configured.

        Raises:
            PokedataApiKeyError: If no API key is available.
        """
        if not self.has_api_key():
            raise PokedataApiKeyError(
                "Pokedata API key is required but not configured. "
                "Set the POKEDATA_API_KEY environment variable or provide an API key directly."
            )

    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic.

        Returns:
            requests.Session: Configured session object.
        """
        session = requests.Session()

        # Configure retry strategy - don't auto-retry 500 errors
        # (Pokedata returns 500 for auth issues)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)

        return session

    def _get_headers(self) -> dict[str, str]:
        """
        Get request headers including authentication.

        Returns:
            Dict[str, str]: Headers for API requests.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,  # Some APIs use this header
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _rate_limit(self) -> None:
        """
        Apply rate limiting between requests.

        Handles both regular rate limiting and exponential backoff
        after receiving 429 responses.
        """
        now = time.time()

        # Check if we're in a backoff period
        if now < self._backoff_until:
            wait_time = self._backoff_until - now
            logger.warning(f"Rate limit backoff: waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        # Apply normal rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)

        self._last_request_time = time.time()

    def _handle_rate_limit(self, retry_after: int | None = None) -> None:
        """
        Handle a rate limit response with exponential backoff.

        Args:
            retry_after: Optional Retry-After header value in seconds.
        """
        self._consecutive_429s += 1

        # Calculate backoff time: use Retry-After header if available,
        # otherwise use exponential backoff (2^n seconds, capped at max)
        if retry_after:
            backoff_time = float(retry_after)
        else:
            backoff_time = min(2**self._consecutive_429s, self._max_backoff)

        self._backoff_until = time.time() + backoff_time
        logger.warning(
            f"Rate limited (attempt {self._consecutive_429s}), "
            f"backing off for {backoff_time:.1f}s"
        )

    def _reset_backoff(self) -> None:
        """Reset backoff state after a successful request."""
        if self._consecutive_429s > 0:
            logger.info("Rate limit backoff reset after successful request")
        self._consecutive_429s = 0
        self._backoff_until = 0

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            params: Query parameters.
            data: Request body data.
            timeout: Request timeout in seconds.

        Returns:
            Dict[str, Any]: Parsed JSON response.

        Raises:
            PokedataAuthError: If authentication fails.
            PokedataRateLimitError: If rate limit exceeded.
            PokedataClientError: For other API errors.
        """
        max_retries = 3

        for attempt in range(max_retries):
            self._rate_limit()

            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()

            logger.info(f"Making request to: {url}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Params: {params}")

            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=data,
                    timeout=timeout,
                )
                logger.info(f"Response status: {response.status_code}")

                if response.status_code == 401:
                    raise PokedataAuthError("Invalid API key or authentication failed")

                if response.status_code == 500:
                    # Pokedata often returns 500 for auth issues
                    try:
                        error_body = response.json()
                        error_msg = error_body.get("message", "Internal Server Error")
                    except Exception:
                        error_msg = (
                            response.text[:200] if response.text else "Internal Server Error"
                        )
                    raise PokedataAuthError(f"API Error (500): {error_msg}. Check your API key.")

                if response.status_code == 429:
                    # Get Retry-After header if available
                    retry_after = response.headers.get("Retry-After")
                    retry_after_int = (
                        int(retry_after) if retry_after and retry_after.isdigit() else None
                    )

                    self._handle_rate_limit(retry_after_int)

                    # Retry if we have attempts left
                    if attempt < max_retries - 1:
                        logger.info(
                            f"Retrying request after rate limit (attempt {attempt + 2}/{max_retries})"
                        )
                        continue
                    else:
                        raise PokedataRateLimitError(
                            f"Rate limit exceeded after {max_retries} attempts. "
                            f"Try again in {self._backoff_until - time.time():.0f}s"
                        )

                response.raise_for_status()

                # Success - reset backoff
                self._reset_backoff()

                return response.json()

            except requests.exceptions.Timeout as e:
                logger.error(f"Request timeout for {endpoint}: {e}")
                raise PokedataClientError(f"Request timeout for {endpoint}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error for {endpoint}: {e}")
                # Check if this is a retry exhaustion
                if "Max retries exceeded" in str(e):
                    raise PokedataClientError(f"API unreachable (v2): {str(e)[:200]}")
                raise PokedataClientError(f"Network error (v2) for {endpoint}: {str(e)[:200]}")
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise PokedataClientError(f"HTTP error: {e}")
            except (PokedataAuthError, PokedataRateLimitError, PokedataClientError):
                # Re-raise our custom errors
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {type(e).__name__}: {e}")
                raise PokedataClientError(f"Unexpected error: {type(e).__name__}: {e}")

        # Should not reach here, but just in case
        raise PokedataClientError(f"Request failed after {max_retries} attempts")

    def verify_account(self) -> bool:
        """
        Verify API key and account status.

        Returns:
            bool: True if authentication is valid.

        Raises:
            PokedataAuthError: If authentication fails.
        """
        try:
            self._make_request("GET", "/v0/account")
            return True
        except PokedataAuthError:
            return False
        except PokedataClientError as e:
            logger.warning(f"Account verification failed: {e}")
            return False

    def get_product(self, product_id: str) -> PokedataProduct | None:
        """
        Get product details by Pokedata product ID.

        Args:
            product_id: The Pokedata product ID.

        Returns:
            PokedataProduct if found, None otherwise.
        """
        try:
            encoded_product_id = quote(str(product_id), safe="")
            data = self._make_request("GET", f"/v0/products/{encoded_product_id}")
            return PokedataProduct.from_api_response(data)
        except PokedataClientError as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            return None

    def get_pricing(
        self,
        product_id: str,
        asset_type: str = "PRODUCT",
        force_refresh: bool = False,
    ) -> PokedataPriceInfo:
        """
        Get current market price for a product.

        Prioritizes price sources:
        1. Pokedata Raw
        2. TCGPlayer
        3. eBay Raw
        4. Average of available sources

        Args:
            product_id: The Pokedata product ID.
            asset_type: Asset type (default: PRODUCT for sealed).
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            PokedataPriceInfo with current pricing (primary_price_usd may be None on error).
        """
        # Check cache first (unless force refresh)
        if self._use_cache and not force_refresh:
            cached = self._cache.get_price(str(product_id))
            if cached:
                logger.debug(f"Cache hit for {product_id}: ${cached.price_usd}")
                return PokedataPriceInfo(
                    product_id=product_id,
                    primary_price_usd=cached.price_usd,
                    source=f"{cached.source} (cached)",
                    raw_prices=cached.raw_prices,
                    cached=True,
                )

        try:
            # Pokedata API v0 pricing endpoint requires:
            # - id: numeric product ID (required)
            # - asset_type: CARD, PRODUCT, or MASTERSET (required)
            endpoint = "/v0/pricing"
            params = {"id": str(product_id), "asset_type": asset_type}

            data = self._make_request("GET", endpoint, params=params)

            # Parse response - Pokedata API v0 format:
            # { "id": 66, "pricing": { "Pokedata Raw": {"currency": "usd", "value": 389.65}, ... } }
            raw_prices = {}

            if isinstance(data, dict):
                # Get pricing dict from response
                pricing_data = data.get("pricing", {})

                if isinstance(pricing_data, dict):
                    # Each key is source name, value has "currency" and "value"
                    for source, price_info in pricing_data.items():
                        if isinstance(price_info, dict) and "value" in price_info:
                            value = price_info.get("value")
                            if value is not None and value != 0:
                                raw_prices[source] = float(value)
                        elif isinstance(price_info, (int, float)):
                            # Direct numeric value
                            if price_info != 0:
                                raw_prices[source] = float(price_info)

            # Determine primary price with priority
            primary_price = None
            source = None

            # Priority: Pokedata Raw > TCGPlayer > eBay Raw > any available
            priority_sources = [
                ("Pokedata Raw", ["pokedata_raw", "Pokedata Raw"]),
                ("TCGPlayer", ["tcgplayer", "TCGPlayer"]),
                ("eBay Raw", ["ebay_raw", "eBay Raw"]),
            ]

            for source_name, keys in priority_sources:
                for key in keys:
                    if key in raw_prices:
                        primary_price = raw_prices[key]
                        source = source_name
                        break
                if primary_price is not None:
                    break

            # Fallback: average of available prices
            if primary_price is None and raw_prices:
                primary_price = sum(raw_prices.values()) / len(raw_prices)
                source = "average"

            # If still no price, try direct price field
            if primary_price is None and isinstance(data, dict):
                direct_price = data.get("price", data.get("market_price"))
                if direct_price is not None:
                    primary_price = float(direct_price)
                    source = data.get("source", "direct")

            price_info = PokedataPriceInfo(
                product_id=product_id,
                primary_price_usd=primary_price,
                source=source,
                raw_prices=raw_prices,
                cached=False,
            )

            # Cache the price if valid
            if primary_price is not None:
                self._cache.set_price(
                    product_id=product_id,
                    price_usd=primary_price,
                    source=source,
                    raw_prices=raw_prices,
                )
                # Also record to history
                self._record_price_history(product_id, asset_type, primary_price)

            return price_info

        except PokedataClientError as e:
            logger.error(f"Failed to get pricing for {product_id}: {e}")
            return PokedataPriceInfo(
                product_id=product_id,
                primary_price_usd=None,
                error=str(e),
            )
        except Exception as e:
            logger.exception(f"Unexpected error getting pricing for {product_id}: {e}")
            return PokedataPriceInfo(
                product_id=product_id,
                primary_price_usd=None,
                error=str(e),
            )

    def _record_price_history(
        self,
        product_id: str,
        asset_type: str,
        price_usd: float,
    ) -> None:
        """
        Record a price to the history store.

        Args:
            product_id: Pokedata product ID.
            asset_type: Asset type.
            price_usd: Price in USD.
        """
        try:
            from src.storage.pokedata_history_store import record_price

            record_price(product_id, asset_type, price_usd)
        except Exception as e:
            logger.debug(f"Failed to record price history: {e}")

    def get_product_pricing(
        self,
        pokedata_id: str,
        asset_type: str = "PRODUCT",
    ) -> PokedataPriceInfo:
        """
        Get pricing for a specific product.

        This is an alias for get_pricing() for explicit naming.

        Args:
            pokedata_id: The Pokedata product ID.
            asset_type: Asset type (default: PRODUCT).

        Returns:
            PokedataPriceInfo with pricing details.
        """
        return self.get_pricing(pokedata_id, asset_type)

    def get_price(
        self,
        product_id: str,
        language: str = "ENGLISH",
    ) -> PokedataPriceData | None:
        """
        Get current market price for a product.

        Args:
            product_id: The Pokedata product ID.
            language: Product language (ENGLISH or JAPANESE).

        Returns:
            PokedataPriceData with current pricing, or None if not available.
        """
        price_info = self.get_pricing(product_id)
        if price_info.primary_price_usd is not None:
            return PokedataPriceData(
                product_id=product_id,
                price_usd=price_info.primary_price_usd,
                source=price_info.source,
            )
        return None

    def get_prices_batch(
        self,
        product_ids: list[str],
        language: str = "ENGLISH",
        max_workers: int = 5,
        force_refresh: bool = False,
    ) -> dict[str, PokedataPriceInfo]:
        """
        Get prices for multiple products with parallel execution.

        Uses ThreadPoolExecutor for concurrent API requests while
        respecting rate limits. Checks cache first to minimize API calls.

        Args:
            product_ids: List of Pokedata product IDs.
            language: Product language.
            max_workers: Maximum concurrent requests (default: 5).
            force_refresh: If True, bypass cache and fetch all fresh.

        Returns:
            Dict mapping product_id -> PokedataPriceInfo.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        valid_ids = [pid for pid in product_ids if pid and pid != "nan"]
        total = len(valid_ids)

        if total == 0:
            return results

        # Check cache first (unless force refresh)
        ids_to_fetch = valid_ids
        if self._use_cache and not force_refresh:
            cached_prices, missing_ids = self._cache.get_prices_batch(valid_ids)

            # Add cached results
            for pid, cached in cached_prices.items():
                results[pid] = PokedataPriceInfo(
                    product_id=pid,
                    primary_price_usd=cached.price_usd,
                    source=f"{cached.source} (cached)",
                    raw_prices=cached.raw_prices,
                    cached=True,
                )

            ids_to_fetch = missing_ids
            cache_hits = len(cached_prices)

            if cache_hits > 0:
                logger.info(
                    f"Cache: {cache_hits}/{total} hits, " f"{len(ids_to_fetch)} API calls needed"
                )

        # Fetch remaining from API
        if not ids_to_fetch:
            logger.info(f"All {total} prices served from cache!")
            return results

        logger.info(f"Fetching {len(ids_to_fetch)} prices from API (max {max_workers} concurrent)")

        def fetch_single(product_id: str) -> tuple:
            """Fetch a single product price."""
            return product_id, self.get_pricing(product_id, force_refresh=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(fetch_single, pid): pid for pid in ids_to_fetch}

            completed = 0
            for future in as_completed(future_to_id):
                product_id = future_to_id[future]
                completed += 1
                try:
                    pid, price_info = future.result()
                    results[pid] = price_info
                    if completed % 10 == 0 or completed == total:
                        logger.debug(f"Progress: {completed}/{total} prices fetched")
                except Exception as e:
                    logger.error(f"Error fetching {product_id}: {e}")
                    results[product_id] = PokedataPriceInfo(
                        product_id=product_id,
                        primary_price_usd=None,
                        error=str(e),
                    )

        logger.info(f"Completed fetching {len(results)} prices")
        return results

    def search_products(
        self,
        query: str,
        language: str | None = None,
        limit: int = 10,
    ) -> list[PokedataSearchResult]:
        """
        Search for products by name or other criteria.

        Args:
            query: Search query string.
            language: Optional language filter.
            limit: Maximum number of results (default: 10).

        Returns:
            List[PokedataSearchResult]: Matching products.
        """
        self.require_api_key()

        try:
            # Pokedata API uses 'query' param and requires 'asset_type'
            params = {
                "query": query,
                "asset_type": "PRODUCT",  # Default to PRODUCT for sealed products
            }
            if language:
                params["language"] = language

            # Get search endpoint from config or use default
            pokedata_config = getattr(self.config, "pokedata", None)
            if pokedata_config:
                # Handle both dict and object config
                if isinstance(pokedata_config, dict):
                    search_endpoint = pokedata_config.get("search_endpoint", "/v0/search")
                elif hasattr(pokedata_config, "search_endpoint"):
                    search_endpoint = pokedata_config.search_endpoint
                else:
                    search_endpoint = "/v0/search"
            else:
                search_endpoint = "/v0/search"

            logger.info(f"Calling search endpoint: {search_endpoint} with params: {params}")
            data = self._make_request("GET", search_endpoint, params=params)
            logger.info(
                f"Search API response type: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}"
            )
            logger.info(f"Search API response (truncated): {str(data)[:500]}")

            results = []

            # Handle different response formats
            if isinstance(data, list):
                # Response is a direct JSON array
                items = data
            elif isinstance(data, dict):
                # Response is wrapped in an object
                items = data.get("results", data.get("data", data.get("products", [])))
            else:
                items = []

            logger.info(f"Found {len(items)} items in response")

            for item in items:
                results.append(PokedataSearchResult.from_api_response(item))

            return results

        except PokedataClientError as e:
            logger.error(f"Product search failed: {e}")
            raise  # Re-raise so the UI can show the error
