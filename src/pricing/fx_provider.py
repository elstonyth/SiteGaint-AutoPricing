"""
FX rate provider module.

Retrieves USD→MYR exchange rates from manual input, Google Finance, or fallback default.
"""

import logging
import re
from typing import Optional, Tuple
from decimal import Decimal

import requests

from src.utils.config_loader import AppConfig


logger = logging.getLogger(__name__)


class FXProviderError(Exception):
    """Exception raised for FX rate retrieval errors."""
    pass


class FXProvider:
    """
    Provider for USD to MYR exchange rates.
    
    Supports:
    - Manual rate input
    - Google Finance live rate fetching
    - Fallback to default rate
    
    Attributes:
        config: Application configuration.
        current_rate: Currently active FX rate.
        source: Source of the current rate (manual/google/default).
    """
    
    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the FX provider.
        
        Args:
            config: Application configuration with FX settings.
        """
        self.config = config
        self.current_rate: Optional[Decimal] = None
        self.source: str = "default"
    
    def get_default_rate(self) -> Decimal:
        """
        Get the default FX rate from configuration.
        
        Returns:
            Decimal: Default USD to MYR rate.
        """
        default_rate = getattr(self.config.fx, 'default_rate', 4.70)
        return Decimal(str(default_rate))
    
    def set_manual_rate(self, rate: float) -> None:
        """
        Set a manual FX rate.
        
        Args:
            rate: USD to MYR exchange rate.
            
        Raises:
            ValueError: If rate is invalid (<=0).
        """
        if rate <= 0:
            raise ValueError(f"Invalid FX rate: {rate}. Must be positive.")
        
        self.current_rate = Decimal(str(rate))
        self.source = "manual_override"
        logger.info(f"Manual FX rate set: {self.current_rate}")
    
    def get_rate(self) -> Decimal:
        """
        Get the current FX rate.
        
        Returns the manually set rate, or fetches based on config mode.
        Falls back to default rate if fetch fails.
        
        Returns:
            Decimal: Current USD to MYR rate.
        """
        if self.current_rate is not None:
            return self.current_rate
        
        return self.get_default_rate()
    
    def fetch_rate_from_api(self) -> Decimal:
        """
        Fetch current FX rate from external API.
        
        Returns:
            Decimal: Live USD to MYR rate.
            
        Raises:
            FXProviderError: If API request fails.
        """
        raise NotImplementedError("fetch_rate_from_api not yet implemented")
    
    def validate_rate(self, rate: float, min_rate: float = 3.0, max_rate: float = 6.0) -> bool:
        """
        Validate that a rate is within reasonable bounds.
        
        Args:
            rate: Rate to validate.
            min_rate: Minimum acceptable rate.
            max_rate: Maximum acceptable rate.
            
        Returns:
            bool: True if rate is valid.
        """
        return min_rate <= rate <= max_rate
    
    def get_rate_info(self) -> dict:
        """
        Get information about the current rate.
        
        Returns:
            dict: Rate value, source, and timestamp.
        """
        return {
            "rate": float(self.get_rate()),
            "source": self.source,
            "is_default": self.current_rate is None,
        }


def fetch_google_fx_rate(config: AppConfig) -> Tuple[Optional[float], str]:
    """
    Fetch live FX rate from Google Finance.
    
    Args:
        config: Application configuration with google FX settings.
        
    Returns:
        Tuple of (rate, source):
            - (float, "google") if successful
            - (None, error_message) if failed
    """
    # Get config values
    fx_config = getattr(config, 'fx', None)
    if fx_config is None:
        return None, "No FX config found"
    
    google_config = getattr(fx_config, 'google', None)
    
    # Build pair symbol
    if google_config:
        pair_symbol = getattr(google_config, 'pair_symbol', 'USD-MYR')
        timeout = getattr(google_config, 'timeout_seconds', 10)
    else:
        pair_symbol = "USD-MYR"
        timeout = 10
    
    # Build Google Finance URL
    url = f"https://www.google.com/finance/quote/{pair_symbol}"
    
    logger.info(f"Fetching live FX rate from Google Finance: {url}")
    
    try:
        # Set headers to mimic browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        html = response.text
        
        # Try multiple regex patterns to extract the FX rate
        # Pattern 1: Look for data-last-price attribute
        patterns = [
            r'data-last-price="([0-9]+\.?[0-9]*)"',
            r'data-value="([0-9]+\.?[0-9]*)"',
            # Pattern 2: Look for the rate in a specific div structure
            r'class="YMlKec fxKbKc"[^>]*>([0-9]+\.?[0-9]*)',
            r'class="fxKbKc"[^>]*>([0-9]+\.?[0-9]*)',
            # Pattern 3: Look for rate near currency text
            r'USD.*?MYR.*?([0-9]+\.[0-9]{2,4})',
            r'([0-9]\.[0-9]{4})',  # Typical FX rate format like 4.4850
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html)
            for match in matches:
                try:
                    rate = float(match)
                    # Validate it's a reasonable USD-MYR rate (typically between 3.5 and 5.5)
                    if 3.0 <= rate <= 6.0:
                        logger.info(f"Successfully fetched Google FX rate: {rate}")
                        return rate, "google"
                except ValueError:
                    continue
        
        # If no valid rate found
        logger.warning("Could not parse FX rate from Google Finance response")
        return None, "Failed to parse rate from HTML"
        
    except requests.exceptions.Timeout:
        error_msg = f"Google Finance request timed out after {timeout}s"
        logger.warning(error_msg)
        return None, error_msg
        
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error fetching Google Finance: {e}"
        logger.warning(error_msg)
        return None, error_msg
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error from Google Finance: {e}"
        logger.warning(error_msg)
        return None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error fetching Google FX rate: {e}"
        logger.warning(error_msg)
        return None, error_msg


def get_fx_rate(config: AppConfig, manual_override: Optional[float] = None) -> Tuple[float, str]:
    """
    Get FX rate based on config mode and optional manual override.
    
    Priority:
    1. manual_override (if provided)
    2. Google Finance (if mode == "google")
    3. default_rate (fallback)
    
    Args:
        config: Application configuration.
        manual_override: Optional manual rate override.
        
    Returns:
        Tuple of (rate, source) where source is one of:
            - "manual_override"
            - "google"
            - "default"
    """
    # Get default rate from config
    fx_config = getattr(config, 'fx', None)
    if fx_config:
        default_rate = getattr(fx_config, 'default_rate', 4.70)
        mode = getattr(fx_config, 'mode', 'manual')
    else:
        default_rate = 4.70
        mode = 'manual'
    
    # Priority 1: Manual override
    if manual_override is not None:
        if manual_override <= 0:
            logger.warning(f"Invalid manual override {manual_override}, using default")
            return default_rate, "default"
        logger.info(f"Using manual FX rate override: {manual_override}")
        return manual_override, "manual_override"
    
    # Priority 2: Google Finance (if mode is google)
    if mode.lower() == "google":
        rate, source = fetch_google_fx_rate(config)
        if rate is not None:
            return rate, source
        else:
            logger.warning(f"Google FX fetch failed ({source}), falling back to default rate: {default_rate}")
            return default_rate, "default (google failed)"
    
    # Priority 3: Default rate (mode == "manual" or unknown)
    logger.info(f"Using default FX rate: {default_rate}")
    return default_rate, "default"
