"""
Pricing module.

Handles FX rate retrieval and USD→MYR price conversion with configurable formulas.
Supports live FX rates from Google Finance with fallback to manual/default rates.
"""

from src.pricing.fx_provider import FXProvider, get_fx_rate, fetch_google_fx_rate
from src.pricing.pricing_engine import PricingEngine, compute_new_price_myr, attach_pricing

__all__ = [
    "FXProvider",
    "PricingEngine",
    "get_fx_rate",
    "fetch_google_fx_rate",
    "compute_new_price_myr",
    "attach_pricing",
]
