"""
Tests for the pricing engine module.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock

from src.pricing.pricing_engine import PricingEngine
from src.pricing.fx_provider import FXProvider
from src.utils.config_loader import AppConfig


class TestPricingEngine:
    """Tests for PricingEngine class."""
    
    @pytest.fixture
    def config(self) -> AppConfig:
        """Create test configuration."""
        return AppConfig()
    
    @pytest.fixture
    def fx_provider(self, config: AppConfig) -> FXProvider:
        """Create test FX provider with fixed rate."""
        provider = FXProvider(config)
        provider.set_manual_rate(4.50)  # Fixed rate for testing
        return provider
    
    @pytest.fixture
    def engine(self, config: AppConfig, fx_provider: FXProvider) -> PricingEngine:
        """Create test pricing engine."""
        return PricingEngine(config, fx_provider)
    
    def test_engine_initialization(self, engine: PricingEngine) -> None:
        """Test engine initializes correctly."""
        assert engine is not None
        assert engine.margin_divisor == Decimal("0.8")
    
    def test_set_margin_divisor_valid(self, engine: PricingEngine) -> None:
        """Test setting a valid margin divisor."""
        engine.set_margin_divisor(0.75)
        assert engine.margin_divisor == Decimal("0.75")
    
    def test_set_margin_divisor_invalid_zero(self, engine: PricingEngine) -> None:
        """Test setting zero margin divisor raises error."""
        with pytest.raises(ValueError):
            engine.set_margin_divisor(0)
    
    def test_set_margin_divisor_invalid_negative(self, engine: PricingEngine) -> None:
        """Test setting negative margin divisor raises error."""
        with pytest.raises(ValueError):
            engine.set_margin_divisor(-0.5)
    
    def test_set_margin_divisor_invalid_greater_than_one(
        self,
        engine: PricingEngine,
    ) -> None:
        """Test setting margin divisor > 1 raises error."""
        with pytest.raises(ValueError):
            engine.set_margin_divisor(1.5)
    
    def test_convert_usd_to_myr_raw(self, engine: PricingEngine) -> None:
        """Test USD to MYR raw conversion."""
        # $100 USD × 4.50 = 450 MYR
        result = engine.convert_usd_to_myr_raw(100.0)
        assert result == Decimal("450")
    
    def test_apply_margin(self, engine: PricingEngine) -> None:
        """Test margin application."""
        # 450 MYR ÷ 0.8 = 562.50 MYR
        result = engine.apply_margin(Decimal("450"))
        assert result == Decimal("562.5")
    
    def test_round_price_default(self, engine: PricingEngine) -> None:
        """Test default rounding (2 decimal places)."""
        result = engine.round_price(Decimal("123.456"))
        assert result == Decimal("123.46")
    
    def test_round_price_floor(self, engine: PricingEngine) -> None:
        """Test floor rounding."""
        result = engine.round_price(Decimal("123.456"), method="floor")
        assert result == Decimal("123.45")
    
    def test_round_price_ceil(self, engine: PricingEngine) -> None:
        """Test ceiling rounding."""
        result = engine.round_price(Decimal("123.451"), method="ceil")
        assert result == Decimal("123.46")
    
    def test_round_price_to_nearest(self, engine: PricingEngine) -> None:
        """Test rounding to nearest increment."""
        # Round to nearest 0.05
        result = engine.round_price(
            Decimal("123.47"),
            round_to_nearest=0.05,
        )
        assert result == Decimal("123.45")
    
    def test_calculate_final_price(self, engine: PricingEngine) -> None:
        """Test full price calculation."""
        # $100 USD → 450 MYR raw → 562.50 MYR final
        result = engine.calculate_final_price(100.0)
        
        assert result["price_usd"] == 100.0
        assert result["price_myr_raw"] == 450.0
        assert result["price_myr_final"] == 562.5
        assert result["fx_rate"] == 4.5
        assert result["margin_divisor"] == 0.8
    
    def test_get_pricing_summary(self, engine: PricingEngine) -> None:
        """Test pricing summary generation."""
        summary = engine.get_pricing_summary(100.0)
        
        assert "USD $100.00" in summary
        assert "4.5000" in summary
        assert "MYR 562.50" in summary


class TestPricingEngineEdgeCases:
    """Edge case tests for PricingEngine."""
    
    @pytest.fixture
    def engine(self) -> PricingEngine:
        """Create test engine."""
        config = AppConfig()
        fx_provider = FXProvider(config)
        fx_provider.set_manual_rate(4.50)
        return PricingEngine(config, fx_provider)
    
    def test_zero_price(self, engine: PricingEngine) -> None:
        """Test calculation with zero price."""
        result = engine.calculate_final_price(0.0)
        assert result["price_myr_final"] == 0.0
    
    def test_small_price(self, engine: PricingEngine) -> None:
        """Test calculation with very small price."""
        result = engine.calculate_final_price(0.01)
        assert result["price_myr_final"] > 0
    
    def test_large_price(self, engine: PricingEngine) -> None:
        """Test calculation with large price."""
        result = engine.calculate_final_price(10000.0)
        assert result["price_myr_final"] == 56250.0
