"""
Tests for the threshold engine module.
"""


import pytest

from src.risk.status_codes import ChangeDirection, PriceStatus
from src.risk.threshold_engine import ThresholdConfig, ThresholdEngine, ThresholdResult
from src.utils.config_loader import AppConfig


class TestThresholdConfig:
    """Tests for ThresholdConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default threshold values."""
        config = ThresholdConfig()

        assert config.soft_vs_sitegiant_pct == 20.0
        assert config.soft_vs_previous_pct == 15.0
        assert config.hard_vs_sitegiant_pct == 50.0
        assert config.hard_vs_previous_pct == 30.0

    def test_custom_values(self) -> None:
        """Test custom threshold values."""
        config = ThresholdConfig(
            soft_vs_sitegiant_pct=10.0,
            soft_vs_previous_pct=5.0,
            hard_vs_sitegiant_pct=25.0,
            hard_vs_previous_pct=15.0,
        )

        assert config.soft_vs_sitegiant_pct == 10.0
        assert config.hard_vs_sitegiant_pct == 25.0


class TestThresholdResult:
    """Tests for ThresholdResult dataclass."""

    def test_ok_result(self) -> None:
        """Test creating an OK result."""
        result = ThresholdResult(
            status=PriceStatus.OK,
            change_pct_vs_sitegiant=5.0,
            direction=ChangeDirection.INCREASE,
        )

        assert result.status == PriceStatus.OK
        assert result.reasons == []

    def test_warning_result_with_reasons(self) -> None:
        """Test creating a warning result with reasons."""
        result = ThresholdResult(
            status=PriceStatus.WARNING,
            change_pct_vs_sitegiant=25.0,
            direction=ChangeDirection.INCREASE,
            reasons=["Change exceeds soft threshold"],
        )

        assert result.status == PriceStatus.WARNING
        assert len(result.reasons) == 1


class TestThresholdEngine:
    """Tests for ThresholdEngine class."""

    @pytest.fixture
    def config(self) -> AppConfig:
        """Create test configuration."""
        return AppConfig()

    @pytest.fixture
    def engine(self, config: AppConfig) -> ThresholdEngine:
        """Create test threshold engine."""
        return ThresholdEngine(config)

    def test_engine_initialization(self, engine: ThresholdEngine) -> None:
        """Test engine initializes correctly."""
        assert engine is not None
        assert engine.thresholds is not None

    def test_calculate_change_percent_increase(
        self,
        engine: ThresholdEngine,
    ) -> None:
        """Test calculating percentage increase."""
        # 100 → 120 = 20% increase
        result = engine.calculate_change_percent(100.0, 120.0)
        assert result == 20.0

    def test_calculate_change_percent_decrease(
        self,
        engine: ThresholdEngine,
    ) -> None:
        """Test calculating percentage decrease."""
        # 100 → 80 = -20% decrease
        result = engine.calculate_change_percent(100.0, 80.0)
        assert result == -20.0

    def test_calculate_change_percent_no_change(
        self,
        engine: ThresholdEngine,
    ) -> None:
        """Test calculating with no change."""
        result = engine.calculate_change_percent(100.0, 100.0)
        assert result == 0.0

    def test_calculate_change_percent_from_zero(
        self,
        engine: ThresholdEngine,
    ) -> None:
        """Test calculating from zero base price."""
        result = engine.calculate_change_percent(0.0, 100.0)
        assert result == 100.0

    def test_get_change_direction_increase(
        self,
        engine: ThresholdEngine,
    ) -> None:
        """Test direction detection for increase."""
        result = engine.get_change_direction(100.0, 120.0)
        assert result == ChangeDirection.INCREASE

    def test_get_change_direction_decrease(
        self,
        engine: ThresholdEngine,
    ) -> None:
        """Test direction detection for decrease."""
        result = engine.get_change_direction(100.0, 80.0)
        assert result == ChangeDirection.DECREASE

    def test_get_change_direction_no_change(
        self,
        engine: ThresholdEngine,
    ) -> None:
        """Test direction detection for no change."""
        result = engine.get_change_direction(100.0, 100.0)
        assert result == ChangeDirection.NO_CHANGE

    def test_check_thresholds_ok(self, engine: ThresholdEngine) -> None:
        """Test threshold check returns OK for small change."""
        result = engine.check_thresholds(
            new_price_myr=105.0,
            current_sitegiant_price=100.0,
        )

        assert result.status == PriceStatus.OK
        assert result.change_pct_vs_sitegiant == 5.0
        assert result.direction == ChangeDirection.INCREASE

    def test_check_thresholds_warning(self, engine: ThresholdEngine) -> None:
        """Test threshold check returns WARNING for moderate change."""
        result = engine.check_thresholds(
            new_price_myr=125.0,  # 25% increase
            current_sitegiant_price=100.0,
        )

        assert result.status == PriceStatus.WARNING
        assert len(result.reasons) > 0

    def test_check_thresholds_blocked(self, engine: ThresholdEngine) -> None:
        """Test threshold check returns BLOCKED for large change."""
        result = engine.check_thresholds(
            new_price_myr=160.0,  # 60% increase
            current_sitegiant_price=100.0,
        )

        assert result.status == PriceStatus.BLOCKED
        assert len(result.reasons) > 0

    def test_check_thresholds_no_current_price(
        self,
        engine: ThresholdEngine,
    ) -> None:
        """Test threshold check with no current price."""
        result = engine.check_thresholds(
            new_price_myr=100.0,
            current_sitegiant_price=None,
        )

        assert result.status == PriceStatus.OK
        assert result.change_pct_vs_sitegiant is None

    def test_get_threshold_summary(self, engine: ThresholdEngine) -> None:
        """Test getting threshold configuration summary."""
        summary = engine.get_threshold_summary()

        assert "soft_vs_sitegiant_pct" in summary
        assert "hard_vs_sitegiant_pct" in summary
        assert summary["soft_vs_sitegiant_pct"] == 20.0
        assert summary["hard_vs_sitegiant_pct"] == 50.0


class TestThresholdEngineWithPreviousPrices:
    """Tests for threshold engine with historical price data."""

    @pytest.fixture
    def engine_with_cache(self) -> ThresholdEngine:
        """Create engine with pre-loaded price cache."""
        config = AppConfig()
        engine = ThresholdEngine(config)

        # Manually set previous prices
        engine.previous_prices = {
            "12345": 100.0,
            "67890": 50.0,
        }

        return engine

    def test_get_previous_price_found(
        self,
        engine_with_cache: ThresholdEngine,
    ) -> None:
        """Test getting existing previous price."""
        result = engine_with_cache.get_previous_price("12345")
        assert result == 100.0

    def test_get_previous_price_not_found(
        self,
        engine_with_cache: ThresholdEngine,
    ) -> None:
        """Test getting non-existent previous price."""
        result = engine_with_cache.get_previous_price("99999")
        assert result is None

    def test_spike_detection_warning(
        self,
        engine_with_cache: ThresholdEngine,
    ) -> None:
        """Test spike detection triggers warning."""
        result = engine_with_cache.check_thresholds(
            new_price_myr=500.0,
            current_sitegiant_price=500.0,  # No change vs SiteGiant
            pokedata_id="12345",
            current_pokedata_price_usd=120.0,  # 20% increase vs previous
        )

        assert result.status == PriceStatus.WARNING
        assert result.change_pct_vs_previous == 20.0

    def test_spike_detection_blocked(
        self,
        engine_with_cache: ThresholdEngine,
    ) -> None:
        """Test large spike triggers block."""
        result = engine_with_cache.check_thresholds(
            new_price_myr=500.0,
            current_sitegiant_price=500.0,
            pokedata_id="12345",
            current_pokedata_price_usd=150.0,  # 50% spike
        )

        assert result.status == PriceStatus.BLOCKED
