"""
Tests for the FX provider module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal

from src.pricing.fx_provider import (
    FXProvider,
    get_fx_rate,
    fetch_google_fx_rate,
)
from src.utils.config_loader import AppConfig


class TestFXProvider:
    """Tests for FXProvider class."""
    
    @pytest.fixture
    def config(self) -> AppConfig:
        """Create test configuration."""
        return AppConfig()
    
    @pytest.fixture
    def provider(self, config: AppConfig) -> FXProvider:
        """Create test FX provider."""
        return FXProvider(config)
    
    def test_provider_initialization(self, provider: FXProvider) -> None:
        """Test provider initializes correctly."""
        assert provider is not None
        assert provider.source == "default"
        assert provider.current_rate is None
    
    def test_get_default_rate(self, provider: FXProvider) -> None:
        """Test getting default rate from config."""
        rate = provider.get_default_rate()
        assert isinstance(rate, Decimal)
        assert rate > 0
    
    def test_set_manual_rate_valid(self, provider: FXProvider) -> None:
        """Test setting a valid manual rate."""
        provider.set_manual_rate(4.75)
        assert provider.current_rate == Decimal("4.75")
        assert provider.source == "manual_override"
    
    def test_set_manual_rate_invalid_zero(self, provider: FXProvider) -> None:
        """Test setting zero rate raises error."""
        with pytest.raises(ValueError):
            provider.set_manual_rate(0)
    
    def test_set_manual_rate_invalid_negative(self, provider: FXProvider) -> None:
        """Test setting negative rate raises error."""
        with pytest.raises(ValueError):
            provider.set_manual_rate(-1.5)
    
    def test_get_rate_returns_manual_if_set(self, provider: FXProvider) -> None:
        """Test get_rate returns manual rate when set."""
        provider.set_manual_rate(4.80)
        rate = provider.get_rate()
        assert rate == Decimal("4.80")
    
    def test_get_rate_returns_default_if_not_set(self, provider: FXProvider) -> None:
        """Test get_rate returns default when no manual rate set."""
        rate = provider.get_rate()
        default = provider.get_default_rate()
        assert rate == default
    
    def test_validate_rate_valid(self, provider: FXProvider) -> None:
        """Test rate validation with valid rate."""
        assert provider.validate_rate(4.50) is True
        assert provider.validate_rate(3.0) is True
        assert provider.validate_rate(6.0) is True
    
    def test_validate_rate_invalid(self, provider: FXProvider) -> None:
        """Test rate validation with invalid rate."""
        assert provider.validate_rate(2.5) is False
        assert provider.validate_rate(7.0) is False
    
    def test_get_rate_info(self, provider: FXProvider) -> None:
        """Test rate info dictionary."""
        info = provider.get_rate_info()
        assert "rate" in info
        assert "source" in info
        assert "is_default" in info


class TestGetFXRate:
    """Tests for get_fx_rate convenience function."""
    
    @pytest.fixture
    def config(self) -> AppConfig:
        """Create test configuration."""
        return AppConfig()
    
    def test_manual_override_takes_priority(self, config: AppConfig) -> None:
        """Test that manual override is used when provided."""
        rate, source = get_fx_rate(config, manual_override=4.85)
        assert rate == 4.85
        assert source == "manual_override"
    
    def test_invalid_manual_override_falls_back(self, config: AppConfig) -> None:
        """Test that invalid manual override falls back to default."""
        rate, source = get_fx_rate(config, manual_override=-1.0)
        assert rate > 0
        assert source == "default"
    
    def test_manual_mode_returns_default(self, config: AppConfig) -> None:
        """Test that manual mode returns default rate."""
        # Manually set mode to manual
        config.fx.mode = "manual"
        rate, source = get_fx_rate(config, manual_override=None)
        assert rate == config.fx.default_rate
        assert source == "default"


class TestFetchGoogleFXRate:
    """Tests for fetch_google_fx_rate function."""
    
    @pytest.fixture
    def config(self) -> AppConfig:
        """Create test configuration."""
        return AppConfig()
    
    @patch('src.pricing.fx_provider.requests.get')
    def test_successful_fetch(self, mock_get: Mock, config: AppConfig) -> None:
        """Test successful Google Finance fetch."""
        # Mock a response with a valid FX rate
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '''
        <html>
        <div data-last-price="4.4850"></div>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        rate, source = fetch_google_fx_rate(config)
        
        assert rate == 4.4850
        assert source == "google"
        mock_get.assert_called_once()
    
    @patch('src.pricing.fx_provider.requests.get')
    def test_fetch_with_different_pattern(self, mock_get: Mock, config: AppConfig) -> None:
        """Test parsing with alternative HTML pattern."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '''
        <html>
        <div class="YMlKec fxKbKc">4.5200</div>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        rate, source = fetch_google_fx_rate(config)
        
        assert rate == 4.52
        assert source == "google"
    
    @patch('src.pricing.fx_provider.requests.get')
    def test_timeout_returns_none(self, mock_get: Mock, config: AppConfig) -> None:
        """Test that timeout returns None with error message."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        rate, source = fetch_google_fx_rate(config)
        
        assert rate is None
        assert "timed out" in source.lower()
    
    @patch('src.pricing.fx_provider.requests.get')
    def test_connection_error_returns_none(self, mock_get: Mock, config: AppConfig) -> None:
        """Test that connection error returns None."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")
        
        rate, source = fetch_google_fx_rate(config)
        
        assert rate is None
        assert "connection" in source.lower()
    
    @patch('src.pricing.fx_provider.requests.get')
    def test_invalid_html_returns_none(self, mock_get: Mock, config: AppConfig) -> None:
        """Test that unparseable HTML returns None."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<html><div>No price here</div></html>'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        rate, source = fetch_google_fx_rate(config)
        
        assert rate is None
        assert "parse" in source.lower() or "failed" in source.lower()
    
    @patch('src.pricing.fx_provider.requests.get')
    def test_http_error_returns_none(self, mock_get: Mock, config: AppConfig) -> None:
        """Test that HTTP error returns None."""
        import requests
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_response
        
        rate, source = fetch_google_fx_rate(config)
        
        assert rate is None
        assert "http" in source.lower() or "error" in source.lower()


class TestGetFXRateGoogleMode:
    """Tests for get_fx_rate with google mode."""
    
    @pytest.fixture
    def google_config(self) -> AppConfig:
        """Create config with google mode."""
        config = AppConfig()
        config.fx.mode = "google"
        config.fx.default_rate = 4.70
        return config
    
    @patch('src.pricing.fx_provider.fetch_google_fx_rate')
    def test_google_mode_uses_fetched_rate(
        self, 
        mock_fetch: Mock, 
        google_config: AppConfig
    ) -> None:
        """Test that google mode uses the fetched rate."""
        mock_fetch.return_value = (4.55, "google")
        
        rate, source = get_fx_rate(google_config, manual_override=None)
        
        assert rate == 4.55
        assert source == "google"
        mock_fetch.assert_called_once()
    
    @patch('src.pricing.fx_provider.fetch_google_fx_rate')
    def test_google_mode_fallback_on_failure(
        self, 
        mock_fetch: Mock, 
        google_config: AppConfig
    ) -> None:
        """Test that google mode falls back to default on failure."""
        mock_fetch.return_value = (None, "Connection error")
        
        rate, source = get_fx_rate(google_config, manual_override=None)
        
        assert rate == 4.70  # default_rate
        assert "default" in source.lower()
        assert "google failed" in source.lower()
    
    @patch('src.pricing.fx_provider.fetch_google_fx_rate')
    def test_manual_override_beats_google(
        self, 
        mock_fetch: Mock, 
        google_config: AppConfig
    ) -> None:
        """Test that manual override takes priority over google mode."""
        rate, source = get_fx_rate(google_config, manual_override=4.90)
        
        assert rate == 4.90
        assert source == "manual_override"
        # fetch_google_fx_rate should not be called
        mock_fetch.assert_not_called()
