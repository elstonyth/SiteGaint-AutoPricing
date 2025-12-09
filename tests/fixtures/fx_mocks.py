"""
Mock responses for FX rate API calls.

Mocks Google Finance FX rate fetching.
"""

from unittest.mock import patch

# Default mock FX rate (USD to MYR)
DEFAULT_MOCK_FX_RATE = 4.72


def mock_google_fx_response(rate: float = DEFAULT_MOCK_FX_RATE) -> str:
    """
    Build a mock Google Finance response HTML.

    The actual Google Finance page structure is complex, so we mock
    the fx_provider function directly in most cases.
    """
    # Simplified mock - in practice, mock the function not the HTTP response
    return f"""
    <html>
        <body>
            <div data-exchange-rate="{rate}">
                {rate}
            </div>
        </body>
    </html>
    """


class FXRateMocker:
    """
    Context manager for mocking FX rate fetching.

    Usage:
        with FXRateMocker(rate=4.75):
            # Code that fetches FX rate will get 4.75
            result = get_fx_rate(config)
    """

    def __init__(self, rate: float = DEFAULT_MOCK_FX_RATE, source: str = "google"):
        self.rate = rate
        self.source = source
        self._patcher = None

    def __enter__(self):
        self._patcher = patch(
            "src.pricing.fx_provider.fetch_google_fx_rate", return_value=(self.rate, self.source)
        )
        self._patcher.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._patcher:
            self._patcher.stop()
        return False


def patch_fx_rate(rate: float = DEFAULT_MOCK_FX_RATE, source: str = "google"):
    """
    Decorator to patch FX rate in a test function.

    Usage:
        @patch_fx_rate(rate=4.75)
        def test_something():
            ...
    """
    return patch("src.pricing.fx_provider.fetch_google_fx_rate", return_value=(rate, source))


def patch_get_fx_rate(rate: float = DEFAULT_MOCK_FX_RATE, source: str = "mock"):
    """
    Decorator to patch the main get_fx_rate function.

    Usage:
        @patch_get_fx_rate(rate=4.75)
        def test_something():
            ...
    """
    return patch("src.pricing.fx_provider.get_fx_rate", return_value=(rate, source))


# Preset FX rate scenarios
FX_SCENARIOS = {
    "normal": {"rate": 4.72, "source": "google"},
    "high": {"rate": 5.00, "source": "google"},
    "low": {"rate": 4.50, "source": "google"},
    "fallback": {"rate": 4.70, "source": "fallback"},
    "error": {"rate": None, "source": "error"},
}


def get_fx_scenario(scenario: str = "normal") -> tuple:
    """
    Get a predefined FX rate scenario.

    Args:
        scenario: One of "normal", "high", "low", "fallback", "error"

    Returns:
        Tuple of (rate, source)
    """
    data = FX_SCENARIOS.get(scenario, FX_SCENARIOS["normal"])
    return data["rate"], data["source"]
