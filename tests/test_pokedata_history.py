"""
Tests for the Pokedata price history storage module.
"""

import os
import tempfile

import pytest

from src.storage.pokedata_history_store import PokedataHistoryStore, get_history, record_price


class TestPokedataHistoryStore:
    """Tests for PokedataHistoryStore class."""

    @pytest.fixture
    def temp_history_file(self):
        """Create a temporary history file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("pokedata_id,asset_type,date,price_usd\n")
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_record_and_get_history(self, temp_history_file):
        """Test recording and retrieving price history."""
        store = PokedataHistoryStore(temp_history_file)

        # Record some prices
        store.record_price("PROD001", "PRODUCT", 99.99, "2024-01-01")
        store.record_price("PROD001", "PRODUCT", 105.50, "2024-01-02")
        store.record_price("PROD001", "PRODUCT", 102.25, "2024-01-03")

        # Get history
        history = store.get_history("PROD001", "PRODUCT", limit=5)

        assert len(history) == 3
        # Should be sorted newest first
        assert history[0].date == "2024-01-03"
        assert history[0].price_usd == 102.25
        assert history[1].date == "2024-01-02"
        assert history[2].date == "2024-01-01"

    def test_update_existing_entry(self, temp_history_file):
        """Test that recording a price on the same date updates the entry."""
        store = PokedataHistoryStore(temp_history_file)

        # Record initial price
        store.record_price("PROD001", "PRODUCT", 99.99, "2024-01-01")

        # Update with new price for same date
        store.record_price("PROD001", "PRODUCT", 110.00, "2024-01-01")

        # Should only have one entry with updated price
        history = store.get_history("PROD001", "PRODUCT")
        assert len(history) == 1
        assert history[0].price_usd == 110.00

    def test_different_products(self, temp_history_file):
        """Test that different products have separate histories."""
        store = PokedataHistoryStore(temp_history_file)

        store.record_price("PROD001", "PRODUCT", 100.00, "2024-01-01")
        store.record_price("PROD002", "PRODUCT", 200.00, "2024-01-01")

        history1 = store.get_history("PROD001", "PRODUCT")
        history2 = store.get_history("PROD002", "PRODUCT")

        assert len(history1) == 1
        assert history1[0].price_usd == 100.00

        assert len(history2) == 1
        assert history2[0].price_usd == 200.00

    def test_limit_parameter(self, temp_history_file):
        """Test that limit parameter works correctly."""
        store = PokedataHistoryStore(temp_history_file)

        # Record 10 prices
        for i in range(10):
            store.record_price("PROD001", "PRODUCT", 100.0 + i, f"2024-01-{i+1:02d}")

        # Get only last 3
        history = store.get_history("PROD001", "PRODUCT", limit=3)
        assert len(history) == 3
        # Should be most recent
        assert history[0].date == "2024-01-10"

    def test_empty_history(self, temp_history_file):
        """Test getting history for non-existent product."""
        store = PokedataHistoryStore(temp_history_file)

        history = store.get_history("NONEXISTENT", "PRODUCT")
        assert len(history) == 0

    def test_creates_file_if_not_exists(self, tmp_path):
        """Test that store creates file if it doesn't exist."""
        history_path = tmp_path / "new_history.csv"

        PokedataHistoryStore(str(history_path))

        assert history_path.exists()

        # Should have header
        with open(history_path) as f:
            header = f.readline().strip()
            assert "pokedata_id" in header


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_record_and_get_functions(self, tmp_path, monkeypatch):
        """Test the module-level record_price and get_history functions."""
        # Patch the default path
        test_path = tmp_path / "test_history.csv"

        import src.storage.pokedata_history_store as store_module

        monkeypatch.setattr(store_module, "_store", None)
        monkeypatch.setattr(store_module, "DEFAULT_HISTORY_PATH", str(test_path))

        # Record via module function
        record_price("TEST001", "PRODUCT", 55.55, "2024-06-15")

        # Get via module function
        history = get_history("TEST001", "PRODUCT")

        assert len(history) == 1
        assert history[0].price_usd == 55.55
