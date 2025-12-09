"""
Tests for the SiteGiant importer module.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.importer.sitegiant_importer import SiteGiantImporter
from src.utils.config_loader import AppConfig


class TestSiteGiantImporter:
    """Tests for SiteGiantImporter class."""

    @pytest.fixture
    def config(self) -> AppConfig:
        """Create a test configuration."""
        return AppConfig()

    @pytest.fixture
    def importer(self, config: AppConfig) -> SiteGiantImporter:
        """Create a test importer instance."""
        return SiteGiantImporter(config)

    def test_importer_initialization(self, importer: SiteGiantImporter) -> None:
        """Test that importer initializes correctly."""
        assert importer is not None
        assert importer.config is not None

    def test_load_file_not_found(self, importer: SiteGiantImporter) -> None:
        """Test that loading a non-existent file raises an error."""
        with pytest.raises(FileNotFoundError):
            importer.load_file(Path("non_existent.xlsx"))

    def test_validate_columns_missing(self, importer: SiteGiantImporter) -> None:
        """Test validation with missing columns."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        missing = importer.validate_columns(df)
        assert "sku" in missing
        assert "name" in missing
        assert "price" in missing

    def test_validate_columns_present(self, importer: SiteGiantImporter) -> None:
        """Test validation with all required columns."""
        df = pd.DataFrame(
            {
                "SKU": ["A", "B"],
                "Product Name": ["P1", "P2"],
                "Price": [100, 200],
            }
        )
        missing = importer.validate_columns(df)
        assert len(missing) == 0

    def test_filter_products(self, importer: SiteGiantImporter) -> None:
        """Test product filtering based on configuration."""
        df = pd.DataFrame(
            {
                "SKU": ["A", "B", "C"],
                "Status": ["Active", "Inactive", "Active"],
                "Stock": [10, 0, 5],
            }
        )
        # Currently no filtering - returns as-is
        result = importer.filter_products(df)
        assert len(result) == 3

    def test_normalize_columns(self, importer: SiteGiantImporter) -> None:
        """Test column name normalization."""
        df = pd.DataFrame(
            {
                "Product Name": ["A", "B"],
                "SKU": ["S1", "S2"],
                "Price": [100, 200],
            }
        )
        result = importer.normalize_columns(df)
        assert "name" in result.columns
        assert "sku" in result.columns
        assert "price" in result.columns
        assert "isku" in result.columns  # Should be added even if empty

    def test_import_products_file_not_found(self, importer: SiteGiantImporter) -> None:
        """Test the full import pipeline with missing file."""
        with pytest.raises(FileNotFoundError):
            importer.import_products(Path("test.xlsx"))


class TestSiteGiantImporterIntegration:
    """Integration tests for SiteGiant importer with real files."""

    @pytest.fixture
    def sample_excel_path(self, tmp_path: Path) -> Path:
        """Create a sample Excel file for testing."""
        file_path = tmp_path / "sample_sitegiant.xlsx"

        df = pd.DataFrame(
            {
                "SKU": ["SKU-001", "SKU-002", "SKU-003"],
                "Product Name": ["Product A", "Product B", "Product C"],
                "Price": [100.00, 200.00, 150.00],
                "Stock": [10, 5, 0],
                "Status": ["Active", "Active", "Inactive"],
            }
        )

        df.to_excel(file_path, index=False)
        return file_path

    @pytest.fixture
    def config(self) -> AppConfig:
        """Create test configuration."""
        return AppConfig()

    def test_load_real_excel_file(
        self,
        sample_excel_path: Path,
        config: AppConfig,
    ) -> None:
        """Test loading a real Excel file."""
        importer = SiteGiantImporter(config)
        df = importer.load_file(sample_excel_path)
        assert len(df) == 3
        assert "SKU" in df.columns

    def test_import_products_full_pipeline(
        self,
        sample_excel_path: Path,
        config: AppConfig,
    ) -> None:
        """Test full import pipeline with real file."""
        importer = SiteGiantImporter(config)
        df = importer.import_products(sample_excel_path)

        # Check normalization
        assert "sku" in df.columns
        assert "name" in df.columns
        assert "price" in df.columns
        assert len(df) == 3
