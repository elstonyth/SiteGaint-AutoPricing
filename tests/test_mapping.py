"""
Tests for the mapping manager module.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from src.mapping.mapping_manager import (
    MappingManager,
    MappingEntry,
    extract_pokedata_id_from_url,
)
from src.utils.config_loader import AppConfig


class TestExtractPokedataIdFromUrl:
    """Tests for extract_pokedata_id_from_url function."""
    
    def test_extract_simple_url(self) -> None:
        """Test extracting ID from simple URL."""
        url = "https://www.pokedata.io/product/Scarlet+Violet+Booster+Box"
        result = extract_pokedata_id_from_url(url)
        assert result == "Scarlet Violet Booster Box"
    
    def test_extract_url_with_special_chars(self) -> None:
        """Test extracting ID from URL with encoded special characters."""
        url = "https://www.pokedata.io/product/Scarlet+%26+Violet+ETB"
        result = extract_pokedata_id_from_url(url)
        assert result == "Scarlet & Violet ETB"
    
    def test_extract_url_without_www(self) -> None:
        """Test extracting ID from URL without www."""
        url = "https://pokedata.io/product/Base+Set+Booster+Box"
        result = extract_pokedata_id_from_url(url)
        assert result == "Base Set Booster Box"
    
    def test_extract_empty_url(self) -> None:
        """Test with empty URL returns empty string."""
        assert extract_pokedata_id_from_url("") == ""
        assert extract_pokedata_id_from_url(None) == ""
    
    def test_extract_invalid_url(self) -> None:
        """Test with URL without /product/ returns empty string."""
        url = "https://www.pokedata.io/sets/Base+Set"
        result = extract_pokedata_id_from_url(url)
        assert result == ""
    
    def test_extract_nan_value(self) -> None:
        """Test with NaN (pandas) value returns empty string."""
        import math
        result = extract_pokedata_id_from_url(float('nan'))
        assert result == ""
    
    def test_extract_url_with_query_params(self) -> None:
        """Test extracting ID from URL with query parameters."""
        url = "https://www.pokedata.io/product/Elite+Trainer+Box?ref=search"
        result = extract_pokedata_id_from_url(url)
        assert result == "Elite Trainer Box"


class TestMappingEntry:
    """Tests for MappingEntry dataclass."""
    
    def test_mapping_entry_creation(self) -> None:
        """Test creating a mapping entry."""
        entry = MappingEntry(
            sku="SKU-001",
            language="ENGLISH",
            pokedata_id="12345",
            asset_type="Booster Box",
            pokedata_url="https://pokedata.io/product/12345",
        )
        
        assert entry.sku == "SKU-001"
        assert entry.language == "ENGLISH"
        assert entry.pokedata_id == "12345"
        assert entry.asset_type == "Booster Box"
        assert entry.pokedata_url == "https://pokedata.io/product/12345"
    
    def test_mapping_entry_optional_url(self) -> None:
        """Test creating entry without optional URL."""
        entry = MappingEntry(
            sku="SKU-002",
            language="JAPANESE",
            pokedata_id="67890",
            asset_type="ETB",
        )
        
        assert entry.pokedata_url is None


class TestMappingManager:
    """Tests for MappingManager class."""
    
    @pytest.fixture
    def config(self) -> AppConfig:
        """Create test configuration."""
        return AppConfig()
    
    @pytest.fixture
    def manager(self, config: AppConfig) -> MappingManager:
        """Create test mapping manager."""
        return MappingManager(config)
    
    def test_manager_initialization(self, manager: MappingManager) -> None:
        """Test manager initializes correctly."""
        assert manager is not None
        assert manager.mappings == {}
    
    def test_get_mapping_not_found(self, manager: MappingManager) -> None:
        """Test getting a mapping that doesn't exist."""
        result = manager.get_mapping("NONEXISTENT")
        assert result is None
    
    def test_get_pokedata_id_not_found(self, manager: MappingManager) -> None:
        """Test getting Pokedata ID for unmapped SKU."""
        result = manager.get_pokedata_id("NONEXISTENT")
        assert result is None
    
    def test_get_language_not_found(self, manager: MappingManager) -> None:
        """Test getting language for unmapped SKU."""
        result = manager.get_language("NONEXISTENT")
        assert result is None
    
    def test_load_mapping_file_not_found(self, manager: MappingManager) -> None:
        """Test loading a non-existent mapping file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            manager.load_mapping_file(Path("nonexistent_file.xlsx"))
    
    def test_join_with_products_no_mapping_loaded(self, manager: MappingManager) -> None:
        """Test joining products without loading mapping first raises ValueError."""
        df = pd.DataFrame({"sku": ["A", "B"]})
        with pytest.raises(ValueError, match="No mapping file loaded"):
            manager.join_with_products(df)
    
    def test_get_unmapped_skus(self, manager: MappingManager) -> None:
        """Test identifying unmapped SKUs returns empty list when no mapping loaded."""
        df = pd.DataFrame({"sku": ["A", "B"]})
        # When no mapping is loaded, all SKUs are unmapped
        unmapped = manager.get_unmapped_skus(df)
        assert set(unmapped) == {"A", "B"}


class TestMappingManagerWithData:
    """Tests for MappingManager with pre-loaded data."""
    
    @pytest.fixture
    def manager_with_data(self) -> MappingManager:
        """Create manager with test data."""
        config = AppConfig()
        manager = MappingManager(config)
        
        # Manually add test mappings
        manager.mappings = {
            "SKU-001": MappingEntry(
                sku="SKU-001",
                language="ENGLISH",
                pokedata_id="12345",
                asset_type="Booster Box",
            ),
            "SKU-002": MappingEntry(
                sku="SKU-002",
                language="JAPANESE",
                pokedata_id="67890",
                asset_type="ETB",
            ),
        }
        
        return manager
    
    def test_get_mapping_found(self, manager_with_data: MappingManager) -> None:
        """Test getting an existing mapping."""
        result = manager_with_data.get_mapping("SKU-001")
        
        assert result is not None
        assert result.pokedata_id == "12345"
        assert result.language == "ENGLISH"
    
    def test_get_pokedata_id_found(self, manager_with_data: MappingManager) -> None:
        """Test getting Pokedata ID for mapped SKU."""
        result = manager_with_data.get_pokedata_id("SKU-002")
        assert result == "67890"
    
    def test_get_language_found(self, manager_with_data: MappingManager) -> None:
        """Test getting language for mapped SKU."""
        result = manager_with_data.get_language("SKU-001")
        assert result == "ENGLISH"


class TestScanDuplicates:
    """Tests for scan_duplicates function."""
    
    @pytest.fixture
    def mapping_file_with_duplicates(self, tmp_path: Path) -> Path:
        """Create a mapping file with duplicate SKUs."""
        df = pd.DataFrame({
            "sku": ["SKU-001", "SKU-002", "SKU-001", "SKU-003", "SKU-002"],
            "pokedata_id": ["100", "200", "101", "300", "201"],
            "pokedata_language": ["ENGLISH", "ENGLISH", "ENGLISH", "JAPANESE", "ENGLISH"],
            "auto_update": ["Y", "Y", "Y", "Y", "Y"],
        })
        file_path = tmp_path / "duplicates.xlsx"
        df.to_excel(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def mapping_file_no_duplicates(self, tmp_path: Path) -> Path:
        """Create a mapping file without duplicates."""
        df = pd.DataFrame({
            "sku": ["SKU-001", "SKU-002", "SKU-003"],
            "pokedata_id": ["100", "200", "300"],
            "pokedata_language": ["ENGLISH", "ENGLISH", "JAPANESE"],
            "auto_update": ["Y", "Y", "Y"],
        })
        file_path = tmp_path / "no_duplicates.xlsx"
        df.to_excel(file_path, index=False)
        return file_path
    
    def test_scan_detects_duplicates(self, mapping_file_with_duplicates: Path) -> None:
        """Test that scan_duplicates detects duplicate SKUs."""
        from src.mapping.mapping_manager import scan_duplicates
        
        result = scan_duplicates(mapping_file_with_duplicates)
        
        assert result.has_duplicates is True
        assert len(result.duplicate_skus) == 2
        assert "SKU-001" in result.duplicate_skus
        assert "SKU-002" in result.duplicate_skus
        assert result.total_rows == 5
        assert result.unique_skus == 3
    
    def test_scan_no_duplicates(self, mapping_file_no_duplicates: Path) -> None:
        """Test that scan_duplicates returns no duplicates when none exist."""
        from src.mapping.mapping_manager import scan_duplicates
        
        result = scan_duplicates(mapping_file_no_duplicates)
        
        assert result.has_duplicates is False
        assert len(result.duplicate_skus) == 0
        assert result.total_rows == 3
        assert result.unique_skus == 3
    
    def test_scan_returns_row_details(self, mapping_file_with_duplicates: Path) -> None:
        """Test that scan_duplicates returns correct row numbers."""
        from src.mapping.mapping_manager import scan_duplicates
        
        result = scan_duplicates(mapping_file_with_duplicates)
        
        # SKU-001 appears in rows 1 and 3 (0-indexed: 0 and 2, +2 for Excel = 2 and 4)
        assert "SKU-001" in result.duplicate_details
        assert 2 in result.duplicate_details["SKU-001"]
        assert 4 in result.duplicate_details["SKU-001"]
    
    def test_scan_to_dict(self, mapping_file_with_duplicates: Path) -> None:
        """Test that to_dict returns correct JSON-serializable format."""
        from src.mapping.mapping_manager import scan_duplicates
        
        result = scan_duplicates(mapping_file_with_duplicates)
        result_dict = result.to_dict()
        
        assert "has_duplicates" in result_dict
        assert result_dict["has_duplicates"] is True
        assert result_dict["duplicate_count"] == 2
        assert "duplicate_skus" in result_dict


class TestDuplicateHandling:
    """Tests for duplicate handling in load_mapping_file."""
    
    @pytest.fixture
    def mapping_file_with_duplicates(self, tmp_path: Path) -> Path:
        """Create a mapping file with duplicate SKUs."""
        df = pd.DataFrame({
            "sku": ["SKU-001", "SKU-002", "SKU-001"],
            "pokedata_id": ["100", "200", "999"],  # SKU-001 has IDs 100 and 999
            "pokedata_language": ["ENGLISH", "ENGLISH", "ENGLISH"],
            "pokedata_asset_type": ["PRODUCT", "PRODUCT", "PRODUCT"],
            "auto_update": ["Y", "Y", "Y"],
            "notes": ["first", "unique", "last"],
        })
        file_path = tmp_path / "duplicates.xlsx"
        df.to_excel(file_path, index=False)
        return file_path
    
    def test_merge_strategy_keeps_last(self, mapping_file_with_duplicates: Path) -> None:
        """Test that 'merge' strategy keeps the last entry."""
        config = AppConfig()
        manager = MappingManager(config)
        
        result = manager.load_mapping_file(mapping_file_with_duplicates, strategy="merge")
        
        assert result.has_duplicates is True
        # After merge, SKU-001 should have the last entry (pokedata_id=999)
        mapping = manager.get_mapping("SKU-001")
        assert mapping is not None
        assert mapping.pokedata_id == "999"
    
    def test_ignore_strategy_keeps_first(self, mapping_file_with_duplicates: Path) -> None:
        """Test that 'ignore' strategy keeps the first entry."""
        config = AppConfig()
        manager = MappingManager(config)
        
        result = manager.load_mapping_file(mapping_file_with_duplicates, strategy="ignore")
        
        assert result.has_duplicates is True
        # After ignore, SKU-001 should have the first entry (pokedata_id=100)
        mapping = manager.get_mapping("SKU-001")
        assert mapping is not None
        assert mapping.pokedata_id == "100"
    
    def test_returns_duplicate_scan_result(self, mapping_file_with_duplicates: Path) -> None:
        """Test that load_mapping_file returns DuplicateScanResult."""
        from src.mapping.mapping_manager import DuplicateScanResult
        
        config = AppConfig()
        manager = MappingManager(config)
        
        result = manager.load_mapping_file(mapping_file_with_duplicates)
        
        assert isinstance(result, DuplicateScanResult)
        assert result.has_duplicates is True
        assert "SKU-001" in result.duplicate_skus
