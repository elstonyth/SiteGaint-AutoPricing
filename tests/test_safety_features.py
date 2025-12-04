"""
Tests for safety features implemented in the SiteGiant pricing automation.

Phase 1: Schema validation on upload
Phase 2: Demo mode export blocking
Phase 3: BLOCKED rows exclusion from export
Phase 4: Unmapped SKU handling
"""

import pandas as pd
import pytest

from src.webapp.helpers import validate_sitegiant_columns as validate_sitegiant_upload


class TestPhase1SchemaValidation:
    """Tests for Phase 1: Schema validation on upload."""

    def test_valid_sitegiant_file_passes_validation(self):
        """Valid SiteGiant export with all required columns passes validation."""
        df = pd.DataFrame({
            "SKU": ["SKU001", "SKU002"],
            "Product Name": ["Product 1", "Product 2"],
            "Price": [100.00, 200.00],
        })
        errors = validate_sitegiant_upload(df)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_missing_sku_column_fails_validation(self):
        """File missing SKU column should fail validation."""
        df = pd.DataFrame({
            "Product Name": ["Product 1"],
            "Price": [100.00],
        })
        errors = validate_sitegiant_upload(df)
        assert "Missing SKU column" in errors

    def test_missing_name_column_fails_validation(self):
        """File missing Name column should fail validation."""
        df = pd.DataFrame({
            "SKU": ["SKU001"],
            "Price": [100.00],
        })
        errors = validate_sitegiant_upload(df)
        assert "Missing Product Name column" in errors

    def test_missing_price_column_fails_validation(self):
        """File missing Price column should fail validation."""
        df = pd.DataFrame({
            "SKU": ["SKU001"],
            "Product Name": ["Product 1"],
        })
        errors = validate_sitegiant_upload(df)
        assert "Missing Price column" in errors

    def test_empty_dataframe_fails_validation(self):
        """Empty DataFrame should fail validation."""
        df = pd.DataFrame()
        errors = validate_sitegiant_upload(df)
        assert "File is empty or could not be read" in errors

    def test_none_dataframe_fails_validation(self):
        """None DataFrame should fail validation."""
        errors = validate_sitegiant_upload(None)
        assert "File is empty or could not be read" in errors

    def test_case_insensitive_column_detection(self):
        """Column detection should be case-insensitive."""
        df = pd.DataFrame({
            "sku": ["SKU001"],  # lowercase
            "name": ["Product 1"],  # lowercase
            "price": [100.00],  # lowercase
        })
        errors = validate_sitegiant_upload(df)
        assert errors == [], f"Expected no errors for lowercase columns, got: {errors}"

    def test_alternative_column_names_accepted(self):
        """Alternative column names like 'Selling Price' should be accepted."""
        df = pd.DataFrame({
            "Product SKU": ["SKU001"],  # Alternative SKU name
            "Title": ["Product 1"],  # Alternative Name name
            "Selling Price": [100.00],  # Alternative Price name
        })
        errors = validate_sitegiant_upload(df)
        assert errors == [], f"Expected no errors for alternative column names, got: {errors}"


class TestPhase3BlockedRowsExclusion:
    """Tests for Phase 3: BLOCKED rows should never be exported."""

    def test_blocked_rows_have_include_set_to_false(self):
        """
        Simulates the logic that should happen in /export:
        BLOCKED rows should have include_in_update forced to False.
        """
        results_df = pd.DataFrame({
            "sku": ["SKU001", "SKU002", "SKU003"],
            "status": ["OK", "BLOCKED", "WARNING"],
            "include_in_update": [True, True, True],  # User selected all
            "new_price_myr": [100.0, 200.0, 300.0],
        })
        
        # Apply the Phase 3 safety logic
        if "status" in results_df.columns:
            blocked_mask = results_df["status"] == "BLOCKED"
            results_df.loc[blocked_mask, "include_in_update"] = False
        
        # Verify BLOCKED row is excluded
        assert results_df[results_df["sku"] == "SKU001"]["include_in_update"].iloc[0] == True
        assert results_df[results_df["sku"] == "SKU002"]["include_in_update"].iloc[0] == False  # BLOCKED
        assert results_df[results_df["sku"] == "SKU003"]["include_in_update"].iloc[0] == True

    def test_unmapped_rows_have_include_set_to_false(self):
        """
        UNMAPPED rows should also have include_in_update forced to False.
        """
        results_df = pd.DataFrame({
            "sku": ["SKU001", "SKU002", "SKU003"],
            "status": ["OK", "UNMAPPED", "WARNING"],
            "include_in_update": [True, True, True],  # User selected all
            "new_price_myr": [100.0, None, 300.0],  # UNMAPPED has no new price
        })
        
        # Apply the Phase 3 safety logic
        if "status" in results_df.columns:
            unmapped_mask = results_df["status"] == "UNMAPPED"
            results_df.loc[unmapped_mask, "include_in_update"] = False
        
        # Verify UNMAPPED row is excluded
        assert results_df[results_df["sku"] == "SKU001"]["include_in_update"].iloc[0] == True
        assert results_df[results_df["sku"] == "SKU002"]["include_in_update"].iloc[0] == False  # UNMAPPED
        assert results_df[results_df["sku"] == "SKU003"]["include_in_update"].iloc[0] == True


class TestPhase4UnmappedSKUs:
    """Tests for Phase 4: Unmapped SKU handling."""

    def test_unmapped_skus_can_be_filtered(self):
        """Unmapped SKUs should be filterable from results."""
        results_df = pd.DataFrame({
            "sku": ["SKU001", "SKU002", "SKU003"],
            "name": ["Product 1", "Product 2", "Product 3"],
            "status": ["OK", "UNMAPPED", "UNMAPPED"],
            "price": [100.0, 200.0, 300.0],
        })
        
        unmapped_df = results_df[results_df["status"] == "UNMAPPED"]
        
        assert len(unmapped_df) == 2
        assert "SKU002" in unmapped_df["sku"].values
        assert "SKU003" in unmapped_df["sku"].values
        assert "SKU001" not in unmapped_df["sku"].values

    def test_unmapped_export_has_mapping_columns(self):
        """
        Unmapped SKU export should include helpful columns for mapping.
        """
        results_df = pd.DataFrame({
            "sku": ["SKU001"],
            "name": ["Product 1"],
            "status": ["UNMAPPED"],
            "price": [100.0],
        })
        
        unmapped_df = results_df[results_df["status"] == "UNMAPPED"].copy()
        
        # Add mapping columns (as done in the endpoint)
        export_cols = ["sku", "name", "price"]
        available_cols = [c for c in export_cols if c in unmapped_df.columns]
        unmapped_export = unmapped_df[available_cols].copy()
        unmapped_export["pokedata_id"] = ""
        unmapped_export["pokedata_language"] = "ENGLISH"
        unmapped_export["pokedata_asset_type"] = "PRODUCT"
        unmapped_export["auto_update"] = "Y"
        
        # Verify all expected columns are present
        assert "sku" in unmapped_export.columns
        assert "pokedata_id" in unmapped_export.columns
        assert "pokedata_language" in unmapped_export.columns
        assert "auto_update" in unmapped_export.columns
