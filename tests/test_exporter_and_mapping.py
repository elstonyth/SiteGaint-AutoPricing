import pandas as pd

from src.exporter.sitegiant_exporter import (
    SiteGiantExporter,
    add_include_in_update_column,
    build_update_dataframe,
    find_price_column,
    find_sku_column,
)
from src.mapping.mapping_manager import MappingManager
from src.utils.config_loader import AppConfig


def test_prepare_export_and_validation_and_backup(tmp_path):
    config = AppConfig()
    exporter = SiteGiantExporter(config, output_dir=tmp_path)

    original = pd.DataFrame({"sku": ["A", "B"], "price": [10.0, 20.0]})
    updated = pd.DataFrame(
        {"sku": ["A", "B"], "new_price_myr": [11.0, 22.0], "include": [True, False]}
    )

    prepared = exporter.prepare_export_data(original, updated)
    assert list(prepared["price"]) == [11.0, 20.0]

    errors = exporter.validate_export_data(prepared)
    assert errors == []

    out_path = exporter.export_to_excel(prepared, output_path=tmp_path / "out.xlsx")
    assert out_path.exists()

    backup = exporter.create_backup(out_path)
    assert backup.exists()


def test_include_and_build_update_dataframe():
    results = pd.DataFrame(
        {
            "SKU": ["A", "B"],
            "status": ["OK", "BLOCKED"],
            "auto_update": ["Y", "Y"],
            "new_price_myr": [15.0, 99.0],
        }
    )
    marked = add_include_in_update_column(
        results, status_column="status", auto_update_column="auto_update"
    )
    assert marked["include_in_update"].tolist() == [True, False]

    original = pd.DataFrame({"SKU": ["A", "B"], "Price": [10.0, 20.0]})
    updated = build_update_dataframe(
        original, marked, price_column="Price", include_column="include_in_update"
    )
    assert updated.loc[updated["SKU"] == "A", "Price"].iloc[0] == 15.0
    assert updated.loc[updated["SKU"] == "B", "Price"].iloc[0] == 20.0


def test_find_price_and_sku_column():
    df = pd.DataFrame({"Selling Price": [1], "product_sku": ["X"]})
    assert find_price_column(df) == "Selling Price"
    assert find_sku_column(df) == "product_sku"


def test_mapping_manager_load_and_join(tmp_path):
    config = AppConfig()
    mapping_path = tmp_path / "mapping.xlsx"
    # Note: pokedata_id must be provided directly (numeric ID from API)
    # URL extraction only populates pokedata_name (slug), not pokedata_id
    mapping_df = pd.DataFrame(
        {
            "SKU": ["A"],
            "pokedata_url": ["https://pokedata.io/product/item-123"],
            "pokedata_id": ["12345"],  # Numeric ID from API
            "pokedata_language": ["english"],
            "auto_update": ["Y"],
            "pokedata_asset_type": ["PRODUCT"],
            "pokedata_name": [""],
            "notes": [""],
        }
    )
    mapping_df.to_excel(mapping_path, index=False)

    manager = MappingManager(config)
    manager.load_mapping_file(mapping_path)

    entry = manager.get_mapping("A")
    assert entry is not None
    assert entry.pokedata_id == "12345"  # Now expects numeric ID
    assert entry.language == "ENGLISH"

    products = pd.DataFrame({"sku": ["A", "B"], "price": [1.0, 2.0]})
    joined = manager.join_with_products(products)
    assert bool(joined.loc[joined["sku"] == "A", "is_mapped"].iloc[0]) is True
    assert bool(joined.loc[joined["sku"] == "B", "is_mapped"].iloc[0]) is False

    unmapped = set(manager.get_unmapped_skus(products))
    assert unmapped == {"B"}
