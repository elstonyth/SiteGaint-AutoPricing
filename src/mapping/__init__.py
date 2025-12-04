"""
SKU to Pokedata ID mapping module.

Handles loading mapping files and joining SiteGiant SKUs with Pokedata product IDs.
Supports auto-extraction of Pokedata product IDs from URLs.
"""

from src.mapping.mapping_manager import (
    MappingManager,
    MappingEntry,
    load_mapping,
    apply_mapping,
    extract_pokedata_id_from_url,
)

__all__ = [
    "MappingManager",
    "MappingEntry",
    "load_mapping",
    "apply_mapping",
    "extract_pokedata_id_from_url",
]
