"""
Storage modules for data persistence.
"""

from src.storage.pokedata_history_store import (
    PokedataHistoryStore,
    get_history,
    record_price,
)
from src.storage.settings_store import (
    AppSettings,
    SettingsStore,
    clear_api_key,
    get_api_key,
    get_settings,
    has_api_key,
    set_api_key,
    update_settings,
)
from src.storage.stats_store import (
    DashboardStats,
    StatsStore,
    get_dashboard_stats,
    record_processing_run,
)

__all__ = [
    "PokedataHistoryStore",
    "record_price",
    "get_history",
    "StatsStore",
    "DashboardStats",
    "record_processing_run",
    "get_dashboard_stats",
    "SettingsStore",
    "AppSettings",
    "get_settings",
    "update_settings",
    "get_api_key",
    "set_api_key",
    "clear_api_key",
    "has_api_key",
]
