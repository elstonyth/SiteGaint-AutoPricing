"""
Storage modules for data persistence.
"""

from src.storage.pokedata_history_store import (
    PokedataHistoryStore,
    record_price,
    get_history,
)
from src.storage.stats_store import (
    StatsStore,
    DashboardStats,
    record_processing_run,
    get_dashboard_stats,
)
from src.storage.settings_store import (
    SettingsStore,
    AppSettings,
    get_settings,
    update_settings,
    get_api_key,
    set_api_key,
    clear_api_key,
    has_api_key,
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
