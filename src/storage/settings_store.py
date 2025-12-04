"""
Settings storage module.

Manages persistent storage of application settings including:
- Pokedata API key (encrypted at rest)
- FX rate configuration
- Threshold settings
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.storage.encryption import encrypt, decrypt, is_encrypted

logger = logging.getLogger(__name__)

# Default path for settings file
DEFAULT_SETTINGS_PATH = "data/cache/app_settings.json"


@dataclass
class AppSettings:
    """Application settings stored persistently."""
    # API Key (stored securely - not in version control)
    pokedata_api_key: str = ""
    
    # FX Rate settings
    fx_rate: float = 4.50
    fx_source: str = "manual"  # "manual" or "google"
    
    # Pricing settings
    margin_divisor: float = 0.8
    
    # Threshold settings
    soft_threshold: float = 20.0
    hard_threshold: float = 50.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        """Create from dictionary."""
        return cls(
            pokedata_api_key=data.get("pokedata_api_key", ""),
            fx_rate=float(data.get("fx_rate", 4.50)),
            fx_source=data.get("fx_source", "manual"),
            margin_divisor=float(data.get("margin_divisor", 0.8)),
            soft_threshold=float(data.get("soft_threshold", 20.0)),
            hard_threshold=float(data.get("hard_threshold", 50.0)),
        )


class SettingsStore:
    """Manages persistent storage of application settings."""
    
    def __init__(self, settings_path: Optional[str] = None) -> None:
        """Initialize the settings store."""
        self.settings_path = Path(settings_path or DEFAULT_SETTINGS_PATH)
        self._settings: Optional[AppSettings] = None
    
    def _ensure_file_exists(self) -> None:
        """Create the settings file with defaults if it doesn't exist."""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.settings_path.exists():
            self._save(AppSettings())
            logger.info(f"Created default settings file: {self.settings_path}")
    
    def _load(self) -> AppSettings:
        """Load settings from file."""
        self._ensure_file_exists()
        
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return AppSettings.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load settings, using defaults: {e}")
            return AppSettings()
    
    def _save(self, settings: AppSettings) -> None:
        """Save settings to file."""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(settings.to_dict(), f, indent=2)
        
        self._settings = settings
        logger.info("Settings saved")
    
    def get(self) -> AppSettings:
        """Get current settings."""
        if self._settings is None:
            self._settings = self._load()
        return self._settings
    
    def update(self, **kwargs) -> AppSettings:
        """Update specific settings."""
        settings = self.get()
        
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        
        self._save(settings)
        return settings
    
    def set_api_key(self, api_key: str) -> None:
        """Set the Pokedata API key (encrypted at rest)."""
        # Encrypt before storing
        encrypted_key = encrypt(api_key.strip())
        self.update(pokedata_api_key=encrypted_key)
        logger.info("API key updated (encrypted)")
    
    def clear_api_key(self) -> None:
        """Clear the Pokedata API key."""
        self.update(pokedata_api_key="")
        logger.info("API key cleared")
    
    def has_api_key(self) -> bool:
        """Check if an API key is configured."""
        return bool(self.get().pokedata_api_key)
    
    def get_api_key(self) -> str:
        """Get the API key (decrypted)."""
        stored_key = self.get().pokedata_api_key
        if not stored_key:
            return ""
        # Decrypt if encrypted, otherwise return as-is (legacy)
        if is_encrypted(stored_key):
            return decrypt(stored_key)
        return stored_key


# Module-level singleton
_store: Optional[SettingsStore] = None


def _get_store() -> SettingsStore:
    """Get or create the singleton store instance."""
    global _store
    if _store is None:
        _store = SettingsStore()
    return _store


def get_settings() -> AppSettings:
    """Get current settings (convenience function)."""
    return _get_store().get()


def update_settings(**kwargs) -> AppSettings:
    """Update settings (convenience function)."""
    return _get_store().update(**kwargs)


def get_api_key() -> str:
    """Get the API key (convenience function).
    
    Checks environment variable first (for Docker), then falls back to stored key.
    """
    import os
    # Check environment variable first (preferred for Docker)
    env_key = os.environ.get("POKEDATA_API_KEY", "")
    if env_key:
        return env_key
    return _get_store().get_api_key()


def set_api_key(api_key: str) -> None:
    """Set the API key (convenience function)."""
    _get_store().set_api_key(api_key)


def clear_api_key() -> None:
    """Clear the API key (convenience function)."""
    _get_store().clear_api_key()


def has_api_key() -> bool:
    """Check if API key exists (convenience function)."""
    return _get_store().has_api_key()
