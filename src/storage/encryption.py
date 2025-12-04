"""
Simple encryption utilities for sensitive data at rest.

Uses Fernet symmetric encryption with a machine-derived key.
This provides basic protection - not suitable for highly sensitive data
but sufficient for API keys in a local application.
"""

import base64
import hashlib
import logging
import os
import platform
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import cryptography, fall back to base64 encoding if not available
try:
    from cryptography.fernet import Fernet, InvalidToken
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not installed, using basic obfuscation only")


def _get_machine_id() -> str:
    """
    Get a machine-specific identifier for key derivation.
    
    Combines multiple machine attributes to create a stable identifier
    that's unique to this machine but consistent across runs.
    """
    components = []
    
    # Platform info
    components.append(platform.node())
    components.append(platform.machine())
    components.append(platform.system())
    
    # Try to get more hardware-specific info
    try:
        # Windows: use machine GUID
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography"
            )
            machine_guid, _ = winreg.QueryValueEx(key, "MachineGuid")
            components.append(machine_guid)
            winreg.CloseKey(key)
    except Exception:
        pass
    
    try:
        # Add username as additional entropy
        components.append(os.getlogin())
    except Exception:
        pass
    
    return "|".join(components)


def _derive_key(salt: bytes = b"sitegiant_pricing_v1") -> bytes:
    """
    Derive an encryption key from machine-specific data.
    
    Args:
        salt: Additional salt for key derivation.
        
    Returns:
        32-byte key suitable for Fernet.
    """
    machine_id = _get_machine_id()
    
    # Use PBKDF2-like derivation
    key_material = hashlib.pbkdf2_hmac(
        "sha256",
        machine_id.encode(),
        salt,
        iterations=100000,
        dklen=32,
    )
    
    # Fernet requires URL-safe base64 encoded 32-byte key
    return base64.urlsafe_b64encode(key_material)


class SecureStorage:
    """
    Provides encryption/decryption for sensitive strings.
    
    Uses Fernet symmetric encryption if available,
    falls back to base64 obfuscation if not.
    """
    
    def __init__(self) -> None:
        """Initialize secure storage."""
        if CRYPTO_AVAILABLE:
            self._key = _derive_key()
            self._fernet = Fernet(self._key)
        else:
            self._fernet = None
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.
        
        Args:
            plaintext: String to encrypt.
            
        Returns:
            Encrypted string (base64 encoded).
        """
        if not plaintext:
            return ""
        
        if self._fernet:
            try:
                encrypted = self._fernet.encrypt(plaintext.encode())
                return encrypted.decode()
            except Exception as e:
                logger.error(f"Encryption failed: {e}")
                return self._obfuscate(plaintext)
        else:
            return self._obfuscate(plaintext)
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a string.
        
        Args:
            ciphertext: Encrypted string.
            
        Returns:
            Decrypted plaintext.
        """
        if not ciphertext:
            return ""
        
        if self._fernet:
            try:
                decrypted = self._fernet.decrypt(ciphertext.encode())
                return decrypted.decode()
            except InvalidToken:
                # Try legacy obfuscation format
                logger.debug("Fernet decryption failed, trying obfuscation format")
                return self._deobfuscate(ciphertext)
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                return self._deobfuscate(ciphertext)
        else:
            return self._deobfuscate(ciphertext)
    
    def _obfuscate(self, plaintext: str) -> str:
        """Simple base64 obfuscation fallback."""
        # Double-encode with a marker prefix
        encoded = base64.b64encode(plaintext.encode()).decode()
        return f"OBF:{encoded}"
    
    def _deobfuscate(self, ciphertext: str) -> str:
        """Reverse obfuscation."""
        try:
            if ciphertext.startswith("OBF:"):
                encoded = ciphertext[4:]
                return base64.b64decode(encoded.encode()).decode()
            else:
                # Try direct base64 decode
                return base64.b64decode(ciphertext.encode()).decode()
        except Exception:
            # Return as-is if we can't decode (might be plaintext)
            return ciphertext
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value appears to be encrypted."""
        if not value:
            return False
        
        # Check for obfuscation prefix
        if value.startswith("OBF:"):
            return True
        
        # Check for Fernet format (starts with gAAAAA)
        if value.startswith("gAAAAA"):
            return True
        
        return False


# Module-level singleton
_storage: Optional[SecureStorage] = None


def get_secure_storage() -> SecureStorage:
    """Get or create the singleton secure storage."""
    global _storage
    if _storage is None:
        _storage = SecureStorage()
    return _storage


def encrypt(plaintext: str) -> str:
    """Encrypt a string (convenience function)."""
    return get_secure_storage().encrypt(plaintext)


def decrypt(ciphertext: str) -> str:
    """Decrypt a string (convenience function)."""
    return get_secure_storage().decrypt(ciphertext)


def is_encrypted(value: str) -> bool:
    """Check if value is encrypted (convenience function)."""
    return get_secure_storage().is_encrypted(value)
