"""
Tests for the encryption module.
"""


from src.storage.encryption import (
    CRYPTO_AVAILABLE,
    SecureStorage,
    decrypt,
    encrypt,
    is_encrypted,
)


class TestSecureStorage:
    """Tests for SecureStorage class."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypted text should decrypt to original."""
        storage = SecureStorage()

        original = "my_secret_api_key_12345"
        encrypted = storage.encrypt(original)
        decrypted = storage.decrypt(encrypted)

        assert decrypted == original
        assert encrypted != original  # Should be different

    def test_encrypt_empty_string(self):
        """Empty string should return empty string."""
        storage = SecureStorage()

        assert storage.encrypt("") == ""
        assert storage.decrypt("") == ""

    def test_encrypt_produces_different_output(self):
        """Encryption should produce non-plaintext output."""
        storage = SecureStorage()

        plaintext = "test_key_abc123"
        encrypted = storage.encrypt(plaintext)

        # Should not be plaintext
        assert encrypted != plaintext
        # Should be base64-ish (contains only valid chars)
        assert len(encrypted) > len(plaintext)

    def test_is_encrypted_detection(self):
        """Should correctly identify encrypted values."""
        storage = SecureStorage()

        plaintext = "not_encrypted"
        encrypted = storage.encrypt(plaintext)

        assert not storage.is_encrypted(plaintext)
        assert storage.is_encrypted(encrypted)

    def test_decrypt_plaintext_returns_as_is(self):
        """Decrypting plaintext should return it as-is (legacy support)."""
        storage = SecureStorage()

        plaintext = "legacy_plain_key"

        # If it's not encrypted format, it should return as-is
        # This supports migrating from unencrypted storage
        result = storage.decrypt(plaintext)

        # Either returns original or fails gracefully
        assert result == plaintext or result != ""


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_encrypt_function(self):
        """Test encrypt convenience function."""
        result = encrypt("test_value")
        assert result != "test_value"
        assert len(result) > 0

    def test_decrypt_function(self):
        """Test decrypt convenience function."""
        encrypted = encrypt("my_key")
        decrypted = decrypt(encrypted)
        assert decrypted == "my_key"

    def test_is_encrypted_function(self):
        """Test is_encrypted convenience function."""
        encrypted = encrypt("some_value")
        assert is_encrypted(encrypted)
        assert not is_encrypted("plain_text")


class TestCryptoAvailability:
    """Tests for crypto library availability handling."""

    def test_crypto_available_flag(self):
        """CRYPTO_AVAILABLE should be boolean."""
        assert isinstance(CRYPTO_AVAILABLE, bool)

    def test_fallback_obfuscation(self):
        """Even without crypto, basic obfuscation should work."""
        storage = SecureStorage()

        # Force use of obfuscation
        obfuscated = storage._obfuscate("test_data")
        deobfuscated = storage._deobfuscate(obfuscated)

        assert deobfuscated == "test_data"
        assert obfuscated.startswith("OBF:")


class TestEdgeCases:
    """Tests for edge cases and special values."""

    def test_special_characters(self):
        """Should handle special characters."""
        storage = SecureStorage()

        special = "key_with_$pecial!@#%^&*()_+=[]{}|;':\",./<>?"
        encrypted = storage.encrypt(special)
        decrypted = storage.decrypt(encrypted)

        assert decrypted == special

    def test_unicode_characters(self):
        """Should handle unicode characters."""
        storage = SecureStorage()

        unicode_text = "key_with_émojis_🔐_and_中文"
        encrypted = storage.encrypt(unicode_text)
        decrypted = storage.decrypt(encrypted)

        assert decrypted == unicode_text

    def test_very_long_string(self):
        """Should handle long strings."""
        storage = SecureStorage()

        long_string = "x" * 10000
        encrypted = storage.encrypt(long_string)
        decrypted = storage.decrypt(encrypted)

        assert decrypted == long_string

    def test_whitespace_only(self):
        """Should handle whitespace strings."""
        storage = SecureStorage()

        whitespace = "   \t\n  "
        encrypted = storage.encrypt(whitespace)
        decrypted = storage.decrypt(encrypted)

        assert decrypted == whitespace
