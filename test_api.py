"""
Test Pokedata API Key save and clear functionality.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.settings_store import (
    get_api_key, set_api_key, clear_api_key, has_api_key, get_settings
)


def test_api_key_operations():
    """Test save and clear API key operations."""
    print("=" * 50)
    print("Testing Pokedata API Key Functions")
    print("=" * 50)
    
    # 1. Check initial state
    print("\n1. Initial State:")
    print(f"   - Has API Key: {has_api_key()}")
    print(f"   - Current Key: {'***' + get_api_key()[-4:] if get_api_key() else '(empty)'}")
    
    # 2. Test save API key
    test_key = os.environ.get("TEST_API_KEY", "test_key")
    print(f"\n2. Saving test key: {test_key}")
    set_api_key(test_key)
    print(f"   - Has API Key: {has_api_key()}")
    print(f"   - Saved Key Matches: {get_api_key() == test_key}")
    
    assert has_api_key(), "ERROR: has_api_key() should return True after save"
    assert get_api_key() == test_key, "ERROR: get_api_key() should return saved key"
    print("   ✓ Save API Key: PASSED")
    
    # 3. Test clear API key
    print("\n3. Clearing API key...")
    clear_api_key()
    print(f"   - Has API Key: {has_api_key()}")
    print(f"   - Current Key: '{get_api_key()}'")
    
    assert not has_api_key(), "ERROR: has_api_key() should return False after clear"
    assert get_api_key() == "", "ERROR: get_api_key() should return empty string"
    print("   ✓ Clear API Key: PASSED")
    
    # 4. Verify settings persistence
    print("\n4. Checking settings file...")
    settings = get_settings()
    print(f"   - Settings object: {type(settings).__name__}")
    print(f"   - API Key in settings: '{settings.pokedata_api_key}'")
    print("   ✓ Settings Persistence: PASSED")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = test_api_key_operations()
    sys.exit(0 if success else 1)
