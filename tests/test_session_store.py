"""
Tests for the file-based session store.
"""

import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.storage.session_store import (
    SessionData,
    SessionStore,
)


@pytest.fixture
def temp_session_dir():
    """Create a temporary directory for sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def session_store(temp_session_dir):
    """Create a session store with temp directory."""
    return SessionStore(session_dir=temp_session_dir, expiry_hours=1)


@pytest.fixture
def sample_dataframes():
    """Create sample DataFrames for testing."""
    original_df = pd.DataFrame(
        {
            "sku": ["SKU001", "SKU002", "SKU003"],
            "name": ["Product 1", "Product 2", "Product 3"],
            "price": [100.0, 200.0, 300.0],
        }
    )

    results_df = pd.DataFrame(
        {
            "sku": ["SKU001", "SKU002", "SKU003"],
            "name": ["Product 1", "Product 2", "Product 3"],
            "price": [100.0, 200.0, 300.0],
            "new_price_myr": [120.0, 240.0, 360.0],
            "status": ["OK", "WARNING", "OK"],
        }
    )

    return original_df, results_df


class TestSessionData:
    """Tests for SessionData class."""

    def test_is_expired_future(self):
        """Session with future expiry should not be expired."""
        session = SessionData(
            session_id="test-123",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert not session.is_expired()

    def test_is_expired_past(self):
        """Session with past expiry should be expired."""
        session = SessionData(
            session_id="test-123",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert session.is_expired()

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = SessionData(
            session_id="test-456",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=2),
            config={"fx_rate": 4.5, "margin": 0.8},
            metadata={"filename": "test.xlsx"},
        )

        data = original.to_dict()
        restored = SessionData.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.config == original.config
        assert restored.metadata == original.metadata


class TestSessionStore:
    """Tests for SessionStore class."""

    def test_create_session(self, session_store, sample_dataframes):
        """Test creating a new session."""
        original_df, results_df = sample_dataframes

        session = session_store.create(
            session_id="create-test-001",
            original_df=original_df,
            results_df=results_df,
            config={"fx_rate": 4.5},
        )

        assert session.session_id == "create-test-001"
        assert session.config == {"fx_rate": 4.5}
        assert not session.is_expired()

    def test_get_session(self, session_store, sample_dataframes):
        """Test retrieving a session."""
        original_df, results_df = sample_dataframes

        session_store.create(
            session_id="get-test-001",
            original_df=original_df,
            results_df=results_df,
            config={"test": True},
        )

        retrieved = session_store.get("get-test-001")

        assert retrieved is not None
        assert retrieved["session_id"] == "get-test-001"
        assert retrieved["config"] == {"test": True}
        assert retrieved["original_df"] is not None
        assert retrieved["results_df"] is not None
        assert len(retrieved["original_df"]) == 3

    def test_get_nonexistent_session(self, session_store):
        """Test retrieving a non-existent session returns None."""
        result = session_store.get("nonexistent-session")
        assert result is None

    def test_delete_session(self, session_store, sample_dataframes):
        """Test deleting a session."""
        original_df, results_df = sample_dataframes

        session_store.create(
            session_id="delete-test-001",
            original_df=original_df,
            results_df=results_df,
            config={},
        )

        # Verify it exists
        assert session_store.exists("delete-test-001")

        # Delete it
        result = session_store.delete("delete-test-001")
        assert result is True

        # Verify it's gone
        assert not session_store.exists("delete-test-001")

    def test_update_results(self, session_store, sample_dataframes):
        """Test updating results DataFrame."""
        original_df, results_df = sample_dataframes

        session_store.create(
            session_id="update-test-001",
            original_df=original_df,
            results_df=results_df,
            config={},
        )

        # Update results
        new_results = results_df.copy()
        new_results["new_price_myr"] = [150.0, 250.0, 350.0]

        success = session_store.update_results("update-test-001", new_results)
        assert success

        # Verify update
        retrieved = session_store.get("update-test-001")
        assert retrieved["results_df"]["new_price_myr"].tolist() == [150.0, 250.0, 350.0]

    def test_list_sessions(self, session_store, sample_dataframes):
        """Test listing all sessions."""
        original_df, results_df = sample_dataframes

        # Create multiple sessions
        for i in range(3):
            session_store.create(
                session_id=f"list-test-{i:03d}",
                original_df=original_df,
                results_df=results_df,
                config={},
            )

        sessions = session_store.list_sessions()
        assert len(sessions) == 3

    def test_get_stats(self, session_store, sample_dataframes):
        """Test getting store statistics."""
        original_df, results_df = sample_dataframes

        session_store.create(
            session_id="stats-test-001",
            original_df=original_df,
            results_df=results_df,
            config={},
        )

        stats = session_store.get_stats()

        assert stats["active_sessions"] == 1
        assert stats["expiry_hours"] == 1
        assert "total_size_mb" in stats


class TestSessionExpiry:
    """Tests for session expiry behavior."""

    def test_expired_session_not_returned(self, temp_session_dir, sample_dataframes):
        """Expired sessions should return None on get."""
        # Create store with very short expiry
        store = SessionStore(session_dir=temp_session_dir, expiry_hours=0)
        original_df, results_df = sample_dataframes

        store.create(
            session_id="expiry-test-001",
            original_df=original_df,
            results_df=results_df,
            config={},
        )

        # Session should be immediately expired (0 hours expiry)
        result = store.get("expiry-test-001")
        assert result is None

    def test_cleanup_removes_expired(self, temp_session_dir, sample_dataframes):
        """Cleanup should remove expired sessions."""
        store = SessionStore(session_dir=temp_session_dir, expiry_hours=0)
        original_df, results_df = sample_dataframes

        # Create some sessions (will be immediately expired)
        for i in range(3):
            store.create(
                session_id=f"cleanup-test-{i:03d}",
                original_df=original_df,
                results_df=results_df,
                config={},
            )

        # Cleanup
        removed = store.cleanup()
        assert removed == 3

        # Verify directory is clean
        sessions = store.list_sessions()
        assert len(sessions) == 0
