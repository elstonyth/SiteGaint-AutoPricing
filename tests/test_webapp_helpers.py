import asyncio
import datetime
import tempfile

import pandas as pd
import pytest

from src.webapp.main import health_check, readiness_check
from src.webapp.routes import (
    cleanup_expired_sessions,
    sessions,
)
from src.storage.session_store import (
    DEFAULT_SESSION_EXPIRY_HOURS,
    SessionStore,
)
from src.webapp.helpers import (
    format_results_for_display,
    validate_mapping_columns,
)
from src.webapp.schemas import StatusCounts


@pytest.fixture(autouse=True)
def clear_sessions():
    sessions.clear()
    yield
    sessions.clear()


def test_cleanup_expired_sessions_removes_old_entries():
    """Test that expired sessions are cleaned up by the file-based store."""
    # Use a temporary directory for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(session_dir=tmpdir, expiry_hours=0)
        
        # Create a sample dataframe
        df = pd.DataFrame({"sku": ["A"], "price": [100.0]})
        
        # Create a session (will be immediately expired with 0 hours expiry)
        store.create(
            session_id="test-old",
            original_df=df,
            results_df=df,
            config={},
        )
        
        # Cleanup should remove it
        removed = store.cleanup()
        assert removed == 1


def test_validate_mapping_columns_reports_missing_required_fields():
    df = pd.DataFrame({"sku": ["A"], "pokedata_id": [1]})
    missing = validate_mapping_columns(df)
    assert set(missing) == {"pokedata_language", "auto_update"}


def test_get_status_counts_and_format_results():
    df = pd.DataFrame(
        {
            "status": ["OK", "WARNING", "OK"],
            "include_in_update": [True, False, True],
            "sku": ["A", "B", "C"],
            "name": ["n1", "n2", "n3"],
            "price": [1, 2, 3],
            "new_price_myr": [1.5, 2.5, 3.5],
            "abs_change": [0.5, 0.5, 0.5],
            "pct_change": [50, 25, 15],
        }
    )
    counts = StatusCounts.from_dataframe(df)
    assert counts.ok == 2
    assert counts.warning == 1
    assert counts.to_update == 2

    formatted = format_results_for_display(df.copy())
    assert len(formatted) == 3
    assert formatted[0]["sku"] == "A"


@pytest.mark.anyio
async def test_health_and_readiness_endpoints():
    health = await health_check()
    assert health["status"] == "healthy"
    assert "timestamp" in health

    ready = await readiness_check()
    assert ready["status"] == "ready"
