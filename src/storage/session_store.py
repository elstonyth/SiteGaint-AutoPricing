"""
File-based session storage module.

Provides persistent session storage that survives application restarts.
Sessions are stored as individual JSON files with automatic expiry cleanup.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

import pandas as pd

# Check if pyarrow is available for parquet support
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default session directory
DEFAULT_SESSION_DIR = "data/sessions"

# Default session expiry (hours)
DEFAULT_SESSION_EXPIRY_HOURS = 2


@dataclass
class SessionData:
    """Session data container."""
    session_id: str
    created_at: datetime
    expires_at: datetime
    original_df_path: str = ""
    results_df_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "original_df_path": self.original_df_path,
            "results_df_path": self.results_df_path,
            "config": self.config,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            original_df_path=data.get("original_df_path", ""),
            results_df_path=data.get("results_df_path", ""),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )


class SessionStore:
    """
    File-based session storage with automatic expiry cleanup.
    
    Sessions are stored as:
    - Metadata: JSON file with session info
    - DataFrames: Parquet files for efficient storage
    
    Thread-safe with file locking for concurrent access.
    """
    
    def __init__(
        self,
        session_dir: Optional[str] = None,
        expiry_hours: int = DEFAULT_SESSION_EXPIRY_HOURS,
    ) -> None:
        """
        Initialize the session store.
        
        Args:
            session_dir: Directory for session files.
            expiry_hours: Hours until session expires.
        """
        self.session_dir = Path(session_dir or DEFAULT_SESSION_DIR)
        self.expiry_hours = expiry_hours
        self._lock = threading.Lock()
        
        # Ensure directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Run initial cleanup
        self._cleanup_expired()
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get path for session metadata file."""
        return self.session_dir / f"{session_id}.json"
    
    def _get_df_path(self, session_id: str, name: str) -> Path:
        """Get path for DataFrame file."""
        ext = "parquet" if PARQUET_AVAILABLE else "pkl"
        return self.session_dir / f"{session_id}_{name}.{ext}"
    
    def _save_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame using parquet or pickle fallback."""
        if PARQUET_AVAILABLE:
            # Convert object columns with mixed types to string to avoid parquet errors
            df_copy = df.copy()
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    # Convert mixed-type object columns to string
                    df_copy[col] = df_copy[col].astype(str).replace('nan', '')
            df_copy.to_parquet(path, index=False)
        else:
            with open(path, "wb") as f:
                pickle.dump(df, f)
    
    def _load_dataframe(self, path: str) -> pd.DataFrame:
        """Load DataFrame from parquet or pickle."""
        path = Path(path)
        if path.suffix == ".parquet" and PARQUET_AVAILABLE:
            return pd.read_parquet(path)
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif PARQUET_AVAILABLE:
            # Try parquet first
            return pd.read_parquet(path)
        else:
            # Fallback to pickle
            with open(path, "rb") as f:
                return pickle.load(f)
    
    def create(
        self,
        session_id: str,
        original_df: pd.DataFrame,
        results_df: pd.DataFrame,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionData:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier.
            original_df: Original uploaded DataFrame.
            results_df: Processed results DataFrame.
            config: Processing configuration.
            metadata: Optional additional metadata.
            
        Returns:
            SessionData object.
        """
        with self._lock:
            now = datetime.now()
            expires_at = now + timedelta(hours=self.expiry_hours)
            
            # Save DataFrames (use parquet if available, else pickle)
            original_path = self._get_df_path(session_id, "original")
            results_path = self._get_df_path(session_id, "results")
            
            self._save_dataframe(original_df, original_path)
            self._save_dataframe(results_df, results_path)
            
            # Create session data
            session = SessionData(
                session_id=session_id,
                created_at=now,
                expires_at=expires_at,
                original_df_path=str(original_path),
                results_df_path=str(results_path),
                config=config,
                metadata=metadata or {},
            )
            
            # Save metadata
            meta_path = self._get_session_path(session_id)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
            
            logger.info(f"Created session {session_id}, expires at {expires_at}")
            return session
    
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data including DataFrames.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Dictionary with session data, or None if not found/expired.
        """
        meta_path = self._get_session_path(session_id)
        
        if not meta_path.exists():
            return None
        
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            session = SessionData.from_dict(data)
            
            # Check expiry
            if session.is_expired():
                logger.info(f"Session {session_id} has expired, removing")
                self.delete(session_id)
                return None
            
            # Load DataFrames
            original_df = None
            results_df = None
            
            if session.original_df_path and Path(session.original_df_path).exists():
                original_df = self._load_dataframe(session.original_df_path)
            
            if session.results_df_path and Path(session.results_df_path).exists():
                results_df = self._load_dataframe(session.results_df_path)
            
            return {
                "session_id": session.session_id,
                "original_df": original_df,
                "results_df": results_df,
                "config": session.config,
                "created_at": session.created_at,
                "metadata": session.metadata,
            }
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def update_results(self, session_id: str, results_df: pd.DataFrame) -> bool:
        """
        Update the results DataFrame for a session.
        
        Args:
            session_id: Session identifier.
            results_df: Updated results DataFrame.
            
        Returns:
            True if successful, False otherwise.
        """
        meta_path = self._get_session_path(session_id)
        
        if not meta_path.exists():
            return False
        
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            session = SessionData.from_dict(data)
            
            if session.is_expired():
                return False
            
            # Update DataFrame
            results_path = Path(session.results_df_path)
            self._save_dataframe(results_df, results_path)
            
            logger.info(f"Updated results for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session and its files.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            meta_path = self._get_session_path(session_id)
            
            if not meta_path.exists():
                return False
            
            try:
                # Load session to get DataFrame paths
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = SessionData.from_dict(data)
                
                # Delete DataFrame files
                for path_str in [session.original_df_path, session.results_df_path]:
                    if path_str:
                        path = Path(path_str)
                        if path.exists():
                            path.unlink()
                
                # Delete metadata file
                meta_path.unlink()
                
                logger.info(f"Deleted session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")
                return False
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists and is not expired."""
        session = self.get(session_id)
        return session is not None
    
    def _cleanup_expired(self) -> int:
        """
        Remove expired sessions.
        
        Returns:
            Number of sessions removed.
        """
        removed = 0
        
        try:
            for meta_file in self.session_dir.glob("*.json"):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    session = SessionData.from_dict(data)
                    
                    if session.is_expired():
                        self.delete(session.session_id)
                        removed += 1
                        
                except Exception as e:
                    logger.warning(f"Error checking session {meta_file}: {e}")
            
            if removed > 0:
                logger.info(f"Cleaned up {removed} expired sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
        
        return removed
    
    def cleanup(self) -> int:
        """Public method to trigger cleanup."""
        return self._cleanup_expired()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.
        
        Returns:
            List of session summaries.
        """
        sessions = []
        
        for meta_file in self.session_dir.glob("*.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                session = SessionData.from_dict(data)
                
                if not session.is_expired():
                    sessions.append({
                        "session_id": session.session_id,
                        "created_at": session.created_at.isoformat(),
                        "expires_at": session.expires_at.isoformat(),
                        "metadata": session.metadata,
                    })
                    
            except Exception as e:
                logger.warning(f"Error reading session {meta_file}: {e}")
        
        return sessions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session store statistics."""
        sessions = self.list_sessions()
        
        # Calculate total size
        total_size = 0
        for f in self.session_dir.glob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        
        return {
            "active_sessions": len(sessions),
            "session_dir": str(self.session_dir),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "expiry_hours": self.expiry_hours,
        }


# Module-level singleton
_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get or create the singleton session store."""
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def create_session(
    session_id: str,
    original_df: pd.DataFrame,
    results_df: pd.DataFrame,
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> SessionData:
    """Create a new session (convenience function)."""
    return get_session_store().create(session_id, original_df, results_df, config, metadata)


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data (convenience function)."""
    return get_session_store().get(session_id)


def delete_session(session_id: str) -> bool:
    """Delete a session (convenience function)."""
    return get_session_store().delete(session_id)


def session_exists(session_id: str) -> bool:
    """Check if session exists (convenience function)."""
    return get_session_store().exists(session_id)


def cleanup_sessions() -> int:
    """Cleanup expired sessions (convenience function)."""
    return get_session_store().cleanup()
