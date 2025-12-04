"""
Health check service for monitoring application status.

Provides detailed health information about all system components.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config_loader import AppConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    
    name: str
    status: str  # "ok", "degraded", "error"
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        result = {
            "status": self.status,
        }
        if self.message:
            result["message"] = self.message
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.details:
            result.update(self.details)
        return result


@dataclass
class HealthReport:
    """Complete health report for the application."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    version: str
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "version": self.version,
            "components": {
                name: comp.to_dict() 
                for name, comp in self.components.items()
            },
        }


class HealthService:
    """Service for checking application health."""
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.project_root = Path(__file__).parent.parent.parent
    
    def get_full_health(self) -> HealthReport:
        """
        Get complete health report for all components.
        
        Returns:
            HealthReport with status of all components.
        """
        components = {}
        
        # Check each component
        components["api_key"] = self._check_api_key()
        components["mapping_file"] = self._check_mapping_file()
        components["fx_rate"] = self._check_fx_rate()
        components["cache"] = self._check_cache()
        components["storage"] = self._check_storage()
        
        # Determine overall status
        statuses = [c.status for c in components.values()]
        if all(s == "ok" for s in statuses):
            overall_status = "healthy"
        elif any(s == "error" for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return HealthReport(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            version=self.VERSION,
            components=components,
        )
    
    def get_simple_health(self) -> dict:
        """Get simple health check (for load balancers)."""
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    
    def _check_api_key(self) -> ComponentHealth:
        """Check if Pokedata API key is configured."""
        from src.storage.settings_store import get_api_key
        
        try:
            api_key = get_api_key()
            env_key = os.environ.get("POKEDATA_API_KEY", "")
            
            if api_key or env_key:
                return ComponentHealth(
                    name="api_key",
                    status="ok",
                    message="API key configured",
                    details={"source": "stored" if api_key else "environment"},
                )
            else:
                return ComponentHealth(
                    name="api_key",
                    status="degraded",
                    message="No API key - running in demo mode",
                )
        except Exception as e:
            return ComponentHealth(
                name="api_key",
                status="error",
                message=str(e),
            )
    
    def _check_mapping_file(self) -> ComponentHealth:
        """Check if mapping file exists and is valid."""
        try:
            mapping_path = self.project_root / getattr(
                self.config.paths, 
                'mapping_master_path', 
                'data/mapping/master_mapping.xlsx'
            )
            
            if not mapping_path.exists():
                return ComponentHealth(
                    name="mapping_file",
                    status="degraded",
                    message="Mapping file not found",
                    details={"path": str(mapping_path)},
                )
            
            # Try to read the file
            import pandas as pd
            from src.utils.excel_utils import read_excel_file
            
            start = time.time()
            df = read_excel_file(mapping_path)
            latency = (time.time() - start) * 1000
            
            return ComponentHealth(
                name="mapping_file",
                status="ok",
                message=f"{len(df)} products mapped",
                latency_ms=latency,
                details={
                    "row_count": len(df),
                    "path": str(mapping_path),
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="mapping_file",
                status="error",
                message=str(e),
            )
    
    def _check_fx_rate(self) -> ComponentHealth:
        """Check FX rate availability."""
        try:
            from src.pricing.fx_provider import get_fx_rate
            
            start = time.time()
            rate, source = get_fx_rate(self.config)
            latency = (time.time() - start) * 1000
            
            if rate and rate > 0:
                return ComponentHealth(
                    name="fx_rate",
                    status="ok",
                    message=f"Rate: {rate:.4f}",
                    latency_ms=latency,
                    details={
                        "rate": rate,
                        "source": source,
                        "currency_pair": "USD/MYR",
                    },
                )
            else:
                return ComponentHealth(
                    name="fx_rate",
                    status="degraded",
                    message="Using fallback rate",
                    details={"rate": self.config.fx.default_rate},
                )
        except Exception as e:
            return ComponentHealth(
                name="fx_rate",
                status="error",
                message=str(e),
            )
    
    def _check_cache(self) -> ComponentHealth:
        """Check cache directory status."""
        try:
            cache_config = getattr(self.config, 'cache', None)
            if not cache_config:
                return ComponentHealth(
                    name="cache",
                    status="ok",
                    message="Cache not configured",
                )
            cache_dir = self.project_root / getattr(cache_config, 'cache_dir', 'data/cache')
            
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Count cache entries
            cache_files = list(cache_dir.glob("*.json")) + list(cache_dir.glob("*.csv"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return ComponentHealth(
                name="cache",
                status="ok",
                message=f"{len(cache_files)} entries",
                details={
                    "entries": len(cache_files),
                    "size_kb": round(total_size / 1024, 2),
                    "enabled": self.config.cache.enabled,
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="cache",
                status="error",
                message=str(e),
            )
    
    def _check_storage(self) -> ComponentHealth:
        """Check storage directories."""
        try:
            paths_config = getattr(self.config, 'paths', None)
            dirs_to_check = [
                ("input", getattr(paths_config, 'sitegiant_export_dir', 'data/input') if paths_config else 'data/input'),
                ("output", getattr(paths_config, 'output_dir', 'data/output') if paths_config else 'data/output'),
                ("mapping", getattr(paths_config, 'mapping_dir', 'data/mapping') if paths_config else 'data/mapping'),
            ]
            
            missing = []
            for name, rel_path in dirs_to_check:
                path = self.project_root / rel_path
                if not path.exists():
                    missing.append(name)
                    path.mkdir(parents=True, exist_ok=True)
            
            if missing:
                return ComponentHealth(
                    name="storage",
                    status="degraded",
                    message=f"Created missing directories: {', '.join(missing)}",
                )
            
            return ComponentHealth(
                name="storage",
                status="ok",
                message="All directories exist",
            )
        except Exception as e:
            return ComponentHealth(
                name="storage",
                status="error",
                message=str(e),
            )


# Singleton instance
_health_service: Optional[HealthService] = None


def get_health_service() -> HealthService:
    """Get or create the health service singleton."""
    global _health_service
    if _health_service is None:
        _health_service = HealthService()
    return _health_service
