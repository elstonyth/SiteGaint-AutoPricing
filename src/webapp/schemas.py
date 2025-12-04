"""
Pydantic models for form/config inputs in the web application.

Provides request validation with sensible defaults and constraints.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any


class ProcessingConfig(BaseModel):
    """
    Configuration for price processing.
    
    Validates thresholds and ensures hard > soft.
    """
    
    fx_rate: Optional[float] = Field(
        None,
        ge=1.0,
        le=10.0,
        description="USD to MYR exchange rate override (1.0-10.0)"
    )
    soft_threshold: float = Field(
        20.0,
        ge=0.0,
        le=100.0,
        description="Soft threshold percentage - shows warning"
    )
    hard_threshold: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Hard threshold percentage - blocks export"
    )
    margin_divisor: float = Field(
        0.8,
        ge=0.1,
        le=1.0,
        description="Margin divisor for price calculation (0.1-1.0)"
    )
    api_key_override: Optional[str] = Field(
        None,
        description="Override API key for this session only"
    )
    force_refresh: bool = Field(
        False,
        description="Force refresh prices from API (bypass cache)"
    )
    
    @validator('hard_threshold')
    def hard_must_be_gte_soft(cls, v, values):
        """Ensure hard threshold is greater than or equal to soft threshold."""
        soft = values.get('soft_threshold', 20.0)
        if v < soft:
            raise ValueError(f'hard_threshold ({v}) must be >= soft_threshold ({soft})')
        return v
    
    @validator('api_key_override')
    def strip_api_key(cls, v):
        """Strip whitespace from API key."""
        if v:
            v = v.strip()
            return v if v else None
        return None
    
    class Config:
        """Pydantic config."""
        
        extra = "ignore"  # Ignore extra fields from form


class ExportRequest(BaseModel):
    """Request model for export endpoint."""
    
    session_id: str = Field(
        ...,
        min_length=1,
        description="Session ID for the processing results"
    )
    selected_skus: List[str] = Field(
        default_factory=list,
        description="List of SKUs to include in export"
    )
    edited_prices: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional price overrides: {sku: new_price}"
    )
    
    @validator('selected_skus', pre=True)
    def ensure_list(cls, v):
        """Ensure selected_skus is a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        return list(v)


class StatusCounts(BaseModel):
    """Summary counts by status."""
    
    ok: int = Field(0, ge=0)
    warning: int = Field(0, ge=0)
    blocked: int = Field(0, ge=0)
    no_data: int = Field(0, ge=0)
    unmapped: int = Field(0, ge=0)
    total: int = Field(0, ge=0)
    to_update: int = Field(0, ge=0)
    
    @classmethod
    def from_dataframe(cls, df) -> "StatusCounts":
        """Create StatusCounts from a results DataFrame."""
        import pandas as pd
        
        if df is None or df.empty:
            return cls()
        
        status_counts = df["status"].value_counts() if "status" in df.columns else {}
        include_count = df["include_in_update"].sum() if "include_in_update" in df.columns else 0
        
        return cls(
            ok=int(status_counts.get("OK", 0)),
            warning=int(status_counts.get("WARNING", 0)),
            blocked=int(status_counts.get("BLOCKED", 0)),
            no_data=int(status_counts.get("NO_DATA", 0)),
            unmapped=int(status_counts.get("UNMAPPED", 0)),
            total=len(df),
            to_update=int(include_count),
        )


class SettingsUpdate(BaseModel):
    """Request model for updating settings."""
    
    fx_rate: float = Field(4.50, ge=1.0, le=10.0)
    margin_divisor: float = Field(0.8, ge=0.1, le=1.0)
    soft_threshold: float = Field(20.0, ge=0.0, le=100.0)
    hard_threshold: float = Field(50.0, ge=0.0, le=100.0)
    
    @validator('hard_threshold')
    def hard_must_be_gte_soft(cls, v, values):
        """Ensure hard threshold is greater than or equal to soft threshold."""
        soft = values.get('soft_threshold', 20.0)
        if v < soft:
            raise ValueError(f'hard_threshold ({v}) must be >= soft_threshold ({soft})')
        return v


class CacheStats(BaseModel):
    """Cache statistics response."""
    
    prices: int = 0
    products: int = 0
    searches: int = 0
    total_entries: int = 0
    cache_size_kb: float = 0.0


class MappingInfo(BaseModel):
    """Information about mapping file."""
    
    exists: bool = False
    path: str = ""
    row_count: int = 0
    columns: List[str] = Field(default_factory=list)
    last_modified: Optional[str] = None
    file_size: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# API Response Models
# ============================================================================

class FXRateResponse(BaseModel):
    """Response model for FX rate endpoint."""
    
    success: bool
    rate: float
    source: str
    error: Optional[str] = None


class CacheClearResponse(BaseModel):
    """Response model for cache clear operations."""
    
    success: bool
    message: str
    cleared: Optional[Dict[str, int]] = None
    count: Optional[int] = None


class SessionClearResponse(BaseModel):
    """Response model for session clear operations."""
    
    success: bool
    message: str


class MappingEntryRequest(BaseModel):
    """Request model for adding/updating mapping entries."""
    
    sku: str = Field(..., min_length=1, description="Product SKU")
    name: Optional[str] = Field(None, description="Product name")
    pokedata_id: Optional[str] = Field(None, description="Pokedata product ID")
    pokedata_url: Optional[str] = Field(None, description="Pokedata product URL")
    pokedata_name: Optional[str] = Field(None, description="Product name on Pokedata")
    pokedata_language: str = Field("ENGLISH", description="ENGLISH or JAPANESE")
    pokedata_asset_type: str = Field("PRODUCT", description="PRODUCT or CARD")
    auto_update: str = Field("Y", description="Y or N")
    
    @validator('pokedata_language')
    def validate_language(cls, v):
        v = v.upper().strip()
        if v not in ("ENGLISH", "JAPANESE"):
            raise ValueError("Language must be ENGLISH or JAPANESE")
        return v
    
    @validator('pokedata_asset_type')
    def validate_asset_type(cls, v):
        v = v.upper().strip()
        if v not in ("PRODUCT", "CARD"):
            raise ValueError("Asset type must be PRODUCT or CARD")
        return v
    
    @validator('auto_update')
    def validate_auto_update(cls, v):
        v = v.upper().strip()
        if v not in ("Y", "N"):
            raise ValueError("Auto update must be Y or N")
        return v
