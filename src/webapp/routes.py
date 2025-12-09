"""
FastAPI routes for the SiteGiant Pricing Automation web application.

Handles:
- Main page with SiteGiant export upload (mapping loaded from config)
- Price processing
- Results preview with checkboxes
- Excel export
- Settings page for mapping file management
"""

import logging
import os
import uuid
from functools import lru_cache
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, Body
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.exporter.sitegiant_exporter import (
    build_update_dataframe,
    write_update_file,
)
from src.mapping.mapping_manager import load_mapping, scan_duplicates, DuplicateScanResult
from src.pokedata_client.api_client import PokedataClient
from src.pricing.fx_provider import fetch_google_fx_rate, get_fx_rate
from src.services.pricing_service import PricingService, ProcessingConfig as ServiceProcessingConfig
from src.storage.settings_store import (
    clear_api_key,
    get_api_key,
    get_settings,
    has_api_key as stored_has_api_key,
    set_api_key,
    update_settings,
)
from src.storage.session_store import (
    create_session,
    get_session as get_stored_session,
    delete_session,
    cleanup_sessions,
)
from src.storage.stats_store import get_dashboard_stats, record_processing_run
from src.utils.config_loader import AppConfig, load_config, load_env
from src.webapp.exceptions import SessionNotFoundError
from src.webapp.helpers import (
    get_mapping_info,
    validate_sitegiant_columns,
    validate_mapping_columns,
    format_results_for_display,
)
from src.webapp.schemas import (
    ProcessingConfig,
    StatusCounts,
    FXRateResponse,
    CacheClearResponse,
    CacheStats,
)
from src.utils.pokedata_url_parser import (
    parse_pokedata_url,
    parse_pokedata_urls_bulk,
    build_pokedata_url,
    validate_pokedata_id,
)


logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Initialize templates
templates_dir = PROJECT_ROOT / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Session configuration - now uses file-based storage (see src/storage/session_store.py)
# Sessions persist across application restarts

# Legacy in-memory fallback (kept for backwards compatibility during migration)
sessions: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Dependency Injection
# ============================================================================

@lru_cache()
def get_app_config() -> AppConfig:
    """
    Get application config (cached).
    
    Use as a FastAPI dependency to avoid repeated config loading.
    Clear cache with get_app_config.cache_clear() if config changes.
    """
    load_env()
    return load_config()


def get_fresh_config() -> AppConfig:
    """Get fresh config without caching (for settings updates)."""
    load_env()
    return load_config()


# ============================================================================
# Template Context Helpers
# ============================================================================

def build_index_context(
    request: Request,
    config: AppConfig,
    **overrides,
) -> Dict[str, Any]:
    """
    Build standard context for index.html template.
    
    Reduces code duplication in error handlers.
    
    Args:
        request: FastAPI request object.
        config: Application configuration.
        **overrides: Additional context values to override defaults.
        
    Returns:
        Dict with template context.
    """
    fx_rate, fx_source = get_fx_rate(config)
    mapping_path = get_mapping_master_path(config)
    mapping_info = get_mapping_info(mapping_path)
    settings = get_settings()
    settings.fx_rate = fx_rate
    
    context = {
        "request": request,
        "fx_rate": fx_rate,
        "fx_source": fx_source,
        "api_key_set": stored_has_api_key(),
        "mapping_exists": mapping_info["exists"],
        "mapping_row_count": mapping_info.get("row_count", 0),
        "stats": get_dashboard_stats(),
        "settings": settings,
        "soft_threshold": 20.0,
        "hard_threshold": 50.0,
        "margin_divisor": 0.8,
    }
    context.update(overrides)
    return context


def cleanup_expired_sessions() -> int:
    """Remove expired sessions (delegates to file-based store)."""
    return cleanup_sessions()


router = APIRouter()


def get_mapping_master_path(config: AppConfig) -> Path:
    """Get the master mapping file path from config."""
    mapping_path = getattr(config.paths, 'mapping_master_path', 'data/mapping/master_mapping.xlsx')
    return PROJECT_ROOT / mapping_path


def get_session(session_id: str) -> Dict[str, Any]:
    """
    Get session by ID or raise SessionNotFoundError.
    
    Uses file-based session storage with fallback to legacy in-memory.
    
    Args:
        session_id: Session identifier.
        
    Returns:
        Session data dictionary.
        
    Raises:
        SessionNotFoundError: If session not found.
    """
    # Try file-based store first
    session = get_stored_session(session_id)
    if session is not None:
        return session
    
    # Fallback to legacy in-memory (for active sessions during migration)
    if session_id in sessions:
        return sessions[session_id]
    
    raise SessionNotFoundError(session_id)


# Note: process_prices logic has been moved to PricingService
# Use: pricing_service = PricingService(app_config)
#      result = pricing_service.process(sitegiant_df, mapping_df, config)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main upload page (SiteGiant export only, mapping from config)."""
    load_env()
    config = load_config()
    
    # Get stored settings
    settings = get_settings()
    api_key_set = stored_has_api_key()
    
    # Check if mapping file exists
    mapping_path = get_mapping_master_path(config)
    mapping_info = get_mapping_info(mapping_path)
    
    # Get dashboard stats
    stats = get_dashboard_stats()
    
    # Fetch live FX rate (mode is "google" in config)
    live_fx_rate, fx_source = get_fx_rate(config)
    
    # Update settings with live rate for display
    settings.fx_rate = live_fx_rate
    settings.fx_source = fx_source
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "api_key_set": api_key_set,
        "mapping_exists": mapping_info["exists"],
        "mapping_row_count": mapping_info.get("row_count", 0),
        "stats": stats,
        "settings": settings,
        "fx_source": fx_source,
    })


@router.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    sitegiant_file: UploadFile = File(...),
    fx_rate: Optional[float] = Form(None),
    soft_threshold: float = Form(20.0),
    hard_threshold: float = Form(50.0),
    margin_divisor: float = Form(0.8),
    api_key_override: Optional[str] = Form(None),
    force_refresh: bool = Form(False),
):
    """Process SiteGiant export using master mapping from config."""
    # Use fresh config (not cached) in case settings were updated
    app_config = get_fresh_config()
    
    # Check for mapping file first
    mapping_path = get_mapping_master_path(app_config)
    mapping_info = get_mapping_info(mapping_path)
    
    if not mapping_info["exists"]:
        # No mapping file - show error using helper
        context = build_index_context(
            request, app_config,
            error="No mapping file configured. Go to Settings → Mapping to upload one.",
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            margin_divisor=margin_divisor,
        )
        return templates.TemplateResponse("index.html", context)
    
    try:
        # Read uploaded SiteGiant file
        sitegiant_content = await sitegiant_file.read()
        
        # Load SiteGiant export
        filename = sitegiant_file.filename or "unknown.xlsx"
        filename_lower = filename.lower()
        if filename_lower.endswith('.csv'):
            sitegiant_df = pd.read_csv(BytesIO(sitegiant_content))
        elif filename_lower.endswith('.xlsx'):
            sitegiant_df = pd.read_excel(BytesIO(sitegiant_content), engine='openpyxl')
        elif filename_lower.endswith('.xls'):
            sitegiant_df = pd.read_excel(BytesIO(sitegiant_content), engine='xlrd')
        else:
            raise ValueError(f"Unsupported file format: {sitegiant_file.filename}. Please upload .xlsx, .xls, or .csv files.")
        
        # ============================================================
        # Phase 1: Validate uploaded file has required columns
        # ============================================================
        validation_errors = validate_sitegiant_columns(sitegiant_df)
        if validation_errors:
            context = build_index_context(
                request, app_config,
                error=f"Invalid SiteGiant export file: {'; '.join(validation_errors)}. "
                      f"Please use the standard SiteGiant product export template.",
                soft_threshold=soft_threshold,
                hard_threshold=hard_threshold,
                margin_divisor=margin_divisor,
            )
            return templates.TemplateResponse("index.html", context)
        
        # Load mapping from master mapping path
        mapping_df = load_mapping(mapping_path, app_config)
        
        # Create processing config for service
        service_config = ServiceProcessingConfig(
            fx_rate=fx_rate if fx_rate and fx_rate > 0 else None,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            margin_divisor=margin_divisor,
            api_key_override=api_key_override if api_key_override and api_key_override.strip() else None,
            force_refresh=force_refresh,
        )
        
        # Process prices using PricingService
        pricing_service = PricingService(app_config)
        result = pricing_service.process(sitegiant_df, mapping_df, service_config)
        results_df = result.results_df
        
        # Generate session ID and store results (file-based for persistence)
        session_id = str(uuid.uuid4())
        
        # Convert config to dict for serialization
        config_dict = {
            "fx_rate": service_config.fx_rate,
            "soft_threshold": service_config.soft_threshold,
            "hard_threshold": service_config.hard_threshold,
            "margin_divisor": service_config.margin_divisor,
            "force_refresh": service_config.force_refresh,
        }
        
        # Store in file-based session store
        create_session(
            session_id=session_id,
            original_df=sitegiant_df,
            results_df=results_df,
            config=config_dict,
            metadata={"filename": sitegiant_file.filename},
        )
        
        # Get status counts and format results
        status_counts = StatusCounts.from_dataframe(results_df)
        results_list = format_results_for_display(results_df)
        
        # Get FX info from results (use result object's values)
        fx_rate_used = result.fx_rate_used
        fx_source_used = result.fx_source
        
        # Record processing stats
        avg_pct_change = 0.0
        if "pct_change" in results_df.columns:
            valid_changes = results_df["pct_change"].dropna()
            if len(valid_changes) > 0:
                avg_pct_change = valid_changes.mean()
        
        products_updated = results_df["include_in_update"].sum() if "include_in_update" in results_df.columns else 0
        
        record_processing_run(
            products_processed=len(results_df),
            products_updated=int(products_updated),
            ok_count=status_counts.ok,
            warning_count=status_counts.warning,
            blocked_count=status_counts.blocked,
            no_data_count=status_counts.no_data,
            unmapped_count=status_counts.unmapped,
            avg_price_change_pct=avg_pct_change,
            fx_rate=fx_rate_used if fx_rate_used else 0.0,
        )
        
        # Use result object for demo mode info
        return templates.TemplateResponse("preview.html", {
            "request": request,
            "session_id": session_id,
            "results": results_list,
            "status_counts": status_counts,
            "fx_rate_used": fx_rate_used,
            "fx_source": fx_source_used,
            "soft_threshold": soft_threshold,
            "hard_threshold": hard_threshold,
            "sitegiant_filename": sitegiant_file.filename,
            "mapping_row_count": mapping_info.get("row_count", 0),
            "demo_mode": result.demo_mode,
            "demo_warning": result.demo_warning,
        })
        
    except Exception as e:
        logger.exception(f"Processing error: {e}")
        context = build_index_context(
            request, app_config,
            error=str(e),
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            margin_divisor=margin_divisor,
        )
        return templates.TemplateResponse("index.html", context)


@router.post("/export")
async def export(
    request: Request,
    session_id: str = Form(...),
    selected_skus: List[str] = Form(default=[]),
):
    """Export selected rows to Excel file with optional price edits."""
    try:
        session = get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    original_df = session["original_df"]
    results_df = session["results_df"].copy()
    config = session["config"]
    
    # ============================================================
    # Parse edited prices from form data (edited_prices[SKU] = value)
    # ============================================================
    form_data = await request.form()
    edited_prices = {}
    for key, value in form_data.multi_items():
        if key.startswith("edited_prices[") and key.endswith("]"):
            sku = key[14:-1]  # Extract SKU from "edited_prices[SKU]"
            try:
                edited_prices[sku] = float(value)
            except (ValueError, TypeError):
                pass
    
    # Apply edited prices to results_df
    if edited_prices and "sku" in results_df.columns:
        edited_count = 0
        for sku, new_price in edited_prices.items():
            mask = results_df["sku"] == sku
            if mask.any():
                old_price = results_df.loc[mask, "new_price_myr"].iloc[0]
                if pd.notna(old_price) and abs(float(old_price) - new_price) > 0.01:
                    results_df.loc[mask, "new_price_myr"] = new_price
                    # Recalculate abs_change and pct_change
                    current_price = results_df.loc[mask, "price"].iloc[0]
                    if pd.notna(current_price) and current_price > 0:
                        results_df.loc[mask, "abs_change"] = new_price - float(current_price)
                        results_df.loc[mask, "pct_change"] = ((new_price - float(current_price)) / float(current_price)) * 100
                    edited_count += 1
        if edited_count > 0:
            logger.info(f"Applied {edited_count} manual price edits from user")
    
    # ============================================================
    # Phase 2: Block export if DEMO MODE is active
    # ============================================================
    demo_mode = results_df["demo_mode"].iloc[0] if "demo_mode" in results_df.columns else False
    if demo_mode:
        logger.warning("Export blocked: DEMO MODE is active")
        # Get data needed for preview page
        status_counts = StatusCounts.from_dataframe(results_df)
        results_list = format_results_for_display(results_df)
        fx_rate_used = results_df["fx_rate_used"].iloc[0] if "fx_rate_used" in results_df.columns else config.get("fx_rate", 0)
        fx_source_used = results_df["fx_source"].iloc[0] if "fx_source" in results_df.columns else "unknown"
        demo_warning = results_df["demo_mode_warning"].iloc[0] if "demo_mode_warning" in results_df.columns else ""
        
        return templates.TemplateResponse("preview.html", {
            "request": request,
            "session_id": session_id,
            "results": results_list,
            "status_counts": status_counts,
            "fx_rate_used": fx_rate_used,
            "fx_source": fx_source_used,
            "soft_threshold": config.get("soft_threshold", 20.0),
            "hard_threshold": config.get("hard_threshold", 50.0),
            "sitegiant_filename": "uploaded file",
            "mapping_row_count": 0,
            "demo_mode": True,
            "demo_warning": demo_warning,
            "export_error": "Export is disabled in DEMO MODE. Configure your Pokedata API key in Settings to fetch real prices before exporting.",
        })
    
    # Update include_in_update based on selected SKUs
    if selected_skus:
        # Handle both normalized 'sku' and original column names
        if "sku" in results_df.columns:
            results_df["include_in_update"] = results_df["sku"].isin(selected_skus)
        elif "SKU" in results_df.columns:
            results_df["include_in_update"] = results_df["SKU"].isin(selected_skus)
        else:
            results_df["include_in_update"] = False
    else:
        # If no SKUs provided, include based on form checkboxes (all unchecked = none)
        results_df["include_in_update"] = False
    
    # ============================================================
    # Phase 3: Force BLOCKED rows to NEVER be exported
    # Even if user selected them, we override to False
    # ============================================================
    if "status" in results_df.columns:
        blocked_mask = results_df["status"] == "BLOCKED"
        blocked_count = blocked_mask.sum()
        if blocked_count > 0:
            results_df.loc[blocked_mask, "include_in_update"] = False
            logger.info(f"Phase 3 safety: Excluded {blocked_count} BLOCKED rows from export")
    
    # Also exclude UNMAPPED rows (they have no valid new price anyway)
    if "status" in results_df.columns:
        unmapped_mask = results_df["status"] == "UNMAPPED"
        unmapped_count = unmapped_mask.sum()
        if unmapped_count > 0:
            results_df.loc[unmapped_mask, "include_in_update"] = False
            logger.info(f"Phase 3 safety: Excluded {unmapped_count} UNMAPPED rows from export")
    
    # Build update dataframe - preserves original SiteGiant column names for re-import
    update_df = build_update_dataframe(original_df, results_df)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        update_df.to_excel(writer, sheet_name='Products', index=False)
        
        # Add summary sheet
        config = session["config"]
        summary_data = {
            "Metric": [
                "Total Products",
                "Products Updated",
                "FX Rate Used",
                "Soft Threshold %",
                "Hard Threshold %",
                "Export Date",
            ],
            "Value": [
                len(update_df),
                results_df["include_in_update"].sum(),
                results_df["fx_rate_used"].iloc[0] if "fx_rate_used" in results_df.columns else "",
                config.get("soft_threshold", 20.0),
                config.get("hard_threshold", 50.0),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    
    # Also save to data/output directory
    try:
        output_dir = Path(__file__).parent.parent.parent / "data" / "output"
        write_update_file(update_df, output_dir)
    except Exception as e:
        logger.warning(f"Failed to save local copy: {e}")
    
    # ============================================================
    # Log price exports to history
    # ============================================================
    try:
        from src.storage.price_history import log_price_export
        
        # Helper to safely get value from row, handling NaN
        def safe_get(row, col, default=''):
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    return default
                return val
            return default
        
        # Build export records for history
        export_records = []
        exported_df = results_df[results_df["include_in_update"] == True]
        
        for _, row in exported_df.iterrows():
            export_records.append({
                'sku': str(safe_get(row, 'sku', '')),
                'isku': str(safe_get(row, 'isku', '')),
                'product_name': str(safe_get(row, 'name', '')),
                'old_price': float(safe_get(row, 'price', 0) or 0),
                'new_price': float(safe_get(row, 'new_price_myr', 0) or 0),
                'pokedata_id': str(safe_get(row, 'pokedata_id', '')),
                'pokedata_price_usd': safe_get(row, 'pokedata_price_usd', ''),
            })
        
        if export_records:
            fx_rate_used = results_df["fx_rate_used"].iloc[0] if "fx_rate_used" in results_df.columns else config.get("fx_rate", 0)
            logged_count = log_price_export(export_records, fx_rate=float(fx_rate_used))
            logger.info(f"Logged {logged_count} exports to price history")
    except Exception as e:
        logger.warning(f"Failed to log price history: {e}")
    
    # Generate filename
    filename = f"sitegiant_price_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/fetch-fx-rate", response_model=FXRateResponse)
async def fetch_fx_rate_endpoint(
    config: AppConfig = Depends(get_app_config),
) -> FXRateResponse:
    """Fetch live FX rate from Google Finance."""
    rate, source = fetch_google_fx_rate(config)
    
    if rate is not None:
        return FXRateResponse(success=True, rate=rate, source=source)
    else:
        # Return default rate on failure
        default_rate, _ = get_fx_rate(config)
        return FXRateResponse(success=False, rate=default_rate, source="default", error=source)


@router.get("/clear-session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session and redirect to home."""
    # Try file-based store first
    delete_session(session_id)
    # Also clear from legacy in-memory store
    if session_id in sessions:
        del sessions[session_id]
    return RedirectResponse(url="/", status_code=303)


# ============================================================================
# Phase 4: Download Unmapped SKUs
# ============================================================================

@router.get("/download-unmapped/{session_id}")
async def download_unmapped(session_id: str):
    """
    Download a list of unmapped SKUs from the current session.
    
    Phase 4 feature: Helps user identify which SKUs need to be added
    to the mapping file.
    """
    try:
        session = get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    results_df = session["results_df"]
    
    # Filter to only UNMAPPED rows
    if "status" not in results_df.columns:
        raise HTTPException(status_code=400, detail="No status information in session")
    
    unmapped_df = results_df[results_df["status"] == "UNMAPPED"].copy()
    
    if unmapped_df.empty:
        raise HTTPException(status_code=404, detail="No unmapped SKUs found")
    
    # Select relevant columns for the user
    export_cols = ["sku", "name", "price"]
    available_cols = [c for c in export_cols if c in unmapped_df.columns]
    
    # Add helpful columns for mapping
    unmapped_export = unmapped_df[available_cols].copy()
    unmapped_export["pokedata_id"] = ""  # Empty for user to fill
    unmapped_export["pokedata_language"] = "ENGLISH"  # Default
    unmapped_export["pokedata_asset_type"] = "PRODUCT"  # Default
    unmapped_export["auto_update"] = "Y"  # Default to yes
    
    # Create Excel file in memory
    output = BytesIO()
    unmapped_export.to_excel(output, index=False, engine='openpyxl', sheet_name='Unmapped SKUs')
    output.seek(0)
    
    filename = f"unmapped_skus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    logger.info(f"Downloaded {len(unmapped_export)} unmapped SKUs")
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============================================================================
# Settings Routes - Mapping Management
# ============================================================================

@router.get("/settings/mapping", response_class=HTMLResponse)
async def settings_mapping(request: Request, success: Optional[str] = None, error: Optional[str] = None):
    """Render the mapping settings page."""
    load_env()
    config = load_config()
    
    mapping_path = get_mapping_master_path(config)
    mapping_info = get_mapping_info(mapping_path)
    
    # Get stored settings
    settings = get_settings()
    api_key_set = stored_has_api_key()
    
    return templates.TemplateResponse("settings_mapping.html", {
        "request": request,
        "mapping_info": mapping_info,
        "api_key_set": api_key_set,
        "settings": settings,
        "success": success,
        "error": error,
    })


@router.post("/settings/api-key", response_class=HTMLResponse)
async def save_api_key(
    request: Request,
    api_key: str = Form(...),
):
    """Save the Pokedata API key."""
    if api_key and api_key.strip():
        set_api_key(api_key.strip())
        return RedirectResponse(
            url="/settings/mapping?success=API+key+saved+successfully.",
            status_code=303
        )
    else:
        return RedirectResponse(
            url="/settings/mapping?error=API+key+cannot+be+empty.",
            status_code=303
        )


@router.post("/settings/api-key/clear")
async def clear_api_key_route():
    """Clear the Pokedata API key."""
    clear_api_key()
    return RedirectResponse(
        url="/settings/mapping?success=API+key+cleared.",
        status_code=303
    )


@router.post("/settings/config", response_class=HTMLResponse)
async def save_config(
    request: Request,
    fx_rate: float = Form(4.50),
    margin_divisor: float = Form(0.8),
    soft_threshold: float = Form(20.0),
    hard_threshold: float = Form(50.0),
):
    """Save configuration settings."""
    update_settings(
        fx_rate=fx_rate,
        margin_divisor=margin_divisor,
        soft_threshold=soft_threshold,
        hard_threshold=hard_threshold,
    )
    return RedirectResponse(
        url="/settings/mapping?success=Configuration+saved+successfully.",
        status_code=303
    )


@router.get("/settings/cache/stats", response_model=CacheStats)
async def get_cache_stats() -> CacheStats:
    """Get cache statistics."""
    from src.storage.price_cache import get_cache
    cache = get_cache()
    stats = cache.get_stats()
    return CacheStats(**stats)


@router.post("/settings/cache/clear", response_model=CacheClearResponse)
async def clear_price_cache() -> CacheClearResponse:
    """Clear all cached data."""
    from src.storage.price_cache import get_cache
    cache = get_cache()
    result = cache.clear_all()
    return CacheClearResponse(
        success=True,
        message=f"Cleared {result['prices']} prices, {result['products']} products, {result['searches']} searches",
        cleared=result,
    )


@router.post("/settings/cache/clear-prices", response_model=CacheClearResponse)
async def clear_prices_only() -> CacheClearResponse:
    """Clear only price cache."""
    from src.storage.price_cache import get_cache
    cache = get_cache()
    count = cache.clear_prices()
    return CacheClearResponse(
        success=True,
        message=f"Cleared {count} cached prices",
        count=count,
    )


@router.post("/settings/mapping/upload", response_class=HTMLResponse)
async def upload_mapping(
    request: Request,
    mapping_file: UploadFile = File(...),
    auto_lookup: bool = Form(False),
):
    """Upload and save a new master mapping file with optional auto-lookup of Pokedata IDs."""
    load_env()
    config = load_config()
    
    mapping_path = get_mapping_master_path(config)
    
    # Check if API key is set when auto_lookup is enabled
    api_key = get_api_key() or os.environ.get("POKEDATA_API_KEY", "")
    api_key_set = stored_has_api_key() or bool(os.environ.get("POKEDATA_API_KEY", ""))
    
    if auto_lookup and not api_key:
        mapping_info = get_mapping_info(mapping_path)
        settings = get_settings()
        return templates.TemplateResponse("settings_mapping.html", {
            "request": request,
            "mapping_info": mapping_info,
            "api_key_set": api_key_set,
            "settings": settings,
            "error": "API key is required for auto-mapping. Please set your Pokedata API key in the Config tab first.",
        })
    
    try:
        # Read uploaded file
        content = await mapping_file.read()
        
        # Load into DataFrame to validate
        filename = mapping_file.filename or "unknown.xlsx"
        if filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content), engine='openpyxl')
        
        # Normalize column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Validate required columns
        missing_cols = validate_mapping_columns(df)
        if missing_cols:
            mapping_info = get_mapping_info(mapping_path)
            settings = get_settings()
            return templates.TemplateResponse("settings_mapping.html", {
                "request": request,
                "mapping_info": mapping_info,
                "api_key_set": api_key_set,
                "settings": settings,
                "error": f"Missing required columns: {', '.join(missing_cols)}. "
                         f"Required: sku, pokedata_id, pokedata_language, pokedata_asset_type, auto_update",
            })
        
        # Scan for duplicate SKUs
        df["_sku_normalized"] = df["sku"].astype(str).str.strip()
        sku_counts = df["_sku_normalized"].value_counts()
        duplicate_skus = sku_counts[sku_counts > 1].index.tolist()
        
        if duplicate_skus:
            # Build duplicate details
            duplicate_details = {}
            for sku in duplicate_skus:
                rows = df[df["_sku_normalized"] == sku].index.tolist()
                duplicate_details[sku] = [r + 2 for r in rows]  # Excel row numbers
            
            # Save to temp file for later confirmation
            temp_filename = f"temp_mapping_{uuid.uuid4().hex[:8]}.xlsx"
            temp_path = PROJECT_ROOT / "data" / "cache" / temp_filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            df.drop(columns=["_sku_normalized"]).to_excel(temp_path, index=False, engine='openpyxl')
            
            logger.warning(f"Duplicate SKUs detected: {duplicate_skus}")
            
            mapping_info = get_mapping_info(mapping_path)
            settings = get_settings()
            return templates.TemplateResponse("settings_mapping.html", {
                "request": request,
                "mapping_info": mapping_info,
                "api_key_set": api_key_set,
                "settings": settings,
                "duplicates_found": True,
                "duplicate_skus": duplicate_skus,
                "duplicate_details": duplicate_details,
                "duplicate_count": len(duplicate_skus),
                "total_rows": len(df),
                "temp_file": temp_filename,
                "auto_lookup": auto_lookup,
                "default_strategy": config.mapping.duplicate_handling,
            })
        
        # Clean up temp column if no duplicates
        df = df.drop(columns=["_sku_normalized"])
        
        # Auto-lookup Pokedata IDs using product name
        lookup_results = {
            "found": [],      # List of {sku, name, product_name, pokedata_id}
            "not_found": [],  # List of {sku, name, product_name, reason}
            "skipped": 0,     # Already had ID
            "errors": [],     # List of {sku, product_name, error}
        }
        
        if auto_lookup and api_key:
            client = PokedataClient(config, api_key=api_key)
            
            for idx in df.index:
                # Safely get values, handling NaN
                sku_val = df.at[idx, "sku"] if "sku" in df.columns else None
                sku = str(sku_val) if pd.notna(sku_val) else f"Row {idx}"
                
                # Get the product name from 'name' column (SiteGiant product name)
                name_val = df.at[idx, "name"] if "name" in df.columns else None
                name = str(name_val) if pd.notna(name_val) else ""
                
                pokedata_id = df.at[idx, "pokedata_id"] if "pokedata_id" in df.columns else None
                language = df.at[idx, "pokedata_language"] if "pokedata_language" in df.columns else "ENGLISH"
                
                # Skip if pokedata_id already has a value
                if pd.notna(pokedata_id) and str(pokedata_id).strip() != "":
                    lookup_results["skipped"] += 1
                    continue
                
                # Skip if no product name to search with
                if not name or name.strip() == "":
                    lookup_results["skipped"] += 1
                    continue
                
                # Search Pokedata API using the product name
                try:
                    search_results = client.search_products(
                        query=name.strip(),
                        language=str(language).upper() if pd.notna(language) else None,
                        limit=1,
                    )
                    
                    if search_results and len(search_results) > 0:
                        # Get the numeric ID from first result
                        numeric_id = str(search_results[0].product_id)
                        matched_name = search_results[0].name
                        df.at[idx, "pokedata_id"] = numeric_id
                        # Also update pokedata_name if column exists
                        if "pokedata_name" in df.columns:
                            df.at[idx, "pokedata_name"] = matched_name
                        lookup_results["found"].append({
                            "sku": sku,
                            "name": name,
                            "product_name": matched_name,
                            "pokedata_id": numeric_id,
                        })
                        logger.info(f"Auto-mapping: '{name}' -> ID {numeric_id} ({matched_name})")
                    else:
                        lookup_results["not_found"].append({
                            "sku": sku,
                            "name": name,
                            "product_name": name,
                            "reason": "No matching product found in Pokedata",
                        })
                        logger.warning(f"Auto-mapping: No results for '{name}'")
                except Exception as e:
                    lookup_results["errors"].append({
                        "sku": sku,
                        "product_name": name,
                        "error": str(e),
                    })
                    logger.error(f"Auto-mapping error for '{name}': {e}")
        
        # Ensure directory exists
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file (always save as xlsx for consistency)
        df.to_excel(mapping_path, index=False, engine='openpyxl')
        
        logger.info(f"Saved new mapping file with {len(df)} rows to {mapping_path}")
        
        # If auto-lookup was used, show detailed results page
        if auto_lookup and (lookup_results["found"] or lookup_results["not_found"] or lookup_results["errors"]):
            mapping_info = get_mapping_info(mapping_path)
            settings = get_settings()
            api_key_set = stored_has_api_key()
            return templates.TemplateResponse("settings_mapping.html", {
                "request": request,
                "mapping_info": mapping_info,
                "api_key_set": api_key_set,
                "settings": settings,
                "success": f"Mapping file uploaded successfully. {len(df)} rows saved.",
                "lookup_results": lookup_results,
            })
        
        # Simple redirect for non-lookup uploads
        return RedirectResponse(
            url=f"/settings/mapping?success=Mapping+file+uploaded+successfully.+{len(df)}+rows+saved.",
            status_code=303
        )
        
    except Exception as e:
        logger.exception(f"Failed to upload mapping: {e}")
        mapping_info = get_mapping_info(mapping_path)
        settings = get_settings()
        api_key_set = stored_has_api_key() or bool(os.environ.get("POKEDATA_API_KEY", ""))
        return templates.TemplateResponse("settings_mapping.html", {
            "request": request,
            "mapping_info": mapping_info,
            "api_key_set": api_key_set,
            "settings": settings,
            "error": f"Failed to upload mapping file: {str(e)}",
        })


@router.post("/settings/mapping/confirm", response_class=HTMLResponse)
async def confirm_mapping_upload(
    request: Request,
    temp_file: str = Form(...),
    strategy: str = Form("merge"),
    auto_lookup: bool = Form(False),
):
    """
    Confirm mapping upload after duplicate detection.
    
    Applies the user's chosen strategy (merge/ignore) and saves the file.
    """
    load_env()
    config = load_config()
    
    mapping_path = get_mapping_master_path(config)
    temp_path = PROJECT_ROOT / "data" / "cache" / temp_file
    
    # Check if API key is set
    api_key = get_api_key() or os.environ.get("POKEDATA_API_KEY", "")
    api_key_set = stored_has_api_key() or bool(os.environ.get("POKEDATA_API_KEY", ""))
    
    if not temp_path.exists():
        mapping_info = get_mapping_info(mapping_path)
        settings = get_settings()
        return templates.TemplateResponse("settings_mapping.html", {
            "request": request,
            "mapping_info": mapping_info,
            "api_key_set": api_key_set,
            "settings": settings,
            "error": "Temporary file expired or not found. Please upload the mapping file again.",
        })
    
    try:
        # Load temp file
        df = pd.read_excel(temp_path, engine='openpyxl')
        
        # Apply duplicate handling strategy
        df["_sku_normalized"] = df["sku"].astype(str).str.strip()
        original_count = len(df)
        
        if strategy == "ignore":
            # Keep first occurrence only
            df = df.drop_duplicates(subset=["_sku_normalized"], keep="first")
            logger.info(f"Applied 'ignore' strategy: kept first entry for duplicates")
        else:  # "merge" - keep last
            df = df.drop_duplicates(subset=["_sku_normalized"], keep="last")
            logger.info(f"Applied 'merge' strategy: kept last entry for duplicates")
        
        duplicates_removed = original_count - len(df)
        df = df.drop(columns=["_sku_normalized"])
        
        # Auto-lookup if requested
        lookup_results = {
            "found": [],
            "not_found": [],
            "skipped": 0,
            "errors": [],
        }
        
        if auto_lookup and api_key:
            client = PokedataClient(config, api_key=api_key)
            
            for idx in df.index:
                sku_val = df.at[idx, "sku"] if "sku" in df.columns else None
                sku = str(sku_val) if pd.notna(sku_val) else f"Row {idx}"
                
                name_val = df.at[idx, "name"] if "name" in df.columns else None
                name = str(name_val) if pd.notna(name_val) else ""
                
                pokedata_id = df.at[idx, "pokedata_id"] if "pokedata_id" in df.columns else None
                language = df.at[idx, "pokedata_language"] if "pokedata_language" in df.columns else "ENGLISH"
                
                if pd.notna(pokedata_id) and str(pokedata_id).strip() != "":
                    lookup_results["skipped"] += 1
                    continue
                
                if not name or name.strip() == "":
                    lookup_results["skipped"] += 1
                    continue
                
                try:
                    search_results = client.search_products(
                        query=name.strip(),
                        language=str(language).upper() if pd.notna(language) else None,
                        limit=1,
                    )
                    
                    if search_results and len(search_results) > 0:
                        numeric_id = str(search_results[0].product_id)
                        matched_name = search_results[0].name
                        df.at[idx, "pokedata_id"] = numeric_id
                        if "pokedata_name" in df.columns:
                            df.at[idx, "pokedata_name"] = matched_name
                        lookup_results["found"].append({
                            "sku": sku,
                            "name": name,
                            "product_name": matched_name,
                            "pokedata_id": numeric_id,
                        })
                    else:
                        lookup_results["not_found"].append({
                            "sku": sku,
                            "name": name,
                            "product_name": name,
                            "reason": "No matching product found",
                        })
                except Exception as e:
                    lookup_results["errors"].append({
                        "sku": sku,
                        "product_name": name,
                        "error": str(e),
                    })
        
        # Save to master mapping path
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(mapping_path, index=False, engine='openpyxl')
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except Exception:
            pass
        
        logger.info(f"Saved mapping file with {len(df)} rows (removed {duplicates_removed} duplicates using '{strategy}' strategy)")
        
        success_msg = f"Mapping file saved successfully. {len(df)} rows saved"
        if duplicates_removed > 0:
            success_msg += f" ({duplicates_removed} duplicate(s) handled using '{strategy}' strategy)"
        
        # Show results
        if auto_lookup and (lookup_results["found"] or lookup_results["not_found"] or lookup_results["errors"]):
            mapping_info = get_mapping_info(mapping_path)
            settings = get_settings()
            return templates.TemplateResponse("settings_mapping.html", {
                "request": request,
                "mapping_info": mapping_info,
                "api_key_set": api_key_set,
                "settings": settings,
                "success": success_msg,
                "lookup_results": lookup_results,
            })
        
        return RedirectResponse(
            url=f"/settings/mapping?success={success_msg.replace(' ', '+')}",
            status_code=303
        )
        
    except Exception as e:
        logger.exception(f"Failed to confirm mapping upload: {e}")
        mapping_info = get_mapping_info(mapping_path)
        settings = get_settings()
        return templates.TemplateResponse("settings_mapping.html", {
            "request": request,
            "mapping_info": mapping_info,
            "api_key_set": api_key_set,
            "settings": settings,
            "error": f"Failed to save mapping file: {str(e)}",
        })


@router.post("/settings/mapping/refresh", response_class=HTMLResponse)
async def refresh_mapping(request: Request):
    """Re-run auto-mapping on existing mapping file to fill missing Pokedata IDs."""
    load_env()
    config = load_config()
    
    mapping_path = get_mapping_master_path(config)
    
    # Check if API key is set
    api_key = get_api_key() or os.environ.get("POKEDATA_API_KEY", "")
    api_key_set = stored_has_api_key() or bool(os.environ.get("POKEDATA_API_KEY", ""))
    
    if not api_key:
        mapping_info = get_mapping_info(mapping_path)
        settings = get_settings()
        return templates.TemplateResponse("settings_mapping.html", {
            "request": request,
            "mapping_info": mapping_info,
            "api_key_set": api_key_set,
            "settings": settings,
            "error": "API key is required for auto-mapping. Please set your Pokedata API key in the Config tab first.",
        })
    
    if not mapping_path.exists():
        mapping_info = get_mapping_info(mapping_path)
        settings = get_settings()
        return templates.TemplateResponse("settings_mapping.html", {
            "request": request,
            "mapping_info": mapping_info,
            "api_key_set": api_key_set,
            "settings": settings,
            "error": "No mapping file found. Please upload a mapping file first.",
        })
    
    try:
        # Load existing mapping file
        df = pd.read_excel(mapping_path, engine='openpyxl')
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Auto-lookup Pokedata IDs using product name
        lookup_results = {
            "found": [],
            "not_found": [],
            "skipped": 0,
            "errors": [],
        }
        
        client = PokedataClient(config, api_key=api_key)
        
        for idx in df.index:
            # Safely get values
            sku_val = df.at[idx, "sku"] if "sku" in df.columns else None
            sku = str(sku_val) if pd.notna(sku_val) else f"Row {idx}"
            
            name_val = df.at[idx, "name"] if "name" in df.columns else None
            name = str(name_val) if pd.notna(name_val) else ""
            
            pokedata_id = df.at[idx, "pokedata_id"] if "pokedata_id" in df.columns else None
            language = df.at[idx, "pokedata_language"] if "pokedata_language" in df.columns else "ENGLISH"
            
            # Skip if pokedata_id already has a value
            if pd.notna(pokedata_id) and str(pokedata_id).strip() != "":
                lookup_results["skipped"] += 1
                continue
            
            # Skip if no product name to search with
            if not name or name.strip() == "":
                lookup_results["skipped"] += 1
                continue
            
            # Search Pokedata API using the product name
            try:
                search_results = client.search_products(
                    query=name.strip(),
                    language=str(language).upper() if pd.notna(language) else None,
                    limit=1,
                )
                
                if search_results and len(search_results) > 0:
                    numeric_id = str(search_results[0].product_id)
                    matched_name = search_results[0].name
                    df.at[idx, "pokedata_id"] = numeric_id
                    if "pokedata_name" in df.columns:
                        df.at[idx, "pokedata_name"] = matched_name
                    lookup_results["found"].append({
                        "sku": sku,
                        "name": name,
                        "product_name": matched_name,
                        "pokedata_id": numeric_id,
                    })
                    logger.info(f"Refresh mapping: '{name}' -> ID {numeric_id} ({matched_name})")
                else:
                    lookup_results["not_found"].append({
                        "sku": sku,
                        "name": name,
                        "product_name": name,
                        "reason": "No matching product found in Pokedata",
                    })
            except Exception as e:
                lookup_results["errors"].append({
                    "sku": sku,
                    "product_name": name,
                    "error": str(e),
                })
                logger.error(f"Refresh mapping error for '{name}': {e}")
        
        # Save updated mapping file
        df.to_excel(mapping_path, index=False, engine='openpyxl')
        
        mapping_info = get_mapping_info(mapping_path)
        settings = get_settings()
        
        return templates.TemplateResponse("settings_mapping.html", {
            "request": request,
            "mapping_info": mapping_info,
            "api_key_set": api_key_set,
            "settings": settings,
            "success": f"Mapping refreshed. Found {len(lookup_results['found'])} new IDs.",
            "lookup_results": lookup_results,
        })
        
    except Exception as e:
        logger.exception(f"Failed to refresh mapping: {e}")
        mapping_info = get_mapping_info(mapping_path)
        settings = get_settings()
        return templates.TemplateResponse("settings_mapping.html", {
            "request": request,
            "mapping_info": mapping_info,
            "api_key_set": api_key_set,
            "settings": settings,
            "error": f"Failed to refresh mapping: {str(e)}",
        })


@router.get("/settings/mapping/download")
async def download_mapping():
    """Download the current master mapping file."""
    load_env()
    config = load_config()
    
    mapping_path = get_mapping_master_path(config)
    
    if not mapping_path.exists():
        raise HTTPException(status_code=404, detail="No mapping file found")
    
    return FileResponse(
        path=mapping_path,
        filename="master_mapping.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ============================================================================
# Mapping Management API - CRUD Operations
# ============================================================================

@router.get("/api/mapping")
async def get_all_mappings():
    """Get all mappings as JSON for the management UI."""
    load_env()
    config = load_config()
    
    mapping_path = get_mapping_master_path(config)
    
    if not mapping_path.exists():
        return {"mappings": [], "count": 0}
    
    try:
        df = pd.read_excel(mapping_path, engine='openpyxl')
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Convert to list of dicts, handling NaN values
        mappings = []
        for _, row in df.iterrows():
            mapping = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    mapping[col] = ""
                else:
                    mapping[col] = str(val)
            mappings.append(mapping)
        
        return {"mappings": mappings, "count": len(mappings)}
    except Exception as e:
        logger.error(f"Failed to load mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mapping/add")
async def add_mapping(
    sku: str = Form(...),
    name: str = Form(""),
    pokedata_url: str = Form(""),
    pokedata_id: str = Form(""),
    pokedata_name: str = Form(""),
    pokedata_language: str = Form("ENGLISH"),
    pokedata_asset_type: str = Form("PRODUCT"),
    auto_update: str = Form("Y"),
    notes: str = Form(""),
    auto_lookup: bool = Form(True),  # Auto-lookup enabled by default
):
    """Add a new mapping entry."""
    load_env()
    config = load_config()
    mapping_path = get_mapping_master_path(config)
    
    try:
        # Load existing or create new
        if mapping_path.exists():
            df = pd.read_excel(mapping_path, engine='openpyxl')
            df.columns = [c.lower().strip() for c in df.columns]
        else:
            df = pd.DataFrame(columns=[
                'sku', 'name', 'pokedata_url', 'pokedata_id', 'pokedata_name',
                'pokedata_language', 'pokedata_asset_type', 'auto_update', 'notes'
            ])
        
        # Check if SKU already exists
        if 'sku' in df.columns and len(df) > 0 and sku in df['sku'].values:
            raise HTTPException(status_code=400, detail=f"SKU '{sku}' already exists")
        
        # Auto-lookup if requested and URL provided but no ID
        lookup_result = None
        if auto_lookup and pokedata_url and not pokedata_id:
            lookup_result = await _lookup_pokedata_id(config, pokedata_url, pokedata_language)
            if lookup_result.get("found"):
                pokedata_id = lookup_result["pokedata_id"]
        
        # Add new row
        new_row = {
            'sku': sku,
            'name': name,
            'pokedata_url': pokedata_url,
            'pokedata_id': pokedata_id,
            'pokedata_name': pokedata_name,
            'pokedata_language': pokedata_language,
            'pokedata_asset_type': pokedata_asset_type,
            'auto_update': auto_update,
            'notes': notes,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(mapping_path, index=False, engine='openpyxl')
        
        logger.info(f"Added mapping for SKU: {sku}")
        
        return {
            "success": True,
            "message": f"Added mapping for SKU: {sku}",
            "mapping": new_row,
            "lookup_result": lookup_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/mapping/{sku}")
async def update_mapping(
    sku: str,
    name: str = Form(""),
    pokedata_url: str = Form(""),
    pokedata_id: str = Form(""),
    pokedata_name: str = Form(""),
    pokedata_language: str = Form("ENGLISH"),
    pokedata_asset_type: str = Form("PRODUCT"),
    auto_update: str = Form("Y"),
    notes: str = Form(""),
    auto_lookup: bool = Form(True),  # Auto-lookup enabled by default
):
    """Update an existing mapping entry."""
    load_env()
    config = load_config()
    mapping_path = get_mapping_master_path(config)
    
    if not mapping_path.exists():
        raise HTTPException(status_code=404, detail="No mapping file found")
    
    try:
        df = pd.read_excel(mapping_path, engine='openpyxl')
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Find the row
        mask = df['sku'] == sku
        if not mask.any():
            raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found")
        
        # Auto-lookup if requested
        lookup_result = None
        if auto_lookup and pokedata_url and not pokedata_id:
            lookup_result = await _lookup_pokedata_id(config, pokedata_url, pokedata_language)
            if lookup_result.get("found"):
                pokedata_id = lookup_result["pokedata_id"]
        
        # Update row
        idx = df[mask].index[0]
        df.at[idx, 'name'] = name
        df.at[idx, 'pokedata_url'] = pokedata_url
        df.at[idx, 'pokedata_id'] = pokedata_id
        df.at[idx, 'pokedata_name'] = pokedata_name
        df.at[idx, 'pokedata_language'] = pokedata_language
        df.at[idx, 'pokedata_asset_type'] = pokedata_asset_type
        df.at[idx, 'auto_update'] = auto_update
        df.at[idx, 'notes'] = notes
        
        # Save
        df.to_excel(mapping_path, index=False, engine='openpyxl')
        
        logger.info(f"Updated mapping for SKU: {sku}")
        
        return {
            "success": True,
            "message": f"Updated mapping for SKU: {sku}",
            "lookup_result": lookup_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/mapping/{sku}")
async def delete_mapping(sku: str):
    """Delete a mapping entry."""
    load_env()
    config = load_config()
    mapping_path = get_mapping_master_path(config)
    
    if not mapping_path.exists():
        raise HTTPException(status_code=404, detail="No mapping file found")
    
    try:
        df = pd.read_excel(mapping_path, engine='openpyxl')
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Find and remove
        original_len = len(df)
        df = df[df['sku'] != sku]
        
        if len(df) == original_len:
            raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found")
        
        # Save
        df.to_excel(mapping_path, index=False, engine='openpyxl')
        
        logger.info(f"Deleted mapping for SKU: {sku}")
        
        return {"success": True, "message": f"Deleted mapping for SKU: {sku}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mapping/delete-bulk")
async def delete_mappings_bulk(skus: List[str] = Body(...)):
    """Delete multiple mapping entries at once."""
    load_env()
    config = load_config()
    mapping_path = get_mapping_master_path(config)
    
    if not mapping_path.exists():
        raise HTTPException(status_code=404, detail="No mapping file found")
    
    try:
        df = pd.read_excel(mapping_path, engine='openpyxl')
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Find and remove
        original_len = len(df)
        df = df[~df['sku'].isin(skus)]
        
        deleted_count = original_len - len(df)
        
        if deleted_count == 0:
             return {"success": True, "message": "No mappings were deleted", "count": 0}
        
        # Save
        df.to_excel(mapping_path, index=False, engine='openpyxl')
        
        logger.info(f"Deleted {deleted_count} mappings")
        
        return {"success": True, "message": f"Deleted {deleted_count} mappings", "count": deleted_count}
    except Exception as e:
        logger.error(f"Failed to delete mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mapping/lookup")
async def lookup_pokedata_id(
    pokedata_url: str = Form(...),
    pokedata_language: str = Form("ENGLISH"),
):
    """Lookup Pokedata ID from a URL."""
    load_env()
    config = load_config()
    
    result = await _lookup_pokedata_id(config, pokedata_url, pokedata_language)
    return result


async def _lookup_pokedata_id(config, pokedata_url: str, pokedata_language: str = "ENGLISH") -> dict:
    """Internal helper to lookup Pokedata ID from URL."""
    from src.mapping.mapping_manager import extract_pokedata_id_from_url
    
    # Extract product name from URL
    product_name = extract_pokedata_id_from_url(pokedata_url)
    if not product_name:
        return {
            "found": False,
            "product_name": "",
            "error": "Could not extract product name from URL",
        }
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        api_key = os.environ.get("POKEDATA_API_KEY", "")
    
    if not api_key:
        return {
            "found": False,
            "product_name": product_name,
            "error": "No API key configured",
        }
    
    try:
        client = PokedataClient(config, api_key=api_key)
        search_results = client.search_products(
            query=product_name,
            language=pokedata_language.upper() if pokedata_language else None,
            limit=1,
        )
        
        if search_results and len(search_results) > 0:
            return {
                "found": True,
                "product_name": product_name,
                "pokedata_id": str(search_results[0].product_id),  # Convert to string for consistency
                "matched_name": search_results[0].name,
            }
        else:
            return {
                "found": False,
                "product_name": product_name,
                "error": "No matching product found in Pokedata",
            }
    except Exception as e:
        logger.error(f"Lookup error: {e}")
        return {
            "found": False,
            "product_name": product_name,
            "error": str(e),
        }


# ============================================================================
# Price History API
# ============================================================================

@router.get("/api/history")
async def get_price_history(
    sku: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 500,
):
    """Get price export history with optional filters."""
    from src.storage.price_history import get_history_service
    
    service = get_history_service()
    records = service.get_history(
        sku=sku,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    stats = service.get_stats()
    
    return {
        "records": records,
        "count": len(records),
        "stats": stats,
    }


@router.get("/api/history/stats")
async def get_history_stats():
    """Get price history statistics."""
    from src.storage.price_history import get_history_service
    
    service = get_history_service()
    return service.get_stats()


@router.get("/api/history/export")
async def export_history(
    sku: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Export filtered history as Excel file."""
    from src.storage.price_history import get_history_service
    
    service = get_history_service()
    df = service.export_history(sku=sku, start_date=start_date, end_date=end_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No history records found")
    
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    
    filename = f"price_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/api/history/clear")
async def clear_history():
    """Clear all price history."""
    from src.storage.price_history import get_history_service
    
    service = get_history_service()
    count = service.clear_history()
    
    return {"success": True, "message": f"Cleared {count} history records"}


@router.post("/settings/mapping/generate-template")
async def generate_mapping_template(
    request: Request,
    sitegiant_export: UploadFile = File(...),
):
    """
    Generate a mapping template from a SiteGiant export file.
    
    Returns an Excel file with SiteGiant data pre-filled and empty Pokedata columns
    ready for manual mapping.
    """
    from src.mapping.generate_mapping_template import create_mapping_template, SITEGIANT_COLUMN_MAP, find_column
    
    try:
        # Read uploaded file
        content = await sitegiant_export.read()
        
        # Load into DataFrame
        filename = sitegiant_export.filename or "unknown.xlsx"
        if filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content), engine='openpyxl')
        
        # Validate that SKU column exists
        sku_col = find_column(df, SITEGIANT_COLUMN_MAP["sku"])
        if sku_col is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not find SKU column in uploaded file. Available columns: {list(df.columns)}"
            )
        
        # Create template
        template_df = create_mapping_template(df)
        
        # Write to bytes buffer
        output = BytesIO()
        template_df.to_excel(output, index=False, engine='openpyxl', sheet_name='Mapping')
        output.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mapping_template_{timestamp}.xlsx"
        
        logger.info(f"Generated mapping template with {len(template_df)} rows")
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to generate mapping template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate template: {str(e)}")


# ============================================================================
# Pokedata Explorer Routes
# ============================================================================

@router.get("/explorer", response_class=HTMLResponse)
async def explorer(request: Request):
    """Render the Pokedata Explorer page."""
    load_env()
    api_key_set = stored_has_api_key() or bool(os.environ.get("POKEDATA_API_KEY", ""))
    
    return templates.TemplateResponse("explorer.html", {
        "request": request,
        "api_key_set": api_key_set,
        "search_results": [],
        "price_info": None,
        "price_history": [],
    })


@router.post("/explorer/search", response_class=HTMLResponse)
async def explorer_search(
    request: Request,
    search_query: str = Form(...),
    search_mode: str = Form("name"),
    api_key_override: Optional[str] = Form(None),
):
    """Search for products or get pricing by ID."""
    load_env()
    app_config = load_config()
    
    # Get API key: form override > stored settings > env var
    api_key = (api_key_override.strip() if api_key_override and api_key_override.strip() 
               else get_api_key() or os.environ.get("POKEDATA_API_KEY", ""))
    api_key_set = stored_has_api_key() or bool(os.environ.get("POKEDATA_API_KEY", ""))
    
    search_results = []
    price_info = None
    price_history = []
    error = None
    
    if not api_key:
        error = "No API key configured. Enter an API key or set POKEDATA_API_KEY in .env file."
    else:
        try:
            client = PokedataClient(app_config, api_key=api_key)
            
            if search_mode == "name":
                # Search by name
                logger.info(f"Searching Pokedata for: {search_query}")
                results = client.search_products(search_query, limit=10)
                logger.info(f"Search returned {len(results)} results")
                
                if results:
                    search_results = [
                        {
                            "product_id": r.product_id,
                            "name": r.name,
                            "language": r.language or "N/A",
                            "asset_type": r.asset_type,
                            "url": r.url,
                        }
                        for r in results
                    ]
                else:
                    error = f"No products found for '{search_query}'. The API returned empty results."
            else:
                # Search by ID - get pricing directly
                price_result = client.get_product_pricing(search_query)
                if price_result.primary_price_usd is not None:
                    price_info = {
                        "product_id": price_result.product_id,
                        "primary_price_usd": price_result.primary_price_usd,
                        "source": price_result.source or "N/A",
                        "raw_prices": price_result.raw_prices or {},
                    }
                    # Get price history
                    from src.storage.pokedata_history_store import get_history
                    history = get_history(search_query, "PRODUCT", limit=5)
                    price_history = [{"date": h.date, "price_usd": h.price_usd} for h in history]
                elif price_result.error:
                    error = f"API Error: {price_result.error}"
                else:
                    error = f"No pricing data found for ID: {search_query}"
                    
        except Exception as e:
            import traceback
            logger.error(f"Explorer search error: {traceback.format_exc()}")
            error = f"Error: {str(e)}"
    
    return templates.TemplateResponse("explorer.html", {
        "request": request,
        "api_key_set": api_key_set,
        "search_query": search_query,
        "search_mode": search_mode,
        "search_results": search_results,
        "price_info": price_info,
        "price_history": price_history,
        "error": error,
    })


@router.get("/explorer/pricing/{product_id}", response_class=HTMLResponse)
async def explorer_pricing(
    request: Request,
    product_id: str,
    api_key: Optional[str] = None,
):
    """Get pricing for a specific product ID."""
    load_env()
    app_config = load_config()
    
    # Get API key: query param > stored settings > env var
    api_key = api_key if api_key else (get_api_key() or os.environ.get("POKEDATA_API_KEY", ""))
    api_key_set = stored_has_api_key() or bool(os.environ.get("POKEDATA_API_KEY", ""))
    
    price_info = None
    price_history = []
    error = None
    
    if not api_key:
        error = "No API key configured."
    else:
        try:
            client = PokedataClient(app_config, api_key=api_key)
            price_result = client.get_product_pricing(product_id)
            
            if price_result.primary_price_usd is not None:
                price_info = {
                    "product_id": price_result.product_id,
                    "primary_price_usd": price_result.primary_price_usd,
                    "source": price_result.source or "N/A",
                    "raw_prices": price_result.raw_prices or {},
                }
                # Get price history
                from src.storage.pokedata_history_store import get_history
                history = get_history(product_id, "PRODUCT", limit=5)
                price_history = [{"date": h.date, "price_usd": h.price_usd} for h in history]
            elif price_result.error:
                error = f"API Error: {price_result.error}"
            else:
                error = f"No pricing data found for ID: {product_id}"
                
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return templates.TemplateResponse("explorer.html", {
        "request": request,
        "api_key_set": api_key_set,
        "search_query": product_id,
        "search_mode": "id",
        "search_results": [],
        "price_info": price_info,
        "price_history": price_history,
        "error": error,
    })


# ============================================================================
# API Routes - Pokedata URL Parsing
# ============================================================================

@router.post("/api/parse-pokedata-url")
async def api_parse_single_url(url: str = Form(...)):
    """
    Parse a single Pokedata URL to extract product ID and language.
    
    Returns:
        JSON with parsed URL information.
    """
    result = parse_pokedata_url(url)
    return result.to_dict()


@router.post("/api/parse-pokedata-urls")
async def api_parse_bulk_urls(urls: str = Form(...)):
    """
    Parse multiple Pokedata URLs (one per line).
    
    Args:
        urls: Text containing URLs, one per line.
        
    Returns:
        JSON with list of parsed URL results and summary.
    """
    results = parse_pokedata_urls_bulk(urls)
    
    valid_count = sum(1 for r in results if r.is_valid)
    
    return {
        "total": len(results),
        "valid": valid_count,
        "invalid": len(results) - valid_count,
        "results": [r.to_dict() for r in results],
    }


@router.get("/api/build-pokedata-url")
async def api_build_url(product_id: int, language: str = "ENGLISH"):
    """
    Build a Pokedata URL from product ID and language.
    
    Args:
        product_id: Pokedata product ID.
        language: ENGLISH or JAPANESE.
        
    Returns:
        JSON with the constructed URL.
    """
    url = build_pokedata_url(product_id, language)
    return {
        "product_id": product_id,
        "language": language.upper(),
        "url": url,
    }


@router.post("/api/validate-pokedata-id")
async def api_validate_id(pokedata_id: str = Form(...)):
    """
    Validate a Pokedata ID (can be numeric or URL).
    
    Args:
        pokedata_id: ID to validate (number or URL).
        
    Returns:
        JSON with validation result and normalized ID.
    """
    is_valid, normalized_id, error = validate_pokedata_id(pokedata_id)
    
    return {
        "input": pokedata_id,
        "is_valid": is_valid,
        "normalized_id": normalized_id,
        "error": error,
        "url": build_pokedata_url(normalized_id) if is_valid and normalized_id else None,
    }


# ============================================================================
# Export with Format Option (CSV/Excel)
# ============================================================================

@router.post("/export-csv")
async def export_csv(
    request: Request,
    session_id: str = Form(...),
    selected_skus: List[str] = Form(default=[]),
):
    """
    Export selected rows to CSV file.
    
    This is an alternative to the Excel export for users who prefer CSV.
    """
    # Get session (try file-based first, then legacy)
    session = get_stored_session(session_id)
    if session is None and session_id in sessions:
        session = sessions[session_id]
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    original_df = session["original_df"]
    results_df = session["results_df"].copy()
    config = session["config"]
    
    # Check demo mode
    demo_mode = results_df["demo_mode"].iloc[0] if "demo_mode" in results_df.columns else False
    if demo_mode:
        raise HTTPException(
            status_code=400, 
            detail="Export is disabled in DEMO MODE. Configure your Pokedata API key first."
        )
    
    # Update include_in_update based on selected SKUs
    if selected_skus:
        if "sku" in results_df.columns:
            results_df["include_in_update"] = results_df["sku"].isin(selected_skus)
        elif "SKU" in results_df.columns:
            results_df["include_in_update"] = results_df["SKU"].isin(selected_skus)
        else:
            results_df["include_in_update"] = False
    else:
        results_df["include_in_update"] = False
    
    # Exclude BLOCKED and UNMAPPED rows
    if "status" in results_df.columns:
        results_df.loc[results_df["status"] == "BLOCKED", "include_in_update"] = False
        results_df.loc[results_df["status"] == "UNMAPPED", "include_in_update"] = False
    
    # Build update dataframe
    update_df = build_update_dataframe(original_df, results_df)
    
    # Create CSV in memory
    output = BytesIO()
    update_df.to_csv(output, index=False)
    output.seek(0)
    
    # Generate filename
    filename = f"sitegiant_price_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    logger.info(f"CSV export: {len(update_df)} rows")
    
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
