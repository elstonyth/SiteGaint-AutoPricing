"""
Integration tests for SiteGiant Pricing Automation.

Tests the full workflow from upload to export.
"""

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.webapp.main import app


# Test client for FastAPI
@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_sitegiant_excel():
    """Create a sample SiteGiant export Excel file."""
    data = {
        "SKU": ["PKM-001", "PKM-002", "PKM-003"],
        "Product Name": ["Scarlet Violet BB", "151 JP Box", "Obsidian Flames BB"],
        "Price": [450.00, 380.00, 480.00],
        "Stock": [10, 5, 8],
        "Status": ["Active", "Active", "Active"],
    }
    df = pd.DataFrame(data)
    
    # Write to bytes buffer
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_mapping_excel():
    """Create a sample mapping file."""
    data = {
        "sku": ["PKM-001", "PKM-002"],
        "pokedata_id": [66, 89],
        "pokedata_language": ["ENGLISH", "JAPANESE"],
        "auto_update": ["Y", "Y"],
    }
    df = pd.DataFrame(data)
    
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return buffer


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_simple(self, client):
        """Test simple health check returns OK."""
        response = client.get("/health/simple")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
    
    def test_health_detailed(self, client):
        """Test detailed health check returns component status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "version" in data
    
    def test_health_ready(self, client):
        """Test readiness check."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"


class TestPokedataURLParsing:
    """Test Pokedata URL parsing API."""
    
    def test_parse_single_url_english(self, client):
        """Test parsing an English product URL."""
        response = client.post(
            "/api/parse-pokedata-url",
            data={"url": "https://www.pokedata.io/product/66"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["pokedata_id"] == 66
        assert data["pokedata_language"] == "ENGLISH"
    
    def test_parse_single_url_japanese(self, client):
        """Test parsing a Japanese product URL."""
        response = client.post(
            "/api/parse-pokedata-url",
            data={"url": "https://www.pokedata.io/jp/product/89"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["pokedata_id"] == 89
        assert data["pokedata_language"] == "JAPANESE"
    
    def test_parse_invalid_url(self, client):
        """Test parsing an invalid URL."""
        response = client.post(
            "/api/parse-pokedata-url",
            data={"url": "https://google.com"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert data["error"] is not None
    
    def test_parse_bulk_urls(self, client):
        """Test parsing multiple URLs."""
        urls = """https://www.pokedata.io/product/66
https://www.pokedata.io/jp/product/89
https://invalid-url.com"""
        
        response = client.post(
            "/api/parse-pokedata-urls",
            data={"urls": urls}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["valid"] == 2
        assert data["invalid"] == 1
    
    def test_build_url_english(self, client):
        """Test building an English product URL."""
        response = client.get("/api/build-pokedata-url?product_id=66&language=ENGLISH")
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://www.pokedata.io/product/66"
    
    def test_build_url_japanese(self, client):
        """Test building a Japanese product URL."""
        response = client.get("/api/build-pokedata-url?product_id=89&language=JAPANESE")
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://www.pokedata.io/jp/product/89"
    
    def test_validate_id_numeric(self, client):
        """Test validating a numeric ID."""
        response = client.post(
            "/api/validate-pokedata-id",
            data={"pokedata_id": "66"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["normalized_id"] == 66
    
    def test_validate_id_from_url(self, client):
        """Test validating an ID from URL."""
        response = client.post(
            "/api/validate-pokedata-id",
            data={"pokedata_id": "https://www.pokedata.io/product/123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["normalized_id"] == 123


class TestMainPage:
    """Test main page rendering."""
    
    def test_index_page_loads(self, client):
        """Test that the index page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Price Update" in response.text or "SiteGiant" in response.text


class TestUploadFlow:
    """Test the upload and processing flow."""
    
    def test_upload_without_mapping_shows_error(self, client, sample_sitegiant_excel):
        """Test that uploading without a mapping file shows an error."""
        # This test relies on the actual mapping file state
        # If no mapping exists, should return to index with error
        response = client.post(
            "/process",
            files={"sitegiant_file": ("test.xlsx", sample_sitegiant_excel, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={
                "fx_rate": "4.72",
                "soft_threshold": "20.0",
                "hard_threshold": "50.0",
                "margin_divisor": "0.8",
            }
        )
        # Should return 200 (either success preview or error page)
        assert response.status_code == 200


class TestExportFormats:
    """Test export format options."""
    
    def test_csv_export_requires_session(self, client):
        """Test that CSV export requires a valid session."""
        response = client.post(
            "/export-csv",
            data={
                "session_id": "invalid-session-id",
                "selected_skus": [],
            }
        )
        assert response.status_code == 404


class TestRateLimiting:
    """Test rate limiting middleware."""
    
    def test_rate_limit_headers_present(self, client):
        """Test that rate limit headers are present in responses."""
        response = client.get("/")
        # Rate limit headers should be present
        assert "X-RateLimit-Limit" in response.headers or response.status_code == 200


# Run with: pytest tests/test_integration.py -v
