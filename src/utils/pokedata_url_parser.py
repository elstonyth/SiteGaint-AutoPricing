"""
Pokedata URL parser for extracting product IDs and language.

Supports various Pokedata URL formats and extracts:
- pokedata_id: The numeric product identifier
- pokedata_language: ENGLISH or JAPANESE
- pokedata_name: Product name (if available from URL or API)
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


@dataclass
class ParsedPokedataURL:
    """Result of parsing a Pokedata URL."""
    
    original_url: str
    pokedata_id: Optional[int] = None
    pokedata_language: str = "ENGLISH"
    pokedata_name: Optional[str] = None
    is_valid: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_url": self.original_url,
            "pokedata_id": self.pokedata_id,
            "pokedata_language": self.pokedata_language,
            "pokedata_name": self.pokedata_name,
            "is_valid": self.is_valid,
            "error": self.error,
        }


# URL patterns for Pokedata
# Format: https://www.pokedata.io/product/66
# Japanese: https://www.pokedata.io/jp/product/89
# With query: https://pokedata.io/product/123?lang=japanese
POKEDATA_PATTERNS = [
    # Japanese path pattern: /jp/product/ID
    re.compile(r"pokedata\.io/jp/product/(\d+)", re.IGNORECASE),
    # Standard pattern: /product/ID
    re.compile(r"pokedata\.io/product/(\d+)", re.IGNORECASE),
    # API pattern: /v0/product/ID
    re.compile(r"pokedata\.io/v\d+/product/(\d+)", re.IGNORECASE),
]


def parse_pokedata_url(url: str) -> ParsedPokedataURL:
    """
    Parse a single Pokedata URL to extract product ID and language.
    
    Args:
        url: Pokedata product URL.
        
    Returns:
        ParsedPokedataURL with extracted data.
        
    Examples:
        >>> parse_pokedata_url("https://www.pokedata.io/product/66")
        ParsedPokedataURL(pokedata_id=66, pokedata_language="ENGLISH", is_valid=True)
        
        >>> parse_pokedata_url("https://pokedata.io/jp/product/89")
        ParsedPokedataURL(pokedata_id=89, pokedata_language="JAPANESE", is_valid=True)
    """
    url = url.strip()
    result = ParsedPokedataURL(original_url=url)
    
    if not url:
        result.error = "Empty URL"
        return result
    
    # Check if it's a Pokedata URL
    if "pokedata.io" not in url.lower():
        result.error = "Not a Pokedata URL"
        return result
    
    # Try to extract product ID
    product_id = None
    is_japanese = False
    
    for pattern in POKEDATA_PATTERNS:
        match = pattern.search(url)
        if match:
            product_id = int(match.group(1))
            # Check if Japanese path
            if "/jp/" in url.lower():
                is_japanese = True
            break
    
    if product_id is None:
        result.error = "Could not extract product ID from URL"
        return result
    
    # Check query parameters for language override
    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        lang_param = query_params.get("lang", [""])[0].lower()
        
        if lang_param in ("jp", "japanese", "ja"):
            is_japanese = True
        elif lang_param in ("en", "english"):
            is_japanese = False
    except Exception:
        pass  # Ignore query parsing errors
    
    result.pokedata_id = product_id
    result.pokedata_language = "JAPANESE" if is_japanese else "ENGLISH"
    result.is_valid = True
    
    logger.debug(f"Parsed URL: {url} -> ID={product_id}, Lang={result.pokedata_language}")
    
    return result


def parse_pokedata_urls_bulk(urls_text: str) -> List[ParsedPokedataURL]:
    """
    Parse multiple Pokedata URLs from text (one per line).
    
    Args:
        urls_text: Text containing URLs, one per line.
        
    Returns:
        List of ParsedPokedataURL results.
    """
    results = []
    
    # Split by newlines and filter empty lines
    lines = [line.strip() for line in urls_text.strip().split("\n")]
    lines = [line for line in lines if line]
    
    for line in lines:
        # Handle lines with SKU prefix like "SKU-001: https://..."
        if ":" in line and "http" in line:
            url_part = line.split("http", 1)[1]
            url = "http" + url_part.split()[0]  # Get URL part only
        else:
            url = line
        
        result = parse_pokedata_url(url)
        results.append(result)
    
    valid_count = sum(1 for r in results if r.is_valid)
    logger.info(f"Parsed {len(results)} URLs, {valid_count} valid")
    
    return results


def extract_id_from_url(url: str) -> Optional[int]:
    """
    Simple helper to extract just the product ID from a URL.
    
    Args:
        url: Pokedata URL.
        
    Returns:
        Product ID or None if not found.
    """
    result = parse_pokedata_url(url)
    return result.pokedata_id if result.is_valid else None


def build_pokedata_url(product_id: int, language: str = "ENGLISH") -> str:
    """
    Build a Pokedata URL from product ID and language.
    
    Args:
        product_id: Pokedata product ID.
        language: ENGLISH or JAPANESE.
        
    Returns:
        Full Pokedata product URL.
    """
    base_url = "https://www.pokedata.io"
    
    if language.upper() == "JAPANESE":
        return f"{base_url}/jp/product/{product_id}"
    else:
        return f"{base_url}/product/{product_id}"


def validate_pokedata_id(pokedata_id: any) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate and normalize a Pokedata ID.
    
    Args:
        pokedata_id: ID to validate (can be int, str, or URL).
        
    Returns:
        Tuple of (is_valid, normalized_id, error_message).
    """
    if pokedata_id is None:
        return False, None, "ID is None"
    
    # If it's a URL, extract the ID
    if isinstance(pokedata_id, str):
        pokedata_id = pokedata_id.strip()
        
        if "pokedata.io" in pokedata_id.lower():
            extracted = extract_id_from_url(pokedata_id)
            if extracted:
                return True, extracted, None
            else:
                return False, None, "Could not extract ID from URL"
        
        # Try to parse as integer
        try:
            pokedata_id = int(pokedata_id)
        except ValueError:
            return False, None, f"Invalid ID format: {pokedata_id}"
    
    # Validate as integer
    if isinstance(pokedata_id, (int, float)):
        pid = int(pokedata_id)
        if pid > 0:
            return True, pid, None
        else:
            return False, None, "ID must be positive"
    
    return False, None, f"Unexpected ID type: {type(pokedata_id)}"
