"""
Status codes for price change assessment.

Defines constants for different price update statuses after threshold checking.
"""

from enum import Enum, auto


class PriceStatus(str, Enum):
    """
    Status codes for price update decisions.
    
    Values:
        OK: Price change within acceptable limits, safe to update.
        WARNING: Price change exceeds soft threshold, review recommended.
        BLOCKED: Price change exceeds hard threshold, update blocked.
        NO_DATA: No Pokedata price available for the product.
        UNMAPPED: Product SKU not found in mapping file.
        ERROR: Error occurred during price calculation.
        SKIPPED: Product intentionally skipped (e.g., out of stock).
    """
    OK = "OK"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    NO_DATA = "NO_DATA"
    UNMAPPED = "UNMAPPED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class ChangeDirection(str, Enum):
    """
    Direction of price change.
    
    Values:
        INCREASE: New price is higher than current.
        DECREASE: New price is lower than current.
        NO_CHANGE: Prices are equal.
        UNKNOWN: Cannot determine (missing data).
    """
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    NO_CHANGE = "NO_CHANGE"
    UNKNOWN = "UNKNOWN"


# Color codes for GUI display
STATUS_COLORS = {
    PriceStatus.OK: "#28a745",        # Green
    PriceStatus.WARNING: "#ffc107",   # Yellow/Orange
    PriceStatus.BLOCKED: "#dc3545",   # Red
    PriceStatus.NO_DATA: "#6c757d",   # Gray
    PriceStatus.UNMAPPED: "#6c757d",  # Gray
    PriceStatus.ERROR: "#dc3545",     # Red
    PriceStatus.SKIPPED: "#17a2b8",   # Blue/Teal
}


# Human-readable status descriptions
STATUS_DESCRIPTIONS = {
    PriceStatus.OK: "Price change is within acceptable limits.",
    PriceStatus.WARNING: "Price change exceeds soft threshold. Please review.",
    PriceStatus.BLOCKED: "Price change exceeds hard threshold. Update blocked.",
    PriceStatus.NO_DATA: "No pricing data available from Pokedata.",
    PriceStatus.UNMAPPED: "Product SKU not found in mapping file.",
    PriceStatus.ERROR: "An error occurred during price calculation.",
    PriceStatus.SKIPPED: "Product was skipped based on filter settings.",
}


def get_status_color(status: PriceStatus) -> str:
    """
    Get the display color for a status.
    
    Args:
        status: The price status.
        
    Returns:
        str: Hex color code.
    """
    return STATUS_COLORS.get(status, "#6c757d")


def get_status_description(status: PriceStatus) -> str:
    """
    Get a human-readable description for a status.
    
    Args:
        status: The price status.
        
    Returns:
        str: Status description.
    """
    return STATUS_DESCRIPTIONS.get(status, "Unknown status.")
