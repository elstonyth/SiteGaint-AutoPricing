"""
Custom exceptions for the SiteGiant Pricing web application.

Provides a hierarchy of exceptions for clean error handling in routes.
"""

from typing import Optional, Dict, Any


class AppException(Exception):
    """Base exception for application errors."""
    
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if status_code:
            self.status_code = status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(AppException):
    """Raised when input validation fails."""
    
    status_code = 400
    error_code = "VALIDATION_ERROR"


class FileValidationError(ValidationError):
    """Raised when uploaded file validation fails."""
    
    error_code = "FILE_VALIDATION_ERROR"
    
    def __init__(self, message: str, filename: Optional[str] = None, missing_columns: Optional[list] = None):
        details = {}
        if filename:
            details["filename"] = filename
        if missing_columns:
            details["missing_columns"] = missing_columns
        super().__init__(message, details)


class SessionError(AppException):
    """Base class for session-related errors."""
    
    status_code = 404
    error_code = "SESSION_ERROR"


class SessionNotFoundError(SessionError):
    """Raised when a session is not found."""
    
    error_code = "SESSION_NOT_FOUND"
    
    def __init__(self, session_id: str):
        super().__init__(
            f"Session not found or expired: {session_id[:8]}...",
            details={"session_id": session_id}
        )


class SessionExpiredError(SessionError):
    """Raised when a session has expired."""
    
    error_code = "SESSION_EXPIRED"
    
    def __init__(self, session_id: str):
        super().__init__(
            "Session has expired. Please upload your file again.",
            details={"session_id": session_id}
        )


class ConfigurationError(AppException):
    """Raised when configuration is missing or invalid."""
    
    status_code = 500
    error_code = "CONFIGURATION_ERROR"


class MappingError(AppException):
    """Raised when mapping file is missing or invalid."""
    
    status_code = 400
    error_code = "MAPPING_ERROR"
    
    def __init__(self, message: str, path: Optional[str] = None):
        details = {"path": path} if path else {}
        super().__init__(message, details)


class APIKeyError(AppException):
    """Raised when API key is missing or invalid."""
    
    status_code = 401
    error_code = "API_KEY_ERROR"


class ExportBlockedError(AppException):
    """Raised when export is blocked (e.g., demo mode)."""
    
    status_code = 403
    error_code = "EXPORT_BLOCKED"
    
    def __init__(self, reason: str):
        super().__init__(
            f"Export blocked: {reason}",
            details={"reason": reason}
        )


class PricingError(AppException):
    """Raised when price processing fails."""
    
    status_code = 500
    error_code = "PRICING_ERROR"


class ExternalAPIError(AppException):
    """Raised when external API (Pokedata) fails."""
    
    status_code = 502
    error_code = "EXTERNAL_API_ERROR"
    
    def __init__(self, message: str, api_name: str = "Pokedata"):
        super().__init__(message, details={"api": api_name})
