"""Custom exceptions for the application."""

from typing import Any, Dict, Optional


class BaseCustomException(Exception):
    """Base custom exception class."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DataValidationError(BaseCustomException):
    """Raised when data validation fails."""
    pass


class OptimizationError(BaseCustomException):
    """Raised when optimization process fails."""
    pass


class ModelTrainingError(BaseCustomException):
    """Raised when model training fails."""
    pass


class FileProcessingError(BaseCustomException):
    """Raised when file processing fails."""
    pass


class InvalidParameterError(BaseCustomException):
    """Raised when invalid parameters are provided."""
    pass 