"""
Custom exceptions for the database module.
"""
from typing import Optional, Dict, Any


class DatabaseError(Exception):
    """Base exception for all database-related errors."""
    
    def __init__(
        self, 
        message: str, 
        *,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            context: Additional context about the error
            original_exception: The original exception that caused this error
        """
        super().__init__(message)
        self.context = context or {}
        self.original_exception = original_exception


class ConnectionError(DatabaseError):
    """Raised when there are issues connecting to the database."""
    pass


class QueryError(DatabaseError):
    """Raised when there are issues with database queries."""
    pass


class ValidationError(DatabaseError):
    """Raised when there are data validation errors."""
    pass


class NotFoundError(DatabaseError):
    """Raised when a requested resource is not found."""
    pass


class IntegrityError(DatabaseError):
    """Raised when there are database integrity errors."""
    pass


class MigrationError(DatabaseError):
    """Raised when there are issues with database migrations."""
    pass


class TimeoutError(DatabaseError):
    """Raised when a database operation times out."""
    pass


class RecordNotFoundError(DatabaseError):
    """Raised when a requested record is not found."""
    pass

class ConflictError(DatabaseError):
    """Raised when there is a conflict in the database."""
    pass
    

