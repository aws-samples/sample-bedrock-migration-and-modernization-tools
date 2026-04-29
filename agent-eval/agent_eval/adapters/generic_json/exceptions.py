# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Exception taxonomy for Generic JSON adapter.

This module defines a hierarchy of exceptions for the adapter, enabling
precise error handling and clear error messages for different failure modes.

Exception Hierarchy:
    AdapterError (base)
    ├── InputError (file/JSON issues)
    ├── ValidationError (schema/data validation)
    └── AdaptationError (internal logic errors)
"""


class AdapterError(Exception):
    """
    Base exception for all adapter errors.
    
    All adapter-specific exceptions inherit from this class, allowing
    callers to catch all adapter errors with a single except clause.
    """
    pass


class InputError(AdapterError):
    """
    Exception raised for input file or JSON parsing errors.
    
    This exception is raised when:
    - Trace file doesn't exist (FileNotFoundError)
    - JSON is malformed or invalid (JSONDecodeError)
    - File is unreadable due to permissions or encoding issues
    
    The exception message always includes the file path for debugging.
    
    Attributes:
        file_path: Path to the problematic file
        original_error: The underlying exception (if any)
    """
    
    def __init__(self, message: str, file_path: str = None, original_error: Exception = None):
        """
        Initialize InputError with context.
        
        Args:
            message: Human-readable error description
            file_path: Path to the file that caused the error
            original_error: The underlying exception (e.g., JSONDecodeError)
        """
        self.file_path = file_path
        self.original_error = original_error
        
        # Build comprehensive error message
        if file_path:
            full_message = f"{message} (file: {file_path})"
        else:
            full_message = message
        
        if original_error:
            full_message += f" - {type(original_error).__name__}: {str(original_error)}"
        
        super().__init__(full_message)


class ValidationError(AdapterError):
    """
    Exception raised for schema validation or data validation errors.
    
    This exception is raised when:
    - Output doesn't conform to normalized schema
    - Schema file cannot be loaded
    - No events exist in input trace
    - Input is completely unreadable (no valid data)
    - Required fields are missing and cannot be gracefully handled
    
    The exception message includes details about which fields failed validation.
    
    Attributes:
        validation_errors: List of specific validation error messages
        schema_path: Path to the schema file (if applicable)
    """
    
    def __init__(self, message: str, validation_errors: list = None, schema_path: str = None):
        """
        Initialize ValidationError with validation details.
        
        Args:
            message: Human-readable error description
            validation_errors: List of specific validation failures
            schema_path: Path to the schema file (if applicable)
        """
        self.validation_errors = validation_errors or []
        self.schema_path = schema_path
        
        # Build comprehensive error message
        full_message = message
        
        if schema_path:
            full_message += f" (schema: {schema_path})"
        
        if validation_errors:
            full_message += "\nValidation errors:"
            for error in validation_errors[:10]:  # Limit to first 10 errors
                full_message += f"\n  - {error}"
            if len(validation_errors) > 10:
                full_message += f"\n  ... and {len(validation_errors) - 10} more errors"
        
        super().__init__(full_message)


class AdaptationError(AdapterError):
    """
    Exception raised for internal adapter logic errors.
    
    This exception is raised when:
    - Internal adapter logic encounters an unexpected state
    - Configuration is invalid or inconsistent
    - Required internal operations fail unexpectedly
    
    This typically indicates a bug in the adapter implementation or
    an unsupported edge case that wasn't handled gracefully.
    
    Attributes:
        context: Additional context about where the error occurred
    """
    
    def __init__(self, message: str, context: dict = None):
        """
        Initialize AdaptationError with context.
        
        Args:
            message: Human-readable error description
            context: Additional context (e.g., event index, turn ID)
        """
        self.context = context or {}
        
        # Build comprehensive error message
        full_message = message
        
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message += f" (context: {context_str})"
        
        super().__init__(full_message)
