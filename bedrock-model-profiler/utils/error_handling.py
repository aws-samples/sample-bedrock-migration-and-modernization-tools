"""
Error handling utilities for the Amazon Bedrock Expert application.
"""
import streamlit as st
from typing import Dict, Optional
from botocore.exceptions import ClientError


# Error categories for user-friendly messages
ERROR_CATEGORIES = {
    # AWS API errors
    'ExpiredToken': {
        'title': 'AWS Credentials Expired',
        'message': 'Your AWS credentials have expired. Please refresh your credentials in the AWS Credentials section.',
        'suggestion': 'Click "Refresh Credentials" in the AWS Credentials section.'
    },
    'AccessDeniedException': {
        'title': 'Access Denied',
        'message': 'Your AWS credentials don\'t have permission to access the requested resource.',
        'suggestion': 'Check your IAM permissions for Amazon Bedrock services.'
    },
    'UnauthorizedOperation': {
        'title': 'Unauthorized Operation',
        'message': 'Your AWS credentials don\'t have permission to perform this operation.',
        'suggestion': 'Check your IAM permissions for Amazon Bedrock services.'
    },
    'ValidationException': {
        'title': 'Validation Error',
        'message': 'The request parameters are invalid.',
        'suggestion': 'Check your input parameters and try again.'
    },
    'ResourceNotFoundException': {
        'title': 'Resource Not Found',
        'message': 'The requested resource was not found.',
        'suggestion': 'Check that the resource exists and try again.'
    },
    'ThrottlingException': {
        'title': 'API Rate Limit Exceeded',
        'message': 'You have exceeded the API rate limit for this operation.',
        'suggestion': 'Wait a moment and try again, or reduce the frequency of your requests.'
    },
    'ServiceUnavailableException': {
        'title': 'Service Unavailable',
        'message': 'The AWS service is currently unavailable.',
        'suggestion': 'Wait a moment and try again, or check the AWS Service Health Dashboard.'
    },

    # File system errors
    'FileNotFoundError': {
        'title': 'File Not Found',
        'message': 'The specified file was not found.',
        'suggestion': 'Check the file path and try again.'
    },
    'PermissionError': {
        'title': 'Permission Denied',
        'message': 'You don\'t have permission to access the specified file.',
        'suggestion': 'Check file permissions and try again.'
    },
    'IsADirectoryError': {
        'title': 'Is a Directory',
        'message': 'The specified path is a directory, not a file.',
        'suggestion': 'Check the path and try again.'
    },

    # Network errors
    'ConnectionError': {
        'title': 'Connection Error',
        'message': 'Failed to connect to the server.',
        'suggestion': 'Check your internet connection and try again.'
    },
    'TimeoutError': {
        'title': 'Connection Timeout',
        'message': 'The connection timed out.',
        'suggestion': 'Check your internet connection and try again.'
    },

    # JSON errors
    'JSONDecodeError': {
        'title': 'Invalid JSON',
        'message': 'Failed to parse JSON data.',
        'suggestion': 'Check the JSON format and try again.'
    },

    # Default error
    'default': {
        'title': 'An Error Occurred',
        'message': 'An unexpected error occurred.',
        'suggestion': 'Please try again or contact support if the problem persists.'
    }
}


def get_error_details(error: Exception) -> Dict[str, str]:
    """
    Get user-friendly error details based on the exception type.

    Args:
        error: The exception object

    Returns:
        Dictionary with error details
    """
    error_type = type(error).__name__
    error_message = str(error)

    # Check for boto3/botocore errors
    if isinstance(error, ClientError):
        error_code = error.response.get('Error', {}).get('Code', 'Unknown')
        error_message = error.response.get('Error', {}).get('Message', str(error))

        if error_code in ERROR_CATEGORIES:
            details = ERROR_CATEGORIES[error_code].copy()
            details['message'] = f"{details['message']} Details: {error_message}"
            return details

    # Check for known error types
    if error_type in ERROR_CATEGORIES:
        details = ERROR_CATEGORIES[error_type].copy()
        details['message'] = f"{details['message']} Details: {error_message}"
        return details

    # Default error details
    details = ERROR_CATEGORIES['default'].copy()
    details['message'] = f"{details['message']} Details: {error_message}"
    return details


def show_error_message(error: Exception, context: Optional[str] = None) -> None:
    """
    Display a user-friendly error message in the Streamlit UI.

    Args:
        error: The exception object
        context: Optional context information
    """
    # Get error details
    details = get_error_details(error)

    # Display error message
    st.error(f"**{details['title']}**")
    st.write(details['message'])

    # Display suggestion
    st.info(f"**Suggestion:** {details['suggestion']}")

    # Display technical details in an expander
    with st.expander("Technical Details", expanded=False):
        st.code(f"Error Type: {type(error).__name__}\nError Message: {str(error)}\nContext: {context or 'Not provided'}")