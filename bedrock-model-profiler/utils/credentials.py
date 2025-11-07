"""
Simplified credential management utilities for the Amazon Bedrock Expert application.
"""
import boto3
import configparser
import os
from typing import List, Optional, Tuple
from botocore.exceptions import ClientError, ProfileNotFound


def get_available_aws_profiles() -> List[str]:
    """
    Get list of available AWS profiles from AWS config file.
    Returns profiles with credentials first, then others.

    Returns:
        List of available profile names
    """
    profiles_with_creds = []
    profiles_without_creds = []

    try:
        # Check AWS config file
        config_path = os.path.expanduser('~/.aws/config')
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)

            # Check default profile
            if config.has_section('default'):
                if (config.has_option('default', 'aws_access_key_id') or
                    config.has_option('default', 'credential_process')):
                    profiles_with_creds.append('default')
                else:
                    profiles_without_creds.append('default')
            else:
                profiles_without_creds.append('default')  # Include default even if not in config

            # Check named profiles
            for section in config.sections():
                if section.startswith('profile '):
                    # Remove 'profile ' prefix
                    profile_name = section[8:]
                    if (config.has_option(section, 'aws_access_key_id') or
                        config.has_option(section, 'credential_process')):
                        profiles_with_creds.append(profile_name)
                    else:
                        profiles_without_creds.append(profile_name)

        # Also check credentials file
        creds_path = os.path.expanduser('~/.aws/credentials')
        if os.path.exists(creds_path):
            config = configparser.ConfigParser()
            config.read(creds_path)

            for section in config.sections():
                if section not in profiles_with_creds and section not in profiles_without_creds:
                    profiles_with_creds.append(section)

    except Exception:
        # Fallback to default if there's an error
        return ['default']

    # Return profiles with credentials first, then others
    return profiles_with_creds + profiles_without_creds


def validate_aws_credentials(profile_name: Optional[str] = None, region: str = 'us-east-1') -> Tuple[bool, str]:
    """
    Validate AWS credentials by attempting to list Bedrock models.

    Args:
        profile_name: AWS profile name to use (None for default)
        region: AWS region to use

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Create a session with the specified profile
        if profile_name and profile_name != 'default':
            session = boto3.Session(profile_name=profile_name, region_name=region)
        else:
            session = boto3.Session(region_name=region)

        # Try to create a Bedrock client and list models
        bedrock = session.client('bedrock', region_name=region)
        response = bedrock.list_foundation_models()

        model_count = len(response.get('modelSummaries', []))
        return True, f"✅ Credentials validated successfully. Found {model_count} model(s)."

    except ProfileNotFound:
        return False, f"❌ Profile '{profile_name}' not found."
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        return False, f"❌ AWS error: {error_code}"
    except Exception as e:
        return False, f"❌ Validation failed: {str(e)}"


def create_boto3_session(profile_name: Optional[str] = None, region: str = 'us-east-1') -> boto3.Session:
    """
    Create a boto3 session with the specified profile and region.

    Args:
        profile_name: AWS profile name to use (None for default)
        region: AWS region to use

    Returns:
        boto3.Session object
    """
    if profile_name and profile_name != 'default':
        return boto3.Session(profile_name=profile_name, region_name=region)
    else:
        return boto3.Session(region_name=region)
