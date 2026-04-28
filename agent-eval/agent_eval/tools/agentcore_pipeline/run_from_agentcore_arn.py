#!/usr/bin/env python3
"""
AgentCore ARN-based Trace Extraction Wrapper

This module provides a thin wrapper orchestrator that simplifies extracting AgentCore traces
from CloudWatch. Users provide only an AgentCore Runtime ARN and optional parameters, and the
system handles all discovery and orchestration automatically.

Overview:
    The wrapper performs the following steps:
    1. Parse the AgentCore Runtime ARN
    2. Fetch runtime metadata from bedrock-agentcore-control API
    3. Discover CloudWatch log groups for the runtime
    4. Orchestrate the 3-script pipeline to extract and merge traces
    5. Write all outputs to a specified directory

Key Features:
    - ARN Parsing: Validates and extracts region, account ID, and resource ID from AgentCore ARNs
    - Runtime Metadata Resolution: Calls GetAgentRuntime API to retrieve authoritative runtime ID
    - Log Group Discovery: Automatically discovers runtime and OTEL log groups using CloudWatch APIs
    - Script Orchestration: Executes the 3-script pipeline in sequence with proper error handling
    - Output Management: Creates organized output directories with discovery artifacts
    - Error Handling: Provides clear, actionable error messages for all failure scenarios

CLI Usage Examples:
    # Basic usage with ARN only
    python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \\
        --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \\
        --region us-east-1 \\
        --output-dir ./out/run-001

    # With specific session and custom time window
    python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \\
        --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \\
        --region us-east-1 \\
        --session-id 44b744ad-db29-418d-9e57-fd1107face44 \\
        --minutes 180 \\
        --output-dir ./out/run-002

    # With AWS profile and debug mode
    python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \\
        --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \\
        --region us-east-1 \\
        --profile myprofile \\
        --minutes 300 \\
        --pad-seconds 7200 \\
        --output-dir ./out/run-003 \\
        --debug

Programmatic Usage Example:
    from agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn import (
        parse_agentcore_arn,
        get_runtime_metadata,
        discover_log_groups,
        execute_pipeline,
        PipelineConfig
    )

    # Parse ARN
    arn = parse_agentcore_arn("arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1")
    
    # Get runtime metadata
    runtime = get_runtime_metadata(arn, region="us-east-1")
    
    # Discover log groups
    log_groups = discover_log_groups(runtime.agent_runtime_id, region="us-east-1")
    
    # Execute pipeline
    config = PipelineConfig(
        arn=arn,
        runtime=runtime,
        log_groups=log_groups,
        region="us-east-1",
        profile=None,
        session_id=None,
        minutes=180,
        output_dir="./out/run-001",
        pad_seconds=7200,
        debug=False
    )
    
    result = execute_pipeline(config)
    
    if result.success:
        print(f"Pipeline completed successfully!")
        print(f"Final output: {result.script3_output}")
    else:
        print(f"Pipeline failed: {result.error}")

Output Files:
    The wrapper produces 4 output files in the specified output directory:
    - discovery.json: ARN parsing and log group discovery results
    - 01_session_turns.json: Script 1 output (extracted turns from runtime logs)
    - 02_session_enriched_runtime.json: Script 2 output (OTEL enrichment)
    - 03_turns_merged_normalized.json: Script 3 final output (merged and normalized traces)

Required IAM Permissions:
    - bedrock-agentcore-control:GetAgentRuntime
    - logs:DescribeLogGroups
    - logs:StartQuery
    - logs:GetQueryResults
"""
import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import re
import boto3

# Data structures

@dataclass
class ParsedARN:
    """Parsed AgentCore Runtime ARN components.

    This dataclass holds the parsed components of an AgentCore Runtime ARN.
    The resource_id field contains the full resource portion without further
    parsing to avoid fragility as ARN formats may evolve.

    Note: The example below is illustrative. Actual AgentCore runtime ARN
    structures may vary. The parser intentionally keeps resource_id unparsed
    for forward compatibility.

    Attributes:
        raw_arn: The original unparsed ARN string
        region: AWS region extracted from the ARN
        account_id: AWS account ID (12-digit string)
        resource_id: Full resource identifier (not further parsed for forward compatibility)
    """
    raw_arn: str
    region: str
    account_id: str
    resource_id: str


@dataclass
class RuntimeMetadata:
    """Metadata from GetAgentRuntime API response.

    This dataclass stores metadata retrieved from the bedrock-agentcore-control
    GetAgentRuntime API. It includes the authoritative agent_runtime_id which is
    used for log group discovery.

    Attributes:
        agent_runtime_id: The authoritative runtime ID from the API response
        status: Runtime status (e.g., "ACTIVE", "INACTIVE")
        raw_response: Complete API response dictionary for reference and debugging
    """
    agent_runtime_id: str
    status: str
    raw_response: Dict[str, Any]


@dataclass
class DiscoveredLogGroups:
    """Discovered CloudWatch log groups for AgentCore runtime.

    This dataclass stores the discovered CloudWatch log group names for both
    runtime logs and OTEL logs. It also tracks how the log groups were discovered
    (e.g., via API discovery or user override).

    Attributes:
        runtime_logs: Log group name for runtime logs (suffix: /runtime-logs)
        otel_logs: Log group name for OTEL logs (suffix: /otel-rt-logs)
        discovery_method: How the log groups were discovered (e.g., "describe_log_groups with prefix" or "override")
    """
    runtime_logs: str
    otel_logs: str
    discovery_method: str


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    This dataclass stores all configuration needed for executing the complete
    trace extraction pipeline. It includes parsed ARN data, runtime metadata,
    discovered log groups, and all user-provided parameters.

    The optional runtime_log_group_override and otel_log_group_override fields
    provide escape hatches for users who want to bypass automatic log group
    discovery and specify log groups directly. This is useful for:
    - Testing with non-standard log group names
    - Working with custom logging configurations
    - Debugging log group discovery issues

    Attributes:
        arn: Parsed AgentCore Runtime ARN components
        runtime: Runtime metadata from GetAgentRuntime API
        log_groups: Discovered CloudWatch log groups
        region: AWS region for API calls
        profile: Optional AWS profile name for credential selection
        session_id: Optional specific session ID to filter traces
        minutes: Lookback window in minutes for CloudWatch queries
        output_dir: Directory path for all output files
        pad_seconds: Time padding in seconds for CloudWatch queries
        debug: Enable debug output in scripts
        log_stream_kind: Log stream kind for Script 1 ("runtime" or "application", default: "runtime")
        runtime_log_group_override: Optional escape hatch to bypass runtime log group discovery
        otel_log_group_override: Optional escape hatch to bypass OTEL log group discovery
    """
    arn: ParsedARN
    runtime: RuntimeMetadata
    log_groups: DiscoveredLogGroups
    region: str
    profile: Optional[str]
    session_id: Optional[str]
    minutes: int
    output_dir: str
    pad_seconds: int
    debug: bool
    log_stream_kind: str = "runtime"
    runtime_log_group_override: Optional[str] = None
    otel_log_group_override: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    This dataclass stores the result of executing the complete trace extraction
    pipeline. It includes success status, paths to all output files, and any
    error message if the pipeline failed.

    Attributes:
        success: Whether the pipeline completed successfully
        discovery_file: Path to discovery.json file containing ARN parsing and log group discovery results (None if not created)
        script1_output: Path to Script 1 output file (01_session_turns.json) (None if not created)
        script2_output: Path to Script 2 output file (02_session_enriched_runtime.json) (None if not created)
        script3_output: Path to Script 3 output file (03_turns_merged_normalized.json) (None if not created)
        error: Error message if pipeline failed, None if successful
    """
    success: bool
    discovery_file: Optional[str]
    script1_output: Optional[str]
    script2_output: Optional[str]
    script3_output: Optional[str]
    error: Optional[str]


# Custom exceptions

class LogGroupNotFoundError(Exception):
    """Exception raised when log group discovery fails to find required log groups.

    This exception is raised by the discover_log_groups() function when it cannot
    find one or both of the required CloudWatch log groups for an AgentCore runtime:
    - Runtime logs (suffix: /runtime-logs)
    - OTEL logs (suffix: /otel-rt-logs)

    The exception message should provide clear information about:
    - Which log group(s) could not be found
    - The prefix used for discovery
    - Suggestions for troubleshooting (e.g., verify runtime has logging enabled)

    Example:
        raise LogGroupNotFoundError(
            "Runtime logs not found with prefix: /aws/bedrock-agentcore/runtimes/abc-123-"
        )
    """
    pass


class RuntimeNotFoundError(Exception):
    """Exception raised when GetAgentRuntime API returns ResourceNotFoundException.

    This exception is raised by the get_runtime_metadata() function when the
    bedrock-agentcore-control GetAgentRuntime API call returns a ResourceNotFoundException,
    indicating that the specified AgentCore runtime does not exist or is not accessible
    in the specified region.

    The exception message should provide clear information about:
    - The ARN that was not found
    - The region where the lookup was attempted
    - Suggestions for troubleshooting (e.g., verify ARN is correct, check region)

    Example:
        raise RuntimeNotFoundError(
            "Runtime not found: arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1. "
            "Verify runtime exists in region: us-east-1"
        )
    """
    pass


class ScriptExecutionError(Exception):
    """Exception raised when one of the 3 pipeline scripts fails during execution.

    This exception is raised by the script execution helper functions (execute_script1,
    execute_script2, execute_script3) when a script fails during execution. It provides
    clear error messages including which script failed and the error details from stderr.

    The exception message should provide clear information about:
    - Which script failed (Script 1, 2, or 3)
    - The command that was executed
    - The error details from stderr
    - The return code from the failed script

    This exception is used to distinguish script execution failures from other types of
    errors (user errors, system errors) and allows the pipeline to provide appropriate
    exit codes and error messages.

    Example:
        raise ScriptExecutionError(
            "Script 1 (01_export_turns_from_app_logs.py) failed with return code 1: "
            "Error: Invalid log group name"
        )
    """
    pass




# Core functions

def parse_agentcore_arn(arn: str) -> ParsedARN:
    """Parse and validate an AgentCore Runtime ARN.

    This function parses an AgentCore Runtime ARN and extracts its components:
    region, account ID, and resource ID. It performs comprehensive validation
    to ensure the ARN is well-formed and contains all required components.

    The function intentionally avoids over-parsing the resource_id field to
    maintain forward compatibility as ARN formats may evolve. The resource_id
    is kept as-is (e.g., "agent/abc-def-123:v1") without further decomposition.

    Expected ARN format:
        arn:aws:bedrock-agentcore:<region>:<account-id>:<resource-id>
        
    Example:
        arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/customersupportagent-crGhIpFJYP

    Args:
        arn: The AgentCore Runtime ARN string to parse

    Returns:
        ParsedARN: Dataclass containing parsed components:
            - raw_arn: Original ARN string
            - region: AWS region (e.g., "us-east-1")
            - account_id: 12-digit AWS account ID
            - resource_id: Full resource identifier (e.g., "agent/abc-def-123:v1")

    Raises:
        ValueError: If ARN validation fails for any of the following reasons:
            - ARN is None or empty string
            - ARN does not start with "arn:aws:bedrock-agentcore:"
            - ARN has fewer than 6 components when split by ":"
            - Account ID is not exactly 12 digits
            - Region is empty
            - Resource ID is empty

    Examples:
        >>> arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/customersupportagent-crGhIpFJYP"
        >>> parsed = parse_agentcore_arn(arn)
        >>> parsed.region
        'us-east-1'
        >>> parsed.account_id
        '123456789012'
        >>> parsed.resource_id
        'runtime/customersupportagent-crGhIpFJYP'

        >>> parse_agentcore_arn("invalid-arn")
        ValueError: Invalid ARN prefix. Expected ARN to start with 'arn:aws:bedrock-agentcore:', got: invalid-arn

        >>> parse_agentcore_arn("arn:aws:bedrock-agentcore:us-east-1:12345:agent/abc")
        ValueError: Invalid account ID format. Expected 12-digit account ID, got: 12345
    """
    # Validate input is not None or empty
    if not arn:
        raise ValueError("ARN cannot be None or empty string")

    # Task 2.1.2: Validate ARN prefix
    expected_prefix = "arn:aws:bedrock-agentcore:"
    if not arn.startswith(expected_prefix):
        raise ValueError(
            f"Invalid ARN prefix. Expected ARN to start with '{expected_prefix}', got: {arn}"
        )

    # Task 2.1.3: Split ARN by ":" and validate minimum component count
    parts = arn.split(":")
    if len(parts) < 6:
        raise ValueError(
            f"Invalid ARN format. Expected at least 6 components separated by ':', got {len(parts)} components. "
            f"Expected format: arn:aws:bedrock-agentcore:<region>:<account-id>:<resource-id>"
        )

    # Task 2.1.4: Extract region from component 3 (0-indexed)
    region = parts[3]
    if not region:
        raise ValueError("Invalid ARN format. Region component (index 3) is empty")

    # Task 2.1.5: Extract account_id from component 4 and validate 12-digit format
    account_id = parts[4]
    if not re.match(r'^\d{12}$', account_id):
        raise ValueError(
            f"Invalid account ID format. Expected 12-digit account ID, got: {account_id}"
        )

    # Task 2.1.6: Extract resource_id from component 5 onwards (keep as-is, don't over-parse)
    # Join remaining components with ":" to preserve any colons in the resource ID
    # (e.g., "agent/abc-def-123:v1" or potentially more complex formats in the future)
    resource_id = ":".join(parts[5:])
    if not resource_id:
        raise ValueError("Invalid ARN format. Resource ID component (index 5+) is empty")

    # Task 2.1.7: Return ParsedARN with raw_arn, region, account_id, resource_id
    return ParsedARN(
        raw_arn=arn,
        region=region,
        account_id=account_id,
        resource_id=resource_id
    )


def get_runtime_metadata(
    arn: ParsedARN,
    region: str,
    profile: Optional[str] = None
) -> RuntimeMetadata:
    """Fetch runtime metadata from bedrock-agentcore-control API.

    This function calls the bedrock-agentcore-control GetAgentRuntime API to retrieve
    authoritative metadata about an AgentCore runtime. The most important piece of
    metadata is the agent_runtime_id, which is used for CloudWatch log group discovery.

    The function handles common AWS API errors gracefully and provides clear, actionable
    error messages with troubleshooting guidance.

    Args:
        arn: Parsed AgentCore Runtime ARN containing the ARN string and components
        region: AWS region for the API call (should match ARN region)
        profile: Optional AWS profile name for credential selection. If None, uses
                default credential chain (environment variables, ~/.aws/credentials, IAM role)

    Returns:
        RuntimeMetadata: Dataclass containing:
            - agent_runtime_id: Authoritative runtime ID from API response
            - status: Runtime status (e.g., "ACTIVE", "INACTIVE")
            - raw_response: Complete API response dictionary for reference

    Raises:
        RuntimeNotFoundError: If the runtime does not exist or is not accessible
            (GetAgentRuntime API returns ResourceNotFoundException)
        RuntimeError: If AWS credentials are not configured (wraps NoCredentialsError)
        botocore.exceptions.ClientError: For other AWS API errors (network issues,
            permission errors, etc.)

    Examples:
        >>> arn = parse_agentcore_arn("arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/customersupportagent-crGhIpFJYP")
        >>> metadata = get_runtime_metadata(arn, region="us-east-1")
        >>> metadata.agent_runtime_id
        'customersupportagent-crGhIpFJYP'
        >>> metadata.status
        'READY'

        >>> # With AWS profile
        >>> metadata = get_runtime_metadata(arn, region="us-east-1", profile="myprofile")

        >>> # Runtime not found
        >>> get_runtime_metadata(invalid_arn, region="us-east-1")
        RuntimeNotFoundError: Runtime not found: arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/invalid:v1
        Verify runtime exists in region: us-east-1

        >>> # No credentials configured
        >>> get_runtime_metadata(arn, region="us-east-1")
        NoCredentialsError: Unable to locate credentials
    """
    from botocore.exceptions import ClientError, NoCredentialsError

    try:
        # Task 3.1.2: Initialize boto3 session with region and profile
        session = boto3.Session(region_name=region, profile_name=profile)

        # Task 3.1.3: Create bedrock-agentcore-control client
        client = session.client('bedrock-agentcore-control')

        # Task 3.1.3.5: Extract runtime ID from ARN resource_id
        # ARN resource_id format: "runtime/customersupportagent-crGhIpFJYP"
        # Extract the ID part after the last "/"
        runtime_id = arn.resource_id.split('/')[-1]

        # Task 3.1.4: Call GetAgentRuntime API with runtime ID (not full ARN)
        response = client.get_agent_runtime(agentRuntimeId=runtime_id)

        # Task 3.1.5: Extract agentRuntimeId from response
        agent_runtime_id = response.get('agentRuntimeId')
        if not agent_runtime_id:
            raise ValueError(
                f"GetAgentRuntime API response missing 'agentRuntimeId' field. "
                f"Response: {response}"
            )

        # Task 3.1.6: Extract status, agent_id, agent_version from response
        status = response.get('status', 'UNKNOWN')

        # Task 3.1.7: Return RuntimeMetadata with all fields
        return RuntimeMetadata(
            agent_runtime_id=agent_runtime_id,
            status=status,
            raw_response=response
        )

    # Task 3.1.9: Handle NoCredentialsError with setup instructions
    except NoCredentialsError as e:
        # Re-raise with additional context in a RuntimeError since NoCredentialsError
        # doesn't accept custom messages
        raise RuntimeError(
            "AWS credentials not found. Configure credentials using one of:\n"
            "  1. AWS CLI: aws configure --profile <profile>\n"
            "  2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
            "  3. AWS credentials file: ~/.aws/credentials\n"
            "  4. IAM role (if running on EC2/Lambda)\n"
            f"\nAttempted to access runtime: {arn.raw_arn}"
        ) from e

    # Task 3.1.8: Handle ResourceNotFoundException with user-friendly message
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        
        if error_code == 'ResourceNotFoundException':
            raise RuntimeNotFoundError(
                f"Runtime not found: {arn.raw_arn}\n"
                f"Verify runtime exists in region: {region}\n"
                f"Troubleshooting:\n"
                f"  - Check that the ARN is correct\n"
                f"  - Verify the runtime exists in the specified region\n"
                f"  - Ensure you have permission to access the runtime"
            ) from e
        
        # Task 3.1.10: Handle generic ClientError with error details
        error_message = e.response.get('Error', {}).get('Message', str(e))
        raise ClientError(
            {
                'Error': {
                    'Code': error_code,
                    'Message': f"Failed to get runtime metadata: {error_message}\n"
                               f"ARN: {arn.raw_arn}\n"
                               f"Region: {region}\n"
                               f"Error Code: {error_code}"
                }
            },
            'GetAgentRuntime'
        ) from e



def discover_log_groups(
    agent_runtime_id: str,
    region: str,
    profile: Optional[str] = None,
    runtime_override: Optional[str] = None,
    otel_override: Optional[str] = None
) -> DiscoveredLogGroups:
    """Discover CloudWatch log groups for an AgentCore runtime.

    This function discovers the CloudWatch log group associated with an AgentCore
    runtime by querying the CloudWatch Logs API with a prefix filter. 
    
    IMPORTANT: AgentCore uses a SINGLE log group containing multiple streams:
    - Runtime logs: Multiple streams with pattern [runtime-logs]
    - OTEL logs: Single stream named "otel-rt-logs"
    
    The function returns the same log group for both runtime_logs and otel_logs fields
    to maintain backward compatibility with the existing wrapper interface.

    Expected log group naming pattern:
        /aws/bedrock-agentcore/runtimes/<agentRuntimeId>-DEFAULT
        
    Within this log group:
        - Multiple [runtime-logs] streams (plain text Python logs)
        - One otel-rt-logs stream (JSON OTEL data)

    Args:
        agent_runtime_id: The authoritative agent runtime ID from GetAgentRuntime API
        region: AWS region for CloudWatch Logs API calls
        profile: Optional AWS profile name for credential selection. If None, uses
                default credential chain
        runtime_override: Optional escape hatch to bypass runtime log group discovery.
                         If provided along with otel_override, skips API calls entirely
        otel_override: Optional escape hatch to bypass OTEL log group discovery.
                      If provided along with runtime_override, skips API calls entirely

    Returns:
        DiscoveredLogGroups: Dataclass containing:
            - runtime_logs: Log group name (same as otel_logs for AgentCore)
            - otel_logs: Log group name (same as runtime_logs for AgentCore)
            - discovery_method: How the log groups were discovered
                              ("override" if overrides used, "describe_log_groups with prefix" otherwise)

    Raises:
        LogGroupNotFoundError: If the required log group is not found and no override
                              is provided. The error message includes:
                              - The prefix used for discovery
                              - Troubleshooting suggestions

    Examples:
        >>> # Normal discovery
        >>> log_groups = discover_log_groups(
        ...     agent_runtime_id="customersupportagent-crGhIpFJYP",
        ...     region="us-east-1"
        ... )
        >>> log_groups.runtime_logs
        '/aws/bedrock-agentcore/runtimes/customersupportagent-crGhIpFJYP-DEFAULT'
        >>> log_groups.otel_logs
        '/aws/bedrock-agentcore/runtimes/customersupportagent-crGhIpFJYP-DEFAULT'
        >>> log_groups.discovery_method
        'describe_log_groups with prefix'

        >>> # With overrides (escape hatch)
        >>> log_groups = discover_log_groups(
        ...     agent_runtime_id="customersupportagent-crGhIpFJYP",
        ...     region="us-east-1",
        ...     runtime_override="/custom/runtime/logs",
        ...     otel_override="/custom/otel/logs"
        ... )
        >>> log_groups.runtime_logs
        '/custom/runtime/logs'
        >>> log_groups.otel_logs
        '/custom/otel/logs'
        >>> log_groups.discovery_method
        'override'

        >>> # Missing log groups
        >>> discover_log_groups("nonexistent-runtime-id", "us-east-1")
        LogGroupNotFoundError: Runtime log group not found with prefix: /aws/bedrock-agentcore/runtimes/nonexistent-runtime-id-
        Verify runtime has logging enabled in AgentCore configuration
    """
    # Task 5.1.2: If runtime_override and otel_override provided, return immediately (escape hatch)
    if runtime_override and otel_override:
        return DiscoveredLogGroups(
            runtime_logs=runtime_override,
            otel_logs=otel_override,
            discovery_method="override"
        )

    # Task 5.1.3: Initialize boto3 session with region and profile
    session = boto3.Session(region_name=region, profile_name=profile)

    # Task 5.1.4: Create CloudWatch Logs client
    logs_client = session.client('logs')

    # Task 5.1.5: Build log group prefix
    prefix = f"/aws/bedrock-agentcore/runtimes/{agent_runtime_id}-"

    # Task 5.1.6 & 5.1.7: Call describe_log_groups with prefix filter and paginate through all results
    log_groups = []
    paginator = logs_client.get_paginator('describe_log_groups')
    
    for page in paginator.paginate(logGroupNamePrefix=prefix):
        for log_group in page.get('logGroups', []):
            log_groups.append(log_group['logGroupName'])
    
    # Bug fix: Sort log groups alphabetically for deterministic selection
    log_groups.sort()

    # Task 5.1.8: Find log group (prefer -DEFAULT suffix)
    runtime_log_group = None
    
    # First, try to find a group ending with -DEFAULT
    for group in log_groups:
        if group.endswith('-DEFAULT'):
            runtime_log_group = group
            break
    
    # If no -DEFAULT group found, take the first one
    if not runtime_log_group and log_groups:
        runtime_log_group = log_groups[0]

    # Task 5.1.10 & 5.1.11: Validate log group found (or use overrides if provided)
    # If runtime_override provided but runtime_log_group not found, use override
    if not runtime_log_group and runtime_override:
        runtime_log_group = runtime_override
    
    # For backward compatibility, use the same group for OTEL unless override provided
    otel_log_group = otel_override if otel_override else runtime_log_group

    # Raise error if missing and no override
    if not runtime_log_group:
        raise LogGroupNotFoundError(
            f"Runtime log group not found with prefix: {prefix}\n"
            f"Expected log group with pattern: {prefix}DEFAULT or {prefix}*\n"
            f"Found log groups: {log_groups}\n"
            f"Troubleshooting:\n"
            f"  - Verify runtime has logging enabled in AgentCore configuration\n"
            f"  - Check that the runtime has been invoked at least once\n"
            f"  - Verify you have logs:DescribeLogGroups permission\n"
            f"  - Use --runtime-log-group override if using custom log group names"
        )

    # Task 5.1.12: Return DiscoveredLogGroups with log group name
    # Note: Both fields point to the same log group for AgentCore
    # The OTEL data is in the "otel-rt-logs" stream within this group
    discovery_method = "describe_log_groups with prefix"
    if runtime_override or otel_override:
        discovery_method = "partial override"
    
    return DiscoveredLogGroups(
        runtime_logs=runtime_log_group,
        otel_logs=otel_log_group,
        discovery_method=discovery_method
    )


# Script execution helpers (MVP: subprocess orchestration)
# NOTE: This is an MVP approach using subprocess.run to execute scripts.
# Future refactoring should consider direct module imports for better error handling
# and performance. Marked for future enhancement.

def execute_script1(
    log_group: str,
    session_id: Optional[str],
    minutes: int,
    output: str,
    region: str,
    profile: Optional[str],
    output_dir: str,
    log_stream_kind: str = "runtime"
) -> str:
    """Execute Script 1 (01_export_turns_from_app_logs.py) to extract turns from runtime logs.

    This function builds and executes the command-line for Script 1, which queries
    CloudWatch Logs to extract turn data from AgentCore runtime logs. It uses
    subprocess.run for execution (MVP approach - marked for future refactoring).

    Args:
        log_group: CloudWatch log group name for runtime logs
        session_id: Optional specific session ID to filter
        minutes: Lookback window in minutes for CloudWatch query
        output: Output file path for Script 1 results
        region: AWS region for CloudWatch Logs API
        profile: Optional AWS profile name for credentials
        output_dir: Output directory for intermediate files
        log_stream_kind: Log stream kind ("runtime" or "application", default: "runtime")

    Returns:
        str: Path to Script 1 output file

    Raises:
        ScriptExecutionError: If Script 1 execution fails
    """
    import sys
    import os
    
    # Build path to Script 1
    script_path = os.path.join(
        os.path.dirname(__file__),
        "01_export_turns_from_app_logs.py"
    )
    
    # Build command-line arguments for Script 1
    # Bug fixes:
    # - Use sys.executable with script path instead of python -m (fixes digit-prefixed module issue)
    # - Use --out instead of --output (matches Script 1's actual argument)
    # - Add --log-stream-kind to support runtime vs application log streams
    cmd = [
        sys.executable, script_path,
        "--log-group", log_group,
        "--minutes", str(minutes),
        "--out", output,
        "--region", region,
        "--output-dir", output_dir,
        "--log-stream-kind", log_stream_kind
    ]
    
    if session_id:
        cmd.extend(["--session-id", session_id])
    
    if profile:
        cmd.extend(["--profile", profile])
    
    # Execute Script 1 using subprocess.run
    # Security: cmd is a list built from validated CLI args and internal paths — no shell injection risk
    try:
        result = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit — list-form call, no shell=True
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return output
    except subprocess.CalledProcessError as e:
        raise ScriptExecutionError(
            f"Script 1 (01_export_turns_from_app_logs.py) failed with return code {e.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error output:\n{e.stderr}"
        ) from e


def execute_script2(
    turns: str,
    otel_log_group: str,
    region: str,
    pad_seconds: int,
    output: str,
    profile: Optional[str] = None
) -> str:
    """Execute Script 2 (02_build_session_trace_index.py) to enrich turns with OTEL data.

    This function builds and executes the command-line for Script 2, which enriches
    the turn data from Script 1 with OTEL trace data from CloudWatch Logs. It uses
    subprocess.run for execution (MVP approach - marked for future refactoring).

    Note: Script 2 doesn't support --profile flag, so we pass it via AWS_PROFILE
    environment variable to ensure consistent credentials with Script 1.

    Args:
        turns: Path to Script 1 output file (01_session_turns.json)
        otel_log_group: CloudWatch log group name for OTEL logs
        region: AWS region for CloudWatch Logs API
        pad_seconds: Time padding in seconds for CloudWatch queries
        output: Output file path for Script 2 results
        profile: Optional AWS profile name (passed via AWS_PROFILE env var)

    Returns:
        str: Path to Script 2 output file

    Raises:
        ScriptExecutionError: If Script 2 execution fails
    """
    import sys
    import os
    
    # Build path to Script 2
    script_path = os.path.join(
        os.path.dirname(__file__),
        "02_build_session_trace_index.py"
    )
    
    # Build command-line arguments for Script 2
    # Bug fixes:
    # - Use sys.executable with script path instead of python -m (fixes digit-prefixed module issue)
    # - Use --out instead of --output (matches Script 2's actual argument)
    # - Remove --profile (Script 2 doesn't support this flag)
    # - Remove --debug (Script 2 doesn't support this flag, only --debug-sample-traces)
    cmd = [
        sys.executable, script_path,
        "--turns", turns,
        "--otel-log-group", otel_log_group,
        "--region", region,
        "--pad-seconds", str(pad_seconds),
        "--out", output
    ]
    
    # Pass profile and region via environment variables for consistent credentials
    # AWS_REGION ensures boto3 uses the correct region even if internal code doesn't use --region arg
    env = os.environ.copy()
    if profile:
        env["AWS_PROFILE"] = profile
    env["AWS_REGION"] = region
    
    # Execute Script 2 using subprocess.run
    # Security: cmd is a list built from validated CLI args and internal paths — no shell injection risk
    try:
        result = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit — list-form call, no shell=True
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        return output
    except subprocess.CalledProcessError as e:
        raise ScriptExecutionError(
            f"Script 2 (02_build_session_trace_index.py) failed with return code {e.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error output:\n{e.stderr}"
        ) from e


def execute_script3(
    index: str,
    detail: str,
    output: str,
    debug: bool
) -> str:
    """Execute Script 3 (03_add_xray_steps_and_latency.py) to merge and normalize traces.

    This function builds and executes the command-line for Script 3, which merges
    the turn data from Script 1 with the OTEL enrichment from Script 2 and produces
    the final normalized trace output. It uses subprocess.run for execution
    (MVP approach - marked for future refactoring).

    Args:
        index: Path to Script 1 output file (01_session_turns.json)
        detail: Path to Script 2 output file (02_session_enriched_runtime.json)
        output: Output file path for Script 3 results
        debug: Enable debug output

    Returns:
        str: Path to Script 3 output file

    Raises:
        ScriptExecutionError: If Script 3 execution fails
    """
    import sys
    import os
    
    # Build path to Script 3
    script_path = os.path.join(
        os.path.dirname(__file__),
        "03_add_xray_steps_and_latency.py"
    )
    
    # Build command-line arguments for Script 3
    # Bug fixes:
    # - Use sys.executable with script path instead of python -m (fixes digit-prefixed module issue)
    # - Use --out instead of --output (matches Script 3's actual argument)
    cmd = [
        sys.executable, script_path,
        "--index", index,
        "--detail", detail,
        "--out", output
    ]
    
    if debug:
        cmd.append("--debug")
    
    # Execute Script 3 using subprocess.run
    # Security: cmd is a list built from validated CLI args and internal paths — no shell injection risk
    try:
        result = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit — list-form call, no shell=True
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return output
    except subprocess.CalledProcessError as e:
        raise ScriptExecutionError(
            f"Script 3 (03_add_xray_steps_and_latency.py) failed with return code {e.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error output:\n{e.stderr}"
        ) from e


# Artifact validation functions

def validate_script1_output(file_path: str) -> bool:
    """Validate Script 1 output file has required schema.

    This function validates that the Script 1 output file exists, is valid JSON,
    and contains the required fields: run_id, window, and turns. It also verifies
    that turns is a non-empty list.

    Args:
        file_path: Path to Script 1 output file

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails with clear error message
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Verify required fields
        if 'run_id' not in data:
            raise ValueError(f"Script 1 output missing required field 'run_id'")
        
        if 'window' not in data:
            raise ValueError(f"Script 1 output missing required field 'window'")
        
        if 'turns' not in data:
            raise ValueError(f"Script 1 output missing required field 'turns'")
        
        # Verify turns is a list
        if not isinstance(data['turns'], list):
            raise ValueError(f"Script 1 output field 'turns' must be a list, got {type(data['turns'])}")
        
        # Bug fix: Zero turns may be valid (no data in time window), but print strong warning
        # Empty turns is often the first sign that Script 1's query doesn't match the log schema
        if len(data['turns']) == 0:
            import warnings
            import sys
            
            # Print visible warning to stderr in addition to warnings.warn
            warning_msg = (
                "\n" + "="*80 + "\n"
                "WARNING: Script 1 returned ZERO turns!\n"
                "="*80 + "\n"
                "This often indicates one of the following issues:\n"
                "  1. Script 1 query template doesn't match your CloudWatch log schema\n"
                "  2. No data exists in the specified time window (--minutes)\n"
                "  3. Session ID filter excluded all data (--session-id)\n"
                "  4. Log group contains no matching log streams\n"
                "\n"
                "Troubleshooting steps:\n"
                "  - Verify log group contains data in the time window\n"
                "  - Check that --log-stream-kind matches your log type (runtime vs application)\n"
                "  - Increase --minutes to expand the time window\n"
                "  - Remove --session-id to see all sessions\n"
                "  - Manually inspect CloudWatch Logs Insights with the query from Script 1\n"
                "="*80 + "\n"
            )
            
            print(warning_msg, file=sys.stderr)
            warnings.warn(
                "Script 1 output field 'turns' is empty. "
                "This may indicate query/log schema mismatch or no data in time window."
            )
        
        return True
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Script 1 output is not valid JSON: {e}") from e
    except FileNotFoundError as e:
        raise ValueError(f"Script 1 output file not found: {file_path}") from e


def validate_script2_output(file_path: str) -> bool:
    """Validate Script 2 output file has required schema.

    This function validates that the Script 2 output file exists, is valid JSON,
    and contains the required fields: sessions and enrich_stats.

    Args:
        file_path: Path to Script 2 output file

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails with clear error message
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Verify required fields
        if 'sessions' not in data:
            raise ValueError(f"Script 2 output missing required field 'sessions'")
        
        if 'enrich_stats' not in data:
            raise ValueError(f"Script 2 output missing required field 'enrich_stats'")
        
        return True
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Script 2 output is not valid JSON: {e}") from e
    except FileNotFoundError as e:
        raise ValueError(f"Script 2 output file not found: {file_path}") from e


def validate_script3_output(file_path: str) -> bool:
    """Validate Script 3 output file has required schema.

    This function validates that the Script 3 output file exists, is valid JSON,
    and contains the required fields: turns_merged_normalized and merge_stats.

    Args:
        file_path: Path to Script 3 output file

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails with clear error message
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Verify required fields
        if 'turns_merged_normalized' not in data:
            raise ValueError(f"Script 3 output missing required field 'turns_merged_normalized'")
        
        if 'merge_stats' not in data:
            raise ValueError(f"Script 3 output missing required field 'merge_stats'")
        
        return True
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Script 3 output is not valid JSON: {e}") from e
    except FileNotFoundError as e:
        raise ValueError(f"Script 3 output file not found: {file_path}") from e


# Pipeline execution

def execute_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the complete trace extraction pipeline.

    This function orchestrates the execution of all 3 scripts in sequence:
    1. Create output directory
    2. Write discovery.json artifact
    3. Execute Script 1 (export turns from runtime logs)
    4. Validate Script 1 output
    5. Execute Script 2 (enrich with OTEL data)
    6. Validate Script 2 output
    7. Execute Script 3 (merge and normalize)
    8. Validate Script 3 output

    The function handles errors at each step and returns a PipelineResult with
    success status and error details if any step fails.

    Args:
        config: PipelineConfig containing all configuration for pipeline execution

    Returns:
        PipelineResult: Result object containing:
            - success: True if all steps completed successfully
            - discovery_file: Path to discovery.json
            - script1_output: Path to Script 1 output
            - script2_output: Path to Script 2 output
            - script3_output: Path to Script 3 output
            - error: Error message if any step failed, None otherwise

    Examples:
        >>> config = PipelineConfig(...)
        >>> result = execute_pipeline(config)
        >>> if result.success:
        ...     print(f"Pipeline completed! Final output: {result.script3_output}")
        ... else:
        ...     print(f"Pipeline failed: {result.error}")
    """
    import datetime
    
    try:
        # Create output directory if it doesn't exist
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build discovery data dictionary
        discovery_data = {
            "arn": {
                "raw_arn": config.arn.raw_arn,
                "region": config.arn.region,
                "account_id": config.arn.account_id,
                "resource_id": config.arn.resource_id
            },
            "runtime_metadata": {
                "agent_runtime_id": config.runtime.agent_runtime_id,
                "status": config.runtime.status,
                "raw_response": config.runtime.raw_response
            },
            "log_groups": {
                "runtime_logs": config.log_groups.runtime_logs,
                "otel_logs": config.log_groups.otel_logs,
                "discovery_method": config.log_groups.discovery_method
            },
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "config": {
                "region": config.region,
                "profile": config.profile,
                "session_id": config.session_id,
                "minutes": config.minutes,
                "pad_seconds": config.pad_seconds,
                "debug": config.debug,
                "log_stream_kind": config.log_stream_kind
            }
        }
        
        # Write discovery.json file
        discovery_file = str(output_path / "discovery.json")
        with open(discovery_file, 'w') as f:
            json.dump(discovery_data, f, indent=2, default=str)
        
        # Execute Script 1 and capture output path
        script1_output = str(output_path / "01_session_turns.json")
        print(f"Executing Script 1: Extracting turns from runtime logs...")
        execute_script1(
            log_group=config.log_groups.runtime_logs,
            session_id=config.session_id,
            minutes=config.minutes,
            output=script1_output,
            region=config.region,
            profile=config.profile,
            output_dir=config.output_dir,
            log_stream_kind=config.log_stream_kind
        )
        
        # Verify Script 1 output file exists and validate schema
        if not Path(script1_output).exists():
            raise ScriptExecutionError(f"Script 1 completed but output file not found: {script1_output}")
        
        print(f"Validating Script 1 output...")
        validate_script1_output(script1_output)
        print(f"✓ Script 1 completed: {script1_output}")
        
        # Execute Script 2 and capture output path
        script2_output = str(output_path / "02_session_enriched_runtime.json")
        print(f"Executing Script 2: Enriching with OTEL data...")
        execute_script2(
            turns=script1_output,
            otel_log_group=config.log_groups.otel_logs,
            region=config.region,
            pad_seconds=config.pad_seconds,
            output=script2_output,
            profile=config.profile
        )
        
        # Verify Script 2 output file exists and validate schema
        if not Path(script2_output).exists():
            raise ScriptExecutionError(f"Script 2 completed but output file not found: {script2_output}")
        
        print(f"Validating Script 2 output...")
        validate_script2_output(script2_output)
        print(f"✓ Script 2 completed: {script2_output}")
        
        # Execute Script 3 and capture output path
        script3_output = str(output_path / "03_turns_merged_normalized.json")
        print(f"Executing Script 3: Merging and normalizing traces...")
        execute_script3(
            index=script1_output,
            detail=script2_output,
            output=script3_output,
            debug=config.debug
        )
        
        # Verify Script 3 output file exists and validate schema
        if not Path(script3_output).exists():
            raise ScriptExecutionError(f"Script 3 completed but output file not found: {script3_output}")
        
        print(f"Validating Script 3 output...")
        validate_script3_output(script3_output)
        print(f"✓ Script 3 completed: {script3_output}")
        
        # Return PipelineResult with success=True
        return PipelineResult(
            success=True,
            discovery_file=discovery_file,
            script1_output=script1_output,
            script2_output=script2_output,
            script3_output=script3_output,
            error=None
        )
    
    except (ScriptExecutionError, ValueError, OSError) as e:
        # Handle script failures and return PipelineResult with success=False and error message
        return PipelineResult(
            success=False,
            discovery_file=discovery_file if 'discovery_file' in locals() else None,
            script1_output=script1_output if 'script1_output' in locals() else None,
            script2_output=script2_output if 'script2_output' in locals() else None,
            script3_output=script3_output if 'script3_output' in locals() else None,
            error=str(e)
        )


# CLI implementation

def print_summary(config: PipelineConfig, result: PipelineResult, execution_time: float) -> None:
    """Print a summary of the pipeline execution.

    This function prints a formatted summary of the pipeline execution including:
    - ARN used
    - Runtime ID discovered
    - Log groups discovered
    - Output file paths
    - Execution time

    Args:
        config: PipelineConfig used for execution
        result: PipelineResult from pipeline execution
        execution_time: Total execution time in seconds
    """
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"\nARN: {config.arn.raw_arn}")
    
    # Bug fix: Clearly indicate when runtime lookup was skipped due to overrides
    if config.runtime.agent_runtime_id == "overridden":
        print(f"Runtime ID: (runtime lookup skipped due to log group overrides)")
        print(f"Status: (runtime lookup skipped due to log group overrides)")
    else:
        print(f"Runtime ID: {config.runtime.agent_runtime_id}")
        print(f"Status: {config.runtime.status}")
    
    print(f"\nLog Groups:")
    print(f"  Runtime Logs: {config.log_groups.runtime_logs}")
    print(f"  OTEL Logs: {config.log_groups.otel_logs}")
    print(f"  Discovery Method: {config.log_groups.discovery_method}")
    print(f"\nConfiguration:")
    print(f"  Region: {config.region}")
    if config.profile:
        print(f"  Profile: {config.profile}")
    if config.session_id:
        print(f"  Session ID: {config.session_id}")
    print(f"  Time Window: {config.minutes} minutes")
    print(f"  Pad Seconds: {config.pad_seconds}")
    print(f"  Log Stream Kind: {config.log_stream_kind}")
    print(f"\nOutput Files:")
    print(f"  Discovery: {result.discovery_file or 'N/A'}")
    print(f"  Script 1: {result.script1_output or 'N/A'}")
    print(f"  Script 2: {result.script2_output or 'N/A'}")
    print(f"  Script 3: {result.script3_output or 'N/A'}")
    print(f"\nExecution Time: {execution_time:.2f} seconds")
    print("="*80)


def main():
    """Main CLI entry point for the AgentCore ARN-based trace extraction wrapper.

    This function implements the complete CLI workflow:
    1. Parse command-line arguments
    2. Parse and validate ARN
    3. Fetch runtime metadata from API (unless overrides provided)
    4. Discover log groups (unless overrides provided)
    5. Execute pipeline
    6. Print summary

    Exit codes:
        0: Success
        1: User error (invalid ARN, missing credentials, etc.)
        2: System error (API failures, network issues, etc.)
        3: Script error (script execution failures)
    """
    import sys
    import time
    from datetime import datetime
    
    # Create ArgumentParser with description
    parser = argparse.ArgumentParser(
        description="Extract AgentCore traces from CloudWatch using only an ARN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with ARN only
  python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \\
    --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \\
    --region us-east-1

  # With specific session and custom time window
  python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \\
    --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \\
    --session-id 44b744ad-db29-418d-9e57-fd1107face44 \\
    --minutes 180

  # With log group overrides (escape hatch)
  python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \\
    --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \\
    --runtime-log-group /custom/runtime/logs \\
    --otel-log-group /custom/otel/logs

Required IAM Permissions:
  - bedrock-agentcore-control:GetAgentRuntime
  - logs:DescribeLogGroups
  - logs:StartQuery
  - logs:GetQueryResults
        """
    )
    
    # Add CLI arguments
    parser.add_argument(
        "--agent-runtime-arn",
        required=True,
        help="AgentCore Runtime ARN (e.g., arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1)"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--profile",
        help="AWS profile name for credential selection (optional)"
    )
    parser.add_argument(
        "--session-id",
        help="Specific session ID to extract (optional, extracts all sessions if not specified)"
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=180,
        help="Lookback window in minutes for CloudWatch queries (default: 180)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory path (default: ./out/run-<timestamp>)"
    )
    parser.add_argument(
        "--pad-seconds",
        type=int,
        default=7200,
        help="Time padding in seconds for CloudWatch queries (default: 7200)"
    )
    parser.add_argument(
        "--runtime-log-group",
        help="Override runtime log group discovery with explicit log group name. "
             "Can be used alone (partial override) or with --otel-log-group (full override). "
             "Partial override still requires API calls for runtime metadata and missing log group."
    )
    parser.add_argument(
        "--otel-log-group",
        help="Override OTEL log group discovery with explicit log group name. "
             "Can be used alone (partial override) or with --runtime-log-group (full override). "
             "Partial override still requires API calls for runtime metadata and missing log group."
    )
    parser.add_argument(
        "--log-stream-kind",
        choices=["runtime", "application"],
        default="runtime",
        help="Log stream kind for Script 1: 'runtime' for direct AgentCore runtime logs, "
             "'application' for APPLICATION_LOGS filter (default: runtime)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output in Script 3 only (Scripts 1 and 2 don't support this flag)"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./out/run-{timestamp}"
    
    start_time = time.time()
    
    try:
        # Step 1: Parse ARN
        print("Parsing ARN...")
        parsed_arn = parse_agentcore_arn(args.agent_runtime_arn)
        print(f"✓ ARN parsed successfully")
        print(f"  Region: {parsed_arn.region}")
        print(f"  Account ID: {parsed_arn.account_id}")
        print(f"  Resource ID: {parsed_arn.resource_id}")
        
        # Step 2: Check if escape hatches are used
        using_overrides = args.runtime_log_group and args.otel_log_group
        
        if using_overrides:
            print("\nUsing log group overrides (escape hatch mode)")
            print(f"  Runtime Log Group: {args.runtime_log_group}")
            print(f"  OTEL Log Group: {args.otel_log_group}")
            
            # Create minimal runtime metadata (skip GetAgentRuntime API)
            runtime_metadata = RuntimeMetadata(
                agent_runtime_id="overridden",
                status="UNKNOWN",
                raw_response={"note": "Skipped GetAgentRuntime API due to log group overrides"}
            )
            
            # Create log groups from overrides
            log_groups = DiscoveredLogGroups(
                runtime_logs=args.runtime_log_group,
                otel_logs=args.otel_log_group,
                discovery_method="override"
            )
        else:
            # Step 3: Get runtime metadata from API
            print("\nFetching runtime metadata...")
            runtime_metadata = get_runtime_metadata(
                parsed_arn,
                region=args.region,
                profile=args.profile
            )
            print(f"✓ Runtime metadata retrieved")
            print(f"  Agent Runtime ID: {runtime_metadata.agent_runtime_id}")
            print(f"  Status: {runtime_metadata.status}")
            
            # Step 4: Discover log groups
            print("\nDiscovering log groups...")
            log_groups = discover_log_groups(
                runtime_metadata.agent_runtime_id,
                region=args.region,
                profile=args.profile,
                runtime_override=args.runtime_log_group,
                otel_override=args.otel_log_group
            )
            print(f"✓ Log groups discovered")
            print(f"  Runtime Logs: {log_groups.runtime_logs}")
            print(f"  OTEL Logs: {log_groups.otel_logs}")
            print(f"  Discovery Method: {log_groups.discovery_method}")
        
        # Step 5: Build PipelineConfig
        config = PipelineConfig(
            arn=parsed_arn,
            runtime=runtime_metadata,
            log_groups=log_groups,
            region=args.region,
            profile=args.profile,
            session_id=args.session_id,
            minutes=args.minutes,
            output_dir=args.output_dir,
            pad_seconds=args.pad_seconds,
            debug=args.debug,
            log_stream_kind=args.log_stream_kind,
            runtime_log_group_override=args.runtime_log_group,
            otel_log_group_override=args.otel_log_group
        )
        
        # Step 6: Execute pipeline
        print(f"\nExecuting pipeline...")
        print(f"Output directory: {args.output_dir}")
        result = execute_pipeline(config)
        
        # Step 7: Print summary
        execution_time = time.time() - start_time
        
        if result.success:
            print_summary(config, result, execution_time)
            print("\n✓ Pipeline completed successfully!")
            sys.exit(0)
        else:
            print(f"\n✗ Pipeline failed: {result.error}")
            sys.exit(3)  # Script error
    
    except ValueError as e:
        # User error (invalid ARN, validation failures, etc.)
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    except (RuntimeNotFoundError, LogGroupNotFoundError) as e:
        # User error (runtime or log groups not found)
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    except RuntimeError as e:
        # System error (credentials, API failures, etc.)
        print(f"\n✗ System Error: {e}", file=sys.stderr)
        sys.exit(2)
    
    except Exception as e:
        # Unexpected error
        print(f"\n✗ Unexpected Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
