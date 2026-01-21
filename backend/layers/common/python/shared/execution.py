"""
Execution ID utilities.
"""


def parse_execution_id(execution_id_or_arn: str) -> str:
    """
    Extract execution ID from ARN if needed.

    Step Functions can pass either a direct execution ID or a full ARN.
    This function normalizes both formats to just the execution ID portion.

    Args:
        execution_id_or_arn: Either a direct execution ID (e.g., "exec-123")
                            or a full ARN (e.g., "arn:aws:states:region:account:execution:name:id")

    Returns:
        The execution ID portion only.

    Examples:
        >>> parse_execution_id("exec-123")
        'exec-123'
        >>> parse_execution_id("arn:aws:states:us-east-1:123456789:execution:MyStateMachine:exec-123")
        'exec-123'
    """
    if ':' in execution_id_or_arn:
        return execution_id_or_arn.split(':')[-1]
    return execution_id_or_arn
