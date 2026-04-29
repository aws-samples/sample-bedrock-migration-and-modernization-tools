# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Amazon Bedrock judge client implementation.

This module provides integration with Amazon Bedrock for LLM-based
rubric evaluation using boto3 with support for modern Bedrock APIs
including Converse API and Messages format.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Optional, List, Union

try:
    import boto3
    from botocore.exceptions import ClientError, ReadTimeoutError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    # Define fallback types to prevent NameError when boto3 is missing
    ClientError = Exception  # type: ignore
    ReadTimeoutError = Exception  # type: ignore

from agent_eval.judges.judge_client import JudgeClient, JudgeResponse
from agent_eval.judges.exceptions import (
    ValidationResult,
    ValidationError,
    APIError,
    TimeoutError as JudgeTimeoutError
)

logger = logging.getLogger(__name__)


class BedrockJudgeClient(JudgeClient):
    """
    Amazon Bedrock implementation of JudgeClient.
    
    Supports modern Amazon Bedrock APIs including:
    - Converse API (recommended for Claude 3+, Mistral, etc.)
    - InvokeModel API with Messages format (Claude 3+)
    - Legacy InvokeModel API (Claude 2.x, Titan)
    - Streaming responses
    
    Automatically detects model family and uses appropriate API format.
    """
    
    # Model family detection patterns
    CLAUDE_3_PATTERN = re.compile(r'(anthropic\.claude-[34]|us\.anthropic\.claude)')
    CLAUDE_LEGACY_PATTERN = re.compile(r'anthropic\.claude-[v]?2')
    TITAN_PATTERN = re.compile(r'amazon\.titan')
    MISTRAL_PATTERN = re.compile(r'mistral')
    
    def __init__(
        self,
        judge_id: str,
        model_id: str,
        params: Dict[str, Any],
        timeout_seconds: int = 30,
        region_name: Optional[str] = None,
        streaming: bool = False,
        use_converse_api: bool = True
    ):
        """
        Initialize Amazon Bedrock judge client.
        
        Args:
            judge_id: Unique identifier for this judge
            model_id: Amazon Bedrock model ID (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
            params: Model parameters (temperature, max_tokens, etc.)
            timeout_seconds: Maximum execution time per request
            region_name: AWS region (uses default if None)
            streaming: Enable streaming responses
            use_converse_api: Use Converse API when available (recommended)
            
        Raises:
            ImportError: If boto3 is not installed
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for BedrockJudgeClient. "
                "Install with: pip install boto3"
            )
        
        super().__init__(judge_id, model_id, params, timeout_seconds)
        
        self.region_name = region_name
        self.streaming = streaming
        self.use_converse_api = use_converse_api
        self.client = None
        
        # Detect model family for API format selection
        self.model_family = self._detect_model_family()
        
        self._initialize_client()
    
    def _detect_model_family(self) -> str:
        """
        Detect model family from model_id.
        
        Returns:
            Model family: "claude3", "claude_legacy", "titan", "mistral", "generic"
        """
        model_id_lower = self.model_id.lower()
        
        if self.CLAUDE_3_PATTERN.search(model_id_lower):
            return "claude3"
        elif self.CLAUDE_LEGACY_PATTERN.search(model_id_lower):
            return "claude_legacy"
        elif self.TITAN_PATTERN.search(model_id_lower):
            return "titan"
        elif self.MISTRAL_PATTERN.search(model_id_lower):
            return "mistral"
        else:
            return "generic"
    
    def _initialize_client(self) -> None:
        """Initialize boto3 Amazon Bedrock client with authentication."""
        try:
            session = boto3.Session()
            self.client = session.client(
                'bedrock-runtime',
                region_name=self.region_name
            )
            logger.info(
                f"Initialized Amazon Bedrock client for judge {self.judge_id} "
                f"(model: {self.model_id}, family: {self.model_family}, "
                f"region: {self.region_name or 'default'})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Amazon Bedrock client: {e}")
            raise APIError(
                message=f"Failed to initialize Amazon Bedrock client: {str(e)}",
                error_code="BEDROCK_INIT_FAILED",
                retryable=False
            )
    
    async def execute_judge(
        self,
        prompt: str,
        rubric_id: str,
        scoring_scale: Dict[str, Any]
    ) -> JudgeResponse:
        """
        Execute judge evaluation via Amazon Bedrock API.
        
        Retries up to 3 times on invalid JSON responses.
        
        Args:
            prompt: Evaluation prompt
            rubric_id: Rubric identifier
            scoring_scale: Scoring scale definition
            
        Returns:
            JudgeResponse with score and reasoning
            
        Raises:
            JudgeTimeoutError: If execution exceeds timeout
            ValidationError: If response invalid after retries
            APIError: If Amazon Bedrock API call fails
        """
        max_json_retries = 3
        last_validation_result = None
        last_raw_response = None
        
        for attempt in range(max_json_retries):
            try:
                start_time = time.time()
                
                # Execute with timeout
                response = await asyncio.wait_for(
                    self._invoke_model(prompt),
                    timeout=self.timeout_seconds
                )
                
                latency_ms = (time.time() - start_time) * 1000
                last_raw_response = response
                
                # Parse and validate response
                parsed = self._parse_response(response)
                
                validation_result = await self.validate_response(parsed, scoring_scale)
                
                if validation_result.is_valid:
                    return JudgeResponse(
                        score=parsed.get('score'),
                        reasoning=parsed.get('reasoning'),
                        raw_response=response,  # Keep original response (dict, str, or list)
                        latency_ms=latency_ms,
                        metadata={
                            'model_id': self.model_id,
                            'model_family': self.model_family,
                            'judge_id': self.judge_id,
                            'rubric_id': rubric_id,
                            'attempt': attempt + 1
                        }
                    )
                else:
                    last_validation_result = validation_result
                    logger.warning(
                        f"Invalid response on attempt {attempt + 1}/{max_json_retries}: "
                        f"{validation_result.error_code} - {validation_result.message}"
                    )
                    
            except asyncio.TimeoutError:
                raise JudgeTimeoutError(
                    message=f"Bedrock call exceeded {self.timeout_seconds}s timeout",
                    timeout_seconds=self.timeout_seconds,
                    context={
                        'judge_id': self.judge_id,
                        'model_id': self.model_id,
                        'rubric_id': rubric_id,
                        'attempt': attempt + 1
                    }
                )
            except json.JSONDecodeError as e:
                last_validation_result = ValidationResult.failure(
                    error_code="INVALID_JSON",
                    message=f"Response is not valid JSON: {str(e)}",
                    actual=str(last_raw_response)[:200] if last_raw_response else "N/A"
                )
                logger.warning(
                    f"JSON decode error on attempt {attempt + 1}/{max_json_retries}: {e}"
                )
                # Continue to retry
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'UNKNOWN')
                status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                # Check if retryable
                retryable = (
                    status_code >= 500 if status_code else True
                ) or error_code in ['ThrottlingException', 'ServiceUnavailableException']
                
                if not retryable or attempt == max_json_retries - 1:
                    raise APIError(
                        message=f"Bedrock API error: {error_message}",
                        error_code=f"BEDROCK_{error_code}",
                        status_code=status_code,
                        retryable=retryable,
                        context={
                            'judge_id': self.judge_id,
                            'model_id': self.model_id,
                            'rubric_id': rubric_id,
                            'attempt': attempt + 1
                        }
                    )
                else:
                    logger.warning(
                        f"Retryable API error on attempt {attempt + 1}/{max_json_retries}: "
                        f"{error_code} - {error_message}"
                    )
                    
            except ReadTimeoutError as e:
                raise JudgeTimeoutError(
                    message=f"Bedrock read timeout: {str(e)}",
                    timeout_seconds=self.timeout_seconds,
                    context={
                        'judge_id': self.judge_id,
                        'model_id': self.model_id,
                        'rubric_id': rubric_id,
                        'attempt': attempt + 1
                    }
                )
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{max_json_retries}: {e}")
                if attempt == max_json_retries - 1:
                    raise APIError(
                        message=f"Unexpected Amazon Bedrock error: {str(e)}",
                        error_code="BEDROCK_UNKNOWN_ERROR",
                        context={
                            'judge_id': self.judge_id,
                            'model_id': self.model_id,
                            'rubric_id': rubric_id,
                            'attempt': attempt + 1
                        }
                    )
        
        # All retries exhausted - raise ValidationError with details
        if last_validation_result:
            raise ValidationError(
                message=f"Failed to get valid response after {max_json_retries} attempts",
                error_code=last_validation_result.error_code,
                field=last_validation_result.field,
                expected=last_validation_result.expected,
                actual=last_validation_result.actual,
                context={
                    'judge_id': self.judge_id,
                    'model_id': self.model_id,
                    'rubric_id': rubric_id,
                    'attempts': max_json_retries
                }
            )
        else:
            raise ValidationError(
                message=f"Failed to get valid response after {max_json_retries} attempts",
                error_code="VALIDATION_FAILED",
                context={
                    'judge_id': self.judge_id,
                    'model_id': self.model_id,
                    'rubric_id': rubric_id,
                    'attempts': max_json_retries
                }
            )
    
    async def _invoke_model(self, prompt: str) -> Union[Dict[str, Any], str]:
        """
        Invoke Amazon Bedrock model with prompt.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Raw response from Bedrock API (dict or string)
            
        Raises:
            JudgeTimeoutError: On timeout
            APIError: On API failure
        """
        try:
            # Use Converse API for supported models (Claude 3+, Mistral)
            if self.use_converse_api and self.model_family in ['claude3', 'mistral']:
                return await self._invoke_converse(prompt)
            # Use InvokeModel with appropriate format
            elif self.streaming:
                return await self._invoke_streaming(prompt)
            else:
                return await self._invoke_non_streaming(prompt)
                
        except ClientError as e:
            logger.error(f"Bedrock ClientError: {e}")
            raise
        except ReadTimeoutError as e:
            logger.error(f"Bedrock ReadTimeoutError: {e}")
            raise JudgeTimeoutError(
                message=f"Bedrock read timeout: {str(e)}",
                timeout_seconds=self.timeout_seconds
            )
        except Exception as e:
            logger.error(f"Unexpected error in _invoke_model: {e}")
            raise
    
    async def _invoke_converse(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke model using Amazon Bedrock Converse API (recommended for Claude 3+).
        
        This is the modern, unified API that works across model families.
        Uses the correct boto3 client.converse() signature.
        """
        loop = asyncio.get_event_loop()
        
        # Build Converse API request using correct boto3 signature
        # Note: modelId, messages, and inferenceConfig are separate parameters
        inference_config = {
            'temperature': self.params.get('temperature', 0.0),
            'maxTokens': self.params.get('max_tokens', 2048),
            'topP': self.params.get('top_p', 1.0)
        }
        
        # Add stop sequences if provided
        stop_sequences = self.params.get('stop_sequences')
        if stop_sequences:
            inference_config['stopSequences'] = stop_sequences
        
        # Messages must be a list of message dicts with role and content
        messages = [
            {
                'role': 'user',
                'content': [{'text': prompt}]  # content is a list of content blocks
            }
        ]
        
        response = await loop.run_in_executor(
            None,
            lambda: self.client.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig=inference_config
            )
        )
        
        return response
    
    async def _invoke_non_streaming(self, prompt: str) -> Dict[str, Any]:
        """Invoke model without streaming using InvokeModel API."""
        loop = asyncio.get_event_loop()
        
        # Build request body based on model family
        body = self._build_request_body(prompt)
        
        response = await loop.run_in_executor(
            None,
            lambda: self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
        )
        
        response_body = json.loads(response['body'].read())
        return response_body
    
    async def _invoke_streaming(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke model with streaming using InvokeModelWithResponseStream API.
        
        Properly handles Amazon Bedrock event stream format with chunk decoding.
        """
        loop = asyncio.get_event_loop()
        
        # Build request body based on model family
        body = self._build_request_body(prompt)
        
        response = await loop.run_in_executor(
            None,
            lambda: self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
        )
        
        # Collect streaming chunks with proper error handling
        chunks = []
        stream = response['body']
        
        try:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_bytes = chunk.get('bytes')
                    if chunk_bytes:
                        try:
                            # Decode bytes and parse JSON
                            chunk_str = chunk_bytes.decode('utf-8')
                            chunk_data = json.loads(chunk_str)
                            chunks.append(chunk_data)
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            # Log but continue - some chunks may be partial/non-JSON
                            logger.warning(f"Failed to decode chunk: {e}")
                            continue
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            raise APIError(
                message=f"Failed to process streaming response: {str(e)}",
                error_code="BEDROCK_STREAM_ERROR"
            )
        
        if not chunks:
            raise APIError(
                message="No valid chunks received from streaming response",
                error_code="BEDROCK_STREAM_EMPTY"
            )
        
        # Combine chunks into final response
        return self._combine_chunks(chunks)
    
    def _build_request_body(self, prompt: str) -> Dict[str, Any]:
        """
        Build request body for Amazon Bedrock InvokeModel API.
        
        Formats request based on model family using appropriate API format.
        """
        if self.model_family == 'claude3':
            # Claude 3+ uses Messages API format
            # content must be a list of content blocks
            return {
                'anthropic_version': 'bedrock-2023-05-31',
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': prompt
                            }
                        ]
                    }
                ],
                'max_tokens': self.params.get('max_tokens', 2048),
                'temperature': self.params.get('temperature', 0.0),
                'top_p': self.params.get('top_p', 1.0),
                'stop_sequences': self.params.get('stop_sequences', [])
            }
        elif self.model_family == 'claude_legacy':
            # Claude 2.x uses legacy prompt format
            return {
                'prompt': f"\n\nHuman: {prompt}\n\nAssistant:",
                'max_tokens_to_sample': self.params.get('max_tokens', 2048),
                'temperature': self.params.get('temperature', 0.0),
                'top_p': self.params.get('top_p', 1.0),
                'stop_sequences': self.params.get('stop_sequences', ['\n\nHuman:'])
            }
        elif self.model_family == 'titan':
            # Amazon Titan format
            return {
                'inputText': prompt,
                'textGenerationConfig': {
                    'maxTokenCount': self.params.get('max_tokens', 2048),
                    'temperature': self.params.get('temperature', 0.0),
                    'topP': self.params.get('top_p', 1.0),
                    'stopSequences': self.params.get('stop_sequences', [])
                }
            }
        elif self.model_family == 'mistral':
            # Mistral uses similar format to Claude 3
            return {
                'prompt': prompt,
                'max_tokens': self.params.get('max_tokens', 2048),
                'temperature': self.params.get('temperature', 0.0),
                'top_p': self.params.get('top_p', 1.0),
                'stop': self.params.get('stop_sequences', [])
            }
        else:
            # Generic format
            return {
                'prompt': prompt,
                'max_tokens': self.params.get('max_tokens', 2048),
                'temperature': self.params.get('temperature', 0.0),
                **self.params
            }
    
    def _parse_response(self, response: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Parse Amazon Bedrock response to extract completion text and convert to JSON.
        
        Handles different response formats from various model families and APIs.
        Uses robust JSON extraction to handle prose around JSON objects.
        
        Args:
            response: Raw response from Bedrock (dict or string)
            
        Returns:
            Parsed JSON dict with score and reasoning
            
        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Handle Converse API response
        if isinstance(response, dict) and 'output' in response:
            message = response['output'].get('message', {})
            content = message.get('content', [])
            if content and isinstance(content, list):
                # Concatenate ALL text blocks (not just first)
                text = ''.join(
                    part.get('text', '') 
                    for part in content 
                    if isinstance(part, dict) and 'text' in part
                )
            else:
                text = str(response)
        
        # Handle InvokeModel response formats
        elif isinstance(response, dict):
            # Claude 3+ Messages API format (InvokeModel body)
            # Response body structure: {"content": [{"type": "text", "text": "..."}], ...}
            if 'content' in response:
                content = response['content']
                if isinstance(content, list) and content:
                    # Concatenate all text blocks
                    text = ''.join(
                        part.get('text', '') 
                        for part in content 
                        if isinstance(part, dict) and 'text' in part
                    )
                elif isinstance(content, str):
                    text = content
                else:
                    text = str(content)
            
            # Claude legacy format
            elif 'completion' in response:
                text = response['completion']
            
            # Titan format
            elif 'results' in response:
                results = response['results']
                if isinstance(results, list) and results:
                    text = results[0].get('outputText', '')
                else:
                    text = str(results)
            
            # Mistral format
            elif 'outputs' in response:
                outputs = response['outputs']
                if isinstance(outputs, list) and outputs:
                    text = outputs[0].get('text', '')
                else:
                    text = str(outputs)
            
            # Generic format
            elif 'generated_text' in response:
                text = response['generated_text']
            
            # Fallback: try to find any text-like field
            else:
                # Look for common text field names
                for field in ['text', 'output', 'response', 'message']:
                    if field in response:
                        text = response[field]
                        break
                else:
                    # Last resort: stringify entire response
                    text = str(response)
                    logger.warning(
                        f"Unknown response format, using str(response). "
                        f"Keys: {list(response.keys())}"
                    )
        else:
            text = str(response)
        
        # Extract JSON from text using robust extraction
        return self._extract_json_from_text(text)
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Robustly extract JSON object from text that may contain prose.
        
        Handles:
        - JSON wrapped in markdown code blocks
        - JSON with surrounding prose
        - Nested braces in reasoning text
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON dict
            
        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Try 1: Parse entire text as JSON (fast path)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try 2: Extract from markdown code block
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try 3: Find first complete JSON object with balanced braces
        # This handles nested braces correctly and guards against negative counts
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                if brace_count > 0:  # Guard against negative count
                    brace_count -= 1
                    if brace_count == 0 and start_idx >= 0:
                        # Found complete JSON object
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            # Continue searching for next object
                            start_idx = -1
                # If brace_count <= 0 and we see }, ignore it (stray closing brace)
        
        # Try 4: Simple first { to last } (fallback, less robust)
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        
        # No valid JSON found
        raise json.JSONDecodeError(
            f"No valid JSON object found in response",
            text,
            0
        )
    
    def _combine_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine streaming chunks into final response.
        
        Handles different streaming formats from various model families.
        Note: Bedrock streaming format may differ from direct Anthropic API.
        """
        if not chunks:
            return {}
        
        # Check first chunk to determine format
        first_chunk = chunks[0]
        
        # Claude 3+ InvokeModel streaming (delta format)
        if 'delta' in first_chunk:
            combined_text = ''
            for chunk in chunks:
                delta = chunk.get('delta', {})
                if 'text' in delta:
                    combined_text += delta['text']
                elif 'completion' in delta:
                    combined_text += delta['completion']
            return {'content': [{'text': combined_text}]}
        
        # Claude legacy streaming (completion field)
        if 'completion' in first_chunk:
            combined_text = ''.join(
                chunk.get('completion', '') for chunk in chunks
            )
            return {'completion': combined_text}
        
        # Titan streaming (outputText field)
        if 'outputText' in first_chunk:
            combined_text = ''.join(
                chunk.get('outputText', '') for chunk in chunks
            )
            return {'results': [{'outputText': combined_text}]}
        
        # Generic combination - concatenate any text-like fields
        # or return last complete chunk
        text_fields = ['text', 'generated_text', 'output']
        for field in text_fields:
            if field in first_chunk:
                combined_text = ''.join(
                    chunk.get(field, '') for chunk in chunks
                )
                return {field: combined_text}
        
        # Fallback: return last chunk (may be complete response)
        return chunks[-1] if chunks else {}
    
    async def validate_response(
        self,
        response: Dict[str, Any],
        scoring_scale: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate response contains required fields and valid score.
        
        Returns structured validation result with detailed error information.
        
        Args:
            response: Parsed response dictionary
            scoring_scale: Expected scoring scale
            
        Returns:
            ValidationResult with validation status and error details
        """
        # Check required fields
        if 'score' not in response:
            return ValidationResult.failure(
                error_code="MISSING_FIELD",
                message="Response missing required 'score' field",
                field="score",
                expected="score field present",
                actual="score field missing"
            )
        
        score = response['score']
        scale_type = scoring_scale.get('type', 'numeric')
        
        # Validate score against scale
        if scale_type == 'numeric':
            min_val = scoring_scale.get('min', 0)
            max_val = scoring_scale.get('max', 5)
            
            try:
                score_num = float(score)
                if not (min_val <= score_num <= max_val):
                    return ValidationResult.failure(
                        error_code="INVALID_SCORE",
                        message=f"Score {score_num} outside valid range",
                        field="score",
                        expected=f"number between {min_val} and {max_val}",
                        actual=str(score_num)
                    )
            except (ValueError, TypeError):
                return ValidationResult.failure(
                    error_code="INVALID_SCORE",
                    message=f"Score '{score}' is not a valid number",
                    field="score",
                    expected="numeric value",
                    actual=str(score)
                )
                
        elif scale_type == 'categorical':
            valid_values = scoring_scale.get('values', [])
            if score not in valid_values:
                return ValidationResult.failure(
                    error_code="INVALID_SCORE",
                    message=f"Score not in allowed categorical values",
                    field="score",
                    expected=f"one of {valid_values}",
                    actual=str(score)
                )
        
        # Optional: Check reasoning field (warning only)
        if 'reasoning' not in response:
            logger.warning("Response missing optional 'reasoning' field")
        
        return ValidationResult.success()
