"""
Bedrock Advanced Prompt Optimization (APO) client.

Single self-contained module — builds records, submits jobs, polls until terminal,
downloads + parses results. Two evaluator modes in v1: LLM-as-Judge and Steering
criteria. Lambda mode is intentionally deferred (requires Lambda deployment infra).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)

API_VERSION = "bedrock-2026-05-14"
# APO runs in the same region as the rest of 360-eval's assets (DynamoDB, S3, KMS,
# ECS). Source it from AWS_REGION like every other client (web-ui/aws/*.py) so APO
# follows the deployment region instead of being pinned; us-east-1 is the shared
# fallback default.
APO_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_N_SAMPLES = 5
DEFAULT_TIMEOUT_SEC = 3600   # 60 min per job
DEFAULT_POLL_SEC = 60        # check status every minute


# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #

def _base_record(template_id: str, system_prompt: str,
                 sample_rows: list[dict]) -> dict:
    """Build the shared shape of an APO record.

    Args:
        template_id: A synthetic identifier for this template (e.g. "360eval-<eval_id>").
        system_prompt: The detected system-prompt portion to optimize.
        sample_rows: List of dicts shaped `{"variable_part": str, "golden": str}`.

    Returns a dict ready to receive mode-specific fields.
    """
    template = system_prompt.rstrip() + "\n\n{{input}}"
    samples = []
    for row in sample_rows:
        samples.append({
            "inputVariables": [{"input": row.get("variable_part", "")}],
            "referenceResponse": row.get("golden", "") or "",
        })
    return {
        "version": API_VERSION,
        "templateId": template_id,
        "promptTemplate": template,
        "evaluationSamples": samples,
    }


def build_apo_record_llmj(template_id: str, system_prompt: str,
                          sample_rows: list[dict], *,
                          rubric: str, judge_model_id: str,
                          metric_label: str = "360eval_llmj") -> dict:
    """Record for LLM-as-Judge evaluator.

    The judge model scores each candidate optimized prompt using the rubric.
    """
    if not rubric or not rubric.strip():
        raise ValueError("LLM-as-Judge requires a non-empty rubric")
    if not judge_model_id:
        raise ValueError("LLM-as-Judge requires a judge_model_id")
    record = _base_record(template_id, system_prompt, sample_rows)
    record["customLLMJConfig"] = {
        "customLLMJPrompt": rubric,
        "customLLMJModelId": judge_model_id,
    }
    record["customEvaluationMetricLabel"] = metric_label
    return record


def build_apo_record_steering(template_id: str, system_prompt: str,
                              sample_rows: list[dict], *,
                              criteria: list[str]) -> dict:
    """Record for Steering-criteria evaluator.

    Up to 5 natural-language steering rules. No metric label needed.
    """
    cleaned = [c.strip() for c in (criteria or []) if c and c.strip()]
    if not cleaned:
        raise ValueError("Steering mode requires at least one non-empty criterion")
    if len(cleaned) > 5:
        raise ValueError(f"Steering caps at 5 criteria; got {len(cleaned)}")
    record = _base_record(template_id, system_prompt, sample_rows)
    record["steeringCriteria"] = cleaned
    return record


# --------------------------------------------------------------------------- #
# Submit / poll / download
# --------------------------------------------------------------------------- #

def make_clients(region: str = APO_REGION) -> tuple[Any, Any]:
    """Return `(bedrock_client, s3_client)` for the APO region."""
    bedrock = boto3.client("bedrock", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    return bedrock, s3


def upload_record(record: dict, bucket: str, key: str,
                  s3_client=None) -> str:
    """Serialize record to JSONL (single line) and upload to S3."""
    if s3_client is None:
        s3_client = boto3.client("s3", region_name=APO_REGION)
    body = (json.dumps(record) + "\n").encode("utf-8")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body)
    return f"s3://{bucket}/{key}"


def submit_apo_job(record: dict, *, bucket: str, eval_id: str,
                   model_id: str, s3_input_key: str | None = None,
                   bedrock_client=None, s3_client=None) -> dict:
    """Upload the record + create the APO job.

    Returns `{"jobArn", "jobName", "inputS3Uri", "outputS3Uri", "modelId"}`.
    """
    if bedrock_client is None:
        bedrock_client = boto3.client("bedrock", region_name=APO_REGION)
    if s3_client is None:
        s3_client = boto3.client("s3", region_name=APO_REGION)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # APO jobName must match [a-zA-Z0-9][a-zA-Z0-9.+-]* — NO underscores. Replace the
    # path/ARN separators ('/' and ':') with hyphens and keep dots (which are allowed).
    # The previous version used underscores, so CreateAdvancedPromptOptimizationJob
    # rejected EVERY job with a ValidationException. (Hyphens/dots are also S3-key safe.)
    safe_model = model_id.replace("/", "-").replace(":", "-")[:50]
    job_name = f"360eval-{eval_id[:8]}-{safe_model}-{ts}"
    if s3_input_key is None:
        s3_input_key = f"apo-jobs/{eval_id}/inputs/{safe_model}.jsonl"
    s3_output_prefix = f"apo-jobs/{eval_id}/outputs/{safe_model}-{ts}/"
    input_uri = upload_record(record, bucket, s3_input_key, s3_client=s3_client)
    output_uri = f"s3://{bucket}/{s3_output_prefix}"
    resp = bedrock_client.create_advanced_prompt_optimization_job(
        jobName=job_name,
        modelConfigurations=[{"modelId": model_id}],
        inputConfig={"s3Uri": input_uri},
        outputConfig={"s3Uri": output_uri},
    )
    return {
        "jobArn": resp["jobArn"],
        "jobName": job_name,
        "inputS3Uri": input_uri,
        "outputS3Uri": output_uri,
        "modelId": model_id,
    }


def poll_apo_job(job_arn: str, *, timeout_sec: int = DEFAULT_TIMEOUT_SEC,
                 poll_sec: int = DEFAULT_POLL_SEC,
                 bedrock_client=None) -> dict:
    """Poll the APO job until terminal status. Raises TimeoutError after timeout_sec."""
    if bedrock_client is None:
        bedrock_client = boto3.client("bedrock", region_name=APO_REGION)
    start = time.time()
    info: dict = {}
    while time.time() - start < timeout_sec:
        info = bedrock_client.get_advanced_prompt_optimization_job(jobIdentifier=job_arn)
        status = info.get("jobStatus", "")
        if status in ("Completed", "Failed", "Stopped"):
            return info
        time.sleep(poll_sec)
    raise TimeoutError(f"APO job did not finish within {timeout_sec}s: {job_arn}")


def download_apo_result(job_info: dict, *, bucket: str, local_path: Path,
                        s3_client=None) -> Path:
    """Fetch the result JSONL emitted by APO."""
    if s3_client is None:
        s3_client = boto3.client("s3", region_name=APO_REGION)
    out_uri = job_info["outputConfig"]["s3Uri"].rstrip("/")
    job_id = job_info["jobArn"].rsplit("/", 1)[-1]
    out_prefix = out_uri.split(f"s3://{bucket}/", 1)[1]
    key = f"{out_prefix}/{job_id}/advanced_prompt_optimization_results.jsonl"
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(local_path))
    return local_path


def parse_apo_result(path: Path) -> dict:
    """Parse the result JSONL and return:

        {
          "optimized_template": "...",
          "original_score": float | None,
          "optimized_score": float | None,
          "status": "Success" | "Failed" | ...,
          "model_id": str,
          "metric_label": str | None,
        }

    Returns the first result row's content (APO emits one result per model per
    invocation; we submit one model per job so there's one row).
    """
    out = {
        "optimized_template": None,
        "original_score": None,
        "optimized_score": None,
        "status": None,
        "model_id": None,
        "metric_label": None,
    }
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out["metric_label"] = d.get("customEvaluationMetricLabel")
            results = d.get("promptOptimizationResults", [])
            if not results:
                continue
            r = results[0]
            out["status"] = r.get("status")
            out["model_id"] = r.get("modelId")
            out["optimized_template"] = r.get("optimizedPromptTemplate")
            orig = (r.get("originalPromptMetrics") or {}).get("averageScore")
            opt = (r.get("optimizedPromptMetrics") or {}).get("averageScore")
            out["original_score"] = orig
            out["optimized_score"] = opt
            break
    return out


# --------------------------------------------------------------------------- #
# One-shot orchestration
# --------------------------------------------------------------------------- #

def run_one_apo_job(record: dict, *, bucket: str, eval_id: str,
                    model_id: str, local_result_dir: Path,
                    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
                    poll_sec: int = DEFAULT_POLL_SEC,
                    bedrock_client=None, s3_client=None) -> dict:
    """End-to-end for one (record, model) pair.

    Returns a dict with the parse_apo_result fields plus `jobArn`,
    `submittedAt`, `completedAt`, `error` (if any).
    """
    bedrock_client = bedrock_client or boto3.client("bedrock", region_name=APO_REGION)
    s3_client = s3_client or boto3.client("s3", region_name=APO_REGION)
    submitted_at = datetime.utcnow().isoformat() + "Z"
    out = {
        "model_id": model_id,
        "submittedAt": submitted_at,
        "completedAt": None,
        "jobArn": None,
        "status": None,
        "optimized_template": None,
        "original_score": None,
        "optimized_score": None,
        "error": None,
    }
    try:
        submit = submit_apo_job(record, bucket=bucket, eval_id=eval_id,
                                model_id=model_id,
                                bedrock_client=bedrock_client,
                                s3_client=s3_client)
        out["jobArn"] = submit["jobArn"]
        logger.info(f"[APO] submitted {submit['jobName']} for {model_id}")
        info = poll_apo_job(submit["jobArn"], timeout_sec=timeout_sec,
                            poll_sec=poll_sec, bedrock_client=bedrock_client)
        out["completedAt"] = datetime.utcnow().isoformat() + "Z"
        out["status"] = info.get("jobStatus")
        if info.get("jobStatus") != "Completed":
            # Surface the service's failureMessage (from GetAdvancedPromptOptimizationJob)
            # so the real reason flows through to apo_status, not just "Failed".
            fail_msg = info.get("failureMessage")
            out["error"] = (
                f"APO job {info.get('jobStatus')}: {fail_msg}" if fail_msg
                else f"APO job terminal status: {info.get('jobStatus')}"
            )
            return out
        local_path = Path(local_result_dir) / f"apo_result_{model_id.replace('/', '_').replace(':', '_')}.jsonl"
        download_apo_result(info, bucket=bucket, local_path=local_path, s3_client=s3_client)
        parsed = parse_apo_result(local_path)
        # Merge in the parsed fields, but keep our submitted/job metadata.
        for k in ("optimized_template", "original_score", "optimized_score"):
            out[k] = parsed.get(k)
        # status from result row takes precedence (sometimes Failed at the row level even if job is Completed)
        if parsed.get("status"):
            out["status"] = parsed["status"]
    except (ClientError, TimeoutError) as e:
        out["completedAt"] = datetime.utcnow().isoformat() + "Z"
        out["error"] = f"{type(e).__name__}: {e}"
        logger.exception(f"[APO] failed for {model_id}: {e}")
    except Exception as e:
        out["completedAt"] = datetime.utcnow().isoformat() + "Z"
        out["error"] = f"{type(e).__name__}: {e}"
        logger.exception(f"[APO] unexpected error for {model_id}: {e}")
    return out
