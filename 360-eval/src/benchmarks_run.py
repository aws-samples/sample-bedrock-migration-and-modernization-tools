import os
import time
import concurrent.futures
import json
import logging
import uuid
import pandas as pd
import argparse
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from utils import (get_timestamp,
                   setup_logging,
                   calculate_average_scores,
                   run_inference,
                   extract_json_response,
                   extract_json_from_text,
                   llm_judge_template,
                   extract_system_prompt_hybrid,
                   judge_cache_get,
                   judge_cache_put,
                   flush_judge_cache,
                   validate_custom_metrics,
                   STANDARD_METRICS,
                   METRIC_DEFINITIONS)
from apo_client import (build_apo_record_llmj, build_apo_record_steering,
                        run_one_apo_job, APO_REGION)
from config_validator import validate_jsonl_file, validate_prompt_jsonl_file
from rate_limiter import TokenBucketRateLimiter

env = load_dotenv()


# ----------------------------------------
# Single LLM‑as‑judge call
# ----------------------------------------
def evaluate_with_llm_judge(judge_model_id,
                            judge_region,
                            prompt,
                            model_response,
                            golden_answer,
                            task_types,
                            task_criteria,
                            custom_metrics=None,
                            yard_stick=3,
                            structure_validation=None,
                            success_criteria=None):
    """
     Runs the target model on `prompt`, then has three jury models
     evaluate its response against `golden_response` using the
     specified metrics. Returns per-juror scores, aggregated scores,
     and a final pass/fail decision by majority vote.
     """
    standard_metrics = ["Correctness", "Completeness", "Relevance", "Format", "Coherence", "Following-instructions"]
    # custom_metrics can be a list of strings or dicts; extract names for the metric list
    custom_metric_names = []
    custom_metric_defs = []
    for cm in (custom_metrics or []):
        if isinstance(cm, dict):
            custom_metric_names.append(cm.get("metric_name", "custom"))
            custom_metric_defs.append(cm)
        else:
            custom_metric_names.append(cm)
    all_metrics = standard_metrics + custom_metric_names
    eval_template = llm_judge_template(all_metrics,
                                       task_types,
                                       task_criteria,
                                       prompt,
                                       model_response,
                                       golden_answer,
                                       structure_validation=structure_validation,
                                       custom_metric_definitions=custom_metric_defs or None,
                                       success_criteria=success_criteria)

    cfg = {"maxTokens": 1500, "aws_region_name": judge_region}
    try:
        resp = run_inference(model_name=judge_model_id,
                             prompt_text=eval_template,
                             provider_params=cfg,
                             stream=False,
                             judge_eval=True)
        text = resp['text']
    except Exception as e:
        logging.error(f"Judge inference failed ({judge_model_id}): {e}", exc_info=True)
        return {
            "judgment": "Error inference response",
            "explanation": str(e),
            "full_response": "",
            "scores": {"score": "NULL"},
            "judge_input_tokens": 0,
            "judge_output_tokens": 0,
            "error_type": "inference_failure",
            "original_error": str(e)
        }

    try:
        eval_results = extract_json_response(all_metrics, text, judge_model_id, cfg)
        if not eval_results:
            logging.warning(f"MalformedJudgeResponseError: Judge {judge_model_id} response did not contain valid JSON scores")
            return {
                "judgment": "Error Parsing response",
                "explanation": "MalformedJudgeResponseError: Judge response did not contain valid JSON scores",
                "full_response": text,
                "scores": {"score": "NULL"},
                "judge_input_tokens": resp.get('inputTokens', 0),
                "judge_output_tokens": resp.get('outputTokens', 0),
                "error_type": "MalformedJudgeResponseError"
            }

        # Normalize scores: flatten {"Metric": {"score": int, "rationale": "..."}} to {"Metric": int}
        if "scores" in eval_results:
            rationales = {}
            normalized_scores = {}
            for metric, val in eval_results["scores"].items():
                if isinstance(val, dict):
                    score = val.get("score")
                    # Only accept a numeric score. A missing/non-numeric score must be
                    # excluded (not coerced to 0) — 0 is not a valid 1-5 rating and would
                    # silently drag the cross-judge average down (fabricated low score).
                    if isinstance(score, (int, float)) and not isinstance(score, bool):
                        normalized_scores[metric] = score
                        rationales[metric] = val.get("rationale", "")
                    else:
                        logging.warning(
                            f"Judge {judge_model_id} returned no numeric score for "
                            f"metric '{metric}'; excluding it from aggregation")
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    normalized_scores[metric] = val
                else:
                    logging.warning(
                        f"Judge {judge_model_id} returned non-numeric score for "
                        f"metric '{metric}' ({val!r}); excluding it from aggregation")
            eval_results["scores"] = normalized_scores
            if rationales:
                eval_results["rationales"] = rationales

        judgment = "PASS"
        if isinstance(yard_stick, dict):
            explanation = [key for key, val in eval_results["scores"].items()
                          if val < yard_stick.get(key, 3)]
        else:
            explanation = [key for key, val in eval_results["scores"].items() if val < yard_stick]
        if len(explanation) > 0:
            judgment = "FAIL"

        eval_results["judgment"] = judgment
        payload = {
            "judgment": eval_results["judgment"],
            "scores": eval_results["scores"],
            "explanation": ";".join(explanation),
            "full_response": text,
            "judge_input_tokens": resp['inputTokens'],
            "judge_output_tokens": resp['outputTokens']
        }
    except Exception as e:
        logging.error(f"Judge evaluation parsing failed ({judge_model_id}): {e}", exc_info=True)
        return {
            "judgment": "Error Parsing response",
            "explanation": str(e),
            "full_response": text,
            "scores": {"score": "NULL"},
            "judge_input_tokens": resp.get('inputTokens', 0),
            "judge_output_tokens": resp.get('outputTokens', 0),
            "error_type": "parsing_failure",
            "original_error": str(e)
        }

    return payload


# ----------------------------------------
# Multi‑judge + majority‑vote (parallelized)
# ----------------------------------------
def _extract_model_family(model_id):
    """
    Extract model family prefix from a model ID for self-eval comparison.
    E.g. 'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0' → 'anthropic.claude'
         'bedrock/us.meta.llama3-2-90b-instruct-v1:0' → 'meta.llama'
         'openai/gpt-4o' → 'openai/gpt-4'
    """
    if not model_id:
        return ''
    name = model_id.lower()
    # Strip provider routing prefixes
    for prefix in ('bedrock/', 'openai/', 'azure/', 'gemini/'):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Strip regional prefixes like 'us.' or 'eu.'
    if len(name) > 3 and name[2] == '.' and name[:2].isalpha():
        name = name[3:]
    # Take provider.model-family (first two dot-separated or hyphen-separated parts)
    # e.g. 'anthropic.claude-3-5-sonnet-...' → 'anthropic.claude'
    # e.g. 'meta.llama3-2-90b-...' → 'meta.llama'
    parts = name.split('.')
    if len(parts) >= 2:
        provider = parts[0]
        model_part = parts[1].split('-')[0]  # 'claude' from 'claude-3-5-sonnet-...'
        # For llama, strip trailing digits: 'llama3' → 'llama'
        model_part = model_part.rstrip('0123456789')
        return f"{provider}.{model_part}"
    return name.split('-')[0]


def evaluate_with_judges(judges,
                         prompt,
                         model_response,
                         golden_answer,
                         task_types,
                         task_criteria,
                         user_defined_metrics,
                         yard_stick=3,
                         structure_validation=None,
                         target_model_id=None,
                         success_criteria=None):
    """
    Evaluate model response using multiple LLM judges in parallel.

    Each judge evaluation is independent and thread-safe, allowing parallel execution
    for ~3x speedup with 3 judges. Results are aggregated using majority voting.
    """
    results = []

    def evaluate_single_judge(j):
        """Evaluate with a single judge - thread-safe inner function."""
        try:
            logging.debug(f"Evaluating with judge model {j['model_id']}")
            # Model ID preparation for litellm is now handled centrally in run_inference()
            r = evaluate_with_llm_judge(
                judge_model_id=j["model_id"],
                judge_region=j["region"],
                prompt=prompt,
                model_response=model_response,
                golden_answer=golden_answer,
                task_types=task_types,
                task_criteria=task_criteria,
                custom_metrics=user_defined_metrics,
                yard_stick=yard_stick,
                success_criteria=success_criteria,
                structure_validation=structure_validation,
            )

            # Check for various error indicators
            if "error" in r or r.get("judgment") == "Error inference response" or r.get(
                    "judgment") == "Error Parsing response":
                logging.warning(
                    f"Judge {j['model_id']} returned an error response: {r.get('explanation', 'Unknown error')}")
                return {"model": j["model_id"], **r}

            # Check if scores are valid
            if not r.get("scores") or r.get("scores", {}).get("score") == "NULL":
                logging.warning(f"Judge {j['model_id']} returned invalid scores: {r.get('scores', 'None')}")
                return {"model": j["model_id"], **r}

            # Defensive cost calculation with defaults
            judge_input_tokens = r.get("judge_input_tokens", 0)
            judge_output_tokens = r.get("judge_output_tokens", 0)
            r['judge_input_token_cost'] = judge_input_tokens * (j["input_token_cost"] / 1_000_000)
            r['judge_output_token_cost'] = judge_output_tokens * (j["output_token_cost"] / 1_000_000)
            logging.debug(
                f"Successfully evaluated with judge {j['model_id']}, judgment: {r.get('judgment', 'Unknown')}")
            return {"model": j["model_id"], **r}

        except Exception as e:
            logging.error(f"Exception evaluating with judge {j['model_id']}: {str(e)}", exc_info=True)
            return {
                "model": j["model_id"],
                "judgment": "Judge Exception",
                "explanation": str(e),
                "scores": {"score": "NULL"},
                "judge_input_tokens": 0,
                "judge_output_tokens": 0,
                "error_type": "judge_exception"
            }

    # Handle edge case of empty judges list
    if not judges:
        return {"majority_judgment": "FAIL", "majority_explanations": [], "judge_details": [],
                "majority_score": {}, "eval_cost": 0}

    # Execute judge evaluations in parallel with timeout
    # Use max_workers equal to number of judges since each judge is independent
    with ThreadPoolExecutor(max_workers=len(judges)) as executor:
        future_to_judge = {executor.submit(evaluate_single_judge, j): j for j in judges}

        # No timeout on as_completed itself: an iterator-level TimeoutError would escape
        # this loop and discard judges that already returned valid scores. The per-future
        # result(timeout=...) below still bounds each individual judge.
        for future in concurrent.futures.as_completed(future_to_judge):
            judge = future_to_judge[future]
            try:
                result = future.result(timeout=120)  # 120-second timeout per judge
                results.append(result)
            except concurrent.futures.TimeoutError:
                logging.error(f"Judge {judge['model_id']} timed out after 120 seconds")
                results.append({
                    "model": judge["model_id"],
                    "judgment": "Judge Exception",
                    "explanation": "Timeout after 120 seconds",
                    "scores": {"score": "NULL"},
                    "judge_input_tokens": 0,
                    "judge_output_tokens": 0,
                    "error_type": "timeout"
                })
            except Exception as e:
                logging.error(f"Exception getting result for judge {judge['model_id']}: {str(e)}", exc_info=True)
                results.append({
                    "model": judge["model_id"],
                    "judgment": "Judge Exception",
                    "explanation": str(e),
                    "scores": {"score": "NULL"},
                    "judge_input_tokens": 0,
                    "judge_output_tokens": 0,
                    "error_type": "judge_exception"
                })

    # Filter valid results (non-errored judges with numeric scores)
    valid_results = []
    for r in results:
        if r.get("judgment") in ("PASS", "FAIL"):
            scores = r.get("scores", {})
            if scores and all(isinstance(v, (int, float)) for v in scores.values()):
                valid_results.append(r)

    if not valid_results:
        logging.warning("All judges errored — no valid results to average")
        tot_cost = sum(r.get("judge_input_token_cost", 0) + r.get('judge_output_token_cost', 0) for r in results)
        return {"majority_judgment": "FAIL", "majority_explanations": ["All judges failed"],
                "judge_details": results, "majority_score": {}, "eval_cost": tot_cost,
                "has_self_eval_judges": False}

    pass_ct = sum(1 for r in valid_results if r.get("judgment") == "PASS")
    fail_ct = sum(1 for r in valid_results if r.get("judgment") == "FAIL")
    tot_cost = sum(r.get("judge_input_token_cost", 0) + r.get('judge_output_token_cost', 0) for r in results)

    avg_scores = calculate_average_scores([r['scores'] for r in valid_results])
    maj = "PASS" if pass_ct > fail_ct else "FAIL"
    exps = [r["explanation"] for r in valid_results if r["judgment"] == maj]

    # Self-evaluation bias detection
    has_self_eval_judges = False
    if target_model_id:
        target_family = _extract_model_family(target_model_id)
        for r in results:
            judge_family = _extract_model_family(r.get("model", ""))
            is_self = bool(target_family and judge_family and target_family == judge_family)
            r["self_eval_warning"] = is_self
            if is_self:
                has_self_eval_judges = True
                logging.warning(
                    f"Self-evaluation detected: judge '{r.get('model')}' shares family "
                    f"'{target_family}' with target model '{target_model_id}'"
                )

    return {"majority_judgment": maj, "majority_explanations": exps, "judge_details": results,
            "majority_score": avg_scores, "eval_cost": tot_cost,
            "has_self_eval_judges": has_self_eval_judges}


# ----------------------------------------
# Detect evaluation mode from judge config
# ----------------------------------------
def detect_eval_mode(judge_models):
    """
    Detect whether judge_models is old-style flat list (bundled mode)
    or new metric-to-model mapping (specialist mode).

    Flat list format (bundled):
      [{"model_id": "...", "region": "...", "input_token_cost": ..., "output_token_cost": ...}, ...]

    Metric mapping format (specialist):
      {"Correctness": {"primary": {...}, "secondary": {...}}, "Completeness": {"primary": {...}}, ...}

    Returns: "specialist" or "bundled"
    """
    if isinstance(judge_models, dict):
        # Check if any key is a known metric name
        if any(k in STANDARD_METRICS for k in judge_models):
            return "specialist"
    return "bundled"


# ----------------------------------------
# Single-metric specialist judge call
# ----------------------------------------
def _evaluate_single_metric(metric_name, judge_config, prompt, model_response,
                            golden_answer, task_types, task_criteria,
                            yard_stick_value, structure_validation=None,
                            custom_metric_definition=None, success_criteria=None):
    """
    Evaluate a single metric with a single judge model.
    Returns a dict with score, rationale, model, cost, etc.

    Args:
        custom_metric_definition: For custom metrics, the full definition dict
    """
    judge_model_id = judge_config["model_id"]
    judge_region = judge_config["region"]

    # Check judge cache before invoking
    cached = judge_cache_get(prompt, model_response, metric_name, judge_model_id)
    if cached is not None:
        cached["cache_hit"] = True
        cached["role"] = "primary"
        return cached

    # Build single-metric prompt — structure validation only for Format
    sv = structure_validation if metric_name == "Format" else None
    custom_defs = [custom_metric_definition] if custom_metric_definition else None
    eval_template = llm_judge_template(
        [metric_name], task_types, task_criteria,
        prompt, model_response, golden_answer,
        structure_validation=sv,
        custom_metric_definitions=custom_defs,
        success_criteria=success_criteria,
    )

    cfg = {"maxTokens": 1500, "aws_region_name": judge_region}
    try:
        resp = run_inference(model_name=judge_model_id,
                             prompt_text=eval_template,
                             provider_params=cfg,
                             stream=False,
                             judge_eval=True)
        text = resp['text']
    except Exception as e:
        logging.error(f"Specialist judge inference failed ({judge_model_id} for {metric_name}): {e}", exc_info=True)
        return {
            "metric": metric_name,
            "model": judge_model_id,
            "role": "primary",
            "judgment": "Judge Exception",
            "score": None,
            "rationale": str(e),
            "judge_input_tokens": 0,
            "judge_output_tokens": 0,
            "error_type": "inference_failure",
        }

    # Parse single-metric JSON response: {"score": int, "rationale": "..."}
    try:
        parsed = extract_json_from_text(text)
        if not parsed:
            return {
                "metric": metric_name, "model": judge_model_id, "role": "primary",
                "judgment": "Error Parsing response", "score": None,
                "rationale": "MalformedJudgeResponseError: Judge response did not contain valid JSON scores", "full_response": text,
                "judge_input_tokens": resp.get('inputTokens', 0),
                "judge_output_tokens": resp.get('outputTokens', 0),
                "error_type": "MalformedJudgeResponseError",
            }

        score = parsed.get("score")
        rationale = parsed.get("rationale", "")

        # Handle bundled-style response that might come back
        if score is None and "scores" in parsed:
            scores_dict = parsed["scores"]
            if metric_name in scores_dict:
                score = scores_dict[metric_name]
                if isinstance(score, dict):
                    rationale = score.get("rationale", "")
                    score = score.get("score")

        if not isinstance(score, (int, float)):
            return {
                "metric": metric_name, "model": judge_model_id, "role": "primary",
                "judgment": "Error Parsing response", "score": None,
                "rationale": f"Invalid score type: {type(score)}", "full_response": text,
                "judge_input_tokens": resp.get('inputTokens', 0),
                "judge_output_tokens": resp.get('outputTokens', 0),
                "error_type": "invalid_score",
            }

        judgment = "PASS" if score >= yard_stick_value else "FAIL"

        # Cost calculation
        input_tokens = resp.get('inputTokens', 0)
        output_tokens = resp.get('outputTokens', 0)
        input_cost = input_tokens * (judge_config.get("input_token_cost", 0) / 1_000_000)
        output_cost = output_tokens * (judge_config.get("output_token_cost", 0) / 1_000_000)

        result = {
            "metric": metric_name,
            "model": judge_model_id,
            "role": "primary",
            "judgment": judgment,
            "score": score,
            "rationale": rationale,
            "full_response": text,
            "judge_input_tokens": input_tokens,
            "judge_output_tokens": output_tokens,
            "judge_input_token_cost": input_cost,
            "judge_output_token_cost": output_cost,
            "cache_hit": False,
        }

        # Store in cache (exclude full_response to keep cache small)
        cache_entry = {k: v for k, v in result.items() if k != "full_response"}
        judge_cache_put(prompt, model_response, metric_name, judge_model_id, cache_entry)

        return result
    except Exception as e:
        logging.error(f"Specialist judge parsing failed ({judge_model_id} for {metric_name}): {e}", exc_info=True)
        return {
            "metric": metric_name, "model": judge_model_id, "role": "primary",
            "judgment": "Error Parsing response", "score": None,
            "rationale": str(e), "full_response": text,
            "judge_input_tokens": resp.get('inputTokens', 0),
            "judge_output_tokens": resp.get('outputTokens', 0),
            "error_type": "parsing_failure",
        }


# ----------------------------------------
# Specialist evaluation (one metric per call)
# ----------------------------------------
def evaluate_specialist(metric_assignments, prompt, model_response, golden_answer,
                        task_types, task_criteria, user_defined_metrics,
                        yard_stick=3, structure_validation=None, target_model_id=None,
                        success_criteria=None):
    """
    Specialist mode: evaluate each metric independently with its assigned model.

    metric_assignments format:
      {"Correctness": {"primary": {...}, "secondary": {...}}, ...}

    Each metric gets 1-2 parallel calls (primary + optional secondary).
    All metric calls run in parallel via ThreadPoolExecutor.
    """
    # Determine per-metric yard stick values
    if isinstance(yard_stick, dict):
        get_ys = lambda m: yard_stick.get(m, 3)
    else:
        _ys = yard_stick
        get_ys = lambda _m: _ys

    # Build lookup for custom metric definitions
    custom_defs = {}
    for cm in (user_defined_metrics or []):
        if isinstance(cm, dict) and cm.get("metric_name"):
            custom_defs[cm["metric_name"]] = cm

    # Build list of all judge calls to dispatch
    calls = []  # (metric_name, judge_config, role, custom_def)
    metrics_to_eval = list(metric_assignments.keys())

    for metric_name, assignment in metric_assignments.items():
        primary = assignment.get("primary")
        secondary = assignment.get("secondary")
        cm_def = custom_defs.get(metric_name)
        if primary:
            calls.append((metric_name, primary, "primary", cm_def))
        if secondary:
            calls.append((metric_name, secondary, "secondary", cm_def))

    if not calls:
        return {"majority_judgment": "FAIL", "majority_explanations": [], "judge_details": [],
                "majority_score": {}, "eval_cost": 0, "has_self_eval_judges": False,
                "eval_mode": "specialist"}

    # Execute all metric evaluations in parallel
    results = []
    with ThreadPoolExecutor(max_workers=min(len(calls), 12)) as executor:
        futures = {}
        for metric_name, judge_config, role, cm_definition in calls:
            f = executor.submit(
                _evaluate_single_metric,
                metric_name, judge_config, prompt, model_response,
                golden_answer, task_types, task_criteria,
                get_ys(metric_name), structure_validation,
                custom_metric_definition=cm_definition,
                success_criteria=success_criteria,
            )
            futures[f] = (metric_name, judge_config, role)

        # No timeout on as_completed itself (see evaluate_with_judges): an iterator-level
        # TimeoutError would discard already-completed metric evaluations. The per-future
        # result(timeout=...) below still bounds each individual specialist judge.
        for future in concurrent.futures.as_completed(futures):
            metric_name, judge_config, role = futures[future]
            try:
                result = future.result(timeout=120)
                result["role"] = role
                results.append(result)
            except concurrent.futures.TimeoutError:
                logging.error(f"Specialist judge timed out: {judge_config['model_id']} for {metric_name}")
                results.append({
                    "metric": metric_name, "model": judge_config["model_id"], "role": role,
                    "judgment": "Judge Exception", "score": None,
                    "rationale": "Timeout after 120 seconds",
                    "judge_input_tokens": 0, "judge_output_tokens": 0,
                    "error_type": "timeout",
                })
            except Exception as e:
                logging.error(f"Specialist judge exception: {judge_config['model_id']} for {metric_name}: {e}")
                results.append({
                    "metric": metric_name, "model": judge_config["model_id"], "role": role,
                    "judgment": "Judge Exception", "score": None,
                    "rationale": str(e),
                    "judge_input_tokens": 0, "judge_output_tokens": 0,
                    "error_type": "judge_exception",
                })

    # Aggregate results: group by metric, average primary+secondary, detect divergence
    metric_results = {}  # metric_name → list of results
    for r in results:
        mn = r["metric"]
        if mn not in metric_results:
            metric_results[mn] = []
        metric_results[mn].append(r)

    final_scores = {}  # metric_name → final averaged score
    all_pass = True
    explanations = []
    total_cost = 0

    for metric_name in metrics_to_eval:
        mrs = metric_results.get(metric_name, [])
        valid = [r for r in mrs if r.get("score") is not None and isinstance(r["score"], (int, float))]

        if not valid:
            # All judges for this metric failed
            final_scores[metric_name] = None
            all_pass = False
            explanations.append(f"{metric_name}: all judges failed")
            continue

        scores = [r["score"] for r in valid]
        avg_score = sum(scores) / len(scores)
        final_scores[metric_name] = round(avg_score, 4)

        # Score divergence detection (primary vs secondary)
        if len(valid) >= 2:
            score_diff = abs(scores[0] - scores[1])
            if score_diff > 2:
                for r in mrs:
                    r["score_divergence"] = True
                logging.warning(
                    f"Score divergence on {metric_name}: {scores[0]} vs {scores[1]} (diff={score_diff})"
                )

        # Pass/fail per metric
        ys = get_ys(metric_name)
        if avg_score < ys:
            all_pass = False
            explanations.append(f"{metric_name}: {avg_score:.1f} < {ys}")

    # Compute costs
    for r in results:
        total_cost += r.get("judge_input_token_cost", 0) + r.get("judge_output_token_cost", 0)

    # Build AVG_ scores dict (same schema as bundled mode)
    avg_scores = {f"AVG_{k}": v for k, v in final_scores.items() if v is not None}

    # Self-evaluation bias detection
    has_self_eval_judges = False
    if target_model_id:
        target_family = _extract_model_family(target_model_id)
        for r in results:
            judge_family = _extract_model_family(r.get("model", ""))
            is_self = bool(target_family and judge_family and target_family == judge_family)
            r["self_eval_warning"] = is_self
            if is_self:
                has_self_eval_judges = True

    majority_judgment = "PASS" if all_pass else "FAIL"

    return {
        "majority_judgment": majority_judgment,
        "majority_explanations": explanations,
        "judge_details": results,
        "majority_score": avg_scores,
        "eval_cost": total_cost,
        "has_self_eval_judges": has_self_eval_judges,
        "eval_mode": "specialist",
    }


# ----------------------------------------
# Core benchmarking function
# ----------------------------------------
def benchmark(
        region,
        prompt, task_types, task_criteria, golden_answer,
        max_tokens, model_id,
        in_cost, out_cost,
        temperature,
        judge_models,
        user_defined_metrics,
        yard_stick=3,
        vision_enabled=None,
        service_tier=None,
        latency_only_mode=False,
        stream_evaluation=True,
        structured_output_format=None,
        success_criteria=None,
        endpoint=None,
):
    logging.debug(f"Starting benchmark for model: {model_id} in region: {region}")
    status = "Success"
    time_to_first_byte = 0
    time_to_last_byte = 0
    ts = get_timestamp()
    perf = {}
    err_code = None
    total_runtime = 0
    throughput_tps = 0
    input_tokens = 0
    output_tokens = 0
    cost = 0
    resp_txt = ""
    thinking_response = ""
    evaluation_cost_data = 0
    inference_request_count = 0
    params = {"max_tokens": max_tokens,
              "temperature": temperature,
              }

    try:
        # Mantle (OpenAI-compatible Responses API) is checked FIRST: its model_id
        # contains "bedrock" so it would otherwise hit the Converse branch, and it must
        # NOT use the direct-OpenAI key. Route it to the Responses path with the Mantle
        # endpoint + the Bedrock API key (BEDROCK_API_KEY — same credential as Converse).
        if endpoint == "bedrock_mantle":
            # Mantle uses the SAME Bedrock API key as Converse models (it's one
            # credential), so source it from BEDROCK_API_KEY — no separate Mantle key.
            params['api_base'] = f"https://bedrock-mantle.{region}.api.aws/openai/v1"
            params['api_key'] = os.getenv('BEDROCK_API_KEY')
        elif "gemini" in model_id:
            params['api_key'] = os.getenv('GOOGLE_API')
        elif 'azure' in model_id:
            params['api_key'] = os.getenv('AZURE_API_KEY')
        elif "bedrock" in model_id:
            params['aws_region_name'] = region
            # Add service tier for Bedrock models if specified and not default
            if service_tier and service_tier != "default":
                params['serviceTier'] = {"type": service_tier}
                logging.info(f"Using service tier '{service_tier}' for model {model_id}")
            # Model ID preparation for litellm is now handled centrally in run_inference()
        elif 'openai/' in model_id:
            params['api_key'] = os.getenv('OPENAI_API')
        else:
            # Sagemaker
            params['aws_region_name'] = region

        r = run_inference(model_id,
                          prompt,
                          in_cost,
                          out_cost,
                          provider_params=params,
                          stream=stream_evaluation,
                          vision_enabled=vision_enabled,
                          endpoint=endpoint)

        # Check for partial/error responses from inference
        if r.get('partial_result'):
            logging.warning(f"Partial response received for {model_id}: {r.get('error', 'Unknown error')}")
            status = f"PartialResponse: {r.get('error_type', 'Unknown')}"
            err_code = "PARTIAL_RESPONSE"

        resp_txt = r.get('model_response', '')
        thinking_response = r.get('thinking_response', '')
        input_tokens = r.get('input_tokens', 0)
        output_tokens = r.get('output_tokens', 0)
        total_runtime = r.get('total_runtime', 0)
        time_to_first_byte = r.get('time_to_first_byte', 0)
        time_to_last_byte = r.get('time_to_last_byte', 0)
        throughput_tps = r.get('throughput_tps', 0)
        cost = r.get('total_cost', 0)
        # retry_count is the number of retries *before* success (0 when the first call
        # succeeds); add 1 so this column reflects the actual number of requests made.
        inference_request_count = r.get('retry_count', 0) + 1

        # Data Structure Analysis — validate response format before judge evaluation
        structure_validation = None
        if structured_output_format and resp_txt:
            from structure_data_checker import validate_structure
            structure_validation = validate_structure(resp_txt, structured_output_format)
            structure_validation["expected_format"] = structured_output_format
            if structure_validation["valid"]:
                logging.info(f"Structure validation PASSED for {model_id} (expected: {structured_output_format})")
            else:
                logging.info(f"Structure validation FAILED for {model_id}: {structure_validation['error']}")

        if resp_txt and not r.get('partial_result'):
            if latency_only_mode:
                # Latency-only mode: Skip judge evaluation and use placeholders
                perf["judge_success"] = "N/A"
                perf["judge_explanation"] = "N/A"
                perf["judge_details"] = []
                perf["judge_scores"] = {}
                evaluation_cost_data = 0
                logging.info(f"Latency-only mode: Skipping judge evaluation for {model_id}")
            else:
                # Full 360 evaluation mode: Run judge evaluation
                eval_mode = detect_eval_mode(judge_models)
                logging.info(f"Judge evaluation mode: {eval_mode} for model {model_id}")

                if eval_mode == "specialist":
                    multi = evaluate_specialist(
                        judge_models,
                        prompt,
                        resp_txt,
                        golden_answer,
                        task_types,
                        task_criteria,
                        user_defined_metrics,
                        yard_stick=yard_stick,
                        structure_validation=structure_validation,
                        target_model_id=model_id,
                        success_criteria=success_criteria,
                    )
                else:
                    multi = evaluate_with_judges(
                        judge_models,
                        prompt,
                        resp_txt,
                        golden_answer,
                        task_types,
                        task_criteria,
                        user_defined_metrics,
                        yard_stick=yard_stick,
                        structure_validation=structure_validation,
                        success_criteria=success_criteria,
                        target_model_id=model_id,
                    )
                perf["judge_success"] = (multi["majority_judgment"] == "PASS")
                perf["judge_explanation"] = ";".join(list(set(multi["majority_explanations"])))
                perf["judge_details"] = multi["judge_details"]
                perf["judge_scores"] = multi["majority_score"]
                perf["has_self_eval_judges"] = multi.get("has_self_eval_judges", False)
                perf["eval_mode"] = multi.get("eval_mode", "bundled")
                evaluation_cost_data = multi["eval_cost"]
        elif r.get('partial_result'):
            # Partial streaming response: skip judging a truncated answer (avoids both
            # wasted judge spend and a misleadingly low score). Status/err_code were
            # already set to PARTIAL_RESPONSE above, which routes this to unprocessed.
            logging.warning(f"Skipping judge evaluation for partial response from {model_id}")
        else:
            # Empty response - set explicit error status
            logging.error(f"Target model error: Model {model_id} returned an empty output.")
            if status == "Success":  # Only override if no prior error
                status = "EmptyResponse: Model returned no output"
                err_code = "EMPTY_RESPONSE"

    except ClientError as err:
        # Log detailed error BEFORE converting to status string
        error_code = err.response["Error"]["Code"]
        error_message = err.response["Error"].get("Message", str(err))
        logging.error(
            f"AWS API ClientError for {model_id}@{region}: {error_code} - {error_message}",
            exc_info=True,
            extra={
                "error_type": "client_error",
                "error_code": error_code,
                "model_id": model_id,
                "region": region
            }
        )
        status = f"{error_code}: {error_message}"
        err_code = error_code
    except KeyError as key_err:
        # Log KeyError with full context to identify root cause
        logging.error(
            f"KeyError for {model_id}@{region}: Missing key '{str(key_err)}' - This indicates a data structure issue",
            exc_info=True,
            extra={
                "error_type": "key_error",
                "missing_key": str(key_err),
                "model_id": model_id,
                "region": region
            }
        )
        status = f"KeyError: {str(key_err)}"
        err_code = "KEY_ERROR"
    except Exception as e:
        # Log general exception with full stack trace
        logging.error(
            f"Unexpected exception for {model_id}@{region}: {type(e).__name__} - {str(e)}",
            exc_info=True,
            extra={
                "error_type": "general_exception",
                "exception_class": type(e).__name__,
                "model_id": model_id,
                "region": region
            }
        )
        status = f"{type(e).__name__}: {str(e)}"
        err_code = type(e).__name__.upper()

    # Extract flattened error info for easier CSV querying
    judge_details = perf.get("judge_details", [])
    has_judge_error = any(j.get("error_type") for j in judge_details) if judge_details else False
    judge_error_types = list(set(j.get("error_type") for j in judge_details if j.get("error_type"))) if has_judge_error else []
    failed_judge_models = [j.get("model") for j in judge_details if j.get("error_type")] if has_judge_error else []

    return {
        "time_to_first_byte": time_to_first_byte,
        "time_to_last_byte": time_to_last_byte,
        "total_runtime": total_runtime,
        "throughput_tps": throughput_tps,
        "job_timestamp_iso": ts,
        "api_call_status": status,
        "error_code": err_code,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response_cost": cost,
        "model_response": resp_txt,
        "performance_metrics": perf,
        "evaluation_cost": evaluation_cost_data,
        "inference_request_count": inference_request_count,
        "eval_type": "latency" if latency_only_mode else "360",
        "stream": stream_evaluation,
        "thinking_response": thinking_response,
        # Flattened error fields for easier CSV querying
        "has_judge_error": has_judge_error,
        "judge_error_types": ";".join(judge_error_types) if judge_error_types else None,
        "failed_judges": ";".join(failed_judge_models) if failed_judge_models else None,
    }


# ----------------------------------------
# Scenario expansion: dynamic temp sweeps
# ----------------------------------------
def expand_scenarios(raw, cfg):
    expanded = []
    for s in raw:
        prompt = s["prompt"]
        region = s["region"]
        base_t = s.get("temperature", s.get("TEMPERATURE", cfg["TEMPERATURE"]))
        param_variants = []
        n_variants = cfg["TEMPERATURE_VARIATIONS"]
        u_diff = 1
        l_diff = 1
        for _ in range(0, n_variants + 1):
            param_variants.append(round(base_t * u_diff, 3))
            param_variants.append(round(base_t * l_diff, 3))
            u_diff += .25
            l_diff -= .25
        # Build a symmetric ±25%-per-step sweep around the base temperature, then
        # clamp to the valid (0, 1.0] range: drop non-positive variants (negative/zero
        # temperatures are invalid API params produced once l_diff goes <= 0) and cap
        # the upper half at 1.0 so high-base sweeps keep their upper point instead of
        # silently discarding it.
        temps = sorted({min(round(t, 3), 1.0) for t in param_variants if t > 0})
        for t in temps:
            sc = s.copy()
            sc["prompt"] = prompt
            sc["region"] = region
            sc["TEMPERATURE"] = round(t, 3)
            expanded.append(sc)
    return expanded
# ----------------------------------------
# Parallel execution
# ----------------------------------------
def execute_benchmark(scenarios, cfg, unprocessed_dir, yard_stick=3, latency_only_mode=False, stream_evaluation=True):
    all_recs = []
    unprocessed_records = []
    lock = Lock()

    # Initialize rate limiter for RPM control
    rate_limiter = TokenBucketRateLimiter()

    def run_scn(scn):
        recs = []
        local_unprocessed = []

        for invocation in range(cfg["invocations_per_scenario"]):
            try:
                # Apply rate limiting if target_rpm is configured for this model
                target_rpm = scn.get("target_rpm")
                if target_rpm:
                    throttle_info = rate_limiter.acquire(scn["model_id"], scn["region"], target_rpm)
                else:
                    throttle_info = {"throttled": False, "wait_time": 0}

                # Smart logging: log first, every 10th, and last invocation to reduce noise
                total_invocations = cfg['invocations_per_scenario']
                is_first = invocation == 0
                is_last = invocation == total_invocations - 1
                is_milestone = (invocation + 1) % 10 == 0

                if is_first or is_last or is_milestone:
                    logging.info(
                        f"Running scenario: {scn['model_id']}@{scn['region']}, temp={scn['TEMPERATURE']}, invocation {invocation + 1}/{total_invocations}")
                else:
                    logging.debug(
                        f"Running scenario: {scn['model_id']}@{scn['region']}, temp={scn['TEMPERATURE']}, invocation {invocation + 1}/{total_invocations}")

                # Use per-scenario user_defined_metrics if available, otherwise fall back to global
                scenario_metrics = scn.get("user_defined_metrics", "")
                if scenario_metrics:
                    # Convert comma-separated string to list
                    user_metrics = [m.strip() for m in scenario_metrics.split(",") if m.strip()]
                else:
                    user_metrics = cfg["user_defined_metrics"]

                r = benchmark(
                    scn["region"],
                    scn["prompt"],
                    scn["task_types"],
                    scn["task_criteria"],
                    scn["golden_answer"],
                    scn["configured_output_tokens_for_request"],
                    scn["model_id"],
                    scn["input_token_cost"],
                    scn["output_token_cost"],
                    scn["TEMPERATURE"],
                    cfg["judge_models"],
                    user_metrics,
                    yard_stick=yard_stick,
                    vision_enabled=scn.get("image_path", None),
                    service_tier=scn.get("service_tier", None),
                    latency_only_mode=latency_only_mode,
                    stream_evaluation=stream_evaluation,
                    structured_output_format=scn.get("structured_output_format"),
                    success_criteria=scn.get("success_criteria"),
                    endpoint=scn.get("endpoint"),
                )

                # Add throttle metrics to result
                r["throttled"] = throttle_info["throttled"]
                r["throttle_wait_time"] = throttle_info["wait_time"]
                r["target_rpm"] = target_rpm

                # Enhanced error detection - check API status, error code, and evaluation success
                perf = r.get("performance_metrics", {})
                judge_details = perf.get("judge_details", [])

                # Check for failures: API errors OR empty performance metrics OR judge evaluation failures
                has_api_error = r["api_call_status"] != "Success" or r["error_code"] is not None
                # In latency-only mode, empty judge_details is expected and not an error
                has_no_evaluation = (not latency_only_mode) and (not perf or not judge_details)
                # Use the flattened has_judge_error field from benchmark() result
                has_judge_errors = r.get("has_judge_error", False)

                if has_api_error or has_no_evaluation or has_judge_errors:
                    # Determine failure reason and error classification for better context
                    if has_api_error:
                        reason = f"API error: {r.get('api_call_status', 'Unknown')} - {r.get('error_code', 'No error code')}"
                        error_classification = "api_failure"
                    elif has_no_evaluation:
                        reason = "Evaluation failure: No judge evaluation performed"
                        error_classification = "evaluation_missing"
                    else:
                        # Use flattened fields for cleaner error reporting
                        failed_judges = r.get("failed_judges", "Unknown")
                        judge_error_types = r.get("judge_error_types", "Unknown")
                        reason = f"Judge evaluation failure: {failed_judges} ({judge_error_types})"
                        error_classification = "judge_failure"

                    logging.warning(
                        f"Record processing failed: {scn['model_id']}@{scn['region']}, reason: {reason}",
                        extra={
                            "error_classification": error_classification,
                            "invocation": invocation,
                            "scenario": scn['model_id']
                        }
                    )

                    # Enhanced unprocessed record with full context
                    local_unprocessed.append({
                        "scenario": scn,
                        "result": r,
                        "reason": reason,
                        "error_classification": error_classification,
                        "timestamp": get_timestamp(),
                        "invocation": invocation,
                        # Use flattened error fields for cleaner unprocessed records
                        "failed_judges": r.get("failed_judges"),
                        "judge_error_types": r.get("judge_error_types"),
                        "judge_errors": [
                            {
                                "judge": j.get("model"),
                                "error_type": j.get("error_type"),
                                "explanation": j.get("explanation")
                            }
                            for j in judge_details if j.get("error_type")
                        ] if has_judge_errors else []
                    })

                    # Also add model inference failures to CSV for error reporting in reports
                    if has_api_error and error_classification == "api_failure":
                        error_record = {**scn, **r}
                        error_record["model_response"] = ""
                        error_record["performance_metrics"] = "{}"
                        error_record["evaluation_cost"] = 0
                        error_record["error_classification"] = "api_failure"
                        error_record["parallel_calls"] = cfg.get("parallel_calls")
                        error_record["invocations_per_scenario"] = cfg.get("invocations_per_scenario")
                        error_record["sleep_between_invocations"] = cfg.get("sleep_between_invocations")
                        error_record["experiment_counts"] = cfg.get("experiment_counts")
                        error_record["experiment_wait_time"] = cfg.get("experiment_wait_time")
                        error_record["yard_stick"] = cfg.get("yard_stick")
                        recs.append(error_record)
                else:
                    # Combine scenario and result
                    result_record = {**scn, **r}

                    # Inject evaluation settings into every record for report visibility
                    result_record["parallel_calls"] = cfg.get("parallel_calls")
                    result_record["invocations_per_scenario"] = cfg.get("invocations_per_scenario")
                    result_record["sleep_between_invocations"] = cfg.get("sleep_between_invocations")
                    result_record["experiment_counts"] = cfg.get("experiment_counts")
                    result_record["experiment_wait_time"] = cfg.get("experiment_wait_time")
                    result_record["yard_stick"] = cfg.get("yard_stick")

                    # If this is an optimized prompt, append label to model_id for display
                    if scn.get("prompt_optimization_label"):
                        result_record["model_id"] = f"{scn['model_id']}_{scn['prompt_optimization_label']}"

                    # If this has a service tier label, append it to model_id for display
                    if scn.get("service_tier_label"):
                        result_record["model_id"] = f"{scn['model_id']}{scn['service_tier_label']}"

                    recs.append(result_record)
                    logging.info(
                        f"Successfully processed: {scn['model_id']}@{scn['region']}, invocation {invocation + 1}")
            except Exception as e:
                # Log exception with full context and stack trace
                logging.error(
                    f"Exception processing record for {scn['model_id']}@{scn['region']}: {type(e).__name__} - {str(e)}",
                    exc_info=True,
                    extra={
                        "error_type": "processing_exception",
                        "exception_class": type(e).__name__,
                        "model_id": scn['model_id'],
                        "region": scn['region'],
                        "invocation": invocation
                    }
                )

                # Enhanced unprocessed record with full error context
                local_unprocessed.append({
                    "scenario": scn,
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                    "reason": f"Processing exception: {type(e).__name__}",
                    "error_classification": "processing_exception",
                    "timestamp": get_timestamp(),
                    "invocation": invocation
                })

            if cfg["sleep_between_invocations"]:
                time.sleep(cfg["sleep_between_invocations"])

        with lock:
            logging.info(
                f"Completed scenario: {scn['model_id']}@{scn['region']} temp={scn['TEMPERATURE']}, processed: {len(recs)}, failed: {len(local_unprocessed)}")
            if local_unprocessed:
                unprocessed_records.extend(local_unprocessed)

        return recs

    with ThreadPoolExecutor(max_workers=cfg["parallel_calls"]) as exe:
        futures = [exe.submit(run_scn, s) for s in scenarios]
        for f in concurrent.futures.as_completed(futures):
            try:
                result = f.result()
                if result:
                    all_recs.extend(result)
                else:
                    logging.warning("Received empty result from a scenario task")
            except Exception as e:
                # Log ThreadPoolExecutor exception with full context
                logging.error(
                    f"Exception in ThreadPoolExecutor task: {type(e).__name__} - {str(e)}",
                    exc_info=True,
                    extra={
                        "error_type": "executor_exception",
                        "exception_class": type(e).__name__
                    }
                )
                # Record the failure but allow other tasks to continue
                with lock:
                    unprocessed_records.append({
                        "scenario": "Unknown (future failed)",
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "reason": f"ThreadPoolExecutor exception: {type(e).__name__}",
                        "error_classification": "executor_exception",
                        "timestamp": get_timestamp()
                    })

    # Log RPM metrics for models with rate limiting
    rpm_metrics = rate_limiter.get_all_metrics()
    if rpm_metrics:
        logging.info("=== RPM Rate Limiting Summary ===")
        for model_key, metrics in rpm_metrics.items():
            logging.info(f"Model: {model_key}")
            logging.info(f"  Target RPM: {metrics.get('target_rpm', 'N/A')}")
            logging.info(f"  Actual RPM: {metrics.get('actual_rpm', 0)}")
            logging.info(f"  Throttle Events: {metrics.get('throttle_count', 0)}")
            logging.info(f"  Total Wait Time: {metrics.get('total_wait_time', 0):.2f}s")

    # Add actual_rpm metrics to results
    for rec in all_recs:
        model_key = f"{rec.get('model_id')}@{rec.get('region')}"
        if model_key in rpm_metrics:
            rec["actual_rpm"] = rpm_metrics[model_key].get("actual_rpm", 0)
            rec["throttle_events_count"] = rpm_metrics[model_key].get("throttle_count", 0)
        else:
            rec["actual_rpm"] = None
            rec["throttle_events_count"] = 0

    # Write unprocessed records to file if any exist
    unprocessed_file_path = None
    if unprocessed_records:
        ts = get_timestamp().replace(':', '-')
        uuid_ = str(uuid.uuid4()).split('-')[-1]
        experiment_name = cfg.get("EXPERIMENT_NAME", "unknown")
        unprocessed_file = os.path.join(unprocessed_dir, f"unprocessed_{experiment_name}_{ts}_{uuid_}.json")
        logging.warning(f"Writing {len(unprocessed_records)} unprocessed records to {unprocessed_file}")
        try:
            with open(unprocessed_file, 'w') as f:
                json.dump(unprocessed_records, f, indent=2, default=str)
            logging.info(f"Successfully wrote unprocessed records to {unprocessed_file}")
            unprocessed_file_path = unprocessed_file
        except Exception as e:
            logging.error(f"Failed to write unprocessed records, file: {str(e)}", exc_info=True)

    # Persist the judge cache once per run (judge_cache_put now only writes to
    # memory, avoiding O(n^2) full-file rewrites during the run).
    flush_judge_cache()

    return all_recs, unprocessed_file_path, len(unprocessed_records)


def model_sanity_check(models):
    from utils import check_model_access
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    
    def check_single_model(model):
        """Check access for a single model"""
        params = {"max_tokens": 10, "temperature": 1}
        model_id = model['model_id']

        # Mantle (Responses API) models can't be access-checked via completion() — the
        # access is gated by the Bedrock API key at run time. Treat as granted.
        if model.get('endpoint') == 'bedrock_mantle':
            return model, 'granted', None

        # Setup params based on model type
        if "gemini" in model_id:
            params['api_key'] = os.getenv('GOOGLE_API')
        elif 'azure' in model_id:
            params['api_key'] = os.getenv('AZURE_API_KEY')
        elif 'bedrock' in model_id:
            # Model ID preparation for litellm is now handled centrally in check_model_access()
            pass
        elif 'openai/' in model_id:
            params['api_key'] = os.getenv('OPENAI_API')
        else:
            params['aws_region_name'] = model['region']
        
        try:
            access = check_model_access(params, model_id)
            return model, access, None
        except Exception as e:
            return model, 'failed', str(e)
    
    logging.info(f"Checking access for {len(models)} models...")
    
    distilled = []
    failed = []
    lock = Lock()
    
    # Run checks in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(10, len(models))) as executor:
        # Submit all model checks
        future_to_model = {
            executor.submit(check_single_model, model): model 
            for model in models
        }

        completed = 0
        total = len(models)
        
        # Process results as they complete
        for future in as_completed(future_to_model):
            completed += 1
            original_model = future_to_model[future]
            
            try:
                model, access, error = future.result(timeout=30)  # 30 second timeout per model
                
                with lock:
                    if access == 'granted':
                        distilled.append(model)
                        region = model.get('region', 'N/A')
                        logging.debug(f"✓ Model access granted: {model['model_id']} @ {region} ({completed}/{total})")
                    else:
                        region = model.get('region', 'N/A')
                        failed.append(f"{model['model_id']} @ {region}")
                        if error:
                            logging.debug(f"✗ Model access failed: {model['model_id']} @ {region} - {error} ({completed}/{total})")
                        else:
                            logging.debug(f"✗ Model access denied: {model['model_id']} @ {region} ({completed}/{total})")
                            
            except Exception as e:
                with lock:
                    region = original_model.get('region', 'N/A')
                    failed.append(f"{original_model['model_id']} @ {region}")
                    logging.error(f"✗ Exception checking model {original_model['model_id']} @ {region}: {str(e)} ({completed}/{total})")
    
    logging.info(f"Model access check complete: {len(distilled)} accessible, {len(failed)} failed")
    return distilled, failed


# ----------------------------------------
# Advanced Prompt Optimization (APO) — pre-eval phase
# ----------------------------------------

def _clean_bedrock_model_id(model_id):
    """Strip routing prefixes for APO modelConfigurations (e.g. 'bedrock/us.amazon.nova-pro-v1:0' -> 'us.amazon.nova-pro-v1:0')."""
    return model_id.replace("bedrock/", "").replace("converse/", "")


def _build_optimized_dataset_csv(local_path, raw_scenarios, original_prompts,
                                 system_prompt, optimized_template):
    """Write a per-model CSV showing each row's optimized text_prompt.

    For each raw scenario row, replace the detected system_prompt segment in
    its text_prompt with the optimized_template (verbatim). If the
    system_prompt isn't found in a row (e.g. extraction was heuristic on a
    sample), fall back to prepending the optimized template.
    """
    import csv as _csv
    with open(local_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["row_index", "original_prompt", "optimized_prompt", "golden_answer"])
        for idx, (orig, golden) in enumerate(zip(original_prompts, raw_scenarios)):
            if system_prompt and system_prompt in orig:
                # Strip the matched system_prompt, prepend optimized template.
                tail = orig.replace(system_prompt, "", 1).strip()
                opt = optimized_template.rstrip() + ("\n\n" + tail if tail else "")
            else:
                opt = optimized_template.rstrip() + "\n\n" + orig
            w.writerow([idx, orig, opt, golden.get("golden_answer", "") if isinstance(golden, dict) else ""])


def _run_apo_phase(scenarios, raw, cfg, output_dir, ts):
    """Run one APO job per selected model in parallel, replace per-model prompts.

    Returns the (possibly mutated) `scenarios` list. Failed APO jobs fall back
    to the original prompt for that model and are recorded in the log.
    """
    mode = cfg.get("prompt_optimization_mode", "none")
    if mode == "none":
        return scenarios

    def _signal(status, detail=""):
        # Parsed by the worker (entrypoint.py) and surfaced on the eval record as
        # apo_status / apo_message. The evaluation always continues with whatever
        # prompts it has — this only reports the optimization outcome.
        logging.warning(f"[APO] RESULT {status} | {detail}".rstrip(" |"))

    evaluator = (cfg.get("apo_evaluator") or "").lower()
    if evaluator not in ("llmj", "steering"):
        # APO was enabled (mode != none) but the evaluator wasn't captured — commonly
        # because the UI's default-selected radio never fired a change event. Infer it
        # from whichever evaluator config is present rather than silently skipping.
        if cfg.get("apo_steering_criteria"):
            evaluator = "steering"
        elif (cfg.get("apo_llmj_rubric") or "").strip():
            evaluator = "llmj"
        if evaluator in ("llmj", "steering"):
            logging.info(f"[APO] evaluator not set; inferred '{evaluator}' from the provided config.")
        else:
            logging.warning("[APO] evaluator not set and none could be inferred; skipping APO.")
            _signal("skipped", "no APO evaluator configured (set a rubric or steering criteria)")
            return scenarios

    bucket = cfg.get("apo_bucket")
    if not bucket:
        bucket = os.environ.get("APO_BUCKET") or os.environ.get("S3_BUCKET")
    if not bucket:
        logging.warning("[APO] no S3 bucket configured; skipping APO.")
        _signal("skipped", "no S3 bucket configured")
        return scenarios

    eval_id = cfg.get("eval_id") or cfg.get("EXPERIMENT_NAME") or "unknown"

    # --- 1. Sample prompts + extract system prompt ---
    sample_count = min(5, len(raw))
    if sample_count == 0:
        logging.warning("[APO] no raw scenarios; skipping APO.")
        _signal("skipped", "no input rows to optimize from")
        return scenarios
    sample_rows = raw[:sample_count]
    sample_prompts = [r.get("prompt", "") for r in sample_rows]

    logging.info(f"[APO] extracting system prompt from {sample_count} sample rows...")
    system_prompt, variable_parts = extract_system_prompt_hybrid(
        sample_prompts,
        min_len=20,
        fallback_model_id="us.amazon.nova-lite-v1:0",
        region=os.environ.get("AWS_REGION", "us-east-1"),
    )

    if not system_prompt:
        logging.warning(
            "[APO] could not extract a system prompt from the dataset; "
            "skipping APO — evaluation will run with original prompts."
        )
        _signal("skipped", "could not extract a shared system prompt from the dataset")
        return scenarios
    logging.info(
        f"[APO] detected system prompt ({len(system_prompt)} chars). Preview: "
        f"{system_prompt[:120]!r}"
    )

    # Build the per-sample rows the APO record expects.
    apo_sample_rows = []
    for i, var in enumerate(variable_parts):
        apo_sample_rows.append({
            "variable_part": var,
            "golden": sample_rows[i].get("golden_answer", "") or "",
        })

    # --- 2. Build records per evaluator mode (record is shared across models) ---
    template_id = f"360eval-{eval_id[:24]}"
    try:
        if evaluator == "llmj":
            rubric = (cfg.get("apo_llmj_rubric") or "").strip()
            judge_id = cfg.get("apo_llmj_judge_model") or ""
            if not rubric or not judge_id:
                logging.warning("[APO] LLM-as-Judge mode missing rubric or judge model; skipping APO.")
                _signal("skipped", "LLM-as-Judge mode is missing a rubric or judge model")
                return scenarios
            record = build_apo_record_llmj(
                template_id, system_prompt, apo_sample_rows,
                rubric=rubric, judge_model_id=_clean_bedrock_model_id(judge_id),
            )
        else:
            criteria = cfg.get("apo_steering_criteria") or []
            record = build_apo_record_steering(
                template_id, system_prompt, apo_sample_rows, criteria=criteria,
            )
    except ValueError as e:
        logging.warning(f"[APO] invalid configuration: {e}; skipping APO.")
        _signal("skipped", f"invalid configuration: {e}")
        return scenarios

    # --- 3. Submit one APO job per unique Bedrock model in parallel ---
    bedrock_models = sorted({
        scn["model_id"] for scn in scenarios
        if "bedrock" in scn.get("model_id", "")
    })
    if not bedrock_models:
        logging.info("[APO] no Bedrock models in scenarios; skipping APO.")
        _signal("skipped", "no Bedrock models selected (APO is Bedrock-only)")
        return scenarios

    # APO service limits concurrent jobs to 5 per account — ThreadPoolExecutor
    # caps in-flight work at 5, others queue and start as slots free up.
    n_models = len(bedrock_models)
    if n_models > 5:
        logging.info(
            f"[APO] {n_models} models requested but APO caps at 5 concurrent jobs. "
            f"Total wall-clock ≈ ceil({n_models}/5) × ~30-50 min."
        )
    logging.info(
        f"[APO] submitting {n_models} job(s) (max 5 concurrent) for: "
        f"{[_clean_bedrock_model_id(m) for m in bedrock_models]}"
    )

    apo_results = {}  # model_id -> result dict from run_one_apo_job
    apo_local_dir = os.path.join(output_dir, "apo")
    os.makedirs(apo_local_dir, exist_ok=True)

    n_models = len(bedrock_models)
    apo_done = 0
    # Emit a parseable progress marker the worker maps to the APO phase band.
    logging.info(f"[APO] PROGRESS {apo_done}/{n_models}")
    max_parallel = min(len(bedrock_models), 5)
    with ThreadPoolExecutor(max_workers=max_parallel) as exe:
        futures = {}
        for model_id in bedrock_models:
            clean = _clean_bedrock_model_id(model_id)
            fut = exe.submit(
                run_one_apo_job, record,
                bucket=bucket, eval_id=eval_id, model_id=clean,
                local_result_dir=Path(apo_local_dir),
            )
            futures[fut] = model_id
        for fut in concurrent.futures.as_completed(futures):
            model_id = futures[fut]
            try:
                result = fut.result(timeout=4200)  # 70 min per job
            except Exception as e:
                logging.error(f"[APO] {model_id} failed: {e}", exc_info=True)
                result = {"model_id": _clean_bedrock_model_id(model_id),
                          "status": "Failed", "error": str(e),
                          "optimized_template": None}
            apo_results[model_id] = result
            apo_done += 1
            logging.info(f"[APO] PROGRESS {apo_done}/{n_models}")

    # --- 4. Persist artifacts: per-model template + dataset CSV + summary log ---
    apo_log = {
        "system_prompt": system_prompt[:500] + ("..." if len(system_prompt) > 500 else ""),
        "system_prompt_full_length": len(system_prompt),
        "evaluator": evaluator,
        "apply_mode": mode,
        "models": {},
    }
    original_prompts_full = [r.get("prompt", "") for r in raw]
    for model_id, result in apo_results.items():
        clean = _clean_bedrock_model_id(model_id)
        safe = model_id.replace("/", "_").replace(":", "_").replace(".", "_")
        opt = result.get("optimized_template")
        apo_log["models"][model_id] = {
            "clean_model_id": clean,
            "status": result.get("status"),
            "job_arn": result.get("jobArn"),
            "error": result.get("error"),
            "original_score": result.get("original_score"),
            "optimized_score": result.get("optimized_score"),
            "submitted_at": result.get("submittedAt"),
            "completed_at": result.get("completedAt"),
            "optimized_template_present": bool(opt),
        }
        if opt:
            tpl_path = os.path.join(apo_local_dir, f"optimized_template_{safe}.txt")
            with open(tpl_path, "w", encoding="utf-8") as f:
                f.write(opt)
            csv_path = os.path.join(apo_local_dir, f"optimized_dataset_{safe}.csv")
            _build_optimized_dataset_csv(csv_path, raw, original_prompts_full,
                                         system_prompt, opt)
            logging.info(f"[APO] wrote artifacts for {clean}: {tpl_path}, {csv_path}")

    log_path = os.path.join(apo_local_dir, "apo_optimization_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(apo_log, f, indent=2, default=str)
    logging.info(f"[APO] wrote summary log: {log_path}")

    # --- 5. Inject optimized prompts into scenarios (per model) ---
    optimized_scenarios = []
    for scn in scenarios:
        model_id = scn["model_id"]
        result = apo_results.get(model_id)
        if (not result) or not result.get("optimized_template"):
            # APO unavailable or failed for this model -- keep original.
            optimized_scenarios.append(scn)
            continue
        opt = result["optimized_template"]
        orig_prompt = scn["prompt"]
        if system_prompt and system_prompt in orig_prompt:
            new_prompt = orig_prompt.replace(
                system_prompt, opt.rstrip(), 1,
            )
        else:
            new_prompt = opt.rstrip() + "\n\n" + orig_prompt

        if mode == "evaluate_both":
            optimized_scenarios.append(scn)
            opt_scn = scn.copy()
            opt_scn["prompt"] = new_prompt
            opt_scn["prompt_optimization_label"] = "Prompt_Optimized"
            optimized_scenarios.append(opt_scn)
        else:  # optimize_only
            scn["prompt"] = new_prompt
            optimized_scenarios.append(scn)

    n_ok = sum(1 for r in apo_results.values() if r.get("optimized_template"))
    n_total = len(bedrock_models)
    logging.info(f"[APO] applied optimized prompts to {n_ok}/{n_total} models")

    # Surface the outcome to the worker -> eval record. The eval ALWAYS continues with
    # whatever prompts were applied (original for any model whose job failed).
    if n_ok == 0:
        # Jobs ran but none produced an optimized template — surface the first error
        # (which now includes the service's failureMessage). Collapse whitespace/newlines
        # to keep the single-line worker marker intact, and cap the length.
        first_err = next((r.get("error") for r in apo_results.values() if r.get("error")), "")
        first_err = " ".join(str(first_err).split())[:400]
        _signal("failed", f"0/{n_total} models optimized; ran with original prompts. {first_err}".strip())
    elif n_ok < n_total:
        _signal("partial", f"{n_ok}/{n_total} models optimized; the rest ran with original prompts")
    else:
        _signal("applied", f"{n_ok}/{n_total} models optimized")
    return optimized_scenarios


# ----------------------------------------
# Main entrypoint
# ----------------------------------------
def _resolve_run_file(name_or_path, eval_dir):
    """Resolve a user-supplied file argument to an absolute path.

    Accepts an absolute path, a bare filename under the project runs/
    directory, or a path relative to the current working directory (so the
    documented `runs/input_file.jsonl` form works from the project root).
    """
    if os.path.isabs(name_or_path):
        return name_or_path
    in_runs = os.path.join(eval_dir, name_or_path)
    if os.path.exists(in_runs):
        return in_runs
    as_given = os.path.abspath(name_or_path)
    if os.path.exists(as_given):
        return as_given
    return in_runs


def main(
        input_file,
        output_dir,
        report,
        parallel_calls,
        invocations_per_scenario,
        sleep_between_invocations,
        temp_variants,
        experiment_counts,
        experiment_name,
        user_defined_metrics=None,
        model_file_name=None,
        judge_file_name=None,
        yard_stick=3,
        vision_enabled=False,
        experiment_wait_time=0,
        prompt_optimization_mode="none",
        latency_only_mode=False,
        stream_evaluation=True,
        apo_evaluator=None,
        apo_llmj_rubric=None,
        apo_llmj_judge_model=None,
        apo_steering_criteria=None,
        apo_bucket=None,
        eval_id=None,
):
    user_defined_metrics_list = None
    if user_defined_metrics:
        user_defined_metrics_list = [metrics.strip().replace(' ', '-') for metrics in user_defined_metrics.split(',') if
                                     metrics != "None"]

    # Get project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Create logs directory with absolute path
    logs_dir = os.path.join(project_root, "logs")
    config_dir = os.path.join(project_root, "config")
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging
    ts, log_file = setup_logging(logs_dir, experiment_name)
    logging.info(f"Starting benchmark run: {experiment_name}")
    print(f"Logs are being saved to: {log_file}")

    uuid_ = str(uuid.uuid4()).split('-')[-1]

    # Ensure output directory is absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create directory for unprocessed records
    unprocessed_dir = os.path.join(output_dir, "unprocessed")
    os.makedirs(unprocessed_dir, exist_ok=True)

    # Use consistent paths for runs directory
    eval_dir = os.path.join(project_root, "runs")
    os.makedirs(eval_dir, exist_ok=True)

    file_path = _resolve_run_file(input_file, eval_dir)

    judge_file_name = judge_file_name if judge_file_name else f"{config_dir}/judge_profiles.jsonl"
    model_file_name = model_file_name if model_file_name else f"{config_dir}/models_profiles.jsonl"
    judge_path = _resolve_run_file(judge_file_name, eval_dir)
    model_path = _resolve_run_file(model_file_name, eval_dir)

    # Validate configuration files before loading
    # Skip judge validation in latency-only mode (empty judge file is expected)
    if not latency_only_mode:
        # Check if specialist format (metric-to-model mapping) or JSONL
        with open(judge_path, 'r', encoding='utf-8') as f:
            content_peek = f.read().strip()
        # Specialist: single JSON object with no "model_id" key (metric names as keys)
        is_specialist = content_peek.startswith('{') and '\n' not in content_peek and '"model_id"' not in content_peek

        if is_specialist:
            logging.info("Specialist judge config detected — skipping JSONL validation")
        else:
            logging.info("Validating judge profiles...")
            judge_errors, judge_warnings = validate_jsonl_file(judge_path, "judge")
            if judge_errors:
                logging.error("Judge profiles validation failed:")
                for error in judge_errors:
                    logging.error(f"  {error}")
                raise ValueError(f"Invalid judge profiles configuration. Found {len(judge_errors)} error(s).")

            if judge_warnings:
                for warning in judge_warnings:
                    logging.warning(warning)
    else:
        logging.info("Latency-only mode: Skipping judge profiles validation")

    logging.info("Validating model profiles...")
    model_errors, model_warnings = validate_jsonl_file(model_path, "model")
    if model_errors:
        logging.error("Model profiles validation failed:")
        for error in model_errors:
            logging.error(f"  {error}")
        raise ValueError(f"Invalid model profiles configuration. Found {len(model_errors)} error(s).")

    if model_warnings:
        for warning in model_warnings:
            logging.warning(warning)

    logging.info("Configuration validation completed successfully")

    # Load judge profiles (skip in latency-only mode)
    # Supports two formats:
    #   - Bundled: JSONL with one judge per line (flat list)
    #   - Specialist: single JSON object mapping metrics to model assignments
    judge_config = []  # list for bundled, dict for specialist
    if not latency_only_mode:
        with open(judge_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content.startswith('{') and '\n' not in content:
            # Could be specialist mode (single JSON dict) or a single-line JSONL
            parsed = json.loads(content)
            if 'model_id' in parsed:
                # Single-judge bundled JSONL (has model_id key)
                judge_config = [parsed]
                logging.info(f"Loaded 1 Juror model (bundled mode, single entry)")
            else:
                # Specialist mode: metric-to-model mapping
                judge_config = parsed
                logging.info(f"Loaded specialist judge config with {len(judge_config)} metric assignment(s)")
        else:
            # Bundled mode: JSONL lines
            for line in content.split('\n'):
                if line.strip():
                    judge_config.append(json.loads(line))
            logging.info(f"Loaded {len(judge_config)} Jurors model(s) (bundled mode)")
    else:
        logging.info("Latency-only mode: Skipping judge profiles loading")

    cfg = {
        "parallel_calls": parallel_calls,
        "invocations_per_scenario": invocations_per_scenario,
        "sleep_between_invocations": sleep_between_invocations,
        "TEMPERATURE": 1.0,
        "TEMPERATURE_VARIATIONS": int(temp_variants),
        "EXPERIMENT_NAME": experiment_name,
        "judge_models": judge_config,
        "user_defined_metrics": user_defined_metrics_list,
        "prompt_optimization_mode": prompt_optimization_mode,
        "latency_only_mode": latency_only_mode,
        "experiment_counts": experiment_counts,
        "experiment_wait_time": experiment_wait_time,
        "yard_stick": yard_stick,
        "apo_evaluator": apo_evaluator,
        "apo_llmj_rubric": apo_llmj_rubric,
        "apo_llmj_judge_model": apo_llmj_judge_model,
        "apo_steering_criteria": apo_steering_criteria or [],
        "apo_bucket": apo_bucket,
        "eval_id": eval_id or experiment_name,
    }

    # Validate prompt JSONL — fatal on errors so the worker exits early with a
    # clear message rather than failing mid-loop with a KeyError on malformed input.
    prompt_errors, prompt_warnings = validate_prompt_jsonl_file(file_path)
    for warning in prompt_warnings:
        logging.warning(warning)
    if prompt_errors:
        for error in prompt_errors:
            logging.error(error)
        raise ValueError(f"Invalid prompt JSONL. Found {len(prompt_errors)} error(s).")

    # Load scenarios — single-shot format: scalar fields per row
    raw = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            js = json.loads(line)
            raw.append({
                "prompt": js.get("text_prompt", ""),
                "task_types": js["task"]["task_type"],
                "task_criteria": js["task"]["task_criteria"],
                "golden_answer": js.get("golden_answer", ""),
                "configured_output_tokens_for_request": js.get("expected_output_tokens", 4500),
                "region": js.get("region", "us-east-1"),
                "temperature": js.get("temperature", 0.7),
                "user_defined_metrics": js.get("user_defined_metrics", ""),
                "success_criteria": js.get("success_criteria"),
            })
            if vision_enabled:
                raw[-1].update({"image_path": js.get("url_image", "")})

    if not raw:
        logging.error("No scenarios found in input.")
        return

    # Ensure models_profiles.jsonl exists and pricing is fresh
    try:
        from bedrock_pricing import ensure_models_profiles
        ensure_models_profiles()
    except Exception as e:
        logging.warning("Failed to ensure models profiles: %s", e)

    raw_with_models = []
    raw_models = []
    with open(model_path, 'r', encoding='utf-8') as f:
        for _line in f:
            raw_models.append(json.loads(_line))

        models, failed = model_sanity_check(raw_models)

        if len(models) == 0:
            logging.error('The following models failed to generate inference, please check Permissions and Access:\n'  + '\n'.join([str(fail) for fail in failed]))
            raise
        if len(failed) > 0:
            logging.warning('The following models failed to generate inference, please check Permissions and Access:\n'  + '\n'.join([str(fail) for fail in failed]))

        for model in models:
            for s in raw:
                raw_with_models.append({**s, **model})

    scenarios = expand_scenarios(raw_with_models, cfg)
    logging.info(f"Expanded to {len(scenarios)} scenarios")

    # Handle Advanced Prompt Optimization (APO) if enabled.
    # Replaces the legacy synchronous optimize_prompt_bedrock with the
    # job-based APO API. One job per selected model, run in parallel; each
    # model gets its own optimized template.
    prompt_optimization_mode = cfg.get("prompt_optimization_mode", "none")
    if prompt_optimization_mode != "none":
        scenarios = _run_apo_phase(scenarios, raw, cfg, output_dir, ts)
        logging.info(f"Final scenario count after APO phase: {len(scenarios)}")

    # Authoritative total work units for the progress bar, emitted after APO has
    # finalized the scenario count: runs × scenarios × invocations. The worker uses
    # this as the denominator for per-invocation progress.
    eval_total_units = experiment_counts * len(scenarios) * invocations_per_scenario
    logging.info(f"[PROGRESS] eval_total_units {eval_total_units}")

    for run in range(1, experiment_counts + 1):
        # Add timestamp for time-based performance tracking
        run_start_time = time.time()
        run_timestamp = datetime.now().isoformat()

        logging.info(f"=== Run {run}/{experiment_counts} (Started: {run_timestamp}) ===")

        try:
            results, unprocessed_file_path, unprocessed_count = execute_benchmark(scenarios, cfg, unprocessed_dir, yard_stick=int(yard_stick), latency_only_mode=latency_only_mode, stream_evaluation=stream_evaluation)

            if not results:
                logging.error(f"Run {run}/{experiment_counts} produced no results. Check the unprocessed records file.")
                if unprocessed_file_path:
                    logging.warning(f"Unprocessed records saved to: {unprocessed_file_path}")
                continue

            try:
                df = pd.DataFrame(results)
                df["run_count"] = run
                df["timestamp"] = pd.Timestamp.now()
                df["run_start_time"] = run_timestamp
                df["run_duration_seconds"] = time.time() - run_start_time
                out_csv = os.path.join(output_dir, f"invocations_{run}_{ts}_{uuid_}_{experiment_name}.csv")
                df.to_csv(out_csv, index=False)

                run_duration = time.time() - run_start_time
                logging.info(f"Run {run} completed in {run_duration:.1f} seconds, results saved to {out_csv}")

            except Exception as e:
                logging.error(f"Error saving results for run {run}: {str(e)}", exc_info=True)

        except Exception as e:
            logging.error(f"Critical error in run {run}: {str(e)}", exc_info=True)
            print(f"\nRun {run} failed with error: {str(e)}. Continuing with next run...")

        # Wait between experiments (except after the last one)
        if experiment_wait_time > 0 and run < experiment_counts:
            wait_minutes = experiment_wait_time / 60
            next_run_time = datetime.fromtimestamp(time.time() + experiment_wait_time)

            logging.info(
                f"Waiting {wait_minutes:.1f} minutes before next experiment (next run at {next_run_time.strftime('%H:%M:%S')})")
            print(f"\nWaiting {wait_minutes:.1f} minutes before run {run + 1}...")
            print(f"Next run scheduled at: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Sleep with progress updates every 5 minutes
            remaining_time = experiment_wait_time
            update_interval = min(300, int(experiment_wait_time / 10))  # Every 5 minutes or 10% of wait time

            while remaining_time > 0:
                sleep_time = min(update_interval, remaining_time)
                time.sleep(sleep_time)
                remaining_time -= sleep_time

                if remaining_time > 0:
                    remaining_minutes = remaining_time / 60
                    logging.info(f"Still waiting... {remaining_minutes:.1f} minutes remaining until run {run + 1}")

            logging.info(f"Wait period completed. Starting run {run + 1}")
            print(f"Starting run {run + 1}...")

        # Add small delay even without experiment wait time to separate runs clearly
        elif run < experiment_counts:
            time.sleep(2)  # 2 second separation between runs

    # Check for unprocessed records
    try:
        unprocessed_files = [f for f in os.listdir(unprocessed_dir) if f.startswith("unprocessed_")]
        if unprocessed_files:
            logging.warning(f"Found {len(unprocessed_files)} files with unprocessed records in {unprocessed_dir}")
            print(f"\nWarning: {len(unprocessed_files)} files with unprocessed records found in {unprocessed_dir}")
    except Exception as e:
        logging.error(f"Error checking for unprocessed records: {str(e)}", exc_info=True)

    if report:
        try:
            from visualize_results import create_html_report
            # Generate report
            report = create_html_report(output_dir, ts)
            print(f"\nBenchmark complete! Report: {report}")
            logging.info(f"Benchmark run complete. Report generated at {report}")
        except ImportError as e:
            logging.error(f"Failed to import visualization module: {str(e)}")
            print("\nBenchmark complete, but report generation failed due to import error.")
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}", exc_info=True)
            print("\nBenchmark complete, but report generation failed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Advanced Unified LLM Benchmarking Tool")
    p.add_argument("input_file", help="JSONL file with scenarios")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--report", type=lambda x: x.lower() == 'true', default=True)
    p.add_argument("--parallel_calls", type=int, default=4)
    p.add_argument("--invocations_per_scenario", type=int, default=2)
    p.add_argument("--sleep_between_invocations", type=int, default=3)
    p.add_argument("--experiment_counts", type=int, default=2)
    p.add_argument("--experiment_name", default=f"Benchmark-{datetime.now().strftime('%Y%m%d')}")
    p.add_argument("--experiment_wait_time", type=int, default=0,
                   help="Wait time in seconds between experiments (0 = no wait)")
    p.add_argument("--temperature_variations", type=int, default=0)
    p.add_argument("--user_defined_metrics", default=None)
    p.add_argument("--model_file_name", default=None)
    p.add_argument("--judge_file_name", default=None)
    p.add_argument("--evaluation_pass_threshold", default=3)
    p.add_argument("--vision_enabled", type=lambda x: x.lower() == 'true', default=False)
    p.add_argument("--prompt_optimization_mode",
                   default="none",
                   choices=["none", "optimize_only", "evaluate_both"],
                   help="Prompt optimization apply mode: none, optimize_only, or evaluate_both")
    p.add_argument("--apo_evaluator", default=None, choices=[None, "llmj", "steering"],
                   help="APO evaluator mode (when prompt_optimization_mode != none)")
    p.add_argument("--apo_llmj_rubric", default=None,
                   help="LLM-as-Judge rubric (when apo_evaluator=llmj)")
    p.add_argument("--apo_llmj_judge_model", default=None,
                   help="Judge model id for LLM-as-Judge mode")
    p.add_argument("--apo_steering_criteria", default=None,
                   help="JSON list of steering criteria (when apo_evaluator=steering)")
    p.add_argument("--apo_bucket", default=None,
                   help="S3 bucket for APO input/output (defaults to env APO_BUCKET or S3_BUCKET)")
    p.add_argument("--eval_id", default=None,
                   help="Evaluation UUID (used to namespace APO artifacts in S3)")
    p.add_argument("--latency_only_mode", type=lambda x: x.lower() == 'true', default=False,
                   help="Enable latency-only evaluation mode (skip LLM judge evaluation)")
    p.add_argument("--stream_evaluation", type=lambda x: x.lower() == 'true', default=True,
                   help="Use streaming mode for model evaluation (True=streaming, False=non-streaming)")
    args = p.parse_args()
    main(
        args.input_file,
        args.output_dir,
        args.report,
        args.parallel_calls,
        args.invocations_per_scenario,
        args.sleep_between_invocations,
        args.temperature_variations,
        args.experiment_counts,
        args.experiment_name,
        args.user_defined_metrics,
        args.model_file_name,
        args.judge_file_name,
        args.evaluation_pass_threshold,
        args.vision_enabled,
        args.experiment_wait_time,
        args.prompt_optimization_mode,
        args.latency_only_mode,
        args.stream_evaluation,
        apo_evaluator=args.apo_evaluator,
        apo_llmj_rubric=args.apo_llmj_rubric,
        apo_llmj_judge_model=args.apo_llmj_judge_model,
        apo_steering_criteria=(json.loads(args.apo_steering_criteria) if args.apo_steering_criteria else None),
        apo_bucket=args.apo_bucket,
        eval_id=args.eval_id,
    )
