"""
Flask web server for 360-eval dashboard.
Hosted version using S3 + DynamoDB for multi-tenant storage.
"""

import functools
import io
import sys
import os
import json
import tempfile
import uuid
from decimal import Decimal
from pathlib import Path
from datetime import datetime, timezone

# Load .env.local for local development (before any other imports that read env vars)
from dotenv import load_dotenv
_env_local = Path(__file__).parent.parent / '.env.local'
if _env_local.exists():
    load_dotenv(_env_local)
    print(f"[CONFIG] Loaded {_env_local}")

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, g, redirect
from flask_cors import CORS
import pandas as pd

# Auth middleware for hosted deployment
from aws.auth import require_user
from aws import dynamo_client as db
from aws import s3_client as s3
from aws import kms_client as kms
from aws import ecs_client as ecs

# Add parent src directory to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Model/judge profiles are loaded on-demand by generate_model_info() in constants.py.
# It reads from local config/ first, falls back to S3 at s3://{bucket}/config/.
# No pricing scraping needed — profiles are maintained centrally in S3.

# Import existing backend modules
from dashboard.utils.constants import (
    PROJECT_ROOT, DEFAULT_OUTPUT_DIR, STATUS_FILES_DIR, DEFAULT_PROMPT_EVAL_DIR,
    AWS_REGIONS, DEFAULT_PARALLEL_CALLS, DEFAULT_INVOCATIONS_PER_SCENARIO,
    DEFAULT_SLEEP_BETWEEN_INVOCATIONS, DEFAULT_EXPERIMENT_COUNTS,
    DEFAULT_TEMPERATURE_VARIATIONS, DEFAULT_FAILURE_THRESHOLD,
    generate_model_info
)

# Import visualization module for report generation
from visualization import create_html_report

app = Flask(__name__)

# Enable logging to stdout for CloudWatch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
app.logger.setLevel(logging.INFO)

# CORS — allow CloudFront origin in production, all origins in dev
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
CORS(app, origins=CORS_ORIGINS.split(','), supports_credentials=True)

# --- Local dev mode ---
# When LOCAL_DEV_MODE=true, auto-inject Midway header so you can test without ALB/Midway
LOCAL_DEV_MODE = os.environ.get('LOCAL_DEV_MODE', '').lower() == 'true'
LOCAL_DEV_USER = os.environ.get('LOCAL_DEV_USER', 'localdev')

if LOCAL_DEV_MODE:
    @app.before_request
    def _inject_local_dev_user():
        """In local dev mode, simulate Midway by injecting the user header."""
        from aws.auth import USER_HEADER, EMAIL_HEADER
        if not request.headers.get(USER_HEADER):
            # Werkzeug EnvironHeaders are immutable, so we patch environ directly
            request.environ[f'HTTP_{USER_HEADER.upper().replace("-", "_")}'] = LOCAL_DEV_USER
            request.environ[f'HTTP_{EMAIL_HEADER.upper().replace("-", "_")}'] = f'{LOCAL_DEV_USER}@amazon.com'

    print(f"[LOCAL DEV MODE] Auto-injecting user: {LOCAL_DEV_USER}")


# --------------- Helpers ---------------

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _decimal_to_native(obj):
    """Convert DynamoDB Decimal types to int/float for JSON serialization."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == int(obj) else float(obj)
    if isinstance(obj, dict):
        return {k: _decimal_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimal_to_native(i) for i in obj]
    return obj


def _launch_local_eval(user_id, eval_id, eval_name, composite_id, config, credentials, cli_args):
    """Local dev mode: run benchmark as a subprocess in a background thread.
    Downloads input from S3, runs benchmarks_run.py locally, uploads results back to S3.
    """
    import threading
    import subprocess
    import shlex

    def _run():
        try:
            db.update_evaluation(user_id, eval_id, status='running', progress=0)

            # Download input files from S3 to a local temp dir
            tmp_dir = tempfile.mkdtemp(prefix=f"eval_{eval_id[:8]}_")
            input_prefix = s3._user_prefix(user_id, f"uploads/{eval_id}/")
            keys = s3.list_objects(input_prefix)
            for key in keys:
                fname = os.path.basename(key)
                s3.download_file(key, os.path.join(tmp_dir, fname))

            output_dir = os.path.join(tmp_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)

            # Set API keys as env vars for the subprocess
            env = os.environ.copy()
            key_mapping = {'openai': 'OPENAI_API', 'google': 'GOOGLE_API', 'azure': 'AZURE_API_KEY', 'bedrock': 'BEDROCK_API_KEY'}
            for provider, env_name in key_mapping.items():
                if provider in credentials and credentials[provider]:
                    env[env_name] = credentials[provider]

            # Find the JSONL input file
            jsonl_files = [f for f in os.listdir(tmp_dir) if f.endswith('.jsonl') and 'profiles' not in f]
            if not jsonl_files:
                db.update_evaluation(user_id, eval_id, status='failed', error='No JSONL input file found')
                return
            jsonl_path = os.path.join(tmp_dir, jsonl_files[0])

            # Build command
            script_path = str(Path(__file__).parent.parent / "src" / "benchmarks_run.py")
            cmd = [
                sys.executable, script_path, jsonl_path,
                '--output_dir', output_dir,
            ] + cli_args

            print(f"[LOCAL EVAL] Running: {' '.join(cmd[:6])}...")

            start_time = datetime.now(timezone.utc)
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                  cwd=str(Path(__file__).parent.parent))

            end_time = datetime.now(timezone.utc)
            duration = int((end_time - start_time).total_seconds())

            # Upload results to S3
            results_s3_key = ''
            unprocessed_keys = []
            for root, _, files in os.walk(output_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    relative = os.path.relpath(fpath, output_dir)
                    with open(fpath, 'rb') as fobj:
                        key = s3.upload_file(user_id, f"results/{eval_id}/{relative}", fobj)
                    if fname.endswith('.csv'):
                        results_s3_key = key
                    if 'unprocessed' in fname:
                        unprocessed_keys.append(key)

            if proc.returncode == 0:
                db.update_evaluation(user_id, eval_id,
                                    status='completed', progress=100,
                                    end_time=end_time.isoformat(),
                                    duration=duration,
                                    results_s3_key=results_s3_key,
                                    unprocessed_s3_keys=unprocessed_keys)
                print(f"[LOCAL EVAL] Completed in {duration}s")
            else:
                error_msg = proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr
                db.update_evaluation(user_id, eval_id,
                                    status='failed',
                                    end_time=end_time.isoformat(),
                                    duration=duration,
                                    error=error_msg)
                print(f"[LOCAL EVAL] Failed: {error_msg[:200]}")

        except Exception as e:
            db.update_evaluation(user_id, eval_id, status='failed', error=str(e))
            print(f"[LOCAL EVAL] Exception: {e}")

    db.update_evaluation(user_id, eval_id, status='queued')
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _serialize_cost_map(cost_map):
    """Convert (model_id, region) tuple keys to 'model_id|region' strings for JSON."""
    return {f"{k[0]}|{k[1]}": v for k, v in cost_map.items()}


def _serialize_service_tiers(tiers_map):
    """Convert (model_id, region) tuple keys to 'model_id|region' strings for JSON."""
    return {f"{k[0]}|{k[1]}": v for k, v in tiers_map.items()}


# --------------- Routes ---------------

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.route('/api/user/profile', methods=['GET'])
@require_user
def get_user_profile():
    """Get the authenticated user's profile."""
    return jsonify({
        "user_id": g.user_id,
        "email": g.user_email,
    })


@app.route('/api/config', methods=['GET'])
@require_user
def get_config():
    """Get application configuration."""
    return jsonify({
        "aws_regions": AWS_REGIONS,
        "defaults": {
            "parallel_calls": DEFAULT_PARALLEL_CALLS,
            "invocations_per_scenario": DEFAULT_INVOCATIONS_PER_SCENARIO,
            "sleep_between_invocations": DEFAULT_SLEEP_BETWEEN_INVOCATIONS,
            "experiment_counts": DEFAULT_EXPERIMENT_COUNTS,
            "temperature_variations": DEFAULT_TEMPERATURE_VARIATIONS,
            "failure_threshold": DEFAULT_FAILURE_THRESHOLD
        },
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(DEFAULT_OUTPUT_DIR)
    })


@app.route('/api/models', methods=['GET'])
@require_user
def get_models():
    """Get available models from models_profiles.jsonl."""
    try:
        model_data = generate_model_info('models_profiles.jsonl')
        return jsonify({
            "bedrock_models": model_data.get("DEFAULT_BEDROCK_MODELS", []),
            "openai_models": model_data.get("DEFAULT_OPENAI_MODELS", []),
            "cost_map": _serialize_cost_map(model_data.get("DEFAULT_COST_MAP", {})),
            "model_to_regions": model_data.get("MODEL_TO_REGIONS", {}),
            "region_to_models": model_data.get("REGION_TO_MODELS", {}),
            "service_tiers": _serialize_service_tiers(model_data.get("MODEL_SERVICE_TIERS", {})),
            "tier_pricing": _serialize_cost_map(model_data.get("MODEL_TIER_PRICING", {})),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/judges', methods=['GET'])
@require_user
def get_judges():
    """Get available judge models from judge_profiles.jsonl."""
    try:
        judge_data = generate_model_info('judge_profiles.jsonl')
        return jsonify({
            "judges": judge_data.get("DEFAULT_BEDROCK_MODELS", []),
            "cost_map": _serialize_cost_map(judge_data.get("DEFAULT_COST_MAP", {})),
            "model_to_regions": judge_data.get("MODEL_TO_REGIONS", {}),
            "region_to_models": judge_data.get("REGION_TO_MODELS", {}),
            "service_tiers": _serialize_service_tiers(judge_data.get("MODEL_SERVICE_TIERS", {})),
            "tier_pricing": _serialize_cost_map(judge_data.get("MODEL_TIER_PRICING", {})),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
@require_user
def get_metrics():
    """Return standard metric definitions, boundaries, and rubrics for the UI."""
    # Import from the evaluation engine
    import sys, os as _os
    src_path = _os.path.join(_os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from utils import METRIC_DEFINITIONS, STANDARD_METRICS, JUDGE_TEMPLATE_VERSION
        return jsonify({
            "metrics": METRIC_DEFINITIONS,
            "standard_metrics": STANDARD_METRICS,
            "template_version": JUDGE_TEMPLATE_VERSION,
        })
    except ImportError:
        # Fallback if utils not importable from API container
        standard_metrics = ["Correctness", "Completeness", "Relevance", "Format", "Coherence", "Following-instructions"]
        return jsonify({
            "metrics": {},
            "standard_metrics": standard_metrics,
            "template_version": "2.0",
        })


# --------------- CSV Upload ---------------

@app.route('/api/upload-csv', methods=['POST'])
@require_user
def upload_csv():
    """Handle CSV file upload — stores to S3."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read CSV into DataFrame for validation and preview
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        preview = df.head(10).to_dict(orient='records')

        # Upload CSV to S3 temp location
        temp_id = str(uuid.uuid4())
        csv_bytes = df.to_csv(index=False)
        temp_s3_key = s3.upload_bytes(
            g.user_id,
            f"uploads/temp_{temp_id}.csv",
            csv_bytes
        )

        return jsonify({
            "success": True,
            "filename": file.filename,
            "columns": columns,
            "preview": preview,
            "row_count": len(df),
            "temp_path": temp_s3_key  # S3 key — frontend passes this back on create
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear-temp-files', methods=['POST'])
@require_user
def clear_temp_files():
    """Delete all temp CSV uploads for the current user."""
    try:
        temp_prefix = s3._user_prefix(g.user_id, "uploads/temp_")
        temp_keys = s3.list_objects(temp_prefix)
        for key in temp_keys:
            s3.delete_object(key)
        return jsonify({"success": True, "deleted": len(temp_keys)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------- Evaluations CRUD ---------------

@app.route('/api/evaluations', methods=['GET'])
@require_user
def get_evaluations():
    """Get all evaluations for the current user from DynamoDB."""
    try:
        items = db.get_evaluations(g.user_id)
        evaluations = []
        for item in items:
            config = item.get('config', {})
            evaluations.append({
                "id": item['eval_id'],
                "name": item.get('eval_name', ''),
                "status": item.get('status', 'unknown'),
                "progress": item.get('progress', 0),
                "task_type": config.get('task_type', ''),
                "task_criteria": config.get('task_criteria', ''),
                "created_at": item.get('created_at', ''),
                "updated_at": item.get('updated_at', ''),
                "start_time": item.get('start_time'),
                "end_time": item.get('end_time'),
                "duration": item.get('duration'),
                "error": item.get('error'),
                # APO outcome (set by the worker): applied | partial | failed | skipped.
                "apo_status": item.get('apo_status'),
                "apo_message": item.get('apo_message'),
                "results": item.get('results_s3_key'),
                "selected_models": config.get('selected_models', []),
                "judge_models": config.get('judge_models', []),
                "parallel_calls": config.get('parallel_calls', DEFAULT_PARALLEL_CALLS),
                "invocations_per_scenario": config.get('invocations_per_scenario', DEFAULT_INVOCATIONS_PER_SCENARIO),
                "sleep_between_invocations": config.get('sleep_between_invocations', DEFAULT_SLEEP_BETWEEN_INVOCATIONS),
                "experiment_counts": config.get('experiment_counts', DEFAULT_EXPERIMENT_COUNTS),
                "temperature_variations": config.get('temperature_variations', DEFAULT_TEMPERATURE_VARIATIONS),
                "failure_threshold": config.get('failure_threshold', DEFAULT_FAILURE_THRESHOLD),
                "experiment_wait_time": config.get('experiment_wait_time', 0),
                "user_defined_metrics": config.get('user_defined_metrics', ''),
                "temperature": config.get('temperature'),
                "csv_file_name": config.get('csv_file_name'),
                "stream_evaluation": config.get('stream_evaluation', True),
                "vision_enabled": config.get('vision_enabled', False),
                "image_column": config.get('image_column'),
                "prompt_column": config.get('prompt_column'),
                "golden_answer_column": config.get('golden_answer_column'),
                "golden_answer_mode": config.get('golden_answer_mode', 'golden_answer'),
                "success_criteria": config.get('success_criteria', {}),
                "eval_mode": config.get('eval_mode', 'bundled'),
                "metric_assignments": config.get('metric_assignments', {}),
                "custom_metrics": config.get('custom_metrics', []),
                "unprocessed_files": item.get('unprocessed_s3_keys', []),
                # Multi-shot pass-through (single-shot evals omit these by leaving defaults)
                "evaluation_mode": config.get('evaluation_mode', 'single_shot'),
                "turns": config.get('turns', []),
                "chain_config": config.get('chain_config', {}),
            })
        evaluations.sort(key=lambda e: e.get('created_at', ''), reverse=True)
        return jsonify({"evaluations": _decimal_to_native(evaluations)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations', methods=['POST'])
@require_user
def create_evaluation():
    """Create a new evaluation configuration in DynamoDB + upload CSV to S3.

    Supports two evaluation_mode values:
      - 'single_shot' (default): the existing behavior — N task_evaluations
        each spawn their own DynamoDB row + JSONL.
      - 'multi_shot': one DynamoDB row carrying a `turns` spec + `chain_config`;
        the worker runs the chain sequentially with early-exit on judge FAIL.
    """
    try:
        data = request.json
        evaluation_mode = data.get("evaluation_mode", "single_shot")
        base_name = data.get("name", f"Evaluation-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
        temp_s3_key = data.get("temp_path")  # S3 key from upload-csv

        created_evaluations = []

        # --- Multi-shot: each turn becomes its own standalone evaluation ---
        # Per Option A — turns are NOT chained; each turn is evaluated independently
        # like any other single-shot eval. Names extended as {base_name}_{i+1}.
        if evaluation_mode == "multi_shot":
            turn_specs = data.get("turns") or []

            if not turn_specs:
                return jsonify({"error": "Multi-shot evaluation requires at least one turn"}), 400
            for i, t in enumerate(turn_specs, start=1):
                if not t.get("prompt_column"):
                    return jsonify({"error": f"Turn {i} missing prompt_column"}), 400
                if not t.get("golden_answer_column"):
                    return jsonify({"error": f"Turn {i} missing golden_answer_column"}), 400

            for i, turn in enumerate(turn_specs):
                eval_id = str(uuid.uuid4())
                eval_name = f"{base_name}_{i+1}"

                csv_s3_key = None
                if temp_s3_key and s3.object_exists(temp_s3_key):
                    csv_data = s3.download_bytes(temp_s3_key)
                    csv_s3_key = s3.upload_bytes(
                        g.user_id,
                        f"uploads/{eval_id}/source_data.csv",
                        csv_data
                    )

                eval_config = {
                    'eval_name': eval_name,
                    'csv_s3_key': csv_s3_key or '',
                    'config': {
                        'evaluation_mode': 'single_shot',
                        # Per-turn column + task config
                        'prompt_column': turn.get('prompt_column'),
                        'golden_answer_column': turn.get('golden_answer_column'),
                        'task_type': turn.get('task_type', ''),
                        'task_criteria': turn.get('task_criteria', ''),
                        'temperature': turn.get('temperature', 0.7),
                        'user_defined_metrics': turn.get('user_defined_metrics', ''),
                        'structured_output_format': turn.get('structured_output_format'),
                        'golden_answer_mode': 'golden_answer',
                        'success_criteria': {},
                        # Common (shared across all turns)
                        'csv_file_name': data.get('csv_file_name'),
                        'vision_enabled': data.get('vision_enabled', False),
                        'image_column': data.get('image_column'),
                        'latency_only_mode': data.get('latency_only_mode', False),
                        'stream_evaluation': data.get('stream_evaluation', True),
                        'prompt_optimization_mode': data.get('prompt_optimization_mode', 'none'),
                    # APO (Advanced Prompt Optimization) — fields are ignored
                    # when prompt_optimization_mode == 'none'.
                    'apo_evaluator': data.get('apo_evaluator'),
                    'apo_llmj_rubric': data.get('apo_llmj_rubric', ''),
                    'apo_llmj_judge_model': data.get('apo_llmj_judge_model', ''),
                    'apo_steering_criteria': data.get('apo_steering_criteria', []),
                        'selected_models': data.get('selected_models', []),
                        'judge_models': data.get('judge_models', []),
                        'eval_mode': data.get('eval_mode', 'bundled'),
                        'metric_assignments': data.get('metric_assignments', {}),
                        'custom_metrics': data.get('custom_metrics', []),
                        'parallel_calls': data.get('parallel_calls', DEFAULT_PARALLEL_CALLS),
                        'invocations_per_scenario': data.get('invocations_per_scenario', DEFAULT_INVOCATIONS_PER_SCENARIO),
                        'sleep_between_invocations': data.get('sleep_between_invocations', DEFAULT_SLEEP_BETWEEN_INVOCATIONS),
                        'experiment_counts': data.get('experiment_counts', DEFAULT_EXPERIMENT_COUNTS),
                        'temperature_variations': data.get('temperature_variations', DEFAULT_TEMPERATURE_VARIATIONS),
                        'failure_threshold': data.get('failure_threshold', DEFAULT_FAILURE_THRESHOLD),
                        'experiment_wait_time': data.get('experiment_wait_time', 0),
                    },
                }
                item = db.put_evaluation(g.user_id, eval_id, eval_config)
                created_evaluations.append({
                    "id": eval_id,
                    "name": eval_name,
                    "status": "configuring",
                    "progress": 0,
                    "created_at": item['created_at'],
                    **eval_config['config'],
                })

            # Clean up temp uploads
            temp_prefix = s3._user_prefix(g.user_id, "uploads/temp_")
            for key in s3.list_objects(temp_prefix):
                s3.delete_object(key)

            return jsonify({"success": True, "evaluations": created_evaluations})

        # --- Single-shot: one row per task (existing behavior) ---
        task_evaluations = data.get("task_evaluations", [{"task_type": "", "task_criteria": "", "temperature": 0.7}])

        for i, task_eval in enumerate(task_evaluations):
            eval_id = str(uuid.uuid4())
            eval_name = f"{base_name}_{i+1}" if len(task_evaluations) > 1 else base_name

            # Move CSV from temp to permanent location in S3
            csv_s3_key = None
            if temp_s3_key and s3.object_exists(temp_s3_key):
                csv_data = s3.download_bytes(temp_s3_key)
                csv_s3_key = s3.upload_bytes(
                    g.user_id,
                    f"uploads/{eval_id}/source_data.csv",
                    csv_data
                )

            eval_config = {
                'eval_name': eval_name,
                'csv_s3_key': csv_s3_key or '',
                'config': {
                    # Mode
                    'evaluation_mode': 'single_shot',
                    # Task-specific
                    'task_type': task_eval.get('task_type', ''),
                    'task_criteria': task_eval.get('task_criteria', ''),
                    'temperature': task_eval.get('temperature', 0.7),
                    'user_defined_metrics': task_eval.get('user_defined_metrics', ''),
                    'structured_output_format': task_eval.get('structured_output_format'),
                    # Common
                    'csv_file_name': data.get('csv_file_name'),
                    'prompt_column': data.get('prompt_column'),
                    'golden_answer_column': data.get('golden_answer_column'),
                    'golden_answer_mode': data.get('golden_answer_mode', 'golden_answer'),
                    'success_criteria': data.get('success_criteria', {}),
                    'vision_enabled': data.get('vision_enabled', False),
                    'image_column': data.get('image_column'),
                    'latency_only_mode': data.get('latency_only_mode', False),
                    'stream_evaluation': data.get('stream_evaluation', True),
                    'prompt_optimization_mode': data.get('prompt_optimization_mode', 'none'),
                    # APO (Advanced Prompt Optimization) — fields are ignored
                    # when prompt_optimization_mode == 'none'.
                    'apo_evaluator': data.get('apo_evaluator'),
                    'apo_llmj_rubric': data.get('apo_llmj_rubric', ''),
                    'apo_llmj_judge_model': data.get('apo_llmj_judge_model', ''),
                    'apo_steering_criteria': data.get('apo_steering_criteria', []),
                    'selected_models': data.get('selected_models', []),
                    'judge_models': data.get('judge_models', []),
                    # Judge evaluation mode
                    'eval_mode': data.get('eval_mode', 'bundled'),
                    'metric_assignments': data.get('metric_assignments', {}),
                    'custom_metrics': data.get('custom_metrics', []),
                    # Advanced
                    'parallel_calls': data.get('parallel_calls', DEFAULT_PARALLEL_CALLS),
                    'invocations_per_scenario': data.get('invocations_per_scenario', DEFAULT_INVOCATIONS_PER_SCENARIO),
                    'sleep_between_invocations': data.get('sleep_between_invocations', DEFAULT_SLEEP_BETWEEN_INVOCATIONS),
                    'experiment_counts': data.get('experiment_counts', DEFAULT_EXPERIMENT_COUNTS),
                    'temperature_variations': data.get('temperature_variations', DEFAULT_TEMPERATURE_VARIATIONS),
                    'failure_threshold': data.get('failure_threshold', DEFAULT_FAILURE_THRESHOLD),
                    'experiment_wait_time': data.get('experiment_wait_time', 0),
                },
            }

            item = db.put_evaluation(g.user_id, eval_id, eval_config)

            # Return in the same shape the frontend expects
            created_evaluations.append({
                "id": eval_id,
                "name": eval_name,
                "status": "configuring",
                "progress": 0,
                "created_at": item['created_at'],
                **eval_config['config'],
            })

        # Clean up ALL temp files for this user (not just the one used)
        temp_prefix = s3._user_prefix(g.user_id, "uploads/temp_")
        temp_keys = s3.list_objects(temp_prefix)
        for key in temp_keys:
            s3.delete_object(key)

        return jsonify({
            "success": True,
            "evaluations": created_evaluations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/<eval_id>', methods=['GET'])
@require_user
def get_evaluation(eval_id):
    """Get a single evaluation by ID from DynamoDB."""
    try:
        item = db.get_evaluation(g.user_id, eval_id)
        if not item:
            return jsonify({"error": "Evaluation not found"}), 404

        config = item.get('config', {})
        evaluation = {
            "id": item['eval_id'],
            "name": item.get('eval_name', ''),
            "status": item.get('status', 'unknown'),
            "progress": item.get('progress', 0),
            "created_at": item.get('created_at', ''),
            "updated_at": item.get('updated_at', ''),
            "start_time": item.get('start_time'),
            "end_time": item.get('end_time'),
            "duration": item.get('duration'),
            "error": item.get('error'),
            "results": item.get('results_s3_key'),
            **config,
        }
        return jsonify({"evaluation": _decimal_to_native(evaluation)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/<eval_id>', methods=['DELETE'])
@require_user
def delete_evaluation(eval_id):
    """Delete an evaluation and all associated S3 objects."""
    try:
        item = db.get_evaluation(g.user_id, eval_id)
        if not item:
            return jsonify({"error": "Evaluation not found"}), 404

        # Delete all S3 objects for this eval
        s3.delete_prefix(s3._user_prefix(g.user_id, f"uploads/{eval_id}/"))
        s3.delete_prefix(s3._user_prefix(g.user_id, f"results/{eval_id}/"))
        s3.delete_prefix(s3._user_prefix(g.user_id, f"unprocessed/{eval_id}/"))
        s3.delete_prefix(s3._user_prefix(g.user_id, f"logs/{eval_id}/"))

        # Delete DynamoDB record
        db.delete_evaluation(g.user_id, eval_id)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------- Evaluation Execution ---------------

def _decrypt_user_credentials(user_id):
    """Decrypt all stored API keys for a user. Returns dict of provider→key."""
    out = {}
    for provider in ['openai', 'google', 'azure', 'bedrock']:
        encrypted = db.get_credential_encrypted(user_id, provider)
        if encrypted:
            try:
                out[provider] = kms.decrypt_api_key(encrypted)
            except Exception as e:
                app.logger.warning(f"Failed to decrypt {provider} key for {user_id}: {e}")
    return out


def _aws_default_creds_available():
    """OFFLINE: True if boto3's default credential chain resolves.

    When it does, Bedrock models authenticate via SigV4 (the Converse path passes
    only aws_region_name, no api_key) — so a stored Bedrock API key is NOT required
    to run evaluations or generate report summaries. A stored key is still honored
    when present (e.g. a real bearer token for Mantle/short-term access)."""
    try:
        import boto3
        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def _bedrock_auth_available(credentials):
    """Bedrock is usable if a key is stored (bearer) OR AWS default creds resolve (SigV4)."""
    return 'bedrock' in credentials or _aws_default_creds_available()


def _build_and_launch_eval(user_id, eval_id, eval_item, credentials, user_email):
    """Build JSONL/profile artifacts, upload to S3, launch ECS task (or local subprocess).

    On ECS path: sets ecs_task_arn on the DDB record. Status stays 'queued'
    until the worker entrypoint flips it to 'running'.
    Raises on failure; caller resets the eval state.
    """
    from dashboard.utils.csv_processor import convert_to_jsonl
    from dashboard.utils.benchmark_runner import create_model_profiles_jsonl, create_judge_profiles_jsonl
    from dashboard.utils.csv_processor import create_specialist_judge_profiles_jsonl

    config = _decimal_to_native(eval_item.get('config', {}))
    eval_name = eval_item.get('eval_name', eval_id[:8])
    composite_id = f"{eval_id}_{eval_name}"
    csv_s3_key = eval_item.get('csv_s3_key', '')
    if not csv_s3_key:
        raise ValueError("No CSV source attached to this evaluation")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_csv = os.path.join(tmp_dir, 'source_data.csv')
        s3.download_file(csv_s3_key, local_csv)
        csv_df = pd.read_csv(local_csv)

        # Legacy multi-shot chain evals (pre-Option A) are no longer supported.
        # Each turn is now stored as its own single-shot eval at create time.
        if config.get('evaluation_mode') == 'multi_shot' and 'turns' in config:
            raise ValueError(
                "This is a legacy multi-shot chain evaluation. Re-create the "
                "evaluation from Setup → Multi-shot — each turn now runs as a "
                "standalone evaluation."
            )
        jsonl_path = convert_to_jsonl(
            df=csv_df,
            prompt_col=config.get('prompt_column', 'prompt'),
            golden_answer_col=config.get('golden_answer_column', 'golden_answer'),
            task_type=config.get('task_type', ''),
            task_criteria=config.get('task_criteria', ''),
            output_dir=tmp_dir,
            name=eval_name,
            temperature=config.get('temperature', 0.7),
            user_defined_metrics=config.get('user_defined_metrics', ''),
            vision_enabled=config.get('vision_enabled', False),
            image_column=config.get('image_column'),
            structured_output_format=config.get('structured_output_format'),
            golden_answer_mode=config.get('golden_answer_mode', 'golden_answer'),
            success_criteria=config.get('success_criteria'),
        )

        model_file_path = create_model_profiles_jsonl(
            config.get('selected_models', []), tmp_dir,
            custom_filename=f"model_profiles_{composite_id}.jsonl")
        eval_mode = config.get('eval_mode', 'bundled')
        if eval_mode == 'specialist':
            judge_file_path = create_specialist_judge_profiles_jsonl(
                config.get('metric_assignments', {}),
                config.get('custom_metrics', []),
                tmp_dir,
                custom_filename=f"judge_profiles_{composite_id}.jsonl")
        else:
            judge_file_path = create_judge_profiles_jsonl(
                config.get('judge_models', []), tmp_dir,
                custom_filename=f"judge_profiles_{composite_id}.jsonl")

        s3_prefix = f"uploads/{eval_id}"
        for fpath in [jsonl_path, model_file_path, judge_file_path]:
            if fpath and os.path.exists(fpath):
                fname = os.path.basename(fpath)
                with open(fpath, 'rb') as fobj:
                    s3.upload_file(user_id, f"{s3_prefix}/{fname}", fobj)

    cli_args = [
        '--parallel_calls', str(config.get('parallel_calls', DEFAULT_PARALLEL_CALLS)),
        '--invocations_per_scenario', str(config.get('invocations_per_scenario', DEFAULT_INVOCATIONS_PER_SCENARIO)),
        '--sleep_between_invocations', str(config.get('sleep_between_invocations', DEFAULT_SLEEP_BETWEEN_INVOCATIONS)),
        '--experiment_counts', str(config.get('experiment_counts', DEFAULT_EXPERIMENT_COUNTS)),
        '--experiment_name', composite_id,
        '--temperature_variations', str(config.get('temperature_variations', DEFAULT_TEMPERATURE_VARIATIONS)),
        '--experiment_wait_time', str(config.get('experiment_wait_time', 0)),
        '--model_file_name', f"model_profiles_{composite_id}.jsonl",
        '--judge_file_name', f"judge_profiles_{composite_id}.jsonl",
        '--evaluation_pass_threshold', str(config.get('failure_threshold', DEFAULT_FAILURE_THRESHOLD)),
        '--stream_evaluation', str(config.get('stream_evaluation', True)),
        '--report', 'False',
    ]
    if config.get('user_defined_metrics'):
        cli_args.extend(['--user_defined_metrics', config['user_defined_metrics']])
    if config.get('vision_enabled'):
        cli_args.extend(['--vision_enabled', 'True'])
    if config.get('prompt_optimization_mode', 'none') != 'none':
        cli_args.extend(['--prompt_optimization_mode', config['prompt_optimization_mode']])
        # APO config — flows to the worker only when prompt opt is enabled.
        cli_args.extend(['--eval_id', eval_id])
        # OFFLINE: Bedrock APO can only read/write a REAL S3 bucket, so it must be
        # supplied via APO_BUCKET (the local S3_BUCKET is just a label here).
        apo_bucket = os.environ.get('APO_BUCKET', '').strip()
        if not apo_bucket:
            raise RuntimeError(
                'Prompt optimization (APO) is enabled but APO_BUCKET is not set. '
                'APO requires a real AWS S3 bucket (plus AWS creds/region). Set '
                'APO_BUCKET in the environment, or disable prompt optimization.'
            )
        cli_args.extend(['--apo_bucket', apo_bucket])
        apo_evaluator = config.get('apo_evaluator')
        if apo_evaluator:
            cli_args.extend(['--apo_evaluator', apo_evaluator])
        if config.get('apo_llmj_rubric'):
            cli_args.extend(['--apo_llmj_rubric', config['apo_llmj_rubric']])
        if config.get('apo_llmj_judge_model'):
            cli_args.extend(['--apo_llmj_judge_model', config['apo_llmj_judge_model']])
        criteria = config.get('apo_steering_criteria') or []
        if criteria:
            cli_args.extend(['--apo_steering_criteria', json.dumps(criteria)])
    if config.get('latency_only_mode'):
        cli_args.extend(['--latency_only_mode', 'True'])

    if LOCAL_DEV_MODE:
        _launch_local_eval(user_id, eval_id, eval_name, composite_id,
                           config, credentials, cli_args)
    else:
        s3_input_prefix = s3._user_prefix(user_id, f"uploads/{eval_id}")
        s3_output_prefix = s3._user_prefix(user_id, f"results/{eval_id}")
        task_arn = ecs.launch_eval_task(
            user_id=user_id, eval_id=eval_id, eval_name=eval_name,
            s3_bucket=s3.S3_BUCKET,
            s3_input_prefix=s3_input_prefix, s3_output_prefix=s3_output_prefix,
            user_email=user_email or f"{user_id}@amazon.com",
            credentials=credentials, cli_args=cli_args,
        )
        # Replace the 'PENDING' sentinel with the real ARN
        db.update_evaluation(user_id, eval_id, status='queued', ecs_task_arn=task_arn)


def _try_launch_next_for_user(user_id, credentials=None, user_email=None):
    """Launch the next queued evaluation for user, if nothing is currently in flight.

    Returns the launched eval_id or None.
    Per-user FIFO ordered by created_at. Uses an atomic DDB claim to prevent
    multiple poller threads (or run_evaluations + poller racing) from launching
    the same eval twice.
    """
    from datetime import datetime, timezone, timedelta
    items = db.get_evaluations(user_id)

    # Anything actively executing → don't start another.
    for it in items:
        status = it.get('status')
        if status == 'running':
            return None
        if status == 'queued' and it.get('ecs_task_arn'):
            # Either ECS is still spinning up the worker, or a prior launch crashed.
            try:
                ts = it.get('updated_at', '')
                updated = datetime.fromisoformat(ts.replace('Z', '+00:00')) if ts else None
            except Exception:
                updated = None
            # Stuck PENDING (claim crashed mid-launch) → mark failed and continue.
            if (it.get('ecs_task_arn') == 'PENDING'
                    and updated
                    and datetime.now(timezone.utc) - updated > timedelta(minutes=2)):
                db.update_evaluation(user_id, it['eval_id'],
                                     status='pre_eval_failed',
                                     error='Launch stalled (claim held >2 min without ECS task)')
                continue
            # Real ARN but no 'running' yet — give worker time to start.
            if updated and datetime.now(timezone.utc) - updated > timedelta(minutes=5):
                db.update_evaluation(user_id, it['eval_id'],
                                     status='pre_eval_failed',
                                     error='ECS task did not transition to running within 5 minutes')
                continue
            return None  # Recent launch in progress, wait.

    # Find candidates: queued + no task arn (truly waiting).
    candidates = [i for i in items
                  if i.get('status') == 'queued' and not i.get('ecs_task_arn')]
    candidates.sort(key=lambda i: i.get('created_at', ''))
    if not candidates:
        return None

    # Decrypt credentials on demand (poller path).
    if credentials is None:
        credentials = _decrypt_user_credentials(user_id)
    if not _bedrock_auth_available(credentials):
        # No Bedrock auth at all (no stored key AND no AWS default creds for SigV4)
        # — fail queued evals so they don't pile up.
        for c in candidates:
            db.update_evaluation(user_id, c['eval_id'],
                                 status='pre_eval_failed',
                                 error='No Bedrock authentication: add a Bedrock API key '
                                       'in Credentials, or configure AWS default credentials '
                                       '(~/.aws) for SigV4.')
        return None

    for cand in candidates:
        eval_id = cand['eval_id']
        if not db.claim_eval_for_launch(user_id, eval_id):
            continue  # Lost race to another poller.
        try:
            _build_and_launch_eval(user_id, eval_id, cand, credentials, user_email)
            return eval_id
        except Exception as e:
            app.logger.error(f"Launch failed for {user_id}/{eval_id}: {e}", exc_info=True)
            db.update_evaluation(user_id, eval_id,
                                 status='pre_eval_failed',
                                 error=f'Launch failed: {e}',
                                 ecs_task_arn=None)
            # Try next candidate so one bad eval doesn't block the queue.
            continue
    return None


# --- Background queue poller ---

QUEUE_POLL_INTERVAL_SEC = int(os.environ.get('QUEUE_POLL_INTERVAL_SEC', '30'))
_queue_poller_started = False
_queue_poller_lock = None  # set in _start_queue_poller


def _drain_queues():
    """Iterate users with queued evals and try to launch the next one for each."""
    try:
        queued_items = db.scan_evaluations_by_status(['queued'])
    except Exception as e:
        app.logger.warning(f"Queue poller scan failed: {e}")
        return
    users = sorted({it['user_id'] for it in queued_items if not it.get('ecs_task_arn')})
    for uid in users:
        try:
            _try_launch_next_for_user(uid)
        except Exception as e:
            app.logger.warning(f"Queue poller failed for user {uid}: {e}", exc_info=True)


def _start_queue_poller():
    """Idempotently start the background queue poller thread."""
    global _queue_poller_started, _queue_poller_lock
    import threading
    if _queue_poller_lock is None:
        _queue_poller_lock = threading.Lock()
    with _queue_poller_lock:
        if _queue_poller_started:
            return
        _queue_poller_started = True

    def _loop():
        import time as _t
        # Small initial jitter so multiple gunicorn workers don't all hit DDB at the same instant
        import random
        _t.sleep(random.uniform(0, QUEUE_POLL_INTERVAL_SEC))
        while True:
            try:
                _drain_queues()
            except Exception as e:
                app.logger.warning(f"Queue poller iteration error: {e}")
            _t.sleep(QUEUE_POLL_INTERVAL_SEC)

    import threading as _th
    t = _th.Thread(target=_loop, daemon=True, name='queue-poller')
    t.start()
    app.logger.info(f"Queue poller started (interval={QUEUE_POLL_INTERVAL_SEC}s)")


# Start poller as soon as the module is imported (works under gunicorn).
_start_queue_poller()


@app.route('/api/evaluations/run', methods=['POST'])
@require_user
def run_evaluations():
    """Queue selected evaluations. The first eligible one launches immediately;
    the rest are picked up by the background poller in FIFO order per user."""
    try:
        data = request.json
        eval_ids = data.get("evaluation_ids", [])

        if not eval_ids:
            return jsonify({"error": "No evaluations selected"}), 400

        credentials = _decrypt_user_credentials(g.user_id)

        # Require a Bedrock API key before queueing (fail fast). The Bedrock key also
        # authenticates OpenAI/Mantle models, so it's the only key needed for AWS models.
        if not _bedrock_auth_available(credentials):
            return jsonify({
                "error": "No Bedrock authentication available. Either add a Bedrock API key "
                         "in the Credentials tab, or configure AWS default credentials "
                         "(~/.aws / environment / SSO) so evaluations can authenticate to "
                         "Bedrock via SigV4. Note: a short-term API key only works in the "
                         "region where it was generated."
            }), 400

        queued = []
        skipped = []
        for eval_id in eval_ids:
            item = db.get_evaluation(g.user_id, eval_id)
            if not item:
                skipped.append({"eval_id": eval_id, "reason": "not found"})
                continue
            status = item.get('status')
            if status in ('queued', 'running'):
                skipped.append({"eval_id": eval_id, "reason": f"already {status}"})
                continue
            # Mark as queued; clear any leftover ecs_task_arn so it's eligible.
            db.update_evaluation(g.user_id, eval_id,
                                 status='queued',
                                 ecs_task_arn=None,
                                 error=None,
                                 progress=0)
            queued.append(eval_id)

        # Kick off the first eligible eval immediately (poller picks up the rest).
        launched_now = _try_launch_next_for_user(g.user_id, credentials=credentials,
                                                 user_email=g.user_email)

        return jsonify({
            "success": True,
            "queued": queued,
            "skipped": skipped,
            "launched_now": launched_now,
            "message": (f"Queued {len(queued)} evaluation(s); "
                        f"{'launched first one immediately' if launched_now else 'waiting for current eval to finish'}"),
            "evaluation_ids": queued,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/queue-status', methods=['GET'])
@require_user
def get_queue_status():
    """Get current evaluation queue status from DynamoDB."""
    try:
        items = db.get_evaluations(g.user_id)
        running = [_decimal_to_native(i) for i in items if i.get('status') == 'running']
        queued = [_decimal_to_native(i) for i in items if i.get('status') == 'queued']

        current = None
        if running:
            r = running[0]
            current = {
                "id": r['eval_id'],
                "name": r.get('eval_name', ''),
                "status": "running",
                "progress": r.get('progress', 0),
                "status_message": r.get('status_message', ''),
            }

        # Keys match what monitor.js reads (current_evaluation / queued_evaluations).
        return jsonify({
            "current_evaluation": current,
            "queued_evaluations": [{"id": q['eval_id'], "name": q.get('eval_name', '')} for q in queued],
            "queue_length": len(queued),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/validate-models', methods=['POST'])
@require_user
def validate_models():
    """Validate model availability."""
    try:
        from model_capability_validator import validate_all_models, is_cache_valid

        data = request.json
        force_refresh = data.get("force_refresh", False)

        cache_valid = is_cache_valid()

        if force_refresh or not cache_valid:
            results = validate_all_models()
            return jsonify({
                "success": True,
                "results": results,
                "cache_valid": False,
                "refreshed": True
            })
        else:
            from model_capability_validator import load_capability_cache
            cache = load_capability_cache()
            return jsonify({
                "success": True,
                "results": cache.get("capabilities", {}),
                "cache_valid": True,
                "refreshed": False,
                "last_updated": cache.get("last_updated")
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------- Reports ---------------

@app.route('/api/reports', methods=['GET'])
@require_user
def get_reports():
    """Get all reports for the current user from DynamoDB."""
    try:
        items = db.get_reports(g.user_id)
        reports = []
        now = datetime.now(timezone.utc)
        for item in items:
            # Legacy rows (created before async generation) have no status — treat as completed.
            status = item.get('status', 'completed')

            created = item.get('created_at', '')
            # Stale guard: a 'generating' report whose worker thread died (e.g. the API
            # container restarted mid-generation) would hang forever — fail it past the timeout.
            if status == 'generating':
                try:
                    age = (now - datetime.fromisoformat(created)).total_seconds()
                except Exception:
                    age = 0
                if age > REPORT_GEN_TIMEOUT_SEC:
                    status = 'failed'
                    try:
                        db.update_report(g.user_id, item['report_id'],
                                         status='failed', error='Report generation timed out')
                    except Exception:
                        pass

            s3_key = item.get('s3_key', '')
            file_size = "Unknown"
            # Only completed reports have an object to size — skip the S3 head otherwise.
            if s3_key and status == 'completed':
                size_bytes = s3.get_object_size(s3_key)
                file_size = _format_file_size(size_bytes) if size_bytes else "Unknown"

            try:
                dt = datetime.fromisoformat(created)
                creation_time_formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                creation_time_formatted = created

            reports.append({
                "report_id": item['report_id'],
                "status_file": item['report_id'],  # Compat — used as identifier for delete
                "status": status,
                "report_path": s3_key,
                "html_path": s3_key,
                "report_name": item.get('report_name', ''),
                "error": item.get('error', ''),
                "creation_time": created,
                "creation_time_formatted": creation_time_formatted,
                "evaluations_used": item.get('evaluations_used', []),
                "models_included": item.get('models_included', []),
                "file_size": file_size,
            })

        reports.sort(key=lambda x: x.get("creation_time", ""), reverse=True)
        return jsonify({
            "reports": _decimal_to_native(reports),
            "report_count": len(reports),
            "report_limit": db.MAX_REPORTS_PER_USER,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# A 'generating' report whose worker thread dies (e.g. API container restart) would
# otherwise hang forever — the list endpoint marks it failed past this age.
REPORT_GEN_TIMEOUT_SEC = int(os.environ.get('REPORT_GEN_TIMEOUT_SEC', '1800'))


def _run_report_generation(user_id, report_id, report_name, timestamp,
                           selected_evaluations, selected_model_ids,
                           summary_model, summary_region, selected_sections,
                           bedrock_api_key):
    """Build the HTML report in a background thread, then flip the report row to
    completed/failed. Runs off the request thread so the HTTP handler returns
    immediately instead of blocking a gunicorn worker for minutes."""
    import threading

    def _run():
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                items = db.get_evaluations(user_id)
                downloaded_any = False
                for item in items:
                    if item.get('status') != 'completed':
                        continue
                    eval_name = item.get('eval_name', '')
                    if selected_evaluations and eval_name not in selected_evaluations:
                        continue
                    result_prefix = s3._user_prefix(user_id, f"results/{item['eval_id']}/")
                    for key in s3.list_objects(result_prefix):
                        if key.endswith('.csv'):
                            s3.download_file(key, os.path.join(tmp_dir, os.path.basename(key)))
                            downloaded_any = True

                if not downloaded_any:
                    db.update_report(user_id, report_id, status='failed',
                                     error='No completed evaluation results found to generate report from')
                    return

                # Backfill model inference errors from unprocessed JSONs. Accumulate
                # ALL error rows first, then do ONE read + concat + write — the previous
                # version re-read and rewrote the growing CSV once per unprocessed key
                # (O(n^2) disk/parse).
                import pandas as pd
                error_rows = []
                for item in items:
                    if item.get('status') != 'completed':
                        continue
                    eval_name = item.get('eval_name', '')
                    if selected_evaluations and eval_name not in selected_evaluations:
                        continue
                    for ukey in item.get('unprocessed_s3_keys', []):
                        if not ukey.endswith('.json'):
                            continue
                        try:
                            records = json.loads(s3.download_bytes(ukey))
                            if isinstance(records, dict):
                                records = [records]
                            for rec in records:
                                if not (isinstance(rec, dict) and rec.get('error_classification') == 'api_failure'):
                                    continue
                                row = {**rec.get('scenario', {}), **rec.get('result', {})}
                                row['model_response'] = ''
                                row['performance_metrics'] = '{}'
                                row['evaluation_cost'] = 0
                                row['error_classification'] = 'api_failure'
                                error_rows.append(row)
                        except Exception as e:
                            print(f"Could not backfill errors from {ukey}: {e}")

                if error_rows:
                    csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
                    if csv_files:
                        target_csv = os.path.join(tmp_dir, csv_files[0])
                        merged = pd.concat([pd.read_csv(target_csv), pd.DataFrame(error_rows)],
                                           ignore_index=True)
                        merged.to_csv(target_csv, index=False)

                report_file = create_html_report(
                    output_dir=Path(tmp_dir),
                    timestamp=timestamp,
                    evaluation_names=selected_evaluations,
                    model_ids=selected_model_ids,
                    bedrock_api_key=bedrock_api_key,
                    summary_model=summary_model,
                    summary_region=summary_region,
                    selected_sections=selected_sections,
                )
                if not report_file:
                    db.update_report(user_id, report_id, status='failed', error='Failed to generate report')
                    return

                with open(report_file, 'rb') as f:
                    report_s3_key = s3.upload_file(
                        user_id, f"reports/{report_id}/{Path(report_file).name}", f)

                db.update_report(user_id, report_id, status='completed', s3_key=report_s3_key)
                print(f"[REPORT] {report_id} completed")
        except Exception as e:
            import traceback
            print(f"[REPORT] {report_id} failed: {e}\n{traceback.format_exc()}")
            try:
                db.update_report(user_id, report_id, status='failed',
                                 error=f"{type(e).__name__}: {str(e)}")
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()


@app.route('/api/reports/generate', methods=['POST'])
@require_user
def generate_report():
    """Kick off async HTML report generation. Returns immediately with a report_id;
    the report row's status flips 'generating' -> 'completed'/'failed' in the
    background. The Reports tab polls the list until the report is ready."""
    try:
        # Check report cap up-front (the 'generating' row created below also counts
        # toward the cap, so put_report re-checks it).
        report_count = db.count_reports(g.user_id)
        if report_count >= db.MAX_REPORTS_PER_USER:
            return jsonify({
                "error": f"Report limit reached ({db.MAX_REPORTS_PER_USER}). Delete existing reports before generating new ones."
            }), 400

        data = request.json or {}
        selected_evaluations = data.get("selected_evaluations")
        selected_model_ids = data.get("selected_model_ids")
        summary_model = data.get("summary_model", "bedrock/global.amazon.nova-2-lite-v1:0")
        summary_region = data.get("summary_region", "us-east-1")
        selected_sections = data.get("selected_sections")  # None = all sections

        # OFFLINE: the report's executive-summary model is a Bedrock model, which
        # authenticates via SigV4 (AWS default creds) — same as eval inference. We do
        # NOT pass a stored Bedrock key here: it isn't needed, and a placeholder/expired
        # key would be used as a bearer token and hang the summary call. Force SigV4.
        bedrock_api_key = None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_id = str(uuid.uuid4())
        report_name = f"Report_{timestamp}"

        # Create the report row in 'generating' state before returning.
        db.put_report(g.user_id, report_id, {
            'report_name': report_name,
            'status': 'generating',
            's3_key': '',
            'evaluations_used': selected_evaluations or ['All'],
            'models_included': selected_model_ids or ['All'],
        })

        _run_report_generation(
            g.user_id, report_id, report_name, timestamp,
            selected_evaluations, selected_model_ids,
            summary_model, summary_region, selected_sections,
            bedrock_api_key,
        )

        return jsonify({
            "success": True,
            "status": "generating",
            "report_id": report_id,
            "report_name": report_name,
        }), 202
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        import traceback
        app.logger.error(f"Report generation kickoff failed: {type(e).__name__}: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


@app.route('/api/evaluations/<eval_id>/apo', methods=['GET'])
@require_user
def list_apo_artifacts(eval_id):
    """List the APO (optimized-prompt) artifacts for one of the caller's evaluations.

    Artifacts live at users/<user>/results/<eval_id>/apo/ — built from g.user_id so a
    user can only ever see their own.
    """
    try:
        prefix = s3._user_prefix(g.user_id, f"results/{eval_id}/apo/")
        artifacts = []
        for key in s3.list_objects(prefix):
            name = key.rsplit('/', 1)[-1]
            if not name:
                continue
            size = s3.get_object_size(key)
            artifacts.append({
                "name": name,
                "size": size,
                "size_human": _format_file_size(size) if size else "Unknown",
            })
        artifacts.sort(key=lambda a: a["name"])
        return jsonify({"artifacts": artifacts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/<eval_id>/apo/<path:filename>', methods=['GET'])
@require_user
def download_apo_artifact(eval_id, filename):
    """Download a single APO artifact as an attachment (ownership + traversal safe)."""
    try:
        # Only a bare filename within the eval's apo/ dir — block path traversal.
        if filename != os.path.basename(filename) or filename in ('', '.', '..'):
            return jsonify({"error": "Invalid filename"}), 400
        key = s3._user_prefix(g.user_id, f"results/{eval_id}/apo/{filename}")
        if not s3.object_exists(key):
            return jsonify({"error": "Artifact not found"}), 404
        content = s3.download_bytes(key)
        ctype = ("text/csv" if filename.endswith(".csv")
                 else "application/json" if filename.endswith(".json")
                 else "text/plain")
        return content, 200, {
            "Content-Type": f"{ctype}; charset=utf-8",
            "Content-Disposition": f'attachment; filename="{filename}"',
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reports/<path:report_path>', methods=['GET'])
@require_user
def get_report_content(report_path):
    """Get the content of an HTML report by streaming from S3."""
    try:
        # Verify report belongs to user by checking if key starts with user prefix
        user_prefix = s3._user_prefix(g.user_id)
        if not report_path.startswith(user_prefix):
            return jsonify({"error": "Report not found"}), 404

        if not s3.object_exists(report_path):
            return jsonify({"error": "Report not found"}), 404

        # Stream the HTML content directly instead of redirecting to S3
        content = s3.download_bytes(report_path)
        return content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reports/delete', methods=['POST'])
@require_user
def delete_report():
    """Delete a report from S3 and DynamoDB."""
    try:
        data = request.json
        report_id = data.get("status_file") or data.get("report_id")

        if not report_id:
            return jsonify({"error": "No report ID provided"}), 400

        # Get report to find S3 key
        reports = db.get_reports(g.user_id)
        target = None
        for r in reports:
            if r['report_id'] == report_id:
                target = r
                break

        if not target:
            return jsonify({"error": "Report not found"}), 404

        # Delete S3 objects
        s3.delete_prefix(s3._user_prefix(g.user_id, f"reports/{report_id}/"))

        # Delete DynamoDB record
        db.delete_report(g.user_id, report_id)

        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Report generation failed: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# --------------- Unprocessed Records ---------------

@app.route('/api/unprocessed', methods=['GET'])
@require_user
def get_unprocessed():
    """Get unprocessed summary — lightweight, no S3 downloads.
    Returns per-evaluation summaries from DynamoDB metadata only."""
    try:
        evals = db.get_evaluations(g.user_id)
        eval_summaries = []
        total_files = 0

        for ev in evals:
            ukeys = ev.get('unprocessed_s3_keys', [])
            if not ukeys:
                continue
            json_keys = [k for k in ukeys if k.endswith('.json')]
            if not json_keys:
                continue
            total_files += len(json_keys)
            config = ev.get('config', {})
            eval_summaries.append({
                'eval_id': ev.get('eval_id', ''),
                'eval_name': ev.get('eval_name', 'Unknown'),
                'file_count': len(json_keys),
                's3_keys': json_keys,
                'status': ev.get('status', ''),
                'created_at': ev.get('created_at', ''),
                'task_type': config.get('task_type', '') if isinstance(config, dict) else '',
            })

        # Return lightweight summary — records loaded on demand via /api/unprocessed/<eval_id>
        return jsonify({
            "files": [],
            "records": [],
            "error_data": [],
            "eval_summaries": eval_summaries,
            "summary": {
                "total_files": total_files,
                "total_records": 0,  # Not counted until detail load
                "affected_experiments": len(eval_summaries),
                "affected_models": 0,
                "affected_task_types": 0
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/unprocessed/<eval_id>', methods=['GET'])
@require_user
def get_unprocessed_detail(eval_id):
    """Load full unprocessed records for a specific evaluation — on-demand."""
    try:
        ev = db.get_evaluation(g.user_id, eval_id)
        if not ev:
            return jsonify({"error": "Evaluation not found"}), 404

        ukeys = ev.get('unprocessed_s3_keys', [])
        json_keys = [k for k in ukeys if k.endswith('.json')]
        if not json_keys:
            return jsonify({"records": [], "summary": {}})

        all_records = []
        for key in json_keys:
            try:
                raw = s3.download_bytes(key)
                data = json.loads(raw)
                filename = os.path.basename(key)

                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    continue

                eval_name = ev.get('eval_name', 'Unknown')
                for record in data:
                    if isinstance(record, dict):
                        record['_file'] = filename
                        record['_experiment_name'] = eval_name
                        all_records.append(record)
            except Exception as e:
                print(f"Could not parse {key}: {str(e)}")

        unique_models = len(set(r.get('scenario', {}).get('model_id', 'Unknown') for r in all_records))
        unique_tasks = len(set(r.get('scenario', {}).get('task_types', 'Unknown') for r in all_records))

        return jsonify({
            "records": all_records,
            "summary": {
                "total_records": len(all_records),
                "affected_models": unique_models,
                "affected_task_types": unique_tasks
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------- Credentials ---------------

@app.route('/api/credentials', methods=['GET'])
@require_user
def get_credentials():
    """Get saved credential providers (masked keys) and notification email for the current user."""
    try:
        creds = db.get_credentials(g.user_id)
        # Get notification email from credentials (stored as provider='notification_email')
        notif = db.get_credential_encrypted(g.user_id, 'notification_email')
        notification_email = notif.decode('utf-8') if isinstance(notif, bytes) else notif if notif else None
        # OFFLINE: evals can run via SigV4 (AWS default creds) without a stored Bedrock
        # key. Tell the UI when Bedrock auth is available so it doesn't force a key.
        bedrock_stored = any(c.get('provider') == 'bedrock' for c in creds)
        bedrock_ready = bedrock_stored or _aws_default_creds_available()
        return jsonify({
            "credentials": creds,
            "notification_email": notification_email,
            "bedrock_ready": bedrock_ready,
            "bedrock_auth_mode": "key" if bedrock_stored else ("sigv4" if bedrock_ready else "none"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _test_credential(provider, api_key, region):
    """Test a single API credential with a minimal inference call. Returns (success, error_msg)."""
    import litellm
    litellm.drop_params = True
    try:
        params = {"max_tokens": 1, "timeout": 10}
        messages = [{"role": "user", "content": "hi"}]

        if provider == 'bedrock':
            test_region = region if region and region != 'N/A' else 'us-east-1'
            params["aws_region_name"] = test_region
            params["api_key"] = api_key
            model = f"bedrock/us.amazon.nova-2-lite-v1:0"
        elif provider == 'google':
            params["api_key"] = api_key
            model = "gemini/gemini-2.0-flash"
        elif provider == 'openai':
            params["api_key"] = api_key
            model = "openai/gpt-4o-mini"
        elif provider == 'azure':
            params["api_key"] = api_key
            model = "azure/gpt-4o-mini"
        else:
            return True, ""

        litellm.completion(model=model, messages=messages, **params)
        return True, ""
    except Exception as e:
        error_str = str(e).lower()
        if 'expired' in error_str or 'invalid' in error_str or 'unauthorized' in error_str or 'bearer token' in error_str:
            region_info = f" in region {region}" if provider == 'bedrock' and region != 'N/A' else ""
            return False, f"Your {provider} API key is expired or invalid{region_info}. Please update it in the Credentials tab."
        elif 'access denied' in error_str or 'not authorized' in error_str:
            region_info = f" for region {region}" if provider == 'bedrock' else ""
            return False, f"Your {provider} API key does not have access{region_info}. Check permissions."
        else:
            region_info = f" (region: {region})" if provider == 'bedrock' and region != 'N/A' else ""
            return False, f"{provider} credential check failed{region_info}: {str(e)[:150]}"


@app.route('/api/validate-credentials', methods=['POST'])
@require_user
def validate_credentials():
    """Validate API credentials for selected models before saving configuration."""
    try:
        from concurrent.futures import ThreadPoolExecutor

        data = request.json
        models = data.get('models', [])

        # OFFLINE: normal Bedrock (Converse) models authenticate via SigV4 (AWS default
        # creds) and need NO stored key. Mantle (OpenAI-on-Bedrock / Responses API)
        # models DO need the Bedrock API key (used as the bearer token). Detect Mantle
        # from the catalog endpoint markers — the UI doesn't pass the endpoint field.
        try:
            from dashboard.utils.constants import generate_model_info
            _endpoint_map = generate_model_info('models_profiles.jsonl').get('MODEL_ENDPOINT', {})
        except Exception:
            _endpoint_map = {}

        def _is_mantle(mid):
            return any(v.get('endpoint') == 'bedrock_mantle'
                       for (m, _r), v in _endpoint_map.items() if m == mid)

        sigv4 = _aws_default_creds_available()

        # Determine required providers and regions
        required = {}
        for model in models:
            model_id = str(model.get('id', '') or model.get('model_id', ''))
            region = str(model.get('region', '') or '')
            if 'bedrock' in model_id:
                # Mantle → key required; normal Bedrock → only if SigV4 is unavailable.
                if _is_mantle(model_id) or not sigv4:
                    required.setdefault('bedrock', set()).add(region or 'us-east-1')
            elif 'gemini' in model_id:
                required.setdefault('google', set()).add('N/A')
            elif 'azure' in model_id:
                required.setdefault('azure', set()).add('N/A')
            elif 'openai/' in model_id:
                required.setdefault('openai', set()).add('N/A')

        if not required:
            return jsonify({"valid": True, "errors": []})

        # Check for missing credentials first (fast)
        errors = []
        keys = {}
        for provider in required:
            encrypted = db.get_credential_encrypted(g.user_id, provider)
            if not encrypted:
                errors.append(f"No {provider} API key configured. Please add it in the Credentials tab.")
            else:
                keys[provider] = kms.decrypt_api_key(encrypted)

        if errors:
            return jsonify({"valid": False, "errors": errors})

        # Test all provider+region combos in parallel
        checks = []
        for provider, regions in required.items():
            for region in regions:
                checks.append((provider, keys[provider], region))

        with ThreadPoolExecutor(max_workers=min(len(checks), 6)) as pool:
            results = list(pool.map(lambda args: _test_credential(*args), checks))

        errors = [msg for success, msg in results if not success]
        return jsonify({"valid": len(errors) == 0, "errors": errors})
    except Exception as e:
        return jsonify({"valid": False, "errors": [f"Credential validation error: {str(e)}"]}), 500


@app.route('/api/credentials', methods=['POST'])
@require_user
def save_credential():
    """Save an encrypted API key for a provider."""
    try:
        data = request.json
        provider = data.get("provider")
        api_key = data.get("api_key")

        if not provider or not api_key:
            return jsonify({"error": "Provider and api_key are required"}), 400

        valid_providers = ['openai', 'google', 'azure', 'bedrock']
        if provider not in valid_providers:
            return jsonify({"error": f"Invalid provider. Must be one of: {valid_providers}"}), 400

        encrypted = kms.encrypt_api_key(api_key)
        key_alias = kms.mask_api_key(api_key)
        db.put_credential(g.user_id, provider, encrypted, key_alias)

        return jsonify({"success": True, "provider": provider, "key_alias": key_alias})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/credentials/<provider>', methods=['DELETE'])
@require_user
def delete_credential(provider):
    """Delete a saved credential."""
    try:
        db.delete_credential(g.user_id, provider)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------- Notifications ---------------

SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')

@app.route('/api/notifications/subscribe', methods=['POST'])
@require_user
def subscribe_notifications():
    """Subscribe user email to SNS notifications."""
    try:
        data = request.json
        email = data.get('email', '').strip()

        if not email or not email.endswith('@amazon.com'):
            return jsonify({"error": "Only @amazon.com email addresses are accepted"}), 400

        # OFFLINE: no SNS. We store the email so the UI keeps working, but no
        # email is actually delivered (evaluations run locally and never notify).
        db.put_credential(g.user_id, 'notification_email', email, email)

        return jsonify({"success": True, "subscription_arn": "local"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/notifications/unsubscribe', methods=['POST'])
@require_user
def unsubscribe_notifications():
    """Unsubscribe user from SNS notifications."""
    try:
        db.delete_credential(g.user_id, 'notification_email')
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------- Admin ---------------

ADMIN_USERS = os.environ.get('ADMIN_USERS', 'claumazz').split(',')


def _require_admin(f):
    """Decorator that ensures user is in the admin allowlist."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if g.user_id not in ADMIN_USERS:
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated


@app.route('/api/admin/check', methods=['GET'])
@require_user
def admin_check():
    """Check if current user is an admin."""
    return jsonify({"is_admin": g.user_id in ADMIN_USERS})


@app.route('/api/admin/dashboard', methods=['GET'])
@require_user
@_require_admin
def admin_dashboard():
    """Get admin dashboard data — all users, evaluations, stats."""
    try:
        # OFFLINE: read all evaluations/reports from the local SQLite store.
        eval_items = db.scan_all_evaluations()
        report_items = db.scan_all_reports()

        # Build per-user stats
        from collections import Counter, defaultdict
        user_stats = defaultdict(lambda: {
            'evaluations': 0, 'completed': 0, 'failed': 0, 'running': 0, 'queued': 0,
            'reports': 0, 'last_activity': None,
        })

        all_evaluations = []
        active_jobs = []

        for item in eval_items:
            uid = item.get('user_id', 'unknown')
            status = item.get('status', 'unknown')
            user_stats[uid]['evaluations'] += 1
            if status in ('completed', 'failed', 'running', 'queued'):
                user_stats[uid][status] += 1

            created = item.get('created_at', '')
            if created and (not user_stats[uid]['last_activity'] or created > user_stats[uid]['last_activity']):
                user_stats[uid]['last_activity'] = created

            config = item.get('config', {})
            eval_entry = {
                'user_id': uid,
                'eval_id': item.get('eval_id', ''),
                'eval_name': item.get('eval_name', ''),
                'status': status,
                'progress': item.get('progress', 0),
                'created_at': created,
                'duration': item.get('duration'),
                'models_count': len(config.get('selected_models', [])) if isinstance(config, dict) else 0,
                'results_s3_key': item.get('results_s3_key', ''),
                'has_unprocessed': len(item.get('unprocessed_s3_keys', [])) > 0,
            }
            all_evaluations.append(eval_entry)

            if status in ('running', 'queued'):
                active_jobs.append(eval_entry)

        for item in report_items:
            uid = item.get('user_id', 'unknown')
            user_stats[uid]['reports'] += 1

        # Sort evaluations by date (newest first)
        all_evaluations.sort(key=lambda e: e.get('created_at', ''), reverse=True)

        # Platform stats
        from datetime import datetime, timedelta, timezone as tz
        now = datetime.now(tz.utc)
        week_ago = (now - timedelta(days=7)).isoformat()
        month_ago = (now - timedelta(days=30)).isoformat()

        stats = {
            'total_users': len(user_stats),
            'total_evaluations': len(eval_items),
            'total_reports': len(report_items),
            'evals_this_week': sum(1 for e in all_evaluations if e.get('created_at', '') >= week_ago),
            'evals_this_month': sum(1 for e in all_evaluations if e.get('created_at', '') >= month_ago),
            'active_jobs': len(active_jobs),
        }

        # Users list
        users = []
        for uid, s in sorted(user_stats.items()):
            users.append({'user_id': uid, **s})

        return jsonify(_decimal_to_native({
            'stats': stats,
            'users': users,
            'evaluations': all_evaluations[:100],  # Last 100
            'active_jobs': active_jobs,
        }))
    except Exception as e:
        app.logger.error(f"Admin dashboard error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/download/<user_id>/<eval_id>/<file_type>', methods=['GET'])
@require_user
@_require_admin
def admin_download_file(user_id, eval_id, file_type):
    """Download evaluation results or unprocessed files for any user's evaluation."""
    try:
        item = db.get_evaluation(user_id, eval_id)
        if not item:
            return jsonify({"error": "Evaluation not found"}), 404

        if file_type == 'results':
            s3_key = item.get('results_s3_key', '')
            if not s3_key:
                return jsonify({"error": "No results file available"}), 404
            url = s3.generate_presigned_url(s3_key, expiry=300)
            return jsonify({"url": url, "filename": os.path.basename(s3_key)})

        elif file_type == 'unprocessed':
            ukeys = item.get('unprocessed_s3_keys', [])
            if not ukeys:
                return jsonify({"error": "No unprocessed files available"}), 404
            # Return presigned URLs for all unprocessed files
            files = []
            for key in ukeys:
                if key.endswith('.json'):
                    files.append({
                        "url": s3.generate_presigned_url(key, expiry=300),
                        "filename": os.path.basename(key)
                    })
            return jsonify({"files": files})

        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/local-file', methods=['GET'])
@require_user
@_require_admin
def admin_local_file():
    """OFFLINE: stream a stored object by key (replaces S3 presigned URLs).

    Backs the URLs returned by s3.generate_presigned_url in the offline build.
    The key is validated/traversal-guarded inside s3._key_to_path.
    """
    key = request.args.get('key', '')
    try:
        path = s3._key_to_path(key)
    except ValueError:
        return jsonify({"error": "Invalid key"}), 400
    if not s3.object_exists(key):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(key))


@app.route('/api/admin/clone-eval', methods=['POST'])
@require_user
@_require_admin
def admin_clone_eval():
    """Clone another user's evaluation config + CSV into the admin's space."""
    try:
        data = request.json
        source_user = data.get('user_id')
        source_eval_id = data.get('eval_id')

        if not source_user or not source_eval_id:
            return jsonify({"error": "user_id and eval_id required"}), 400

        # Get the source evaluation from DynamoDB
        item = db.get_evaluation(source_user, source_eval_id)
        if not item:
            return jsonify({"error": "Evaluation not found"}), 404

        config = _decimal_to_native(item.get('config', {}))
        eval_name = item.get('eval_name', 'unknown')

        # Copy the source CSV to admin's temp space
        csv_s3_key = item.get('csv_s3_key', '')
        temp_s3_key = None
        columns = []
        preview = []
        csv_file_name = 'cloned_data.csv'

        if csv_s3_key and s3.object_exists(csv_s3_key):
            # Download source CSV
            csv_data = s3.download_bytes(csv_s3_key)
            # Upload to admin's temp space
            temp_id = str(uuid.uuid4())
            temp_s3_key = s3.upload_bytes(
                g.user_id,
                f"uploads/temp_{temp_id}.csv",
                csv_data
            )
            # Parse columns and preview
            df = pd.read_csv(io.BytesIO(csv_data))
            columns = df.columns.tolist()
            preview = df.head(10).to_dict(orient='records')
            csv_file_name = f"cloned_from_{source_user}_{eval_name}.csv"

        return jsonify({
            "success": True,
            "eval_name": eval_name,
            "config": config,
            "temp_s3_key": temp_s3_key,
            "csv_file_name": csv_file_name,
            "columns": columns,
            "preview": preview,
        })
    except Exception as e:
        app.logger.error(f"Clone eval error: {e}")
        return jsonify({"error": str(e)}), 500


# --------------- Static file serving (dev only — CloudFront in prod) ---------------

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    """Serve files from the outputs directory (dev/local only)."""
    return send_from_directory(DEFAULT_OUTPUT_DIR, filename)


if __name__ == '__main__':
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATUS_FILES_DIR, exist_ok=True)

    print(f"Starting 360-eval Web UI (hosted mode)")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"S3 Bucket: {s3.S3_BUCKET}")

    app.run(debug=True, port=5000, host='0.0.0.0')
