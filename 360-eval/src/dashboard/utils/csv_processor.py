"""Utilities for processing CSV files."""

import pandas as pd
import json
import os
from pathlib import Path
from uuid import uuid4


def read_csv_file(uploaded_file):
    """Read an uploaded CSV file and return a pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def get_csv_columns(df):
    """Get a list of column names from a DataFrame."""
    if df is not None:
        return df.columns.tolist()
    return []


def preview_csv_data(df, max_rows=5):
    """Return a preview of the CSV data."""
    if df is not None:
        return df.head(max_rows)
    return None


def convert_to_jsonl(df, prompt_col, golden_answer_col, task_type, task_criteria, output_dir, name, temperature=0.7, user_defined_metrics="", vision_enabled=False, image_column=None, structured_output_format=None, golden_answer_mode='golden_answer', success_criteria=None):
    """
    Convert CSV data to JSONL format for LLM benchmarking.
    
    Args:
        df: Pandas DataFrame with CSV data
        prompt_col: Column name for prompts
        golden_answer_col: Column name for golden answers
        task_type: Type of task for evaluation
        task_criteria: Criteria for evaluating the task
        output_dir: Directory to save the JSONL file
        name: Name for the evaluation
        
    Returns:
        Path to the created JSONL file
    """
    if df is None:
        raise ValueError("Invalid CSV data")
    
    if prompt_col is None:
        raise ValueError("Please select a prompt column")
    if golden_answer_mode != 'criteria_only' and golden_answer_col is None:
        raise ValueError("Please select a golden answer column (or use criteria-only mode)")
    
    # Check vision model requirements
    if vision_enabled and image_column is None:
        raise ValueError("Vision model enabled but no image column selected")
        
    all_columns = df.columns.tolist()
    has_golden_col = golden_answer_col and golden_answer_col in all_columns
    if prompt_col not in all_columns or (golden_answer_mode != 'criteria_only' and not has_golden_col):
        # Check if this is potentially a merged dataframe with different column names
        has_prompt_rows = False
        has_answer_rows = False
        
        for col in all_columns:
            if col == prompt_col or "prompt" in col.lower():
                has_prompt_rows = True
            if col == golden_answer_col or "answer" in col.lower() or "golden" in col.lower():
                has_answer_rows = True
                
        if not (has_prompt_rows and has_answer_rows):
            raise ValueError(f"Selected columns not found in CSV: {prompt_col}, {golden_answer_col}")

    # Use the absolute prompt-evaluations directory path from constants
    from ..utils.constants import PROJECT_ROOT, DEFAULT_PROMPT_EVAL_DIR
    prompt_eval_dir = Path(DEFAULT_PROMPT_EVAL_DIR)
    os.makedirs(prompt_eval_dir, exist_ok=True)
    
    # Generate JSONL file path - use a unique name for merged evaluations
    if "merged" in name:
        unique_suffix = str(uuid4()).split('-')[0]
        jsonl_path = prompt_eval_dir / f"{name}_{unique_suffix}.jsonl"
    else:
        jsonl_path = prompt_eval_dir / f"{name}.jsonl"
    
    # Convert DataFrame to JSONL format
    jsonl_data = []
    for _, row in df.iterrows():
        # Skip rows that don't have the prompt column
        if prompt_col not in row:
            continue

        prompt = row[prompt_col]
        if pd.isna(prompt):
            continue

        # Get golden answer (may be empty in criteria-only mode)
        answer = ""
        if has_golden_col and golden_answer_col in row:
            answer = row[golden_answer_col]
            if pd.isna(answer):
                answer = ""
        elif golden_answer_mode != 'criteria_only':
            continue  # Skip rows without golden answer in golden_answer mode

        entry = {
            "text_prompt": prompt,
            "expected_output_tokens": 4000,
            "task": {
                "task_type": task_type,
                "task_criteria": task_criteria
            },
            "golden_answer": answer,
            "temperature": temperature,
            "user_defined_metrics": user_defined_metrics,
            "structured_output_format": structured_output_format
        }

        # Add success criteria for criteria-only mode
        if golden_answer_mode == 'criteria_only' and success_criteria:
            entry["success_criteria"] = success_criteria

        # Add image data directly using the column name (like prompt and golden_answer)
        if vision_enabled and image_column and image_column in row:
            image_data = row[image_column]
            if not pd.isna(image_data):
                entry[image_column] = image_data
        jsonl_data.append(entry)
    
    # Write to JSONL file
    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry) + '\n')
    except IOError as e:
        raise Exception(f"Failed to write JSONL file to {jsonl_path}: {e}")
    
    # Return both the absolute path and the filename for CLI compatibility
    return str(jsonl_path)


def create_model_profiles_jsonl(models, output_dir, custom_filename=None):
    """
    Create a JSONL file with model profiles.
    
    Args:
        models: List of dictionaries with model configuration
        output_dir: Directory to save the JSONL file
        
    Returns:
        Path to the created JSONL file
    """
    # Use the absolute prompt-evaluations directory path from constants
    from ..utils.constants import PROJECT_ROOT, DEFAULT_PROMPT_EVAL_DIR
    prompt_eval_dir = Path(DEFAULT_PROMPT_EVAL_DIR)
    os.makedirs(prompt_eval_dir, exist_ok=True)
    
    jsonl_path = prompt_eval_dir / (custom_filename or "model_profiles.jsonl")

    # Endpoint markers (Mantle / Responses-API models) live in the authoritative
    # catalog, not in the UI-selected model dicts — look them up by (model_id, region)
    # so the per-eval profile carries them through to the engine's routing.
    try:
        from ..utils.constants import generate_model_info
        endpoint_map = generate_model_info('models_profiles.jsonl').get('MODEL_ENDPOINT', {})
    except Exception:
        endpoint_map = {}

    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for model in models:
                entry = {
                    "model_id": model["id"],
                    "region": model["region"],
                    "input_token_cost": model["input_cost"],
                    "output_token_cost": model["output_cost"]
                }
                # Add target_rpm if it's configured (not None)
                if model.get("target_rpm") is not None:
                    entry["target_rpm"] = model["target_rpm"]

                # Add service_tier if it exists (for benchmark processing)
                if model.get("service_tier") is not None:
                    entry["service_tier"] = model["service_tier"]

                # Carry the Mantle/Responses-API endpoint marker if this model has one.
                ep = endpoint_map.get((model["id"], model["region"]))
                if ep:
                    entry["endpoint"] = ep["endpoint"]
                    entry["mantle_region"] = ep.get("mantle_region", model["region"])

                f.write(json.dumps(entry) + '\n')
    except IOError as e:
        raise Exception(f"Failed to write model profiles to {jsonl_path}: {e}")
    
    return str(jsonl_path)


def create_judge_profiles_jsonl(judges, output_dir, custom_filename=None):
    """
    Create a JSONL file with judge model profiles.
    
    Args:
        judges: List of dictionaries with judge model configuration
        output_dir: Directory to save the JSONL file
        
    Returns:
        Path to the created JSONL file
    """
    # Use the absolute prompt-evaluations directory path from constants
    from ..utils.constants import PROJECT_ROOT, DEFAULT_PROMPT_EVAL_DIR
    prompt_eval_dir = Path(DEFAULT_PROMPT_EVAL_DIR)
    os.makedirs(prompt_eval_dir, exist_ok=True)
    
    jsonl_path = prompt_eval_dir / (custom_filename or "judge_profiles.jsonl")
    
    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for judge in judges:
                entry = {
                    "model_id": judge["id"],
                    "region": judge["region"],
                    "input_token_cost": judge["input_cost"],
                    "output_token_cost": judge["output_cost"]
                }
                f.write(json.dumps(entry) + '\n')
    except (IOError, KeyError) as e:
        raise Exception(f"Failed to write judge profiles to {jsonl_path}: {e}")

    return str(jsonl_path)


def create_specialist_judge_profiles_jsonl(metric_assignments, custom_metrics, output_dir, custom_filename=None):
    """
    Create a JSONL file with specialist metric-to-model assignments.

    The file contains one JSON object — a dict mapping metric names to
    their primary/secondary model assignments. This format is detected
    by `detect_eval_mode()` in benchmarks_run.py as "specialist" mode.

    Args:
        metric_assignments: dict of metric_name → { primary: {...}, secondary: {...}, threshold: int }
        custom_metrics: list of custom metric dicts
        output_dir: Directory to save the file
        custom_filename: Optional filename

    Returns:
        Path to the created file
    """
    from ..utils.constants import DEFAULT_PROMPT_EVAL_DIR
    prompt_eval_dir = Path(DEFAULT_PROMPT_EVAL_DIR)
    os.makedirs(prompt_eval_dir, exist_ok=True)

    jsonl_path = prompt_eval_dir / (custom_filename or "judge_profiles.jsonl")

    # Build the specialist config dict
    specialist_config = {}
    for metric_name, assignment in metric_assignments.items():
        entry = {}
        if assignment.get("primary"):
            p = assignment["primary"]
            entry["primary"] = {
                "model_id": p.get("id") or p.get("model_id", ""),
                "region": p.get("region", ""),
                "input_token_cost": p.get("input_cost") or p.get("input_token_cost", 0),
                "output_token_cost": p.get("output_cost") or p.get("output_token_cost", 0),
            }
        if assignment.get("secondary"):
            s = assignment["secondary"]
            entry["secondary"] = {
                "model_id": s.get("id") or s.get("model_id", ""),
                "region": s.get("region", ""),
                "input_token_cost": s.get("input_cost") or s.get("input_token_cost", 0),
                "output_token_cost": s.get("output_cost") or s.get("output_token_cost", 0),
            }
        if entry:
            specialist_config[metric_name] = entry

    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            # Write as a single JSON object (not JSONL lines)
            f.write(json.dumps(specialist_config))
    except (IOError, KeyError) as e:
        raise Exception(f"Failed to write specialist judge profiles to {jsonl_path}: {e}")

    return str(jsonl_path)