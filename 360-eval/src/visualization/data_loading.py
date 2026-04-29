# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Data loading and preprocessing functions for benchmark results.
"""

import ast
import glob
import logging
import re
import uuid
from pathlib import Path

import pandas as pd

from .constants import EPSILON_DIVISION, PERCENTILES

logger = logging.getLogger(__name__)


def extract_model_name(model_id):
    """Extract clean model name from ID.

    Cleans model names by:
    1. Taking everything after the last "/"
    2. Removing version suffixes like ":0", ":1"
    3. Stripping provider prefixes (anthropic., meta., amazon., etc.)
    4. Removing date stamps (YYYYMMDD patterns)
    """
    # Check if this is an optimized prompt variant
    optimization_suffix = ""
    if "_Prompt_Optimized" in model_id:
        optimization_suffix = "_Prompt_Optimized"
        # Remove suffix temporarily for processing
        model_id = model_id.replace("_Prompt_Optimized", "")

    # Check if this is a service tier variant (_priority, _flex, _default)
    # Use abbreviated suffixes: (P), (D), (F)
    service_tier_suffix = ""
    tier_abbreviations = {
        "_priority": " (P)",
        "_default": " (D)",
        "_flex": " (F)",
    }
    for tier, abbrev in tier_abbreviations.items():
        if model_id.endswith(tier):
            service_tier_suffix = abbrev
            # Remove suffix temporarily for processing
            model_id = model_id[:-len(tier)]
            break

    # Simple rule: take everything after last "/"
    if '/' in model_id:
        model_id = model_id.split('/')[-1]

    # Remove version suffix like :0, :1
    if ':' in model_id:
        model_id = model_id.split(':')[0]

    # Strip regional prefix if present (e.g., us., eu., ap., global.)
    # Matches 2-letter codes or "global" followed by a dot
    model_id = re.sub(r'^([a-z]{2}|global)\.', '', model_id)

    # Strip provider/family prefix (first segment before the dot)
    # e.g., "amazon.nova-2-lite" -> "nova-2-lite"
    # Exception: keep prefix if resulting name would be too short (≤8 chars)
    if '.' in model_id:
        remaining = model_id.split('.', 1)[1]
        if len(remaining) > 8:
            model_id = remaining

    # Remove date stamps (YYYYMMDD patterns, typically preceded by - or _)
    model_id = re.sub(r'[-_]?\d{8}', '', model_id)

    return model_id + optimization_suffix + service_tier_suffix


def parse_json_string(json_str):
    """Parse JSON string using ast.literal_eval."""
    try:
        if isinstance(json_str, list):
            json_str = json_str[0]
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        # This handles the single-quoted JSON-like strings
        dict_data = ast.literal_eval(json_str)
        return dict_data
    except Exception as e:
        # Return error information if parsing fails
        return {"error": str(e)}


def get_evaluation_config_signature(df):
    """
    Create a unique signature for evaluation configuration.
    Evaluations with matching signatures can be safely merged.

    Note: Models are intentionally excluded from the signature so that
    evaluations with the same task parameters but different models
    can be aggregated together in reports.

    Returns:
        tuple: Configuration components that define a unique evaluation
    """
    try:
        # Extract unique values for each config component
        # Note: models are NOT included - we want to aggregate across different model sets
        # Filter out NaN values to ensure consistent hashing (NaN != NaN in Python)
        task_criteria = tuple(sorted([x for x in df['task_criteria'].unique() if pd.notna(x)]))
        task_types = tuple(sorted([x for x in df['task_types'].unique() if pd.notna(x)]))

        # Extract judge models from performance_metrics column
        judge_models = set()
        for perf_metrics in df['performance_metrics'].dropna():
            try:
                parsed = parse_json_string(perf_metrics)
                if isinstance(parsed, dict) and 'judge_details' in parsed:
                    for judge in parsed['judge_details']:
                        if 'model' in judge:
                            judge_models.add(judge['model'])
            except:
                pass
        judges = tuple(sorted(judge_models))

        # User-defined metrics (if exists in data) - filter out NaN values to ensure consistent hashing
        if 'user_defined_metrics' in df.columns:
            user_metrics = tuple(sorted([x for x in df['user_defined_metrics'].unique() if pd.notna(x)]))
        else:
            user_metrics = tuple()

        # Temperature values - filter out NaN values to ensure consistent hashing
        if 'TEMPERATURE' in df.columns:
            temperatures = tuple(sorted([x for x in df['TEMPERATURE'].unique() if pd.notna(x)]))
        else:
            temperatures = tuple()

        # Create signature tuple (models excluded to allow cross-model aggregation)
        config_sig = (task_criteria, task_types, judges, user_metrics, temperatures)

        return config_sig
    except Exception as e:
        logger.warning(f"Error creating config signature: {e}")
        # Return a random signature to keep this group separate
        return (str(uuid.uuid4()),)


def load_data(directory, evaluation_names=None, model_ids=None):
    """Load and prepare benchmark data with proper configuration-based grouping.

    Only merges evaluations with identical configurations (models, judges, criteria, etc.)

    Args:
        directory: Directory containing CSV files
        evaluation_names: Optional list of evaluation names to filter by
        model_ids: Optional list of model IDs to filter by (raw model_id values)
    """
    directory = Path(directory)
    logger.info(f"Looking for CSV files in: {directory}")

    # Load CSV files
    all_files = glob.glob(str(directory / "invocations_*.csv"))
    if not all_files:
        logger.error(f"No invocation CSVs found in {directory}")
        raise FileNotFoundError(f"No invocation CSVs found in {directory}")

    # Filter files by evaluation names if specified
    if evaluation_names:
        files = []
        for file_path in all_files:
            file_name = Path(file_path).name
            # Match evaluation name precisely at the end of filename (before .csv)
            # Filename format: invocations_{run}_{timestamp}_{hash}_{eval_id}_{eval_name}.csv
            # Use endswith to avoid substring false positives (e.g., "openai-evaluation"
            # matching "openai-evaluation-data-extraction")
            if any(file_name.endswith(f"_{eval_name}.csv") for eval_name in evaluation_names):
                files.append(file_path)

        if not files:
            logger.warning(f"No CSV files found matching evaluations: {evaluation_names}")
            logger.info(f"Available files: {[Path(f).name for f in all_files]}")
            raise FileNotFoundError(f"No CSV files found for evaluations: {evaluation_names}")

        logger.info(f"Filtered to {len(files)} CSV files matching evaluations {evaluation_names}")
    else:
        files = all_files
        logger.info(f"Found {len(files)} CSV files (no filter applied)")

    # Step 1: Load all files and group by configuration signature
    config_groups = {}
    file_metadata = {}

    for f in files:
        try:
            logger.info(f"Reading file: {f}")
            df_file = pd.read_csv(f)
            logger.info(f"Read {len(df_file)} rows from {f}")

            # Add source file tracking
            df_file['source_file'] = Path(f).name

            # Get configuration signature for this file
            config_sig = get_evaluation_config_signature(df_file)

            # Group files by config signature
            if config_sig not in config_groups:
                config_groups[config_sig] = []
                file_metadata[config_sig] = {
                    'files': [],
                    'task_type': df_file['task_types'].iloc[0] if not df_file.empty else 'Unknown',
                    'total_records': 0
                }

            config_groups[config_sig].append(df_file)
            file_metadata[config_sig]['files'].append(Path(f).name)
            file_metadata[config_sig]['total_records'] += len(df_file)

        except Exception as e:
            logger.error(f"Error reading {f}: {str(e)}")
            continue

    if not config_groups:
        logger.error("No valid data found in any CSV files")
        raise ValueError("No valid data found in any CSV files")

    # Step 2: Log configuration grouping information
    logger.info(f"Found {len(config_groups)} unique evaluation configurations")
    for i, (config_sig, metadata) in enumerate(file_metadata.items(), 1):
        logger.info(f"  Config {i}: task_type='{metadata['task_type']}', "
                   f"files={len(metadata['files'])}, records={metadata['total_records']}")
        for fname in metadata['files']:
            logger.info(f"    - {fname}")

    # Step 3: Merge ONLY within same configuration groups
    merged_dfs = []
    for config_sig, dfs_in_group in config_groups.items():
        # These all have matching configs, safe to merge
        merged_df = pd.concat(dfs_in_group, ignore_index=True)

        # Add config signature as a column for later grouping
        merged_df['config_signature'] = str(hash(config_sig))

        merged_dfs.append(merged_df)
        logger.info(f"Merged {len(dfs_in_group)} files with matching config into {len(merged_df)} records")

    # Step 4: Combine all config groups
    df = pd.concat(merged_dfs, ignore_index=True)
    logger.info(f"Combined data has {len(df)} total rows across {len(config_groups)} unique configurations")

    # Clean and prepare data (optimized with method chaining)
    df = (df[df['api_call_status'] == 'Success']
          .reset_index(drop=True)
          .assign(model_name=lambda x: x['model_id'].apply(extract_model_name)))

    # Filter by model IDs if specified
    if model_ids:
        df = df[df['model_id'].isin(model_ids)].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} rows matching {len(model_ids)} selected models")
        if df.empty:
            raise ValueError(f"No data found for selected models: {model_ids}")

    # Create model_name_with_tier for latency/cost grouping (keep original model_name for accuracy)
    # Use abbreviated tier suffixes: (P), (D), (F)
    tier_abbreviations = {
        'priority': ' (P)',
        'default': ' (D)',
        'flex': ' (F)',
    }
    if 'service_tier' in df.columns:
        df['model_name_with_tier'] = df.apply(
            lambda row: f"{row['model_name']}{tier_abbreviations.get(row['service_tier'], '')}" if pd.notna(row.get('service_tier')) else row['model_name'],
            axis=1
        )
    else:
        df['model_name_with_tier'] = df['model_name']

    parsed_dicts = df['performance_metrics'].apply(parse_json_string)
    del df['performance_metrics']
    # Convert the Series of dictionaries to a DataFrame
    unpacked_findings = pd.DataFrame(list(parsed_dicts))
    df = pd.concat([df, unpacked_findings], axis=1)

    # Check if this is a latency-only evaluation
    has_latency_only = 'eval_type' in df.columns and (df['eval_type'] == 'latency').any()

    # Handle judge_success appropriately based on evaluation type
    if has_latency_only:
        # For latency-only records, judge_success is 'N/A', convert to None for processing
        df['task_success'] = df['judge_success'].apply(lambda x: None if x == "N/A" else x)
    else:
        df['task_success'] = df['judge_success']

    # Calculate tokens per second
    df['OTPS'] = df['output_tokens'] / (df['time_to_last_byte'] + EPSILON_DIVISION)

    # Process judge scores only if we have actual judge evaluations
    if not has_latency_only or (df['judge_success'] != "N/A").any():
        try:
            judge_scores = pd.DataFrame(df['judge_scores'].to_dict()).transpose()
            # Identify numeric index values
            numeric_index_mask = pd.to_numeric(judge_scores.index, errors='coerce').notna()
            # Filter and process judge scores (optimized with method chaining)
            judge_scores_df = (judge_scores[numeric_index_mask]
                               .reset_index(drop=True)
                               .assign(mean_scores=lambda x: x.mean(axis=1)))
            df = pd.concat([df, judge_scores_df], axis=1)
        except Exception as e:
            # If judge_scores parsing fails (e.g., all N/A), create empty columns
            logger.warning(f"Could not parse judge scores (latency-only mode): {e}")
            df['mean_scores'] = None
    else:
        # Latency-only mode: No judge scores to process
        df['mean_scores'] = None
    # ── Cost summary ───────────────────────────────────────────────────────────
    cost_stats = (
        df.groupby(["model_name"])["response_cost"]
          .agg(avg_cost="mean", total_cost="sum", num_invocations="count")
    )

    # ── Latency percentiles (50/90/95/99) ──────────────────────────────────────
    latency_stats = (
        df.groupby(["model_name"])["time_to_last_byte"]
          .quantile(PERCENTILES)         # returns MultiIndex
          .unstack(level=-1)                          # percentiles → columns
    )
    latency_stats.columns = [f"p{int(q*100)}" for q in latency_stats.columns]

    # ── Combine both sets of metrics ──────────────────────────────────────────
    summary = cost_stats.join(latency_stats)

    # Optional: forecast spend per model/profile (30-day projection)
    summary["monthly_forecast"] = (
        summary["avg_cost"]
        * (summary["num_invocations"] / df.shape[0])
        * 30
    )
    df = pd.concat([df, summary], axis=1)

    return df
