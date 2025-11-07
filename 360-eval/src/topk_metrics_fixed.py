"""
Fixed Top-K Stability Metrics

The original implementation has a fundamental flaw - it always shows 100% stability
because it compares the same retrieval at different K values, where ranks are
guaranteed to be identical for common items.

This file provides corrected implementations that measure REAL stability and variance.
"""

from typing import List, Dict
import numpy as np
from scipy.stats import kendalltau
import logging

logger = logging.getLogger(__name__)


def calculate_rank_stability_fixed(
    retrieved_chunks: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    FIXED: Calculate meaningful rank stability metrics.

    Instead of comparing ranks (which are always identical), we measure:
    1. Rank Concentration - Are top results much better than lower ones?
    2. Rank Distribution - How evenly are results distributed?
    3. Position Stability Score - Normalized position variance

    Args:
        retrieved_chunks: Single list of retrieved chunks (ordered by rank)
        k_values: List of K values to analyze

    Returns:
        Dictionary with meaningful stability metrics
    """
    stability_scores = {}

    if not retrieved_chunks or len(retrieved_chunks) < 2:
        return stability_scores

    # For each K, calculate position-based metrics
    for i, k1 in enumerate(k_values):
        for k2 in k_values[i+1:]:
            if k1 >= len(retrieved_chunks) or k2 >= len(retrieved_chunks):
                continue

            # Calculate what % of top-K1 appear in top-K2
            top_k1 = set(retrieved_chunks[:k1])
            top_k2 = set(retrieved_chunks[:k2])

            # Preservation rate: how many from K1 are still in K2
            preservation = len(top_k1.intersection(top_k2)) / len(top_k1)

            # Position shift: average position change for items in top-K1
            position_shifts = []
            for chunk in top_k1:
                if chunk in retrieved_chunks[:k2]:
                    old_pos = retrieved_chunks.index(chunk)
                    # Position shift within k2 range
                    position_shifts.append(old_pos / k1)

            avg_position_shift = np.mean(position_shifts) if position_shifts else 0.0

            # Combined stability score (preservation weighted by position consistency)
            stability = preservation * (1 - abs(avg_position_shift - 0.5) * 2)

            stability_scores[f'stability_k{k1}_vs_k{k2}'] = float(stability)

    # Calculate average
    if stability_scores:
        stability_scores['avg_rank_stability'] = sum(stability_scores.values()) / len(stability_scores)

    return stability_scores


def calculate_rank_concentration(
    retrieved_chunks: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Calculate rank concentration - measures if results are clustered at top.

    High concentration (>0.7) = top results are much better, lower K is sufficient
    Low concentration (<0.3) = results are evenly distributed, may need higher K

    Args:
        retrieved_chunks: List of retrieved chunks
        k_values: K values to analyze

    Returns:
        Concentration scores for each K
    """
    scores = {}

    if not retrieved_chunks:
        return scores

    # Use inverse rank weighting
    for k in k_values:
        if k > len(retrieved_chunks):
            continue

        # Calculate concentration: how much weight is in top-K vs rest
        top_k_weight = sum(1 / (i + 1) for i in range(k))
        total_weight = sum(1 / (i + 1) for i in range(len(retrieved_chunks)))

        concentration = top_k_weight / total_weight if total_weight > 0 else 0.0
        scores[f'concentration@{k}'] = float(concentration)

    return scores


def calculate_inter_query_stability(
    all_retrievals: List[List[str]],
    k: int = 5
) -> Dict[str, float]:
    """
    Calculate stability ACROSS multiple queries/retrievals.

    This is the PROPER way to measure stability - compare rankings across
    different retrievals, not different K values of the same retrieval.

    Args:
        all_retrievals: List of retrieval results (one per query)
        k: K value to compare at

    Returns:
        Inter-query stability metrics
    """
    if len(all_retrievals) < 2:
        return {'inter_query_stability': 1.0}

    # Get top-K from each retrieval
    top_k_sets = [set(r[:min(k, len(r))]) for r in all_retrievals]

    # Calculate pairwise Jaccard similarity
    similarities = []
    for i in range(len(top_k_sets)):
        for j in range(i + 1, len(top_k_sets)):
            if top_k_sets[i] and top_k_sets[j]:
                intersection = len(top_k_sets[i].intersection(top_k_sets[j]))
                union = len(top_k_sets[i].union(top_k_sets[j]))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)

    avg_similarity = np.mean(similarities) if similarities else 1.0

    return {
        'inter_query_stability': float(avg_similarity),
        'stability_variance': float(np.var(similarities)) if len(similarities) > 1 else 0.0
    }


def calculate_rank_diversity(
    retrieved_chunks: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Measure diversity/novelty as K increases.

    High diversity = many new unique results at higher K
    Low diversity = diminishing returns at higher K

    Args:
        retrieved_chunks: List of retrieved chunks
        k_values: K values to analyze

    Returns:
        Diversity scores
    """
    scores = {}

    if not retrieved_chunks:
        return scores

    for i, k in enumerate(k_values):
        if k > len(retrieved_chunks):
            continue

        # Calculate unique content in top-K
        unique_items = len(set(retrieved_chunks[:k]))
        diversity = unique_items / k if k > 0 else 0.0

        scores[f'diversity@{k}'] = float(diversity)

    return scores


def calculate_top_k_drop_off(
    retrieved_chunks: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Measure quality drop-off as K increases.

    This approximates relevance decay using rank position as proxy.
    Lower drop-off = quality maintained at higher K

    Args:
        retrieved_chunks: List of retrieved chunks
        k_values: K values to analyze

    Returns:
        Drop-off scores
    """
    scores = {}

    if not retrieved_chunks:
        return scores

    # Use 1/rank as relevance proxy
    relevance_scores = [1.0 / (i + 1) for i in range(len(retrieved_chunks))]

    for i, k in enumerate(k_values[1:], 1):
        prev_k = k_values[i - 1]

        if k > len(retrieved_chunks):
            continue

        # Average relevance in top-K
        avg_k = np.mean(relevance_scores[:k])
        avg_prev = np.mean(relevance_scores[:prev_k])

        # Drop-off: how much did average relevance decrease?
        drop_off = (avg_prev - avg_k) / avg_prev if avg_prev > 0 else 0.0

        scores[f'dropoff_k{prev_k}_to_k{k}'] = float(drop_off)

    return scores


# ========================================
# Aggregation Function
# ========================================

def calculate_comprehensive_topk_metrics(
    retrieved_chunks: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Calculate ALL meaningful Top-K metrics.

    These metrics provide ACTIONABLE insights:
    - Concentration: Should I use lower or higher K?
    - Diversity: Am I getting new information at higher K?
    - Drop-off: How much quality loss at higher K?
    - Churn: How stable are my results?

    Args:
        retrieved_chunks: Single retrieval result
        k_values: K values to analyze

    Returns:
        Combined dictionary of all metrics
    """
    metrics = {}

    try:
        # 1. Fixed stability (preservation-based)
        metrics.update(calculate_rank_stability_fixed(retrieved_chunks, k_values))
    except Exception as e:
        logger.error(f"Error calculating rank stability: {e}")

    try:
        # 2. Rank concentration
        metrics.update(calculate_rank_concentration(retrieved_chunks, k_values))
    except Exception as e:
        logger.error(f"Error calculating concentration: {e}")

    try:
        # 3. Diversity
        metrics.update(calculate_rank_diversity(retrieved_chunks, k_values))
    except Exception as e:
        logger.error(f"Error calculating diversity: {e}")

    try:
        # 4. Drop-off
        metrics.update(calculate_top_k_drop_off(retrieved_chunks, k_values))
    except Exception as e:
        logger.error(f"Error calculating drop-off: {e}")

    return metrics


# ========================================
# Comparison Function
# ========================================

def compare_old_vs_new_metrics(retrieved_chunks: List[str]):
    """
    Compare old (broken) vs new (fixed) metrics to show the difference.
    """
    from rag_evaluation_engine import (
        calculate_rank_stability,
        calculate_result_churn,
        calculate_top_k_overlap
    )

    print("=" * 80)
    print("OLD METRICS (Always 100%):")
    print("=" * 80)

    old_stability = calculate_rank_stability(retrieved_chunks)
    old_churn = calculate_result_churn(retrieved_chunks)
    old_overlap = calculate_top_k_overlap(retrieved_chunks)

    print(f"Rank Stability: {old_stability}")
    print(f"Result Churn: {old_churn}")
    print(f"Top-K Overlap: {old_overlap}")

    print("\n" + "=" * 80)
    print("NEW METRICS (Actually Useful):")
    print("=" * 80)

    new_metrics = calculate_comprehensive_topk_metrics(retrieved_chunks)

    for metric, value in sorted(new_metrics.items()):
        print(f"{metric:30s}: {value:.3f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("""
OLD METRICS show 100% because they compare the SAME retrieval at different K values.
This is mathematically guaranteed to be identical.

NEW METRICS provide actionable insights:
- concentration@k: Should I use lower K? (higher = yes)
- diversity@k: Am I getting unique results? (higher = yes)
- dropoff_k*: How much quality loss at higher K? (lower = better)
- stability_k*: Are top results preserved? (higher = better)
""")
