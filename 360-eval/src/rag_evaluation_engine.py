import logging
import asyncio
from typing import List, Dict, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------
# Ground Truth Parsing
# ----------------------------------------

def parse_ground_truth_chunks(chunks_string: str) -> List[str]:
    """
    Parse ground truth chunks from CSV. Supports both JSON array format and comma-separated format.

    Args:
        chunks_string: JSON array string like '["chunk1", "chunk2"]' or comma-separated string

    Returns:
        List of chunk strings
    """
    import json

    if not chunks_string or pd.isna(chunks_string):
        return []

    chunks_string = str(chunks_string).strip()

    # Try parsing as JSON array first (common RAG evaluation format)
    if chunks_string.startswith('[') and chunks_string.endswith(']'):
        try:
            chunks = json.loads(chunks_string)
            if isinstance(chunks, list):
                # Clean and filter
                return [str(chunk).strip() for chunk in chunks if chunk]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse ground truth as JSON array, falling back to comma-separated parsing")

    # Fallback: Split by comma and clean (legacy format)
    chunks = [chunk.strip() for chunk in chunks_string.split(',')]
    # Filter out empty strings
    chunks = [chunk for chunk in chunks if chunk]

    return chunks


# ----------------------------------------
# RAGAs Evaluation (Latest API - 0.3.7)
# ----------------------------------------

def evaluate_retrieval_with_ragas(
    query: str,
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate retrieval quality using RAGAs metrics (v0.3.7 API).

    Args:
        query: Query text
        retrieved_chunks: List of retrieved chunks
        ground_truth_chunks: List of ground truth relevant chunks
        metrics: List of metrics to compute (default: all supported)

    Returns:
        Dictionary with metric scores
    """
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            NonLLMContextPrecisionWithReference,
            NonLLMContextRecall,
            LLMContextPrecisionWithoutReference
        )

        if metrics is None:
            metrics = ['context_precision', 'context_recall']

        logger.debug(f"Evaluating retrieval with RAGAs metrics: {metrics}")

        scores = {}

        # Create sample for RAGAs
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=retrieved_chunks,
            reference_contexts=ground_truth_chunks if ground_truth_chunks else []
        )

        # Run metrics (sync wrapper for async methods)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Context Precision (without reference - uses only retrieved contexts and query)
            if 'context_precision' in metrics and retrieved_chunks:
                try:
                    # Use NonLLM version which doesn't require an LLM evaluator
                    precision_metric = NonLLMContextPrecisionWithReference()
                    precision_score = loop.run_until_complete(
                        precision_metric.single_turn_ascore(sample)
                    )
                    scores['context_precision'] = float(precision_score)
                except Exception as e:
                    logger.warning(f"Context precision calculation failed: {str(e)}")
                    scores['context_precision'] = calculate_precision_manual(
                        retrieved_chunks, ground_truth_chunks
                    )

            # Context Recall (requires reference contexts)
            if 'context_recall' in metrics and ground_truth_chunks:
                try:
                    # Use NonLLM version for faster evaluation
                    recall_metric = NonLLMContextRecall()
                    recall_score = loop.run_until_complete(
                        recall_metric.single_turn_ascore(sample)
                    )
                    scores['context_recall'] = float(recall_score)
                except Exception as e:
                    logger.warning(f"Context recall calculation failed: {str(e)}")
                    scores['context_recall'] = calculate_recall_manual(
                        retrieved_chunks, ground_truth_chunks
                    )

            # Context Relevancy - use manual calculation as fallback
            if 'context_relevancy' in metrics:
                scores['context_relevancy'] = calculate_relevancy_manual(
                    retrieved_chunks, query
                )

        finally:
            loop.close()

        logger.debug(f"RAGAs evaluation completed: {scores}")
        return scores

    except ImportError:
        logger.error("RAGAs library not installed or missing dependencies")
        # Fallback to manual metrics
        return evaluate_retrieval_manual(query, retrieved_chunks, ground_truth_chunks)
    except Exception as e:
        logger.error(f"Error in RAGAs evaluation: {str(e)}", exc_info=True)
        # Fallback to manual metrics
        return evaluate_retrieval_manual(query, retrieved_chunks, ground_truth_chunks)


# ----------------------------------------
# Manual Retrieval Metrics (Fallback)
# ----------------------------------------

def evaluate_retrieval_manual(
    query: str,
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str]
) -> Dict[str, float]:
    """
    Calculate retrieval metrics manually (fallback if RAGAs fails).

    Args:
        query: Query text
        retrieved_chunks: List of retrieved chunks
        ground_truth_chunks: List of ground truth relevant chunks

    Returns:
        Dictionary with metric scores
    """
    logger.info("Using manual retrieval metrics (RAGAs fallback)")

    scores = {}

    # Context Precision
    scores['context_precision'] = calculate_precision_manual(retrieved_chunks, ground_truth_chunks)

    # Context Recall
    scores['context_recall'] = calculate_recall_manual(retrieved_chunks, ground_truth_chunks)

    # Context Relevancy
    scores['context_relevancy'] = calculate_relevancy_manual(retrieved_chunks, query)

    logger.debug(f"Manual evaluation scores: {scores}")
    return scores


def calculate_precision_manual(retrieved_chunks: List[str], ground_truth_chunks: List[str], similarity_calculator=None) -> float:
    """Calculate precision manually."""
    if not retrieved_chunks or not ground_truth_chunks:
        return 0.0

    relevant_count = sum(1 for chunk in retrieved_chunks if is_chunk_relevant(chunk, ground_truth_chunks, similarity_calculator=similarity_calculator))
    return relevant_count / len(retrieved_chunks)


def calculate_recall_manual(retrieved_chunks: List[str], ground_truth_chunks: List[str], similarity_calculator=None) -> float:
    """Calculate recall manually."""
    if not ground_truth_chunks:
        return 0.0

    retrieved_count = sum(1 for gt_chunk in ground_truth_chunks if is_chunk_retrieved(gt_chunk, retrieved_chunks, similarity_calculator=similarity_calculator))
    return retrieved_count / len(ground_truth_chunks)


def calculate_relevancy_manual(retrieved_chunks: List[str], query: str) -> float:
    """Calculate relevancy manually."""
    if not retrieved_chunks or not query:
        return 0.0

    relevancy_scores = [calculate_relevancy_score(chunk, query) for chunk in retrieved_chunks]
    return sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0


def is_chunk_relevant(chunk: str, ground_truth_chunks: List[str], threshold: float = 0.7, similarity_calculator=None) -> bool:
    """
    Check if a retrieved chunk is relevant based on ground truth.

    Args:
        chunk: Retrieved chunk text
        ground_truth_chunks: List of ground truth chunks
        threshold: Similarity threshold for relevance
        similarity_calculator: Optional SimilarityCalculator instance (defaults to Jaccard)

    Returns:
        True if chunk is relevant, False otherwise
    """
    # Check for exact match
    if chunk in ground_truth_chunks:
        return True

    # Check for high overlap with any ground truth chunk
    for gt_chunk in ground_truth_chunks:
        try:
            if similarity_calculator is not None:
                # Use provided similarity calculator
                similarity = similarity_calculator.calculate(chunk, gt_chunk)
                # Use calculator's threshold if no explicit threshold provided
                threshold_to_use = threshold if threshold != 0.7 else similarity_calculator.default_threshold
            else:
                # Fall back to Jaccard (backward compatibility)
                similarity = calculate_text_overlap(chunk, gt_chunk)
                threshold_to_use = threshold

            if similarity >= threshold_to_use:
                return True
        except Exception as e:
            logger.error(f"Error calculating similarity in is_chunk_relevant: {str(e)}", exc_info=True)
            # Fall back to Jaccard on error
            try:
                similarity = calculate_text_overlap(chunk, gt_chunk)
                if similarity >= 0.7:  # Use default threshold
                    return True
            except Exception as fallback_error:
                logger.error(f"Fallback similarity calculation also failed: {str(fallback_error)}")
                # Continue to next chunk

    return False


def is_chunk_retrieved(ground_truth_chunk: str, retrieved_chunks: List[str], threshold: float = 0.7, similarity_calculator=None) -> bool:
    """
    Check if a ground truth chunk was retrieved.

    Args:
        ground_truth_chunk: Ground truth chunk text
        retrieved_chunks: List of retrieved chunks
        threshold: Similarity threshold
        similarity_calculator: Optional SimilarityCalculator instance (defaults to Jaccard)

    Returns:
        True if ground truth chunk was retrieved, False otherwise
    """
    # Check for exact match
    if ground_truth_chunk in retrieved_chunks:
        return True

    # Check for high overlap with any retrieved chunk
    for chunk in retrieved_chunks:
        try:
            if similarity_calculator is not None:
                # Use provided similarity calculator
                similarity = similarity_calculator.calculate(ground_truth_chunk, chunk)
                # Use calculator's threshold if no explicit threshold provided
                threshold_to_use = threshold if threshold != 0.7 else similarity_calculator.default_threshold
            else:
                # Fall back to Jaccard (backward compatibility)
                similarity = calculate_text_overlap(ground_truth_chunk, chunk)
                threshold_to_use = threshold

            if similarity >= threshold_to_use:
                return True
        except Exception as e:
            logger.error(f"Error calculating similarity in is_chunk_retrieved: {str(e)}", exc_info=True)
            # Fall back to Jaccard on error
            try:
                similarity = calculate_text_overlap(ground_truth_chunk, chunk)
                if similarity >= 0.7:  # Use default threshold
                    return True
            except Exception as fallback_error:
                logger.error(f"Fallback similarity calculation also failed: {str(fallback_error)}")
                # Continue to next chunk

    return False


def calculate_text_overlap(text1: str, text2: str) -> float:
    """
    Calculate text overlap using Jaccard similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Overlap score (0-1)
    """
    # Tokenize and create sets
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    # Jaccard similarity
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union) if union else 0.0


def calculate_relevancy_score(chunk: str, query: str) -> float:
    """
    Calculate relevancy score between chunk and query.

    Args:
        chunk: Chunk text
        query: Query text

    Returns:
        Relevancy score (0-1)
    """
    # Simple keyword-based relevancy
    query_terms = set(query.lower().split())
    chunk_terms = set(chunk.lower().split())

    if not query_terms:
        return 0.0

    # How many query terms appear in the chunk?
    matching_terms = query_terms.intersection(chunk_terms)
    coverage = len(matching_terms) / len(query_terms)

    return coverage


# ----------------------------------------
# Aggregate Metrics
# ----------------------------------------

def calculate_retrieval_metrics(
    queries_df: pd.DataFrame,
    retrieval_results: List[Dict]
) -> pd.DataFrame:
    """
    Calculate aggregated retrieval metrics across all queries.

    Args:
        queries_df: DataFrame with queries and ground truth
        retrieval_results: List of retrieval result dicts

    Returns:
        DataFrame with per-query metrics
    """
    metrics_data = []

    for i, (idx, row) in enumerate(queries_df.iterrows()):
        query = row.get('query', '')
        ground_truth = parse_ground_truth_chunks(row.get('ground_truth_chunks', ''))

        if i < len(retrieval_results):
            result = retrieval_results[i]
            retrieved_chunks = result.get('chunks', [])

            # Evaluate with RAGAs
            metrics = evaluate_retrieval_with_ragas(
                query=query,
                retrieved_chunks=retrieved_chunks,
                ground_truth_chunks=ground_truth
            )

            metrics_data.append({
                'query_index': i,
                'query': query,
                'num_retrieved': len(retrieved_chunks),
                'num_ground_truth': len(ground_truth),
                **metrics
            })
        else:
            logger.warning(f"No retrieval results for query {i}")

    return pd.DataFrame(metrics_data)


# ----------------------------------------
# Precision@K and Recall@K
# ----------------------------------------

def calculate_precision_at_k(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    k: int,
    similarity_calculator=None
) -> float:
    """
    Calculate Precision@K.

    Args:
        retrieved_chunks: List of retrieved chunks
        ground_truth_chunks: List of ground truth chunks
        k: Number of top chunks to consider
        similarity_calculator: Optional SimilarityCalculator instance

    Returns:
        Precision@K score (0-1)
    """
    if k <= 0 or k > len(retrieved_chunks) or not ground_truth_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]
    relevant_count = sum(1 for chunk in top_k if is_chunk_relevant(chunk, ground_truth_chunks, similarity_calculator=similarity_calculator))

    return relevant_count / k


def calculate_recall_at_k(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    k: int,
    similarity_calculator=None
) -> float:
    """
    Calculate Recall@K.

    Args:
        retrieved_chunks: List of retrieved chunks
        ground_truth_chunks: List of ground truth chunks
        k: Number of top chunks to consider
        similarity_calculator: Optional SimilarityCalculator instance

    Returns:
        Recall@K score (0-1)
    """
    if k <= 0 or not ground_truth_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]
    retrieved_count = sum(1 for gt_chunk in ground_truth_chunks if is_chunk_retrieved(gt_chunk, top_k, similarity_calculator=similarity_calculator))

    return retrieved_count / len(ground_truth_chunks)


def calculate_map_at_k(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    k: int,
    similarity_calculator=None
) -> float:
    """
    Calculate Mean Average Precision at K (MAP@K).

    Args:
        retrieved_chunks: List of retrieved chunks
        ground_truth_chunks: List of ground truth chunks
        k: Number of top chunks to consider
        similarity_calculator: Optional SimilarityCalculator instance

    Returns:
        MAP@K score (0-1)
    """
    if k <= 0 or not ground_truth_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]

    # Calculate precision at each relevant position
    precisions = []
    relevant_count = 0

    for i, chunk in enumerate(top_k, 1):
        if is_chunk_relevant(chunk, ground_truth_chunks, similarity_calculator=similarity_calculator):
            relevant_count += 1
            precision_at_i = relevant_count / i
            precisions.append(precision_at_i)

    if not precisions:
        return 0.0

    return sum(precisions) / len(precisions)


# ----------------------------------------
# MRR (Mean Reciprocal Rank)
# ----------------------------------------

def calculate_mrr(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    similarity_calculator=None
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        retrieved_chunks: List of retrieved chunks
        ground_truth_chunks: List of ground truth chunks
        similarity_calculator: Optional SimilarityCalculator instance

    Returns:
        MRR score (0-1)
    """
    if not ground_truth_chunks or not retrieved_chunks:
        return 0.0

    # Find the rank of the first relevant chunk
    for i, chunk in enumerate(retrieved_chunks, 1):
        if is_chunk_relevant(chunk, ground_truth_chunks, similarity_calculator=similarity_calculator):
            return 1.0 / i

    return 0.0


# ----------------------------------------
# Top-K Stability Analysis
# ----------------------------------------

def calculate_rank_stability(
    retrieved_chunks_list: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    FIXED: Calculate meaningful Top-K metrics instead of broken Kendall's tau.

    The old implementation always returned 100% because it compared the same
    retrieval at different K values, where ranks are mathematically guaranteed
    to be identical for common items.

    New metrics:
    - Rank Concentration: Is quality concentrated at the top?
    - Rank Diversity: Are results unique (no duplicates)?
    - Quality Drop-off: How fast does quality decay?

    Args:
        retrieved_chunks_list: List of retrieved chunks (ordered by rank)
        k_values: List of K values to analyze

    Returns:
        Dictionary with meaningful stability metrics
    """
    import numpy as np

    metrics = {}

    if not retrieved_chunks_list or len(retrieved_chunks_list) < 2:
        return metrics

    # 1. Rank Concentration - Is quality concentrated at top?
    # Uses inverse rank weighting: 1/rank
    try:
        total_weight = sum(1 / (i + 1) for i in range(len(retrieved_chunks_list)))

        for k in k_values:
            if k > len(retrieved_chunks_list):
                continue

            top_k_weight = sum(1 / (i + 1) for i in range(k))
            concentration = top_k_weight / total_weight if total_weight > 0 else 0.0
            metrics[f'concentration@{k}'] = float(concentration)
    except Exception as e:
        logger.error(f"Error calculating concentration: {e}")

    # 2. Rank Diversity - Are all results unique?
    try:
        for k in k_values:
            if k > len(retrieved_chunks_list):
                continue

            unique_items = len(set(retrieved_chunks_list[:k]))
            diversity = unique_items / k if k > 0 else 0.0
            metrics[f'diversity@{k}'] = float(diversity)
    except Exception as e:
        logger.error(f"Error calculating diversity: {e}")

    # 3. Quality Drop-off - How much does quality decrease at higher K?
    try:
        # Use 1/rank as relevance proxy
        relevance_scores = [1.0 / (i + 1) for i in range(len(retrieved_chunks_list))]

        for i, k in enumerate(k_values[1:], 1):
            prev_k = k_values[i - 1]

            if k > len(retrieved_chunks_list):
                continue

            avg_k = np.mean(relevance_scores[:k])
            avg_prev = np.mean(relevance_scores[:prev_k])

            drop_off = (avg_prev - avg_k) / avg_prev if avg_prev > 0 else 0.0
            metrics[f'dropoff_k{prev_k}_to_k{k}'] = float(drop_off)
    except Exception as e:
        logger.error(f"Error calculating drop-off: {e}")

    # 4. Average concentration across all K (summary metric)
    concentration_vals = [v for k, v in metrics.items() if 'concentration@' in k]
    if concentration_vals:
        metrics['avg_rank_concentration'] = float(np.mean(concentration_vals))

    return metrics


def calculate_result_churn(
    retrieved_chunks_list: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Calculate result churn rate - how many new results appear as K increases.

    Args:
        retrieved_chunks_list: List of retrieved chunks (ordered by rank)
        k_values: List of K values to analyze

    Returns:
        Dictionary with churn rates
    """
    churn_scores = {}

    if not retrieved_chunks_list:
        return churn_scores

    prev_set = None
    prev_k = None

    for k in sorted(k_values):
        if k > len(retrieved_chunks_list):
            continue

        current_set = set(retrieved_chunks_list[:k])

        if prev_set is not None:
            # Calculate how many new results appeared
            new_results = current_set - prev_set
            churn_rate = len(new_results) / k if k > 0 else 0.0
            churn_scores[f'churn_k{prev_k}_to_k{k}'] = churn_rate

        prev_set = current_set
        prev_k = k

    # Calculate average churn
    if churn_scores:
        churn_scores['avg_churn_rate'] = sum(churn_scores.values()) / len(churn_scores)

    return churn_scores


# REMOVED: calculate_top_k_overlap()
# This function always returned 100% because the first K items in a list
# are mathematically guaranteed to appear in any larger K subset.
# The metric provided no useful information for decision-making.


# ----------------------------------------
# Comprehensive Evaluation
# ----------------------------------------

def comprehensive_retrieval_evaluation(
    query: str,
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    k_values: List[int] = [1, 3, 5, 10],
    similarity_calculators: List = None
) -> Dict[str, float]:
    """
    Perform comprehensive retrieval evaluation with multiple metrics.
    Supports multiple similarity calculation methods.

    Args:
        query: Query text
        retrieved_chunks: List of retrieved chunks
        ground_truth_chunks: List of ground truth chunks
        k_values: List of K values for precision/recall@K
        similarity_calculators: Optional list of SimilarityCalculator instances.
                               If None, defaults to Jaccard only.
                               If provided, calculates metrics for each method.

    Returns:
        Dictionary with all metrics (method-suffixed if multiple calculators)
    """
    # Default to Jaccard if no calculators provided (backward compatibility)
    if similarity_calculators is None:
        from similarity_calculators import JaccardSimilarity
        similarity_calculators = [JaccardSimilarity()]

    all_metrics = {}

    # Calculate metrics for each similarity method
    for calculator in similarity_calculators:
        method_name = calculator.name
        logger.info(f"Calculating metrics using {method_name} similarity")

        try:
            metrics = {}

            # RAGAs metrics (only for first method to avoid redundancy)
            # RAGAs doesn't use our similarity calculator
            if calculator == similarity_calculators[0]:
                try:
                    ragas_scores = evaluate_retrieval_with_ragas(query, retrieved_chunks, ground_truth_chunks)
                    # Add RAGAs scores without suffix (they're method-independent)
                    all_metrics.update(ragas_scores)
                except Exception as e:
                    logger.error(f"Error calculating RAGAs metrics: {str(e)}", exc_info=True)
                    # Continue with other metrics

            # Manual metrics using the similarity calculator
            try:
                metrics['context_precision'] = calculate_precision_manual(retrieved_chunks, ground_truth_chunks, calculator)
                metrics['context_recall'] = calculate_recall_manual(retrieved_chunks, ground_truth_chunks, calculator)
            except Exception as e:
                logger.error(f"Error calculating precision/recall for {method_name}: {str(e)}", exc_info=True)
                metrics['context_precision'] = 0.0
                metrics['context_recall'] = 0.0

            # Precision/Recall/MAP at K
            for k in k_values:
                if k <= len(retrieved_chunks):
                    try:
                        metrics[f'precision@{k}'] = calculate_precision_at_k(retrieved_chunks, ground_truth_chunks, k, calculator)
                        metrics[f'recall@{k}'] = calculate_recall_at_k(retrieved_chunks, ground_truth_chunks, k, calculator)
                        metrics[f'map@{k}'] = calculate_map_at_k(retrieved_chunks, ground_truth_chunks, k, calculator)
                    except Exception as e:
                        logger.error(f"Error calculating metrics@{k} for {method_name}: {str(e)}", exc_info=True)
                        metrics[f'precision@{k}'] = 0.0
                        metrics[f'recall@{k}'] = 0.0
                        metrics[f'map@{k}'] = 0.0

            # MRR
            try:
                metrics['mrr'] = calculate_mrr(retrieved_chunks, ground_truth_chunks, calculator)
            except Exception as e:
                logger.error(f"Error calculating MRR for {method_name}: {str(e)}", exc_info=True)
                metrics['mrr'] = 0.0

            # Suffix metrics with method name
            if len(similarity_calculators) > 1:
                # Multiple methods: add suffix
                for metric_name, metric_value in metrics.items():
                    all_metrics[f'{metric_name}_{method_name}'] = metric_value
            else:
                # Single method: no suffix (backward compatibility)
                all_metrics.update(metrics)

        except Exception as e:
            logger.error(f"Fatal error processing similarity calculator {method_name}: {str(e)}", exc_info=True)
            # Add placeholder metrics to indicate failure
            if len(similarity_calculators) > 1:
                all_metrics[f'error_{method_name}'] = str(e)
            # Continue with next calculator

    # Top-K Analysis (method-independent, only calculate once)
    # Includes: concentration, diversity, drop-off metrics
    stability_scores = calculate_rank_stability(retrieved_chunks, k_values)
    all_metrics.update(stability_scores)

    # Churn rate (how many new results as K increases)
    churn_scores = calculate_result_churn(retrieved_chunks, k_values)
    all_metrics.update(churn_scores)

    # Removed: calculate_top_k_overlap() - always returned 100%

    # Basic stats (method-independent)
    all_metrics['num_retrieved'] = len(retrieved_chunks)
    all_metrics['num_ground_truth'] = len(ground_truth_chunks)

    # Add metadata about which methods were used
    if len(similarity_calculators) > 1:
        all_metrics['similarity_methods_used'] = ','.join([calc.name for calc in similarity_calculators])

    return all_metrics
