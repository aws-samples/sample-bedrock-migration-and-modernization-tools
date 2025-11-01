import os
import logging
import time
from typing import List, Dict, Tuple
import json

logger = logging.getLogger(__name__)


# ----------------------------------------
# 1. No Re-ranking (Pass-through)
# ----------------------------------------

def rerank_none(
    chunks: List[str],
    query: str,
    distances: List[float] = None
) -> Tuple[List[str], List[float], Dict]:
    """
    Pass-through reranker - returns chunks in original order.

    Args:
        chunks: List of retrieved chunks
        query: Query text (unused)
        distances: Original distances from vector search

    Returns:
        Tuple of (reranked_chunks, reranked_scores, metadata)
    """
    logger.debug(f"No reranking applied to {len(chunks)} chunks")

    # If no distances provided, use rank-based scores
    if distances is None:
        scores = [1.0 - (i / len(chunks)) for i in range(len(chunks))]
    else:
        # Convert distances to similarity scores
        scores = [1.0 - d for d in distances]

    metadata = {
        'reranker_type': 'none',
        'num_chunks': len(chunks),
        'latency_seconds': 0.0,
        'cost': 0.0
    }

    return chunks, scores, metadata


# ----------------------------------------
# 2. Cohere Re-ranking (Bedrock)
# ----------------------------------------

def rerank_cohere(
    chunks: List[str],
    query: str,
    model_id: str = "bedrock/cohere.rerank-v3-5:0",
    region: str = "us-east-1",
    cost_per_1k: float = 0.002,
    top_n: int = None
) -> Tuple[List[str], List[float], Dict]:
    """
    Re-rank chunks using Cohere Rerank model via AWS Bedrock.

    Args:
        chunks: List of retrieved chunks
        query: Query text
        model_id: Cohere rerank model identifier (with bedrock/ prefix)
        region: AWS region for Bedrock
        cost_per_1k: Cost per 1000 tokens
        top_n: Optional number of top results to return (None returns all)

    Returns:
        Tuple of (reranked_chunks, reranked_scores, metadata)
    """
    try:
        import boto3

        logger.debug(f"Re-ranking {len(chunks)} chunks using Cohere Rerank: {model_id}")

        start_time = time.time()

        # Remove 'bedrock/' prefix if present
        bedrock_model_id = model_id.replace("bedrock/", "")

        # Initialize Bedrock client
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )

        # Prepare documents for Cohere Rerank
        # Bedrock Cohere Rerank expects documents as strings, not objects
        documents = chunks

        # Build request body with required api_version
        request_body = {
            "query": query,
            "documents": documents,
            "api_version": 2  # Required by Bedrock Cohere Rerank API (must be integer >= 2)
        }

        if top_n is not None:
            request_body["top_n"] = min(top_n, len(chunks))

        # Call Cohere Rerank
        response = bedrock_runtime.invoke_model(
            modelId=bedrock_model_id,
            body=json.dumps(request_body)
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        results = response_body.get('results', [])

        # Extract reranked chunks and scores
        reranked_chunks = []
        reranked_scores = []

        for result in results:
            index = result['index']
            relevance_score = result['relevance_score']
            reranked_chunks.append(chunks[index])
            reranked_scores.append(float(relevance_score))

        latency = time.time() - start_time

        # Calculate approximate cost
        # Cohere charges per search unit (query + documents)
        search_units = 1  # One query with documents
        cost = (search_units / 1000) * cost_per_1k

        metadata = {
            'reranker_type': 'cohere',
            'model_id': model_id,
            'num_chunks': len(chunks),
            'num_reranked': len(reranked_chunks),
            'latency_seconds': latency,
            'cost': cost,
            'search_units': search_units
        }

        logger.debug(f"Cohere reranking completed in {latency:.2f}s, cost: ${cost:.6f}")

        return reranked_chunks, reranked_scores, metadata

    except ImportError:
        logger.error("boto3 library not installed. Install with: pip install boto3")
        raise
    except Exception as e:
        logger.error(f"Error in Cohere reranking: {str(e)}", exc_info=True)
        raise


# ----------------------------------------
# 3. LLM-based Re-ranking
# ----------------------------------------

def rerank_llm(
    chunks: List[str],
    query: str,
    llm_model_id: str,
    region: str = "us-east-1",
    input_cost_per_1k: float = 0.003,
    output_cost_per_1k: float = 0.015
) -> Tuple[List[str], List[float], Dict]:
    """
    Re-rank chunks using LLM to score relevance.

    Args:
        chunks: List of retrieved chunks
        query: Query text
        llm_model_id: LLM model identifier
        region: AWS region (for Bedrock models)
        input_cost_per_1k: Input cost per 1K tokens
        output_cost_per_1k: Output cost per 1K tokens

    Returns:
        Tuple of (reranked_chunks, reranked_scores, metadata)
    """
    try:
        from utils import run_inference

        logger.debug(f"Re-ranking {len(chunks)} chunks using LLM: {llm_model_id}")

        start_time = time.time()

        # Create reranking prompt
        prompt = create_reranking_prompt(query, chunks)

        # Call LLM
        params = {
            "maxTokens": 2000,
            "temperature": 0.0,
            "topP": 1.0
        }

        if "bedrock" in llm_model_id:
            params["aws_region_name"] = region

        response = run_inference(
            model_name=llm_model_id,
            prompt_text=prompt,
            provider_params=params,
            stream=False
        )

        # Parse scores from response
        scores_text = response['text']
        chunk_scores = parse_llm_reranking_response(scores_text, len(chunks))

        # Sort by scores (descending)
        scored_chunks = list(zip(chunks, chunk_scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        reranked_chunks = [chunk for chunk, _ in scored_chunks]
        reranked_scores = [float(score) for _, score in scored_chunks]

        latency = time.time() - start_time

        # Calculate cost
        input_tokens = response.get('inputTokens', 0)
        output_tokens = response.get('outputTokens', 0)
        cost = (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k

        metadata = {
            'reranker_type': 'llm',
            'model_id': llm_model_id,
            'num_chunks': len(chunks),
            'latency_seconds': latency,
            'cost': cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }

        logger.debug(f"LLM reranking completed in {latency:.2f}s, cost: ${cost:.6f}")

        return reranked_chunks, reranked_scores, metadata

    except Exception as e:
        logger.error(f"Error in LLM reranking: {str(e)}", exc_info=True)
        raise


def create_reranking_prompt(query: str, chunks: List[str]) -> str:
    """
    Create prompt for LLM-based reranking.

    Args:
        query: Query text
        chunks: List of chunks to rank

    Returns:
        Prompt string
    """
    chunks_text = "\n\n".join([
        f"[Chunk {i+1}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    ])

    prompt = f"""You are an expert at evaluating the relevance of text chunks to a query.

Query: {query}

Below are {len(chunks)} text chunks. For each chunk, rate its relevance to the query on a scale of 0-10, where:
- 0 = Completely irrelevant
- 5 = Somewhat relevant
- 10 = Highly relevant and directly answers the query

{chunks_text}

Provide your ratings in JSON format:
{{"ratings": [score1, score2, score3, ...]}}

Only output the JSON, nothing else."""

    return prompt


def parse_llm_reranking_response(response_text: str, expected_count: int) -> List[float]:
    """
    Parse LLM reranking response to extract scores.

    Args:
        response_text: LLM response text
        expected_count: Expected number of scores

    Returns:
        List of relevance scores
    """
    try:
        # Try to extract JSON from response
        # Look for JSON block
        import re

        json_match = re.search(r'\{[^}]*"ratings"[^}]*\[[^\]]*\][^}]*\}', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            ratings = data.get('ratings', [])

            if len(ratings) == expected_count:
                # Normalize scores to 0-1 range
                normalized = [float(score) / 10.0 for score in ratings]
                return normalized
            else:
                logger.warning(f"Expected {expected_count} ratings, got {len(ratings)}")

        # Fallback: try to extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
        if len(numbers) >= expected_count:
            scores = [float(n) / 10.0 for n in numbers[:expected_count]]
            return scores

        # If parsing fails, return uniform scores
        logger.warning("Failed to parse LLM reranking response, using uniform scores")
        return [0.5] * expected_count

    except Exception as e:
        logger.error(f"Error parsing LLM reranking response: {str(e)}")
        return [0.5] * expected_count


# ----------------------------------------
# Unified Re-ranking Interface
# ----------------------------------------

def rerank_chunks(
    chunks: List[str],
    query: str,
    reranker_config: Dict,
    distances: List[float] = None
) -> Tuple[List[str], List[float], Dict]:
    """
    Re-rank chunks using specified reranker.

    Args:
        chunks: List of retrieved chunks
        query: Query text
        reranker_config: Reranker configuration dict
        distances: Original distances from vector search

    Returns:
        Tuple of (reranked_chunks, reranked_scores, metadata)
    """
    reranker_type = reranker_config.get('type', 'none')

    logger.info(f"Re-ranking {len(chunks)} chunks using '{reranker_type}' reranker")

    if reranker_type == 'none' or not chunks:
        return rerank_none(chunks, query, distances)

    elif reranker_type == 'cohere':
        model_id = reranker_config.get('model_id', 'bedrock/cohere.rerank-v3-5:0')
        region = reranker_config.get('region', 'us-east-1')
        cost_per_1k = reranker_config.get('cost_per_1k', 0.002)
        top_n = reranker_config.get('top_n', None)

        return rerank_cohere(chunks, query, model_id, region, cost_per_1k, top_n)

    elif reranker_type == 'llm':
        model_id = reranker_config.get('model_id')
        region = reranker_config.get('region', 'us-east-1')
        input_cost = reranker_config.get('input_cost_per_1k', 0.003)
        output_cost = reranker_config.get('output_cost_per_1k', 0.015)

        if not model_id:
            raise ValueError("LLM reranker requires 'model_id' in config")

        return rerank_llm(chunks, query, model_id, region, input_cost, output_cost)

    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}. Use 'none', 'cohere', or 'llm'")


# ----------------------------------------
# Batch Re-ranking
# ----------------------------------------

def rerank_multiple_queries(
    retrieval_results: List[Dict],
    queries: List[str],
    reranker_config: Dict
) -> List[Tuple[List[str], List[float], Dict]]:
    """
    Re-rank chunks for multiple queries.

    Args:
        retrieval_results: List of retrieval result dicts
        queries: List of query texts
        reranker_config: Reranker configuration

    Returns:
        List of (reranked_chunks, scores, metadata) tuples
    """
    reranked_results = []

    for i, (result, query) in enumerate(zip(retrieval_results, queries)):
        try:
            chunks = result.get('chunks', [])
            distances = result.get('distances', [])

            reranked_chunks, scores, metadata = rerank_chunks(
                chunks, query, reranker_config, distances
            )

            reranked_results.append((reranked_chunks, scores, metadata))

        except Exception as e:
            logger.error(f"Error reranking query {i}: {str(e)}")
            # Return original order on error
            reranked_results.append((
                result.get('chunks', []),
                result.get('distances', []),
                {'error': str(e), 'reranker_type': 'error'}
            ))

    return reranked_results


# ----------------------------------------
# Re-ranking Evaluation
# ----------------------------------------

def evaluate_reranking_quality(
    original_chunks: List[str],
    original_scores: List[float],
    reranked_chunks: List[str],
    reranked_scores: List[float],
    ground_truth_chunks: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate the quality of reranking.

    Args:
        original_chunks: Chunks before reranking
        original_scores: Scores before reranking
        reranked_chunks: Chunks after reranking
        reranked_scores: Scores after reranking
        ground_truth_chunks: Optional ground truth relevant chunks

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}

    # Calculate rank changes
    rank_changes = []
    for i, chunk in enumerate(reranked_chunks):
        original_rank = original_chunks.index(chunk) if chunk in original_chunks else -1
        new_rank = i
        if original_rank >= 0:
            rank_changes.append(abs(new_rank - original_rank))

    if rank_changes:
        metrics['avg_rank_change'] = sum(rank_changes) / len(rank_changes)
        metrics['max_rank_change'] = max(rank_changes)

    # Score improvement
    if original_scores and reranked_scores:
        metrics['score_improvement'] = sum(reranked_scores) - sum(original_scores)

    # If ground truth provided, calculate precision improvement
    if ground_truth_chunks:
        # Precision@k for top chunks
        for k in [1, 3, 5]:
            if len(reranked_chunks) >= k:
                original_precision = calculate_precision_at_k(original_chunks, ground_truth_chunks, k)
                reranked_precision = calculate_precision_at_k(reranked_chunks, ground_truth_chunks, k)
                metrics[f'precision@{k}_improvement'] = reranked_precision - original_precision

    return metrics


def calculate_precision_at_k(chunks: List[str], ground_truth: List[str], k: int) -> float:
    """
    Calculate Precision@k.

    Args:
        chunks: Retrieved chunks
        ground_truth: Ground truth relevant chunks
        k: Number of top chunks to consider

    Returns:
        Precision score (0-1)
    """
    if k <= 0 or k > len(chunks):
        return 0.0

    top_k = chunks[:k]
    relevant_count = sum(1 for chunk in top_k if chunk in ground_truth)

    return relevant_count / k
