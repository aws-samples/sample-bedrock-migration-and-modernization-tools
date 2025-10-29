import os
import time
import concurrent.futures
import json
import logging
import uuid
import pandas as pd
import argparse
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from utils import get_timestamp, setup_logging
from embedding_utils import (
    load_embedding_model_profiles,
    generate_embeddings_batch,
    embedding_model_sanity_check,
    calculate_embedding_cost
)
from document_processors import parse_document
from chunking_strategies import chunk_text, analyze_chunks
from vectorstore_manager import (
    create_chroma_vectorstore,
    retrieve_from_vectorstore,
    get_vectorstore_stats,
    cleanup_vectorstore
)
from reranking import rerank_chunks
from rag_evaluation_engine import (
    parse_ground_truth_chunks,
    comprehensive_retrieval_evaluation
)

env = load_dotenv()


# ----------------------------------------
# Core RAG Benchmark Function
# ----------------------------------------

def rag_benchmark(
    query: str,
    ground_truth_chunks: List[str],
    collection,
    embedding_model_config: Dict,
    reranker_config: Dict,
    top_k: int = 5
) -> Dict:
    """
    Single RAG benchmark iteration.

    Args:
        query: Query text
        ground_truth_chunks: Ground truth relevant chunks
        collection: ChromaDB collection
        embedding_model_config: Embedding model configuration
        reranker_config: Reranker configuration
        top_k: Number of chunks to retrieve

    Returns:
        Dictionary with benchmark results
    """
    status = "Success"
    error_code = None

    # Initialize metrics
    query_embedding_latency = 0
    retrieval_latency = 0
    reranking_latency = 0
    total_latency = 0

    embedding_cost = 0
    reranking_cost = 0
    total_cost = 0

    retrieved_chunks = []
    reranked_chunks = []
    retrieval_distances = []
    reranking_scores = []

    evaluation_metrics = {}

    start_time = time.time()

    try:
        # 1. Generate query embedding
        embed_start = time.time()

        model_id = embedding_model_config['model_id']
        params = {}

        if "openai" in model_id:
            params['api_key'] = os.getenv('OPENAI_API')
        elif 'cohere' in model_id and 'bedrock' not in model_id:
            params['api_key'] = os.getenv('COHERE_API')
        elif 'voyage' in model_id:
            params['api_key'] = os.getenv('VOYAGE_API')
        elif 'bedrock' in model_id:
            params['aws_region_name'] = embedding_model_config.get('region', 'us-east-1')

        from embedding_utils import generate_single_embedding
        query_embedding, embed_metadata = generate_single_embedding(query, model_id, params)

        query_embedding_latency = time.time() - embed_start

        # Calculate embedding cost
        embedding_tokens = embed_metadata.get('total_tokens', 0)
        embedding_cost = calculate_embedding_cost(embedding_tokens, embedding_model_config)

        # 2. Retrieve from vector store
        retrieval_start = time.time()

        retrieval_results = retrieve_from_vectorstore(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k
        )

        retrieval_latency = time.time() - retrieval_start

        retrieved_chunks = retrieval_results['chunks']
        retrieval_distances = retrieval_results['distances']

        # 3. Apply re-ranking
        reranking_start = time.time()

        reranked_chunks, reranking_scores, reranking_metadata = rerank_chunks(
            chunks=retrieved_chunks,
            query=query,
            reranker_config=reranker_config,
            distances=retrieval_distances
        )

        reranking_latency = time.time() - reranking_start
        reranking_cost = reranking_metadata.get('cost', 0)

        # 4. Evaluate with RAGAs
        evaluation_metrics = comprehensive_retrieval_evaluation(
            query=query,
            retrieved_chunks=reranked_chunks,
            ground_truth_chunks=ground_truth_chunks
        )

        total_latency = time.time() - start_time
        total_cost = embedding_cost + reranking_cost

    except Exception as e:
        status = f"{type(e).__name__}: {str(e)}"
        error_code = type(e).__name__.upper()
        logging.error(f"Error in RAG benchmark: {str(e)}", exc_info=True)

    return {
        "job_timestamp_iso": get_timestamp(),
        "api_call_status": status,
        "error_code": error_code,

        # Query info
        "query": query,
        "ground_truth_chunks": ",".join(ground_truth_chunks),

        # Latency metrics
        "query_embedding_latency": query_embedding_latency,
        "retrieval_latency": retrieval_latency,
        "reranking_latency": reranking_latency,
        "total_latency": total_latency,

        # Cost metrics
        "embedding_cost": embedding_cost,
        "reranking_cost": reranking_cost,
        "total_cost": total_cost,

        # Retrieval results
        "retrieved_chunks": json.dumps(reranked_chunks),
        "retrieval_scores": json.dumps(reranking_scores),
        "num_chunks_retrieved": len(reranked_chunks),

        # Evaluation metrics
        **evaluation_metrics
    }


# ----------------------------------------
# Parallel Execution
# ----------------------------------------

def execute_rag_benchmark(scenarios, cfg, unprocessed_dir):
    """
    Execute RAG benchmarks in parallel.

    Args:
        scenarios: List of scenario dicts
        cfg: Configuration dict
        unprocessed_dir: Directory for unprocessed records

    Returns:
        Tuple of (all_records, unprocessed_file_path, unprocessed_count)
    """
    all_recs = []
    unprocessed_records = []
    lock = Lock()

    def run_scn(scn):
        recs = []
        local_unprocessed = []

        for invocation in range(cfg["invocations_per_scenario"]):
            try:
                # Smart logging
                total_invocations = cfg['invocations_per_scenario']
                is_first = invocation == 0
                is_last = invocation == total_invocations - 1
                is_milestone = (invocation + 1) % 10 == 0

                if is_first or is_last or is_milestone:
                    logging.info(
                        f"Running RAG scenario: {scn['embedding_model_id']}, query: '{scn['query'][:50]}...', "
                        f"invocation {invocation + 1}/{total_invocations}"
                    )
                else:
                    logging.debug(
                        f"Running RAG scenario: {scn['embedding_model_id']}, invocation {invocation + 1}/{total_invocations}"
                    )

                r = rag_benchmark(
                    query=scn["query"],
                    ground_truth_chunks=scn["ground_truth_chunks"],
                    collection=scn["collection"],
                    embedding_model_config=scn["embedding_model_config"],
                    reranker_config=scn["reranker_config"],
                    top_k=scn["top_k"]
                )

                # Check for errors
                has_error = r["api_call_status"] != "Success" or r["error_code"] is not None

                if has_error:
                    reason = f"API error: {r.get('api_call_status', 'Unknown')}"
                    logging.warning(f"Record processing failed: {reason}")

                    local_unprocessed.append({
                        "scenario": scn,
                        "result": r,
                        "reason": reason,
                        "timestamp": get_timestamp(),
                        "invocation": invocation
                    })
                else:
                    # Combine scenario and result
                    result_record = {**scn, **r}
                    # Remove collection object (not JSON serializable)
                    result_record.pop("collection", None)
                    result_record.pop("embedding_model_config", None)
                    result_record.pop("reranker_config", None)
                    result_record.pop("ground_truth_chunks", None)  # Already in result

                    # Add aliases for visualization compatibility
                    result_record["embedding_model"] = result_record.get("embedding_model_id")
                    # Convert latency from seconds to milliseconds for visualization
                    if "retrieval_latency" in result_record:
                        result_record["retrieval_latency_ms"] = result_record["retrieval_latency"] * 1000
                    if "total_latency" in result_record:
                        result_record["total_latency_ms"] = result_record["total_latency"] * 1000

                    recs.append(result_record)
                    logging.debug(f"Successfully processed invocation {invocation + 1}")

            except Exception as e:
                logging.error(f"Exception processing record: {str(e)}", exc_info=True)

                local_unprocessed.append({
                    "scenario": scn,
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                    "timestamp": get_timestamp(),
                    "invocation": invocation
                })

            if cfg.get("sleep_between_invocations", 0):
                time.sleep(cfg["sleep_between_invocations"])

        with lock:
            logging.info(
                f"Completed scenario: {scn['embedding_model_id']}, "
                f"processed: {len(recs)}, failed: {len(local_unprocessed)}"
            )
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
                logging.error(f"Exception in ThreadPoolExecutor task: {str(e)}", exc_info=True)
                with lock:
                    unprocessed_records.append({
                        "scenario": "Unknown (future failed)",
                        "exception": str(e),
                        "timestamp": get_timestamp()
                    })

    # Write unprocessed records
    unprocessed_file_path = None
    if unprocessed_records:
        ts = get_timestamp().replace(':', '-')
        uuid_ = str(uuid.uuid4()).split('-')[-1]
        experiment_name = cfg.get("EXPERIMENT_NAME", "unknown")
        unprocessed_file = os.path.join(
            unprocessed_dir,
            f"unprocessed_rag_{experiment_name}_{ts}_{uuid_}.json"
        )
        logging.warning(f"Writing {len(unprocessed_records)} unprocessed records to {unprocessed_file}")
        try:
            with open(unprocessed_file, 'w') as f:
                json.dump(unprocessed_records, f, indent=2, default=str)
            logging.info(f"Successfully wrote unprocessed records to {unprocessed_file}")
            unprocessed_file_path = unprocessed_file
        except Exception as e:
            logging.error(f"Failed to write unprocessed records: {str(e)}", exc_info=True)

    return all_recs, unprocessed_file_path, len(unprocessed_records)


# ----------------------------------------
# Main Entrypoint
# ----------------------------------------

def main(
    queries_file,
    data_source_file,
    output_dir,
    data_format,
    chunking_strategy,
    chunk_size,
    chunk_overlap,
    embedding_models_file,
    reranker_file,
    top_k,
    parallel_calls,
    invocations_per_scenario,
    sleep_between_invocations,
    experiment_counts,
    experiment_name,
    custom_chunking_script=None,
    query_column="query",
    ground_truth_column="ground_truth_chunks",
    data_source_text_field=None,
    reranker_type=None,
    reranker_id=None,
    similarity_threshold=0.5,
    semantic_chunking_model_id=None,
    report=True,
    # Rate limiting parameters
    sleep_between_calls=2.0,
    sleep_between_batches=2.0,
    batch_size=50,
    max_retries=5
):
    """
    Main RAG evaluation orchestrator.
    """
    # Get project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Create directories
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging
    ts, log_file = setup_logging(logs_dir, f"rag_{experiment_name}")
    logging.info(f"Starting RAG benchmark run: {experiment_name}")
    print(f"Logs are being saved to: {log_file}")

    uuid_ = str(uuid.uuid4()).split('-')[-1]

    # Ensure output directory is absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create directories
    unprocessed_dir = os.path.join(output_dir, "unprocessed")
    vectorstore_dir = os.path.join(output_dir, "vectorstores", f"{experiment_name}_{ts}")
    os.makedirs(unprocessed_dir, exist_ok=True)
    os.makedirs(vectorstore_dir, exist_ok=True)

    # Paths
    eval_dir = os.path.join(project_root, "prompt-evaluations")
    queries_path = os.path.join(eval_dir, queries_file)
    data_source_path = os.path.join(eval_dir, data_source_file)

    # Load queries CSV
    logging.info(f"Loading queries from: {queries_path}")
    queries_df = pd.read_csv(queries_path)

    if query_column not in queries_df.columns:
        raise ValueError(f"Query column '{query_column}' not found in CSV. Available: {list(queries_df.columns)}")
    if ground_truth_column not in queries_df.columns:
        raise ValueError(f"Ground truth column '{ground_truth_column}' not found in CSV. Available: {list(queries_df.columns)}")

    logging.info(f"Loaded {len(queries_df)} queries")

    # Parse data source file
    logging.info(f"Parsing data source file: {data_source_path}")
    texts = parse_document(data_source_path, data_format, data_source_text_field)
    logging.info(f"Parsed {len(texts)} text entries from data source")

    # Concatenate all texts for chunking
    full_text = "\n\n".join(texts)

    # Load embedding models FIRST (needed for semantic chunking)
    # Determine if embedding_models_file is a model ID(s) or a file path BEFORE any path manipulation
    # Model IDs contain provider prefixes like "openai/", "cohere/", "bedrock/" and don't end with .jsonl
    # File paths end with .jsonl
    is_model_id_list = (
        embedding_models_file and
        not embedding_models_file.endswith('.jsonl') and
        ('/' in embedding_models_file or ',' in embedding_models_file)
    )

    if is_model_id_list:
        # It's a model ID or comma-separated list of model IDs - DO NOT apply path resolution
        requested_model_ids = [m.strip() for m in embedding_models_file.split(',')]
        logging.info(f"Loading specific embedding models: {requested_model_ids}")

        # Load all models from default profiles file
        default_profiles_path = os.path.join(project_root, "default-config/embedding_models_profiles.jsonl")
        all_models = load_embedding_model_profiles(default_profiles_path)

        # Filter to requested models
        embedding_models = [m for m in all_models if m['model_id'] in requested_model_ids]

        if len(embedding_models) != len(requested_model_ids):
            found_ids = [m['model_id'] for m in embedding_models]
            missing = set(requested_model_ids) - set(found_ids)
            logging.warning(f"Some requested models not found in profiles: {missing}")

        if not embedding_models:
            raise ValueError(f"None of the requested embedding models were found: {requested_model_ids}")
    else:
        # It's a file path - apply path resolution if needed
        if not os.path.isabs(embedding_models_file):
            embedding_models_file = os.path.join(project_root, embedding_models_file)

        logging.info(f"Loading embedding models from file: {embedding_models_file}")
        embedding_models = load_embedding_model_profiles(embedding_models_file)

    # Check embedding model access
    accessible_models, failed_models = embedding_model_sanity_check(embedding_models)

    if len(accessible_models) == 0:
        logging.error(f'All embedding models failed access check: {failed_models}')
        raise RuntimeError("No accessible embedding models")
    if len(failed_models) > 0:
        logging.warning(f'Some embedding models failed: {failed_models}')

    # Chunk the text (after loading embedding models, needed for semantic chunking)
    logging.info(f"Chunking text using '{chunking_strategy}' strategy")

    chunking_kwargs = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

    # For semantic chunking, we need to provide an embedding model
    if chunking_strategy == "semantic":
        # Use specified model if provided, otherwise use first accessible model
        selected_model = None

        if semantic_chunking_model_id:
            # Find the specified model in accessible_models
            for model in accessible_models:
                if model['model_id'] == semantic_chunking_model_id:
                    selected_model = model
                    break

            if not selected_model:
                logging.warning(f"Specified semantic chunking model '{semantic_chunking_model_id}' not found or not accessible. Using first accessible model.")
                selected_model = accessible_models[0]
        else:
            # Default to first accessible model
            selected_model = accessible_models[0]

        model_id = selected_model['model_id']
        logging.info(f"Using embedding model '{model_id}' for semantic chunking with similarity threshold {similarity_threshold}")

        chunking_kwargs["embedding_model_id"] = model_id
        chunking_kwargs["similarity_threshold"] = similarity_threshold

        # Set up provider params
        params = {}
        if "openai" in model_id:
            params['api_key'] = os.getenv('OPENAI_API')
        elif 'cohere' in model_id and 'bedrock' not in model_id:
            params['api_key'] = os.getenv('COHERE_API')
        elif 'voyage' in model_id:
            params['api_key'] = os.getenv('VOYAGE_API')
        elif 'bedrock' in model_id:
            params['aws_region_name'] = selected_model.get('region', 'us-east-1')

        chunking_kwargs["embedding_params"] = params
    elif chunking_strategy == "custom" and custom_chunking_script:
        chunking_kwargs["custom_script_path"] = custom_chunking_script

    chunks = chunk_text(full_text, strategy=chunking_strategy, **chunking_kwargs)

    chunk_stats = analyze_chunks(chunks)
    logging.info(f"Created {chunk_stats['num_chunks']} chunks (avg size: {chunk_stats['avg_chunk_size']:.0f} chars)")

    # Load reranker config
    # If reranker_type is passed directly, use it; otherwise load from file
    if reranker_type is not None:
        logging.info(f"Using reranker type from arguments: {reranker_type}")
        reranker_config = {
            "type": reranker_type,
            "reranker_id": reranker_id if reranker_id else "none"
        }
    else:
        # Apply path resolution for reranker file if needed
        if not os.path.isabs(reranker_file):
            reranker_file = os.path.join(project_root, reranker_file)

        logging.info(f"Loading reranker config from: {reranker_file}")
        with open(reranker_file, 'r') as f:
            reranker_configs = [json.loads(line) for line in f if line.strip()]

        # For now, use first reranker (can be extended to compare multiple)
        reranker_config = reranker_configs[0] if reranker_configs else {"type": "none"}

    logging.info(f"Using reranker: {reranker_config.get('type', 'none')}")

    # Create vector stores for each embedding model
    vectorstores = {}

    for embedding_model in accessible_models:
        model_id = embedding_model['model_id']
        logging.info(f"Creating vector store for embedding model: {model_id}")

        # Generate embeddings for all chunks
        params = {}
        if "openai/" in model_id:
            params['api_key'] = os.getenv('OPENAI_API')
        elif 'cohere/' in model_id:
            # Cohere via API (not Bedrock)
            params['api_key'] = os.getenv('COHERE_API')
        elif 'bedrock/' in model_id:
            # Bedrock models (bedrock/*)
            params['aws_region_name'] = embedding_model.get('region', 'us-east-1')

        embeddings, embed_metadata = generate_embeddings_batch(
            texts=chunks,
            model_id=model_id,
            provider_params=params,
            batch_size=batch_size,
            sleep_between_batches=sleep_between_batches
        )

        logging.info(f"Generated {len(embeddings)} embeddings in {embed_metadata['latency_seconds']:.2f}s")

        # Create ChromaDB collection
        collection_name = f"rag_eval_{model_id.replace('/', '_').replace('.', '_').replace(':', '_')}"
        collection = create_chroma_vectorstore(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name,
            persist_dir=vectorstore_dir
        )

        vectorstores[model_id] = {
            "collection": collection,
            "embedding_model": embedding_model
        }

        stats = get_vectorstore_stats(collection)
        logging.info(f"Vector store stats: {stats}")

    # Create scenarios
    scenarios = []

    for idx, row in queries_df.iterrows():
        query = row[query_column]
        ground_truth = parse_ground_truth_chunks(row[ground_truth_column])

        for model_id, vs_data in vectorstores.items():
            scenarios.append({
                "query": query,
                "ground_truth_chunks": ground_truth,
                "embedding_model_id": model_id,
                "embedding_model_config": vs_data["embedding_model"],
                "collection": vs_data["collection"],
                "reranker_config": reranker_config,
                "reranker_id": reranker_config.get("reranker_id", "none"),
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "top_k": top_k
            })

    logging.info(f"Created {len(scenarios)} scenarios ({len(queries_df)} queries × {len(vectorstores)} models)")

    # Configuration
    cfg = {
        "parallel_calls": parallel_calls,
        "invocations_per_scenario": invocations_per_scenario,
        "sleep_between_invocations": sleep_between_invocations,
        "EXPERIMENT_NAME": experiment_name
    }

    # Run experiments
    for run in range(1, experiment_counts + 1):
        run_start_time = time.time()
        run_timestamp = datetime.now().isoformat()

        logging.info(f"=== Run {run}/{experiment_counts} (Started: {run_timestamp}) ===")

        try:
            results, unprocessed_file_path, unprocessed_count = execute_rag_benchmark(
                scenarios, cfg, unprocessed_dir
            )

            if not results:
                logging.error(f"Run {run}/{experiment_counts} produced no results.")
                if unprocessed_file_path:
                    logging.warning(f"Unprocessed records saved to: {unprocessed_file_path}")
                continue

            try:
                df = pd.DataFrame(results)
                df["run_count"] = run
                df["timestamp"] = pd.Timestamp.now()
                df["run_start_time"] = run_timestamp
                df["run_duration_seconds"] = time.time() - run_start_time

                out_csv = os.path.join(output_dir, f"rag_invocations_{run}_{ts}_{uuid_}_{experiment_name}.csv")
                df.to_csv(out_csv, index=False)

                run_duration = time.time() - run_start_time
                logging.info(f"Run {run} completed in {run_duration:.1f} seconds, results saved to {out_csv}")

            except Exception as e:
                logging.error(f"Error saving results for run {run}: {str(e)}", exc_info=True)

        except Exception as e:
            logging.error(f"Critical error in run {run}: {str(e)}", exc_info=True)
            print(f"\nRun {run} failed with error: {str(e)}. Continuing with next run...")

    # Cleanup vector stores
    logging.info("Cleaning up vector stores...")
    cleanup_vectorstore(vectorstore_dir)

    # Generate report
    if report:
        try:
            from visualize_results import create_unified_html_report
            report_path = create_unified_html_report(output_dir, ts)
            print(f"\nRAG Benchmark complete! Report: {report_path}")
            logging.info(f"RAG benchmark run complete. Report generated at {report_path}")
        except ImportError as e:
            logging.error(f"Failed to import RAG visualization module: {str(e)}")
            print("\nRAG Benchmark complete, but report generation failed due to import error.")
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}", exc_info=True)
            print("\nRAG Benchmark complete, but report generation failed.")


if __name__ == "__main__":
    from typing import List, Dict

    p = argparse.ArgumentParser(description="RAG Evaluation Benchmarking Tool")
    p.add_argument("queries_file", help="CSV file with queries and ground truth")
    p.add_argument("data_source_file", help="Data source file to chunk and embed")
    p.add_argument("--output_dir", default="benchmark-results")
    p.add_argument("--data_format", default="auto", help="Data format (csv/json/word/txt/markdown/auto)")
    p.add_argument("--chunking_strategy", default="recursive", choices=["recursive", "semantic", "fixed", "custom"])
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--chunk_overlap", type=int, default=50)
    p.add_argument("--similarity_threshold", type=float, default=0.5, help="Similarity threshold for semantic chunking (0-1)")
    p.add_argument("--embedding_models_file", default="default-config/embedding_models_profiles.jsonl")
    p.add_argument("--reranker_file", default="default-config/reranker_profiles.jsonl")
    p.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    p.add_argument("--parallel_calls", type=int, default=4)
    p.add_argument("--invocations_per_scenario", type=int, default=2)
    p.add_argument("--invocations_per_query", type=int, default=2, help="Alias for invocations_per_scenario")
    p.add_argument("--sleep_between_invocations", type=int, default=1)
    p.add_argument("--experiment_counts", type=int, default=1)
    p.add_argument("--experiment_name", default=f"RAG_Eval_{datetime.now().strftime('%Y%m%d')}")
    # Rate limiting parameters
    p.add_argument("--sleep_between_calls", type=float, default=2.0, help="Sleep time between embedding API calls (seconds)")
    p.add_argument("--sleep_between_batches", type=float, default=2.0, help="Sleep time between embedding batches (seconds)")
    p.add_argument("--batch_size", type=int, default=50, help="Embedding batch size")
    p.add_argument("--max_retries", type=int, default=5, help="Maximum retries on rate limit errors")
    p.add_argument("--custom_chunking_script", default=None)
    p.add_argument("--query_column", default="query")
    p.add_argument("--ground_truth_column", default="ground_truth_chunks")
    p.add_argument("--data_source_text_field", default=None)
    p.add_argument("--reranker_type", default="none", help="Type of reranker (none, cross-encoder, llm)")
    p.add_argument("--reranker_id", default="none", help="Specific reranker model ID")
    p.add_argument("--embedding_models", default=None, help="Comma-separated list of embedding model IDs")
    p.add_argument("--semantic_chunking_model_id", default=None, help="Specific embedding model ID to use for semantic chunking")
    p.add_argument("--report", type=lambda x: x.lower() == 'true', default=True)

    args = p.parse_args()

    # Handle invocations_per_query as alias for invocations_per_scenario
    invocations = args.invocations_per_query if hasattr(args, 'invocations_per_query') and args.invocations_per_query != 2 else args.invocations_per_scenario

    # Handle embedding models - can be passed as comma-separated string or file
    embedding_models_arg = args.embedding_models if args.embedding_models else args.embedding_models_file

    main(
        args.queries_file,
        args.data_source_file,
        args.output_dir,
        args.data_format,
        args.chunking_strategy,
        args.chunk_size,
        args.chunk_overlap,
        embedding_models_arg,
        args.reranker_file,
        args.top_k,
        args.parallel_calls,
        invocations,
        args.sleep_between_invocations,
        args.experiment_counts,
        args.experiment_name,
        args.custom_chunking_script,
        args.query_column,
        args.ground_truth_column,
        args.data_source_text_field,
        args.reranker_type,
        args.reranker_id,
        args.similarity_threshold,
        args.semantic_chunking_model_id,
        args.report,
        # Rate limiting parameters
        args.sleep_between_calls,
        args.sleep_between_batches,
        args.batch_size,
        args.max_retries
    )
