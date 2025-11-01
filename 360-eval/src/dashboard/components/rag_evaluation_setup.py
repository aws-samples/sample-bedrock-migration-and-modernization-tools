"""RAG Evaluation Setup Component for Streamlit Dashboard."""

import streamlit as st
import pandas as pd
import json
import os
from typing import List, Dict

# Import utilities
import sys
# Current file is at: 360-eval/src/dashboard/components/rag_evaluation_setup.py
# Go up 3 levels to reach 360-eval/ (project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import vectorstore validation and metadata functions
from vectorstore_manager import validate_tool_created_vectorstore, load_vectorstore_metadata


class RAGEvaluationSetupComponent:
    """Component for configuring RAG evaluations."""

    def __init__(self):
        # Current file is at: 360-eval/src/dashboard/components/rag_evaluation_setup.py
        # Go up 3 levels to reach 360-eval/ (project root)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.config_dir = os.path.join(self.project_root, "default-config")
        self.eval_dir = os.path.join(self.project_root, "prompt-evaluations")

    def render(self):
        """Render the RAG evaluation setup interface."""
        st.title("🔍 RAG Evaluation Configuration")
        st.markdown("Configure your Retrieval-Augmented Generation evaluation settings")

        # Get RAG config from session state
        if "rag_config" not in st.session_state.current_evaluation_config:
            st.session_state.current_evaluation_config["rag_config"] = self._get_default_rag_config()

        rag_config = st.session_state.current_evaluation_config["rag_config"]

        # Create tabs for different configuration sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📄 Query & Data Files",
            "✂️ Chunking Strategy",
            "🧮 Embedding Models",
            "🎯 Retrieval & Re-ranking",
            "📊 Similarity Methods"
        ])

        with tab1:
            self._render_file_config(rag_config)

        with tab2:
            self._render_chunking_config(rag_config)

        with tab3:
            self._render_embedding_config(rag_config)

        with tab4:
            self._render_retrieval_config(rag_config)

        with tab5:
            self._render_similarity_methods_config(rag_config)

        # Save configuration button
        st.divider()
        col1, col2, col3 = st.columns([2, 1, 1])

        with col2:
            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                self._save_configuration(rag_config)

        with col3:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                self._reset_configuration()

    def _render_file_config(self, rag_config):
        """Render file upload and configuration section."""
        # Experiment Name - MANDATORY at the top
        st.markdown("### 🏷️ Evaluation Information")

        experiment_name = st.text_input(
            "Experiment Name *",
            value=st.session_state.current_evaluation_config.get("name", f"RAG_Eval_{pd.Timestamp.now().strftime('%Y%m%d')}"),
            key="rag_experiment_name",
            placeholder="e.g., RAG_Eval_20251028, prod-embedding-test",
            help="**Required**: Give this evaluation a descriptive name for easy identification"
        )

        if not experiment_name or not experiment_name.strip():
            st.error("⚠️ Experiment Name is required")
        else:
            st.session_state.current_evaluation_config["name"] = experiment_name.strip()

        st.divider()

        st.markdown("### 📊 Query File (CSV)")
        st.info("Upload a CSV file containing your test queries and ground truth chunks")

        # Query file uploader
        queries_file = st.file_uploader(
            "Upload Queries CSV",
            type=["csv"],
            key="rag_queries_uploader",
            help="CSV file with queries and ground truth chunks"
        )

        if queries_file is not None:
            try:
                queries_df = pd.read_csv(queries_file)
                # Store as dict records for serialization
                rag_config["queries_csv_data"] = queries_df.to_dict('records')
                rag_config["queries_file_name"] = queries_file.name

                st.success(f"✓ Loaded {len(queries_df)} queries from {queries_file.name}")

                # Column selectors
                col1, col2 = st.columns(2)

                with col1:
                    query_col = st.selectbox(
                        "Query Column",
                        options=queries_df.columns.tolist(),
                        index=self._get_column_index(queries_df.columns, rag_config.get("query_column", "query")),
                        key="rag_query_column",
                        help="Column containing the query text"
                    )
                    rag_config["query_column"] = query_col

                with col2:
                    # Auto-detect ground truth column (support both naming conventions)
                    default_gt_col = rag_config.get("ground_truth_column", "ground_truth_chunks")
                    # Check for common column name variations
                    if default_gt_col not in queries_df.columns:
                        for alt_name in ["expected_contexts", "ground_truth", "golden_contexts", "gt_chunks"]:
                            if alt_name in queries_df.columns:
                                default_gt_col = alt_name
                                break

                    gt_col = st.selectbox(
                        "Ground Truth Chunks Column",
                        options=queries_df.columns.tolist(),
                        index=self._get_column_index(queries_df.columns, default_gt_col),
                        key="rag_gt_column",
                        help="Column containing ground truth chunks (auto-detects 'expected_contexts')"
                    )
                    rag_config["ground_truth_column"] = gt_col

                # Preview
                with st.expander("👁️ Preview Query Data"):
                    st.dataframe(queries_df.head(), use_container_width=True)

            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

        st.divider()

        # Data source file uploader
        st.markdown("### 📚 Data Source Files")
        st.info("Upload one or more documents that will be chunked and used for retrieval")

        data_source_files = st.file_uploader(
            "Upload Data Source(s)",
            type=["csv", "json", "jsonl", "txt", "md", "docx", "pdf"],
            accept_multiple_files=True,
            key="rag_data_source_uploader",
            help="Upload one or more documents. All files will be combined and chunked together."
        )

        if data_source_files:
            # Process multiple files
            try:
                import base64

                # Initialize list to store file data
                files_data = []
                file_names = []

                for uploaded_file in data_source_files:
                    file_ext = uploaded_file.name.split(".")[-1].lower()

                    # Read file content
                    if file_ext in ["csv", "json", "jsonl", "txt", "md"]:
                        # Text files - read as string
                        content = uploaded_file.getvalue().decode("utf-8")
                        is_binary = False
                    else:
                        # Binary files (docx, pdf) - store as base64
                        content = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                        is_binary = True

                    # Store file metadata
                    files_data.append({
                        "content": content,
                        "name": uploaded_file.name,
                        "ext": file_ext,
                        "is_binary": is_binary
                    })
                    file_names.append(uploaded_file.name)

                # Store all files in rag_config
                rag_config["data_source_files"] = files_data
                rag_config["data_source_file_names"] = file_names

                # Keep single file fields for backward compatibility (use first file)
                rag_config["data_source_file"] = files_data[0]["content"]
                rag_config["data_source_file_name"] = files_data[0]["name"]
                rag_config["data_source_file_ext"] = files_data[0]["ext"]
                rag_config["data_source_is_binary"] = files_data[0]["is_binary"]

                # Auto-detect format from first file
                format_map = {
                    "csv": "csv", "json": "json", "jsonl": "jsonl",
                    "txt": "txt", "md": "markdown", "docx": "word", "pdf": "pdf"
                }
                detected_format = format_map.get(files_data[0]["ext"], "txt")
                rag_config["data_format"] = detected_format

                # Show success message
                if len(files_data) == 1:
                    st.success(f"✓ Loaded {files_data[0]['name']}")
                else:
                    st.success(f"✓ Loaded {len(files_data)} files: {', '.join(file_names)}")

            except Exception as e:
                st.error(f"Error reading data source files: {str(e)}")

    def _render_chunking_config(self, rag_config):
        """Render chunking strategy configuration."""
        st.markdown("### ✂️ Chunking Strategy")

        strategy = st.radio(
            "Select Chunking Strategy",
            options=["recursive", "semantic", "fixed", "custom"],
            format_func=lambda x: {
                "recursive": "🌲 Recursive (hierarchical separators)",
                "semantic": "🧠 Semantic (meaning-based)",
                "fixed": "📏 Fixed-Size (sliding window)",
                "custom": "🔧 Custom (user script)"
            }[x],
            key="rag_chunking_strategy",
            horizontal=True
        )
        rag_config["chunking_strategy"] = strategy

        st.divider()

        # Strategy-specific configuration
        if strategy == "recursive":
            st.markdown("**Recursive Chunking Parameters**")
            st.caption("Splits text using hierarchical separators (paragraphs → sentences → words)")

            col1, col2 = st.columns(2)

            with col1:
                chunk_size = st.slider(
                    "Chunk Size (characters)",
                    min_value=128,
                    max_value=2048,
                    value=rag_config.get("chunk_size", 512),
                    step=64,
                    key="rag_chunk_size_recursive"
                )
                rag_config["chunk_size"] = chunk_size

            with col2:
                chunk_overlap = st.slider(
                    "Chunk Overlap (characters)",
                    min_value=0,
                    max_value=200,
                    value=rag_config.get("chunk_overlap", 50),
                    step=10,
                    key="rag_chunk_overlap_recursive"
                )
                rag_config["chunk_overlap"] = chunk_overlap

        elif strategy == "semantic":
            # Dropdown to select embedding model for semantic chunking (FIRST)
            st.markdown("**Embedding Model for Semantic Chunking**")
            st.caption("Select ONE model to use for computing semantic similarity during chunking")

            # Load available embedding models
            embedding_profiles_path = os.path.join(self.config_dir, "embedding_models_profiles.jsonl")
            if os.path.exists(embedding_profiles_path):
                try:
                    with open(embedding_profiles_path, 'r') as f:
                        all_models = [json.loads(line) for line in f if line.strip()]

                    model_options = [m["model_id"] for m in all_models]

                    # Get current selection
                    current_selection = rag_config.get("semantic_chunking_model_id")
                    default_index = 0
                    if current_selection and current_selection in model_options:
                        default_index = model_options.index(current_selection)

                    selected_model_id = st.selectbox(
                        "Select model for semantic chunking",
                        options=model_options,
                        index=default_index,
                        key="semantic_chunking_model_select",
                        help="This model will be used to compute semantic similarity for chunking"
                    )

                    rag_config["semantic_chunking_model_id"] = selected_model_id

                    # Display selected model info
                    selected_model = next((m for m in all_models if m["model_id"] == selected_model_id), None)
                    if selected_model:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"📏 Dimensions: {selected_model.get('dimensions', 'N/A')}")
                        with col2:
                            st.caption(f"💰 Cost: ${selected_model.get('input_token_cost', 0)}/1K tokens")

                except Exception as e:
                    st.error(f"Error loading embedding models: {str(e)}")
            else:
                st.warning("Embedding models profile file not found")

            # Semantic Chunking Parameters (SECOND)
            st.markdown("**Semantic Chunking Parameters**")

            col1, col2 = st.columns(2)

            with col1:
                chunk_size = st.slider(
                    "Max Chunk Size",
                    min_value=256,
                    max_value=2048,
                    value=rag_config.get("chunk_size", 1000),
                    step=64,
                    key="rag_chunk_size_semantic"
                )
                rag_config["chunk_size"] = chunk_size

            with col2:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.3,
                    max_value=0.9,
                    value=rag_config.get("similarity_threshold", 0.5),
                    step=0.05,
                    key="rag_similarity_threshold",
                    help="Lower = more chunks, Higher = fewer chunks"
                )
                rag_config["similarity_threshold"] = similarity_threshold

        elif strategy == "fixed":
            st.markdown("**Fixed-Size Chunking Parameters**")
            st.caption("Simple sliding window with fixed size and overlap")

            col1, col2 = st.columns(2)

            with col1:
                chunk_size = st.slider(
                    "Chunk Size (characters)",
                    min_value=128,
                    max_value=2048,
                    value=rag_config.get("chunk_size", 512),
                    step=64,
                    key="rag_chunk_size_fixed"
                )
                rag_config["chunk_size"] = chunk_size

            with col2:
                chunk_overlap = st.slider(
                    "Chunk Overlap (characters)",
                    min_value=0,
                    max_value=200,
                    value=rag_config.get("chunk_overlap", 50),
                    step=10,
                    key="rag_chunk_overlap_fixed"
                )
                rag_config["chunk_overlap"] = chunk_overlap

        elif strategy == "custom":
            st.markdown("**Custom Chunking Script**")
            st.caption("Provide a Python script with a `chunk_text(text: str, **kwargs) -> List[str]` function")

            custom_script = st.file_uploader(
                "Upload Custom Chunking Script (.py)",
                type=["py"],
                key="rag_custom_script"
            )

            if custom_script is not None:
                # Save script temporarily
                script_path = os.path.join(self.eval_dir, f"custom_chunker_{custom_script.name}")
                with open(script_path, "wb") as f:
                    f.write(custom_script.getvalue())
                rag_config["custom_chunking_script"] = script_path
                st.success(f"✓ Loaded custom script: {custom_script.name}")

            with st.expander("📖 Script Requirements"):
                st.code("""
# Your script must define this function:
def chunk_text(text: str, **kwargs) -> List[str]:
    \"\"\"
    Chunk the input text.

    Args:
        text: Full text to chunk
        **kwargs: Additional parameters (chunk_size, etc.)

    Returns:
        List of text chunks
    \"\"\"
    chunks = []
    # Your chunking logic here
    return chunks
                """, language="python")

    def _render_embedding_config(self, rag_config):
        """Render embedding model selection."""
        st.markdown("### 🧮 Embedding Models")

        # Region selector for Bedrock embedding models
        aws_regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "eu-central-1"]
        selected_region = st.selectbox(
            "AWS Region (for Bedrock embeddings)",
            options=aws_regions,
            index=0,
            key="rag_embedding_aws_region",
            help="Select the AWS region for Bedrock embedding models. Only models available in this region will be shown."
        )
        rag_config["aws_embedding_region"] = selected_region

        # Load embedding model profiles
        embedding_profiles_path = os.path.join(self.config_dir, "embedding_models_profiles.jsonl")

        if not os.path.exists(embedding_profiles_path):
            st.error(f"Embedding model profile file not found: {embedding_profiles_path}")
            return

        try:
            with open(embedding_profiles_path, 'r') as f:
                all_embedding_models = [json.loads(line) for line in f if line.strip()]

            # Filter models by selected region for Bedrock models
            region_filtered_models = []
            bedrock_models_count = 0
            non_bedrock_count = 0

            for model in all_embedding_models:
                model_id = model.get("model_id", "")
                # If it's a Bedrock model, check region compatibility
                if model_id.startswith("bedrock/"):
                    model_region = model.get("region")
                    if model_region == selected_region:
                        region_filtered_models.append(model)
                        bedrock_models_count += 1
                else:
                    # Non-Bedrock models (OpenAI, etc.) - always available
                    region_filtered_models.append(model)
                    non_bedrock_count += 1

            # Display region filtering info
            total_available = bedrock_models_count + non_bedrock_count
            if bedrock_models_count > 0:
                st.info(f"📍 **{selected_region}**: {bedrock_models_count} Bedrock model(s) available" +
                       (f" + {non_bedrock_count} non-Bedrock model(s)" if non_bedrock_count > 0 else ""))
            elif non_bedrock_count > 0:
                st.warning(f"⚠️ No Bedrock models available in **{selected_region}**. Showing {non_bedrock_count} non-Bedrock model(s) only.")
            else:
                st.error(f"❌ No embedding models available in **{selected_region}**")

            st.markdown("---")

            # Get already selected model keys (model_id + dimensions for uniqueness)
            selected_keys = [
                f"{m['model_id']}|{m.get('dimensions', 'default')}"
                for m in rag_config.get("embedding_models", [])
            ]

            # Filter to show only unselected models in dropdown
            available_models = [
                m for m in region_filtered_models
                if f"{m['model_id']}|{m.get('dimensions', 'default')}" not in selected_keys
            ]

            # Create display options with dimensions for clarity
            model_options = [
                f"{m['model_id']} ({m.get('dimensions', 'N/A')}D)"
                for m in available_models
            ]
            # Map display back to model index
            model_index_map = {opt: idx for idx, opt in enumerate(model_options)}

            if model_options:
                # Render dropdown + Add button
                col1, col2, col3 = st.columns([4, 1.2, 1.2])

                with col1:
                    selected_display = st.selectbox(
                        "Select Embedding Model",
                        options=model_options,
                        key="embedding_model_select"
                    )

                # Find selected model details
                selected_idx = model_index_map.get(selected_display, 0)
                selected_model = available_models[selected_idx] if available_models else None

                with col2:
                    if selected_model:
                        st.metric("Dimensions", selected_model.get("dimensions", "N/A"))
                        st.caption(f"${selected_model.get('input_token_cost', 0)}/1K")

                with col3:
                    st.button(
                        "Add Model",
                        key="add_embedding_model",
                        on_click=self._add_embedding_model,
                        args=(selected_model, rag_config)
                    )
            else:
                st.info("All available embedding models have been added")

            # Display selected models (same pattern as LLM selection)
            st.markdown("---")
            st.subheader("Selected Embedding Models")

            if not rag_config.get("embedding_models"):
                st.info("No embedding models selected. Please add at least one model.")
            else:
                # Create dataframe for display
                display_data = []
                for model in rag_config["embedding_models"]:
                    row = {
                        "Model ID": model["model_id"],
                        "Dimensions": model.get("dimensions", "N/A"),
                        "Cost (per 1K)": f"${model.get('input_token_cost', 0)}"
                    }
                    # Add region for Bedrock models
                    if model["model_id"].startswith("bedrock/"):
                        row["Region"] = model.get("region", "N/A")
                    display_data.append(row)

                df = pd.DataFrame(display_data)
                st.dataframe(df, hide_index=True)

                # Clear button (same as LLM selection)
                st.button(
                    "Clear Selected Models",
                    on_click=self._clear_embedding_models,
                    args=(rag_config,)
                )

        except Exception as e:
            st.error(f"Error loading embedding models: {str(e)}")

    def _get_available_vectorstores(self) -> List[Dict[str, str]]:
        """
        Get list of available vectorstores from benchmark-results/vectorstores/.

        Returns:
            List of dicts with 'path' and 'display_name' keys, sorted by creation time (newest first)
        """
        vectorstores = []
        vectorstores_dir = os.path.join(self.project_root, "benchmark-results", "vectorstores")

        if not os.path.exists(vectorstores_dir):
            return vectorstores

        try:
            # List all directories in vectorstores/
            for entry in os.listdir(vectorstores_dir):
                entry_path = os.path.join(vectorstores_dir, entry)

                # Check if it's a directory
                if not os.path.isdir(entry_path):
                    continue

                # Check if it has metadata.json (tool-created vectorstore)
                metadata_path = os.path.join(entry_path, "metadata.json")
                if not os.path.exists(metadata_path):
                    continue

                # Load metadata to get creation time and experiment name
                try:
                    metadata = load_vectorstore_metadata(entry_path)
                    if metadata:
                        created_at = metadata.get("created_at", "Unknown")
                        experiment_name = metadata.get("experiment_name", entry)
                        num_chunks = metadata.get("num_chunks", "?")
                        num_models = len(metadata.get("embedding_models", []))

                        # Create display name with useful info
                        display_name = f"{experiment_name} ({num_chunks} chunks, {num_models} models) - {created_at[:10]}"

                        vectorstores.append({
                            "path": entry_path,
                            "relative_path": os.path.relpath(entry_path, self.project_root),
                            "display_name": display_name,
                            "created_at": created_at,
                            "experiment_name": experiment_name
                        })
                except Exception as e:
                    # Skip this vectorstore if metadata is invalid
                    continue

            # Sort by creation time (newest first)
            vectorstores.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        except Exception as e:
            # Return empty list on error
            pass

        return vectorstores

    def _render_retrieval_config(self, rag_config):
        """Render retrieval and re-ranking configuration."""
        st.markdown("### 🎯 Retrieval Parameters")

        col1, col2 = st.columns(2)

        with col1:
            top_k = st.slider(
                "Top-K Retrieval",
                min_value=1,
                max_value=20,
                value=rag_config.get("top_k", 5),
                step=1,
                key="rag_top_k",
                help="Number of chunks to retrieve per query"
            )
            rag_config["top_k"] = top_k

        with col2:
            invocations = st.number_input(
                "Invocations per Query",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                key="rag_invocations",
                help="Repeat each query N times for statistical analysis"
            )
            rag_config["invocations_per_scenario"] = invocations

        st.divider()

        # Re-ranking configuration
        st.markdown("### 🔄 Re-ranking Strategy")

        reranker_profiles_path = os.path.join(self.config_dir, "reranker_profiles.jsonl")

        if not os.path.exists(reranker_profiles_path):
            st.warning(f"Reranker profiles file not found: {reranker_profiles_path}")
            rag_config["reranker_config"] = {"type": "none", "reranker_id": "none"}
            return

        try:
            with open(reranker_profiles_path, 'r') as f:
                rerankers = [json.loads(line) for line in f if line.strip()]

            reranker_options = {r["reranker_id"]: r for r in rerankers}
            selected_reranker_id = st.selectbox(
                "Select Reranker",
                options=list(reranker_options.keys()),
                index=0,
                key="rag_reranker_select",
                format_func=lambda x: f"{x} ({reranker_options[x]['type']})"
            )

            selected_reranker = reranker_options[selected_reranker_id]
            rag_config["reranker_config"] = selected_reranker

            # Show reranker details
            with st.expander("ℹ️ Reranker Details"):
                st.json(selected_reranker)

        except Exception as e:
            st.error(f"Error loading rerankers: {str(e)}")
            rag_config["reranker_config"] = {"type": "none", "reranker_id": "none"}

        st.divider()

        # Advanced settings
        st.markdown("### ⚙️ Advanced Settings")
        st.markdown("Configure rate limiting and execution parameters to avoid API throttling.")

        # Rate Limiting Parameters
        st.markdown("#### 🚦 Rate Limiting")
        col1, col2 = st.columns(2)

        with col1:
            parallel_calls = st.number_input(
                "Parallel Query Execution",
                min_value=1,
                max_value=10,
                value=rag_config.get("parallel_calls", 1),
                step=1,
                key="rag_parallel",
                help="Number of queries to process concurrently. Lower values (1-2) help avoid rate limits. Use 1 for Bedrock APIs with strict rate limits."
            )
            rag_config["parallel_calls"] = parallel_calls

            sleep_between_calls = st.number_input(
                "Sleep Between API Calls (seconds)",
                min_value=0.0,
                max_value=60.0,
                value=rag_config.get("sleep_between_calls", 2.0),
                step=0.5,
                key="rag_sleep_between_calls",
                help="Pause time between embedding API calls. Recommended: 2-5 seconds for Bedrock. Higher values = slower but more reliable."
            )
            rag_config["sleep_between_calls"] = sleep_between_calls

            batch_size = st.number_input(
                "Embedding Batch Size",
                min_value=1,
                max_value=100,
                value=rag_config.get("batch_size", 50),
                step=10,
                key="rag_batch_size",
                help="Number of texts to embed in each batch. Lower values (10-50) help avoid rate limits. Use 50 for most use cases."
            )
            rag_config["batch_size"] = batch_size

        with col2:
            invocations_per_query = st.number_input(
                "Invocations per Query",
                min_value=1,
                max_value=10,
                value=rag_config.get("invocations_per_query", 1),
                step=1,
                key="rag_invocations_per_query",
                help="How many times to run each query. More invocations = more reliable results but longer execution. Use 1 for quick testing, 3-5 for production."
            )
            rag_config["invocations_per_query"] = invocations_per_query

            sleep_between_batches = st.number_input(
                "Sleep Between Embedding Batches (seconds)",
                min_value=0.0,
                max_value=30.0,
                value=rag_config.get("sleep_between_batches", 2.0),
                step=0.5,
                key="rag_sleep_between_batches",
                help="Additional pause between embedding batches. Recommended: 2-5 seconds for Bedrock. Helps prevent burst rate limits."
            )
            rag_config["sleep_between_batches"] = sleep_between_batches

            max_retries = st.number_input(
                "Max Retries on Rate Limit",
                min_value=3,
                max_value=10,
                value=rag_config.get("max_retries", 5),
                step=1,
                key="rag_max_retries",
                help="Maximum retry attempts when hitting rate limits. Higher values = more resilient but slower on failures. Use 5 for production."
            )
            rag_config["max_retries"] = max_retries

        st.divider()

        # Vectorstore Management
        st.markdown("### 🗄️ Vectorstore Management")
        st.markdown("Configure vectorstore lifecycle and reuse options.")

        # Keep Vectorstore checkbox
        keep_vectorstore = st.checkbox(
            "Keep Vectorstore After Completion",
            value=rag_config.get("keep_vectorstore", False),
            key="rag_keep_vectorstore",
            help="If checked, the vectorstore will be preserved after evaluation completes. Otherwise, it will be automatically deleted to save disk space. Vectorstores are always kept if the evaluation fails (for debugging)."
        )
        rag_config["keep_vectorstore"] = keep_vectorstore

        st.markdown("#### 📂 Use Existing Vectorstore")
        st.markdown("Load a previously created vectorstore instead of generating new embeddings.")

        # Use existing vectorstore checkbox
        use_existing = st.checkbox(
            "Use Existing Vectorstore",
            value=rag_config.get("use_existing_vectorstore", False),
            key="rag_use_existing_vectorstore",
            help="Load a tool-created vectorstore to skip embedding generation. The tool validates chunking compatibility and can add new models incrementally."
        )
        rag_config["use_existing_vectorstore"] = use_existing

        if use_existing:
            # Get available vectorstores
            available_vectorstores = self._get_available_vectorstores()

            if not available_vectorstores:
                st.warning("⚠️ No existing vectorstores found in benchmark-results/vectorstores/")
                st.info("💡 Create a new evaluation first with 'Keep Vectorstore After Completion' enabled, then you can reuse it here.")
                rag_config["existing_vectorstore_path"] = ""
            else:
                # Dropdown to select vectorstore
                st.markdown("**Select Vectorstore**")

                # Create display options
                display_options = [vs["display_name"] for vs in available_vectorstores]

                # Get current selection index
                current_path = rag_config.get("existing_vectorstore_path", "")
                default_index = 0

                # Try to find current selection in available vectorstores
                if current_path:
                    for i, vs in enumerate(available_vectorstores):
                        if vs["path"] == current_path or vs["relative_path"] == current_path:
                            default_index = i
                            break

                selected_display_name = st.selectbox(
                    "Available Vectorstores",
                    options=display_options,
                    index=default_index,
                    key="rag_vectorstore_dropdown",
                    help="Select a previously created vectorstore. Sorted by creation time (newest first)."
                )

                # Get the selected vectorstore details
                selected_vs = available_vectorstores[display_options.index(selected_display_name)]

                # Check if vectorstore changed (to auto-populate on change)
                previous_vectorstore = rag_config.get("_previous_vectorstore_path", "")
                current_vectorstore = selected_vs["relative_path"]

                rag_config["existing_vectorstore_path"] = current_vectorstore

                # Show selected path
                st.caption(f"📁 Path: `{selected_vs['relative_path']}`")

                # Auto-populate settings from vectorstore metadata (on selection change or button click)
                st.divider()

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("💡 Click below to load settings from this vectorstore")
                with col2:
                    if st.button("⚡ Load", key="rag_autopop_settings", use_container_width=True):
                        self._autopop_from_vectorstore(rag_config, selected_vs)
                        rag_config["_previous_vectorstore_path"] = current_vectorstore

            # Validation button
            st.divider()
            if st.button("🔍 Validate Vectorstore", key="rag_validate_vectorstore", use_container_width=True):
                existing_path = rag_config.get("existing_vectorstore_path", "")
                if not existing_path:
                    st.error("⚠️ Please select a vectorstore first.")
                else:
                    # Get embedding models and chunking params for validation
                    embedding_models = rag_config.get("embedding_models", [])
                    chunking_params = {
                        "chunk_size": rag_config.get("chunk_size", 1000),
                        "chunk_overlap": rag_config.get("chunk_overlap", 200),
                        "chunking_strategy": rag_config.get("chunking_strategy", "recursive")
                    }

                    if not embedding_models:
                        st.warning("⚠️ Please select embedding models first to validate vectorstore compatibility.")
                    else:
                        # Validate tool-created vectorstore
                        with st.spinner("Validating tool-created vectorstore..."):
                            is_valid, found_models, missing_models, metadata = validate_tool_created_vectorstore(
                                existing_path,
                                embedding_models,
                                chunking_params
                            )

                            if is_valid:
                                # Valid tool-created vectorstore
                                if missing_models:
                                    # Partial match - incremental addition
                                    st.warning(f"⚠️ Partial match: {len(found_models)} collection(s) exist, {len(missing_models)} will be added")
                                    st.info("**Incremental Addition:** Missing collections will be created directly in the existing vectorstore.")
                                else:
                                    # All collections found
                                    st.success(f"✅ All {len(found_models)} collection(s) found! Existing vectorstore will be used.")

                                # Display metadata info
                                if metadata:
                                    st.markdown("**Vectorstore Metadata:**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.info(f"**Created:** {metadata.get('created_at', 'Unknown')[:10]}")
                                        st.info(f"**Experiment:** {metadata.get('experiment_name', 'Unknown')}")
                                    with col2:
                                        st.info(f"**Chunks:** {metadata.get('num_chunks', 'Unknown')}")
                                        st.info(f"**Tool Version:** {metadata.get('tool_version', 'Unknown')}")

                                    # Display chunking params
                                    stored_chunking = metadata.get('chunking_params', {})
                                    st.markdown("**Chunking Parameters:**")
                                    st.text(f"• Size: {stored_chunking.get('chunk_size', 'Unknown')}")
                                    st.text(f"• Overlap: {stored_chunking.get('chunk_overlap', 'Unknown')}")
                                    st.text(f"• Strategy: {stored_chunking.get('chunking_strategy', 'Unknown')}")

                                # Display found models
                                if found_models:
                                    st.markdown("**Found Models:**")
                                    for model in found_models:
                                        st.text(f"✓ {model.get('model_id', 'Unknown')}")

                                # List missing models if any
                                if missing_models:
                                    st.markdown("**Will Add Collections For:**")
                                    for model in missing_models:
                                        st.text(f"+ {model.get('model_id', 'Unknown')}")

                            else:
                                # Invalid or incompatible vectorstore
                                if not metadata:
                                    st.error("❌ This vectorstore was not created by this tool (missing metadata.json). Only tool-created vectorstores can be reused.")
                                else:
                                    st.error("❌ Vectorstore is incompatible:")
                                    st.text("• Chunking parameters don't match current configuration")
                                    st.text("• Please create a new vectorstore or adjust your chunking settings")

            # Show info about tool-created vectorstores
            st.info("ℹ️ **Tool-Created Only:** Only vectorstores created by this tool can be reused. The tool validates chunking compatibility and supports incremental model addition.")

    def _render_similarity_methods_config(self, rag_config):
        """Render similarity methods configuration."""
        st.markdown("### 📊 Similarity Calculation Methods")
        st.info("Select one or more methods to calculate similarity between retrieved and ground truth chunks. Each method will generate separate metrics in the report.")

        # Load similarity profiles
        similarity_profiles_path = os.path.join(self.config_dir, "similarity_profiles.jsonl")

        if not os.path.exists(similarity_profiles_path):
            st.error(f"Similarity profiles file not found: {similarity_profiles_path}")
            return

        try:
            with open(similarity_profiles_path, 'r') as f:
                similarity_methods = [json.loads(line) for line in f if line.strip()]

            # Initialize similarity_methods in config if not present
            if "similarity_methods" not in rag_config:
                rag_config["similarity_methods"] = []

            # Get currently selected methods
            selected_methods = {m["method"]: m for m in rag_config["similarity_methods"]}

            st.markdown("#### Select Similarity Methods")

            # Render checkboxes for each method
            for method_profile in similarity_methods:
                method_name = method_profile["method"]
                display_name = method_profile["display_name"]
                description = method_profile["description"]

                # Checkbox for method selection
                is_selected = st.checkbox(
                    f"**{display_name}**",
                    value=method_name in selected_methods,
                    key=f"similarity_method_{method_name}",
                    help=description
                )

                if is_selected:
                    # Method is selected - show configuration
                    with st.expander(f"⚙️ Configure {display_name}", expanded=False):
                        # Threshold configuration
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            threshold = st.slider(
                                "Similarity Threshold",
                                min_value=0.0,
                                max_value=1.0,
                                value=selected_methods.get(method_name, {}).get("threshold", method_profile["default_threshold"]),
                                step=0.05,
                                key=f"similarity_threshold_{method_name}",
                                help="Minimum similarity score to consider chunks as matching"
                            )

                        with col2:
                            st.metric("Default", f"{method_profile['default_threshold']:.2f}")

                        # Method-specific configuration
                        method_config = {
                            "method": method_name,
                            "threshold": threshold,
                            "display_name": display_name
                        }

                        # Sentence Transformer: model selection
                        if method_name == "sentence_transformer":
                            st.markdown("**Model Selection**")
                            model_options = method_profile["model_options"]
                            current_model = selected_methods.get(method_name, {}).get("model_id", method_profile["default_model"])

                            selected_model = st.selectbox(
                                "Sentence Transformer Model",
                                options=model_options,
                                index=model_options.index(current_model) if current_model in model_options else 0,
                                key=f"similarity_st_model_{method_name}",
                                help="HuggingFace Sentence Transformer model for semantic similarity"
                            )
                            method_config["model_id"] = selected_model

                        # LLM Judge: model selection from models_profiles.jsonl
                        elif method_name == "llm_judge":
                            st.markdown("**LLM Model Selection**")

                            # Load LLM models from models_profiles.jsonl
                            models_profiles_path = os.path.join(self.config_dir, "models_profiles.jsonl")
                            if os.path.exists(models_profiles_path):
                                with open(models_profiles_path, 'r') as f:
                                    llm_models = [json.loads(line) for line in f if line.strip()]

                                llm_model_ids = [m["model_id"] for m in llm_models]
                                current_llm = selected_methods.get(method_name, {}).get("model_id", llm_model_ids[0] if llm_model_ids else None)

                                if llm_model_ids:
                                    selected_llm = st.selectbox(
                                        "LLM Model for Judging",
                                        options=llm_model_ids,
                                        index=llm_model_ids.index(current_llm) if current_llm in llm_model_ids else 0,
                                        key=f"similarity_llm_model_{method_name}",
                                        help="LLM model to use for evaluating semantic similarity"
                                    )
                                    method_config["model_id"] = selected_llm

                                    # Show cost warning
                                    st.warning("⚠️ LLM-as-Judge incurs API costs for each chunk comparison. Use for final evaluation only.")
                                else:
                                    st.error("No LLM models found in models_profiles.jsonl")
                            else:
                                st.error(f"Models profiles file not found: {models_profiles_path}")

                        # Cosine: note that it uses embedding model
                        elif method_name == "cosine":
                            st.info("ℹ️ Cosine similarity will use embeddings from your selected embedding model(s)")

                        # Add or update method in config
                        if method_name in selected_methods:
                            # Update existing
                            for i, m in enumerate(rag_config["similarity_methods"]):
                                if m["method"] == method_name:
                                    rag_config["similarity_methods"][i] = method_config
                                    break
                        else:
                            # Add new
                            rag_config["similarity_methods"].append(method_config)

                else:
                    # Method is not selected - remove from config
                    if method_name in selected_methods:
                        rag_config["similarity_methods"] = [
                            m for m in rag_config["similarity_methods"] if m["method"] != method_name
                        ]

            # Show summary
            st.divider()
            st.markdown("#### Selected Methods Summary")

            if not rag_config["similarity_methods"]:
                st.warning("⚠️ No similarity methods selected. Default to Jaccard similarity.")
            else:
                summary_data = []
                for method in rag_config["similarity_methods"]:
                    summary_data.append({
                        "Method": method["display_name"],
                        "Threshold": f"{method['threshold']:.2f}",
                        "Model": method.get("model_id", "N/A")
                    })

                df = pd.DataFrame(summary_data)
                st.dataframe(df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading similarity methods: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    def _save_configuration(self, rag_config):
        """Save RAG configuration."""
        # Validate
        errors = self._validate_configuration(rag_config)

        if errors:
            st.error("❌ Configuration Invalid:")
            for error in errors:
                st.error(f"  • {error}")
            return

        # Save
        try:
            st.session_state.current_evaluation_config["rag_config"] = rag_config
            st.session_state.current_evaluation_config["evaluation_type"] = "rag"

            # Persist to disk
            from src.dashboard.utils.state_management import save_current_evaluation
            save_current_evaluation()

            st.success("✅ RAG configuration saved successfully!")
            st.info("Go to the **Monitor** tab to run your evaluation")

        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")

    def _validate_configuration(self, rag_config) -> List[str]:
        """Validate RAG configuration."""
        errors = []

        # Validate experiment name (mandatory)
        experiment_name = st.session_state.current_evaluation_config.get("name", "").strip()
        if not experiment_name:
            errors.append("Experiment Name is required")

        if rag_config.get("queries_csv_data") is None:
            errors.append("Query CSV file is required")

        if rag_config.get("data_source_file") is None:
            errors.append("Data source file is required")

        if not rag_config.get("embedding_models"):
            errors.append("At least one embedding model must be selected")

        if rag_config.get("chunk_size", 0) <= 0:
            errors.append("Chunk size must be greater than 0")

        if rag_config.get("chunk_overlap", -1) < 0:
            errors.append("Chunk overlap must be non-negative")

        if rag_config.get("top_k", 0) <= 0:
            errors.append("Top-K must be greater than 0")

        return errors

    def _reset_configuration(self):
        """Reset RAG configuration to defaults."""
        st.session_state.current_evaluation_config["rag_config"] = self._get_default_rag_config()
        st.success("✅ Configuration reset to defaults")
        st.rerun()

    def _get_default_rag_config(self) -> Dict:
        """Get default RAG configuration."""
        return {
            "queries_csv_data": None,
            "queries_file_name": None,
            "query_column": "query",
            "ground_truth_column": "ground_truth_chunks",
            "data_source_file": None,
            "data_source_file_name": None,
            "data_source_file_ext": "txt",
            "data_source_is_binary": False,
            "data_format": "auto",
            "data_source_text_field": None,
            "chunking_strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "similarity_threshold": 0.5,
            "semantic_chunking_model_id": None,
            "custom_chunking_script": None,
            "embedding_models": [],
            "reranker_config": {"type": "none", "reranker_id": "none"},
            "top_k": 5,
            "invocations_per_scenario": 2,
            "parallel_calls": 1,
            # Rate limiting parameters
            "sleep_between_calls": 2.0,
            "sleep_between_batches": 2.0,
            "batch_size": 50,
            "invocations_per_query": 1,
            "max_retries": 5,
            # Similarity methods (defaults to Jaccard if empty)
            "similarity_methods": []
        }

    def _get_column_index(self, columns, target_column):
        """Get index of column in list, default to 0."""
        try:
            return list(columns).index(target_column)
        except (ValueError, AttributeError):
            return 0

    def _check_api_key(self, provider: str) -> bool:
        """Check if API key is available for provider."""
        from dotenv import load_dotenv
        load_dotenv()

        key_map = {
            "openai": "OPENAI_API",
            "cohere": "COHERE_API",
            "voyage": "VOYAGE_API",
            "bedrock": "AWS_ACCESS_KEY_ID"  # Bedrock uses AWS creds
        }

        env_var = key_map.get(provider.lower())
        if not env_var:
            return False

        return bool(os.getenv(env_var))

    def _add_embedding_model(self, model_dict, rag_config):
        """Add an embedding model to selected list."""
        if "embedding_models" not in rag_config:
            rag_config["embedding_models"] = []

        # Check if already exists by model_id + dimensions (for uniqueness)
        model_key = f"{model_dict['model_id']}|{model_dict.get('dimensions', 'default')}"
        existing_keys = [
            f"{m['model_id']}|{m.get('dimensions', 'default')}"
            for m in rag_config["embedding_models"]
        ]

        if model_key not in existing_keys:
            rag_config["embedding_models"].append(model_dict)

    def _clear_embedding_models(self, rag_config):
        """Clear all selected embedding models."""
        rag_config["embedding_models"] = []

    def _autopop_from_vectorstore(self, rag_config, vectorstore_info):
        """
        Auto-populate configuration from vectorstore metadata.

        Args:
            rag_config: The RAG configuration dictionary to update
            vectorstore_info: Dictionary with vectorstore path information
        """
        try:
            # Load metadata from vectorstore
            from vectorstore_manager import load_vectorstore_metadata

            vectorstore_path = os.path.join(self.project_root, vectorstore_info["relative_path"])
            metadata = load_vectorstore_metadata(vectorstore_path)

            if not metadata:
                st.error("❌ Failed to load vectorstore metadata")
                return

            # Extract chunking parameters
            chunking_params = metadata.get("chunking_params", {})
            if chunking_params:
                rag_config["chunk_size"] = chunking_params.get("chunk_size", 512)
                rag_config["chunk_overlap"] = chunking_params.get("chunk_overlap", 50)
                rag_config["chunking_strategy"] = chunking_params.get("chunking_strategy", "recursive")

                # For semantic chunking
                if chunking_params.get("chunking_strategy") == "semantic":
                    if "similarity_threshold" in chunking_params:
                        rag_config["similarity_threshold"] = chunking_params["similarity_threshold"]
                    if "semantic_chunking_model" in chunking_params:
                        rag_config["semantic_chunking_model"] = chunking_params["semantic_chunking_model"]

            # Extract embedding models
            embedding_models_metadata = metadata.get("embedding_models", [])
            if embedding_models_metadata:
                # Clear existing selection
                rag_config["embedding_models"] = []

                # Add each model from metadata
                for model_meta in embedding_models_metadata:
                    # The metadata stores model info, we need to add it to the config
                    rag_config["embedding_models"].append(model_meta)

                # Extract AWS region from first Bedrock model if available
                for model in embedding_models_metadata:
                    if model.get("model_id", "").startswith("bedrock/"):
                        region = model.get("region")
                        if region:
                            rag_config["aws_embedding_region"] = region
                            break

            # Show success message with summary
            st.success("✅ Settings auto-populated from vectorstore!")

            with st.expander("📋 Loaded Configuration", expanded=True):
                st.markdown("**Chunking Parameters:**")
                st.write(f"- Strategy: `{rag_config.get('chunking_strategy', 'N/A')}`")
                st.write(f"- Chunk Size: `{rag_config.get('chunk_size', 'N/A')}`")
                st.write(f"- Chunk Overlap: `{rag_config.get('chunk_overlap', 'N/A')}`")

                if rag_config.get("chunking_strategy") == "semantic":
                    st.write(f"- Similarity Threshold: `{rag_config.get('similarity_threshold', 'N/A')}`")
                    st.write(f"- Semantic Model: `{rag_config.get('semantic_chunking_model', 'N/A')}`")

                st.markdown("**Embedding Models:**")
                if rag_config.get("embedding_models"):
                    for model in rag_config["embedding_models"]:
                        region_info = f" ({model.get('region')})" if model.get("region") else ""
                        st.write(f"- {model['model_id']}{region_info} - {model.get('dimensions', 'N/A')}D")
                else:
                    st.write("- None")

                if rag_config.get("aws_embedding_region"):
                    st.markdown(f"**AWS Region:** `{rag_config['aws_embedding_region']}`")

        except Exception as e:
            st.error(f"❌ Error auto-populating settings: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
