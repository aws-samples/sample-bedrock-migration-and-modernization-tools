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
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Query & Data Files",
            "✂️ Chunking Strategy",
            "🧮 Embedding Models",
            "🎯 Retrieval & Re-ranking"
        ])

        with tab1:
            self._render_file_config(rag_config)

        with tab2:
            self._render_chunking_config(rag_config)

        with tab3:
            self._render_embedding_config(rag_config)

        with tab4:
            self._render_retrieval_config(rag_config)

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
        st.markdown("### 📚 Data Source File")
        st.info("Upload the document(s) that will be chunked and used for retrieval")

        data_source_file = st.file_uploader(
            "Upload Data Source",
            type=["csv", "json", "jsonl", "txt", "md", "docx", "pdf"],
            key="rag_data_source_uploader",
            help="Document to be chunked and embedded"
        )

        if data_source_file is not None:
            # Read the file content as text/bytes depending on type
            try:
                # Read file content - use text mode for text files, binary for others
                file_ext = data_source_file.name.split(".")[-1].lower()

                # Store file extension for later reconstruction
                rag_config["data_source_file_ext"] = file_ext

                if file_ext in ["csv", "json", "jsonl", "txt", "md"]:
                    # Text files - read as string
                    content = data_source_file.getvalue().decode("utf-8")
                    rag_config["data_source_is_binary"] = False
                else:
                    # Binary files (docx, pdf) - store as base64
                    import base64
                    content = base64.b64encode(data_source_file.getvalue()).decode("utf-8")
                    rag_config["data_source_is_binary"] = True

                rag_config["data_source_file"] = content
                rag_config["data_source_file_name"] = data_source_file.name

                # Auto-detect format
                format_map = {
                    "csv": "csv", "json": "json", "jsonl": "jsonl",
                    "txt": "txt", "md": "markdown", "docx": "word", "pdf": "pdf"
                }
                detected_format = format_map.get(file_ext, "txt")
                rag_config["data_format"] = detected_format

                st.success(f"✓ Loaded {data_source_file.name}")
            except Exception as e:
                st.error(f"Error reading data source file: {str(e)}")

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

        # Load embedding model profiles
        embedding_profiles_path = os.path.join(self.config_dir, "embedding_models_profiles.jsonl")

        if not os.path.exists(embedding_profiles_path):
            st.error(f"Embedding model profile file not found: {embedding_profiles_path}")
            return

        try:
            with open(embedding_profiles_path, 'r') as f:
                all_embedding_models = [json.loads(line) for line in f if line.strip()]

            # Get already selected model IDs
            selected_model_ids = [m["model_id"] for m in rag_config.get("embedding_models", [])]

            # Filter to show only unselected models in dropdown
            available_models = [m for m in all_embedding_models if m["model_id"] not in selected_model_ids]
            model_options = [m["model_id"] for m in available_models]

            if model_options:
                # Render dropdown + Add button
                col1, col2, col3 = st.columns([4, 1.2, 1.2])

                with col1:
                    selected_model_id = st.selectbox(
                        "Select Embedding Model",
                        options=model_options,
                        key="embedding_model_select"
                    )

                # Find selected model details
                selected_model = next((m for m in available_models if m["model_id"] == selected_model_id), None)

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
                    display_data.append({
                        "Model ID": model["model_id"],
                        "Dimensions": model.get("dimensions", "N/A"),
                        "Cost (per 1K)": f"${model.get('input_token_cost', 0)}"
                    })

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
            "max_retries": 5
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

        # Check if already exists (shouldn't happen due to filtering, but safeguard)
        if not any(m["model_id"] == model_dict["model_id"] for m in rag_config["embedding_models"]):
            rag_config["embedding_models"].append(model_dict)

    def _clear_embedding_models(self, rag_config):
        """Clear all selected embedding models."""
        rag_config["embedding_models"] = []
