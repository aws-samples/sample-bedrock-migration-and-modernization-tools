"""RAG Configuration Tab Component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any


class RAGConfigurationTab:
    """Component for configuring RAG-specific settings in a dedicated tab."""

    def render(self):
        """Render the RAG configuration tab."""
        st.markdown("## 📚 RAG Configuration")
        st.markdown("Configure document processing, chunking, embeddings, and retrieval settings for RAG evaluation.")

        # Ensure rag_config is initialized
        self._ensure_rag_config_initialized()

        # Document Upload Section
        self._render_document_upload()

        st.markdown("---")

        # Chunking Configuration Section
        self._render_chunking_config()

        st.markdown("---")

        # Embedding Model Configuration Section
        self._render_embedding_config()

        st.markdown("---")

        # Retriever Configuration Section
        self._render_retriever_config()

    def _ensure_rag_config_initialized(self):
        """Ensure rag_config exists in current_evaluation_config with default values."""
        if "rag_config" not in st.session_state.current_evaluation_config:
            st.session_state.current_evaluation_config["rag_config"] = {
                "documents": [],
                "documents_paths": [],
                "chunking_strategy": "fixed_size",
                "chunking_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunking_params": {
                    "chunk_size": 512,
                    "overlap": 50,
                    "size_type": "characters"
                },
                "embedding_models": [],
                "collection_name": None,
                "distance_metric": "cosine",
                "top_k": 5,
                "aws_region": "us-east-1"
            }

    def _render_document_upload(self):
        """Render document upload section."""
        st.markdown("### 📄 Document Upload")
        st.markdown("Upload documents that will be used to create the knowledge base for retrieval.")

        # Document upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md", "docx", "csv", "json"],
            accept_multiple_files=True,
            key="rag_document_upload",
            help="Upload documents in PDF, TXT, MD, DOCX, CSV, or JSON format. These will be chunked and indexed for retrieval."
        )

        if uploaded_files:
            self._handle_document_upload(uploaded_files)
            self._show_uploaded_documents()

    def _handle_document_upload(self, uploaded_files):
        """Handle document upload and save to temp directory."""
        # Create temp directory for uploaded documents
        temp_dir = tempfile.gettempdir()
        rag_docs_dir = os.path.join(temp_dir, "rag_evaluation_docs")
        os.makedirs(rag_docs_dir, exist_ok=True)

        documents = []
        documents_paths = []

        for uploaded_file in uploaded_files:
            # Save file to temp directory
            file_path = os.path.join(rag_docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            documents.append({
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "type": uploaded_file.type,
                "path": file_path
            })
            documents_paths.append(file_path)

        # Update session state
        st.session_state.current_evaluation_config["rag_config"]["documents"] = documents
        st.session_state.current_evaluation_config["rag_config"]["documents_paths"] = documents_paths

    def _show_uploaded_documents(self):
        """Display uploaded documents."""
        documents = st.session_state.current_evaluation_config["rag_config"].get("documents", [])

        if documents:
            st.success(f"✅ {len(documents)} document(s) uploaded successfully")

            with st.expander("📋 View Uploaded Documents"):
                for doc in documents:
                    size_kb = doc["size"] / 1024
                    st.text(f"• {doc['name']} ({size_kb:.1f} KB)")

    def _render_chunking_config(self):
        """Render chunking configuration UI."""
        st.markdown("### ✂️ Chunking Configuration")

        rag_config = st.session_state.current_evaluation_config["rag_config"]

        # Chunking strategy selector
        strategy = st.selectbox(
            "Chunking Strategy",
            options=["fixed_size", "sentence", "recursive", "semantic"],
            index=["fixed_size", "sentence", "recursive", "semantic"].index(
                rag_config.get("chunking_strategy", "fixed_size")
            ),
            key="rag_chunking_strategy",
            on_change=self._update_chunking_strategy,
            help="""
            - **Fixed Size**: Split by fixed character/token count with overlap
            - **Sentence**: Split on sentence boundaries
            - **Recursive**: Recursive splitting (paragraphs → sentences → words)
            - **Semantic**: Split based on semantic similarity (requires embedding model)
            """
        )

        # Strategy-specific parameters
        if strategy == "fixed_size":
            self._render_fixed_size_params()
        elif strategy == "sentence":
            self._render_sentence_params()
        elif strategy == "recursive":
            self._render_recursive_params()
        elif strategy == "semantic":
            self._render_semantic_params()

    def _render_fixed_size_params(self):
        """Render parameters for fixed size chunking."""
        rag_config = st.session_state.current_evaluation_config["rag_config"]
        params = rag_config.get("chunking_params", {})

        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=params.get("chunk_size", 512),
                step=50,
                key="rag_chunk_size",
                on_change=self._update_chunking_params,
                help="Number of characters per chunk"
            )

        with col2:
            overlap = st.number_input(
                "Overlap",
                min_value=0,
                max_value=500,
                value=params.get("overlap", 50),
                step=10,
                key="rag_chunk_overlap",
                on_change=self._update_chunking_params,
                help="Number of overlapping characters between chunks"
            )

        size_type = st.radio(
            "Size Type",
            options=["characters", "tokens"],
            index=0 if params.get("size_type", "characters") == "characters" else 1,
            key="rag_size_type",
            on_change=self._update_chunking_params,
            horizontal=True,
            help="Whether to measure chunk size in characters or tokens"
        )

    def _render_sentence_params(self):
        """Render parameters for sentence chunking."""
        rag_config = st.session_state.current_evaluation_config["rag_config"]
        params = rag_config.get("chunking_params", {})

        col1, col2 = st.columns(2)

        with col1:
            max_sentences = st.number_input(
                "Max Sentences per Chunk",
                min_value=1,
                max_value=20,
                value=params.get("max_sentences_per_chunk", 5),
                step=1,
                key="rag_max_sentences",
                on_change=self._update_chunking_params
            )

        with col2:
            min_chunk_size = st.number_input(
                "Min Chunk Size (chars)",
                min_value=10,
                max_value=500,
                value=params.get("min_chunk_size", 50),
                step=10,
                key="rag_min_chunk_size",
                on_change=self._update_chunking_params
            )

    def _render_recursive_params(self):
        """Render parameters for recursive chunking."""
        rag_config = st.session_state.current_evaluation_config["rag_config"]
        params = rag_config.get("chunking_params", {})

        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.number_input(
                "Target Chunk Size",
                min_value=100,
                max_value=2000,
                value=params.get("chunk_size", 500),
                step=50,
                key="rag_recursive_chunk_size",
                on_change=self._update_chunking_params
            )

        with col2:
            overlap = st.number_input(
                "Overlap",
                min_value=0,
                max_value=500,
                value=params.get("overlap", 50),
                step=10,
                key="rag_recursive_overlap",
                on_change=self._update_chunking_params
            )

        separators = st.text_input(
            "Separators (comma-separated)",
            value=", ".join(params.get("separators", ["\\n\\n", "\\n", ". "])),
            key="rag_separators",
            on_change=self._update_chunking_params,
            help="Separators to try in order (e.g., \\n\\n, \\n, . )"
        )

    def _render_semantic_params(self):
        """Render parameters for semantic chunking."""
        from ..utils.constants import load_embedding_profiles

        rag_config = st.session_state.current_evaluation_config["rag_config"]
        params = rag_config.get("chunking_params", {})

        # Load embedding profiles for chunking model selection
        embedding_profiles = load_embedding_profiles()

        if not embedding_profiles:
            st.warning("⚠️ No embedding profiles found. Please check embedding_profiles.json configuration.")
            return

        st.info("ℹ️ Semantic chunking uses an embedding model to calculate sentence similarity")

        # Embedding model selection for chunking
        st.markdown("**Select Embedding Model for Chunking:**")

        # Create options from embedding profiles
        embedding_options = {
            f"{profile['name']} ({profile['provider']})": profile['model_id']
            for profile in embedding_profiles
        }

        # Get current selection
        current_model = rag_config.get("chunking_embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        # Find the display name for current model
        current_display_name = None
        for display_name, model_id in embedding_options.items():
            if model_id == current_model:
                current_display_name = display_name
                break

        # If current model not found in profiles, use first option
        if not current_display_name:
            current_display_name = list(embedding_options.keys())[0]

        selected_display_name = st.selectbox(
            "Chunking Embedding Model",
            options=list(embedding_options.keys()),
            index=list(embedding_options.keys()).index(current_display_name) if current_display_name in embedding_options else 0,
            key="rag_chunking_embedding_model_select",
            on_change=self._update_chunking_embedding_model,
            help="Embedding model used to calculate semantic similarity between sentences during chunking"
        )

        # Show model details
        selected_model_id = embedding_options[selected_display_name]
        selected_profile = next((p for p in embedding_profiles if p['model_id'] == selected_model_id), None)

        if selected_profile:
            with st.expander("ℹ️ Model Details"):
                st.write(f"**Model ID:** `{selected_profile['model_id']}`")
                st.write(f"**Provider:** {selected_profile['provider']}")
                st.write(f"**Dimensions:** {selected_profile['dimensions']}")
                st.write(f"**Description:** {selected_profile.get('description', 'N/A')}")
                st.write(f"**Use Case:** {selected_profile.get('use_case', 'N/A')}")

        st.markdown("---")
        st.markdown("**Chunking Parameters:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=params.get("similarity_threshold", 0.7),
                step=0.05,
                key="rag_similarity_threshold",
                on_change=self._update_chunking_params,
                help="Sentences with similarity below this threshold will start a new chunk"
            )

        with col2:
            min_chunk_size = st.number_input(
                "Min Chunk Size",
                min_value=50,
                max_value=500,
                value=params.get("min_chunk_size", 100),
                step=50,
                key="rag_semantic_min_chunk",
                on_change=self._update_chunking_params
            )

        with col3:
            max_chunk_size = st.number_input(
                "Max Chunk Size",
                min_value=500,
                max_value=2000,
                value=params.get("max_chunk_size", 1000),
                step=100,
                key="rag_semantic_max_chunk",
                on_change=self._update_chunking_params
            )

    def _render_embedding_config(self):
        """Render embedding model configuration."""
        from ..utils.constants import load_embedding_profiles

        st.markdown("### 🔢 Embedding Model Configuration")

        rag_config = st.session_state.current_evaluation_config["rag_config"]

        # Initialize embedding_models as list of dicts if it's still a list of strings
        if "embedding_models" not in rag_config:
            rag_config["embedding_models"] = []
        elif rag_config["embedding_models"] and isinstance(rag_config["embedding_models"][0], str):
            # Convert old format (list of strings) to new format (list of dicts)
            rag_config["embedding_models"] = []

        # Load embedding profiles
        embedding_profiles = load_embedding_profiles()

        if not embedding_profiles:
            st.warning("⚠️ No embedding profiles found. Please check embedding_profiles.json configuration.")
            return

        st.markdown("Select embedding models to compare for RAG evaluation.")

        # Create dropdown options from embedding profiles
        embedding_options = {
            f"{profile['name']} ({profile['provider']})": profile
            for profile in embedding_profiles
        }

        # Embedding model selection UI
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            selected_display_name = st.selectbox(
                "Select Embedding Model",
                options=list(embedding_options.keys()),
                key="embedding_model_select",
                help="Choose an embedding model from the available profiles"
            )

        # Get the selected profile
        selected_profile = embedding_options[selected_display_name]
        default_input_cost = selected_profile.get("input_cost_per_1k", 0.0)

        with col2:
            input_cost = st.number_input(
                "Input Cost (per 1K tokens)",
                min_value=0.0,
                max_value=1.0,
                value=default_input_cost,
                step=0.00001,
                format="%.6f",
                key="embedding_input_cost",
                help="Cost per 1,000 input tokens"
            )

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add Model", key="add_embedding_model"):
                self._add_embedding_model(
                    selected_profile['model_id'],
                    selected_profile['name'],
                    selected_profile['provider'],
                    selected_profile['dimensions'],
                    input_cost
                )
                st.rerun()

        # Display selected embedding models
        current_models = rag_config.get("embedding_models", [])

        if current_models:
            st.markdown("**Selected Embedding Models:**")

            # Create DataFrame for display
            models_df = pd.DataFrame(current_models)
            models_df = models_df.rename(columns={
                "name": "Model Name",
                "model_id": "Model ID",
                "provider": "Provider",
                "input_cost": "Input Cost (per 1K tokens)",
                "dimensions": "Dimensions"
            })

            st.dataframe(models_df, hide_index=True)

            # Clear button
            if st.button("Clear Selected Embeddings", key="clear_embeddings"):
                self._clear_embedding_models()
                st.rerun()
        else:
            st.info("No embedding models selected. Please select at least one model.")

        # AWS Region for Bedrock models
        st.markdown("---")
        aws_region = st.selectbox(
            "AWS Region (for Bedrock embeddings)",
            options=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
            index=0,
            key="rag_aws_region",
            on_change=self._update_aws_region,
            help="AWS region where Bedrock embedding models will be invoked"
        )

    def _render_retriever_config(self):
        """Render retriever configuration."""
        st.markdown("### ⚙️ Retriever Configuration")

        rag_config = st.session_state.current_evaluation_config["rag_config"]

        col1, col2 = st.columns(2)

        with col1:
            top_k = st.number_input(
                "Top K Results",
                min_value=1,
                max_value=20,
                value=rag_config.get("top_k", 5),
                step=1,
                key="rag_top_k",
                on_change=self._update_top_k,
                help="Number of chunks to retrieve for each query"
            )

        with col2:
            distance_metric = st.selectbox(
                "Distance Metric",
                options=["cosine", "l2", "ip"],
                index=["cosine", "l2", "ip"].index(rag_config.get("distance_metric", "cosine")),
                key="rag_distance_metric",
                on_change=self._update_distance_metric,
                help="Distance metric for similarity search (cosine, L2, inner product)"
            )

    # Callback functions
    def _update_chunking_strategy(self):
        """Update chunking strategy in session state."""
        st.session_state.current_evaluation_config["rag_config"]["chunking_strategy"] = st.session_state.rag_chunking_strategy

    def _update_chunking_embedding_model(self):
        """Update chunking embedding model in session state."""
        from ..utils.constants import load_embedding_profiles

        # Get selected display name
        selected_display_name = st.session_state.rag_chunking_embedding_model_select

        # Load embedding profiles to get model_id from display name
        embedding_profiles = load_embedding_profiles()
        embedding_options = {
            f"{profile['name']} ({profile['provider']})": profile['model_id']
            for profile in embedding_profiles
        }

        # Update session state with model_id
        if selected_display_name in embedding_options:
            st.session_state.current_evaluation_config["rag_config"]["chunking_embedding_model"] = embedding_options[selected_display_name]

    def _update_chunking_params(self):
        """Update chunking parameters based on strategy."""
        strategy = st.session_state.current_evaluation_config["rag_config"]["chunking_strategy"]
        params = {}

        if strategy == "fixed_size":
            params = {
                "chunk_size": st.session_state.get("rag_chunk_size", 512),
                "overlap": st.session_state.get("rag_chunk_overlap", 50),
                "size_type": st.session_state.get("rag_size_type", "characters")
            }
        elif strategy == "sentence":
            params = {
                "max_sentences_per_chunk": st.session_state.get("rag_max_sentences", 5),
                "min_chunk_size": st.session_state.get("rag_min_chunk_size", 50)
            }
        elif strategy == "recursive":
            separators_str = st.session_state.get("rag_separators", "\\n\\n, \\n, . ")
            params = {
                "chunk_size": st.session_state.get("rag_recursive_chunk_size", 500),
                "overlap": st.session_state.get("rag_recursive_overlap", 50),
                "separators": [s.strip() for s in separators_str.split(",")]
            }
        elif strategy == "semantic":
            params = {
                "similarity_threshold": st.session_state.get("rag_similarity_threshold", 0.7),
                "min_chunk_size": st.session_state.get("rag_semantic_min_chunk", 100),
                "max_chunk_size": st.session_state.get("rag_semantic_max_chunk", 1000)
            }

        st.session_state.current_evaluation_config["rag_config"]["chunking_params"] = params

    def _update_aws_region(self):
        """Update AWS region in session state."""
        st.session_state.current_evaluation_config["rag_config"]["aws_region"] = st.session_state.rag_aws_region

    def _update_top_k(self):
        """Update top_k in session state."""
        st.session_state.current_evaluation_config["rag_config"]["top_k"] = st.session_state.rag_top_k

    def _update_distance_metric(self):
        """Update distance metric in session state."""
        st.session_state.current_evaluation_config["rag_config"]["distance_metric"] = st.session_state.rag_distance_metric

    def _add_embedding_model(self, model_id, name, provider, dimensions, input_cost):
        """Add an embedding model to the selected models list with pricing."""
        rag_config = st.session_state.current_evaluation_config["rag_config"]

        # Check if model is already selected
        for model in rag_config.get("embedding_models", []):
            if model["model_id"] == model_id:
                # Update existing model's cost
                model["input_cost"] = input_cost
                return

        # Add new model
        if "embedding_models" not in rag_config:
            rag_config["embedding_models"] = []

        rag_config["embedding_models"].append({
            "model_id": model_id,
            "name": name,
            "provider": provider,
            "dimensions": dimensions,
            "input_cost": input_cost
        })

    def _clear_embedding_models(self):
        """Clear all selected embedding models."""
        st.session_state.current_evaluation_config["rag_config"]["embedding_models"] = []
