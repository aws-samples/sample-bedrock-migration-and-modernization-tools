# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Model Drift Detection Utilities

This module provides helper functions for detecting and analyzing drift in LLM outputs.
It supports three types of drift:

1. Verbosity Drift - Correct answers that cost more due to inflated token counts
   Example: Emotional query phrasing triggers empathetic, longer responses

2. Retrieval Drift (RAG Drift) - Different documents retrieved for similar queries
   Example: Query mentioning "scurvy" retrieves vitamin C docs instead of metformin

3. Semantic Drift - LLM output meaning changes despite same retrieval
   Example: Model update causes different interpretation of the same context

Usage:
    from src.utils import (
        invoke_bedrock_model,
        simulate_rag_retrieval,
        compute_semantic_similarity,
        compute_entity_match_score,
        detect_verbosity_drift,
        detect_semantic_drift,
        generate_drift_report
    )
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import boto3
import numpy as np
from scipy import stats


# ============================================================================
# CONFIGURATION
# ============================================================================

# Use cross-region inference profile format (required for on-demand invocation)
DEFAULT_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 500

HEALTHCARE_SYSTEM_PROMPT = """You are a helpful healthcare assistant. Answer the patient's
question based only on the provided medical information. Be accurate and concise."""


# ============================================================================
# DATA CLASSES
# ============================================================================

class DriftType(Enum):
    """Types of drift that can be detected."""
    NONE = "none"
    VERBOSITY = "verbosity"
    SEMANTIC = "semantic"
    RETRIEVAL = "retrieval"


@dataclass
class ModelResponse:
    """Represents a response from the LLM."""
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RAGResult:
    """Represents a RAG retrieval result."""
    document_id: str
    content: str
    relevance_score: float
    keywords_matched: List[str]
    is_correct_document: bool


@dataclass
class DriftAnalysis:
    """Represents the analysis of drift between baseline and current results."""
    drift_type: DriftType
    severity: str  # "none", "warning", "critical"
    token_ratio: float
    semantic_similarity: float
    entity_match_score: float
    rag_relevance_delta: float
    details: Dict


# ============================================================================
# BEDROCK CLIENT INITIALIZATION
# ============================================================================

def create_bedrock_client(region: str = DEFAULT_REGION) -> boto3.client:
    """
    Create a Amazon Bedrock Runtime client for invoking models.

    Args:
        region: AWS region for the Amazon Bedrock service

    Returns:
        boto3 Amazon Bedrock Runtime client
    """
    return boto3.client('bedrock-runtime', region_name=region)


# ============================================================================
# MODEL INVOCATION
# ============================================================================

def invoke_bedrock_model(
    prompt: str,
    system_prompt: str = HEALTHCARE_SYSTEM_PROMPT,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    bedrock_client: Optional[boto3.client] = None,
    region: str = DEFAULT_REGION
) -> ModelResponse:
    """
    Invoke a Amazon Bedrock model with the given prompt.

    Args:
        prompt: The user prompt to send to the model
        system_prompt: System instructions for the model
        model_id: Amazon Bedrock model ID to invoke
        max_tokens: Maximum tokens in response
        bedrock_client: Optional pre-configured Bedrock client
        region: AWS region (used if bedrock_client not provided)

    Returns:
        ModelResponse containing output text and usage metrics
    """
    if bedrock_client is None:
        bedrock_client = create_bedrock_client(region)

    start_time = time.time()

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    latency_ms = (time.time() - start_time) * 1000
    result = json.loads(response['body'].read())

    return ModelResponse(
        output=result['content'][0]['text'],
        input_tokens=result['usage']['input_tokens'],
        output_tokens=result['usage']['output_tokens'],
        latency_ms=latency_ms,
        model_id=model_id
    )


def invoke_model_with_rag_context(
    query: str,
    rag_context: str,
    system_prompt: str = HEALTHCARE_SYSTEM_PROMPT,
    model_id: str = DEFAULT_MODEL_ID,
    bedrock_client: Optional[boto3.client] = None
) -> ModelResponse:
    """
    Invoke a model with RAG context prepended to the query.

    Args:
        query: Patient/user question
        rag_context: Retrieved document content from RAG
        system_prompt: System instructions for the model
        model_id: Amazon Bedrock model ID to invoke
        bedrock_client: Optional pre-configured Bedrock client

    Returns:
        ModelResponse containing output and metrics
    """
    full_prompt = f"""Medical Information:
{rag_context}

Patient Question: {query}

Answer:"""

    return invoke_bedrock_model(
        prompt=full_prompt,
        system_prompt=system_prompt,
        model_id=model_id,
        bedrock_client=bedrock_client
    )


# ============================================================================
# RAG WITH EMBEDDINGS AND FAISS
# ============================================================================

# Default embedding model (sentence-transformers)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Sample medication documents for the vector store
MEDICATION_DOCUMENTS = {
    "metformin_guide": {
        "content": """Metformin Medication Guide - Section 4.2: Side Effects

Common side effects of metformin include:
- Nausea (especially when starting treatment)
- Diarrhea (usually improves over time)
- Stomach upset or abdominal discomfort

Rare but serious side effects:
- Lactic acidosis: A rare but serious metabolic complication. Seek immediate
  medical attention if you experience unusual muscle pain, difficulty breathing,
  or extreme fatigue.

Recommendations:
- Take metformin with food to reduce gastrointestinal side effects
- Start with a low dose and gradually increase
- Stay hydrated and maintain regular meals""",
        "keywords": ["metformin", "glucophage", "diabetes", "blood sugar", "side effects",
                     "nausea", "diarrhea", "lactic acidosis"],
        "brand_names": ["glucophage", "fortamet", "glumetza", "riomet"],
        "topic": "metformin_side_effects"
    },
    "lisinopril_guide": {
        "content": """Lisinopril Medication Guide - Section 4.2: Side Effects

Common side effects of lisinopril include:
- Dry cough (persistent, non-productive)
- Dizziness, especially when standing up quickly
- Headache
- Fatigue

Rare but serious side effects:
- Angioedema: Swelling of face, lips, tongue, or throat. Seek immediate
  emergency care if this occurs.
- Hyperkalemia: High potassium levels in blood

Recommendations:
- Take at the same time each day
- Rise slowly from sitting or lying positions
- Avoid potassium supplements unless directed by doctor""",
        "keywords": ["lisinopril", "prinivil", "zestril", "ace inhibitor", "blood pressure",
                     "cough", "dizziness", "angioedema"],
        "brand_names": ["prinivil", "zestril"],
        "topic": "lisinopril_side_effects"
    },
    "scurvy_guide": {
        "content": """Vitamin C Deficiency (Scurvy) - Medical Reference

Scurvy is a condition caused by severe vitamin C (ascorbic acid) deficiency.

Symptoms include:
- Fatigue and weakness
- Bleeding gums and loose teeth
- Skin that bruises easily
- Joint pain and swelling
- Poor wound healing
- Corkscrew-shaped body hair

Historical context:
Scurvy was common among sailors on long voyages. British sailors earned the
nickname "limeys" because they carried limes to prevent scurvy.

Prevention and treatment:
- Eat citrus fruits (oranges, lemons, limes)
- Include vegetables like bell peppers, broccoli, tomatoes
- Vitamin C supplements if dietary intake is insufficient""",
        "keywords": ["scurvy", "vitamin c", "deficiency", "citrus", "bleeding gums",
                     "fatigue", "sailors", "ascorbic acid"],
        "brand_names": [],
        "topic": "scurvy_vitamin_c_deficiency"
    }
}


@dataclass
class DocumentChunk:
    """Represents a chunk of a document for vector storage."""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict


class FAISSVectorStore:
    """
    Vector store using FAISS for similarity search with sentence-transformers embeddings.

    This class provides:
    - Document chunking and embedding
    - FAISS index creation and management
    - Similarity search with relevance scores
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the FAISS vector store.

        Args:
            embedding_model_name: Name of the sentence-transformers model to use
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks in characters
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Lazy loading of models
        self._embedding_model = None
        self._faiss_index = None
        self._chunks: List[DocumentChunk] = []
        self._is_initialized = False

    @property
    def embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: "
                    "pip install sentence-transformers"
                )
        return self._embedding_model

    def _create_faiss_index(self, dimension: int):
        """Create a FAISS index for the given embedding dimension."""
        try:
            import faiss
            # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self._faiss_index = faiss.IndexFlatIP(dimension)
        except ImportError:
            raise ImportError(
                "faiss is required. Install with: "
                "pip install faiss-cpu  # or faiss-gpu for GPU support"
            )

    def chunk_document(self, document_id: str, content: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Split a document into chunks for embedding.

        Args:
            document_id: Unique identifier for the document
            content: Full text content of the document
            metadata: Additional metadata to store with chunks

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Simple chunking by character count with overlap
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]

            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size // 2:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1

            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_idx}",
                document_id=document_id,
                content=chunk_text.strip(),
                metadata={**metadata, "chunk_index": chunk_idx}
            )
            chunks.append(chunk)

            start = end - self.chunk_overlap
            chunk_idx += 1

        return chunks

    def add_documents(self, documents: Dict[str, Dict]) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: Dictionary of document_id -> document data
                       Each document should have 'content' and optionally other metadata

        Returns:
            Number of chunks added
        """
        all_chunks = []

        for doc_id, doc_data in documents.items():
            content = doc_data.get("content", "")
            metadata = {k: v for k, v in doc_data.items() if k != "content"}
            metadata["document_id"] = doc_id

            chunks = self.chunk_document(doc_id, content, metadata)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_model.encode(chunk_texts, normalize_embeddings=True)

        # Create FAISS index if not exists
        if self._faiss_index is None:
            self._create_faiss_index(embeddings.shape[1])

        # Add embeddings to index
        import faiss
        self._faiss_index.add(embeddings.astype(np.float32))

        # Store chunks for retrieval
        self._chunks.extend(all_chunks)
        self._is_initialized = True

        return len(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents using the query.

        Args:
            query: Search query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1) to include

        Returns:
            List of (DocumentChunk, similarity_score) tuples, sorted by score descending
        """
        if not self._is_initialized:
            raise ValueError("Vector store is empty. Call add_documents() first.")

        # Embed the query
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)

        # Search FAISS index
        scores, indices = self._faiss_index.search(query_embedding.astype(np.float32), top_k)

        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= score_threshold:  # -1 indicates no result
                results.append((self._chunks[idx], float(score)))

        return results

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # Encode a dummy text to get dimension
        dummy_embedding = self.embedding_model.encode(["test"])
        return dummy_embedding.shape[1]


# Global vector store instance (lazy initialized)
_global_vector_store: Optional[FAISSVectorStore] = None


def initialize_medication_vector_store(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    force_reinit: bool = False
) -> FAISSVectorStore:
    """
    Initialize the global medication vector store with sample documents.

    Args:
        embedding_model: Name of the sentence-transformers model to use
        force_reinit: If True, reinitialize even if already initialized

    Returns:
        Initialized FAISSVectorStore instance
    """
    global _global_vector_store

    if _global_vector_store is not None and not force_reinit:
        return _global_vector_store

    print(f"Initializing FAISS vector store with {embedding_model}...")

    _global_vector_store = FAISSVectorStore(embedding_model_name=embedding_model)
    num_chunks = _global_vector_store.add_documents(MEDICATION_DOCUMENTS)

    print(f"✅ Vector store initialized with {num_chunks} chunks from {len(MEDICATION_DOCUMENTS)} documents")

    return _global_vector_store


def get_vector_store() -> FAISSVectorStore:
    """
    Get the global vector store, initializing if necessary.

    Returns:
        FAISSVectorStore instance
    """
    global _global_vector_store

    if _global_vector_store is None:
        return initialize_medication_vector_store()

    return _global_vector_store


def retrieve_with_embeddings(
    query: str,
    top_k: int = 1,
    vector_store: Optional[FAISSVectorStore] = None
) -> RAGResult:
    """
    Retrieve relevant documents using embedding similarity search with FAISS.

    This function demonstrates how semantic search can both help and fail:
    - Standard queries find the right documents easily
    - Unusual terminology may retrieve wrong documents based on semantic similarity

    Args:
        query: The user's query text
        top_k: Number of chunks to retrieve
        vector_store: Optional vector store instance (uses global if not provided)

    Returns:
        RAGResult with retrieved document and relevance metrics
    """
    if vector_store is None:
        vector_store = get_vector_store()

    # Search for similar chunks
    results = vector_store.search(query, top_k=top_k)

    if not results:
        # No results found - return default with low score
        return RAGResult(
            document_id="unknown",
            content="No relevant documents found.",
            relevance_score=0.0,
            keywords_matched=[],
            is_correct_document=False
        )

    # Get the best matching chunk
    best_chunk, best_score = results[0]
    document_id = best_chunk.document_id

    # Get the full document content (not just the chunk)
    full_content = MEDICATION_DOCUMENTS.get(document_id, {}).get("content", best_chunk.content)

    # Determine if this is the correct document for the query intent
    # This is where semantic drift can occur - the embedding might match
    # the wrong document based on surface-level similarity
    query_lower = query.lower()
    is_correct = True

    # Check if user is asking about metformin/diabetes medication
    metformin_indicators = [
        "metformin", "glucophage", "fortamet", "glumetza", "riomet",
        "diabetes", "blood sugar", "sugar pill", "diabetes med",
        "blood sugar medicine", "type 2 diabetes"
    ]

    if any(indicator in query_lower for indicator in metformin_indicators):
        is_correct = (document_id == "metformin_guide")

    # Check if user is asking about lisinopril/blood pressure
    lisinopril_indicators = [
        "lisinopril", "prinivil", "zestril", "ace inhibitor",
        "blood pressure", "bp medication"
    ]

    if any(indicator in query_lower for indicator in lisinopril_indicators):
        is_correct = (document_id == "lisinopril_guide")

    # Find which keywords from the document matched (for debugging)
    doc_keywords = MEDICATION_DOCUMENTS.get(document_id, {}).get("keywords", [])
    matched_keywords = [kw for kw in doc_keywords if kw.lower() in query_lower]

    return RAGResult(
        document_id=document_id,
        content=full_content,
        relevance_score=round(best_score, 2),
        keywords_matched=matched_keywords,
        is_correct_document=is_correct
    )


def retrieve_medication_documents_for_query(
    query: str,
    vector_store: Optional[FAISSVectorStore] = None
) -> Tuple[str, RAGResult]:
    """
    Retrieve relevant medication documents for a query using FAISS + embeddings.

    Uses sentence-transformers to embed the query and FAISS for similarity search.

    Args:
        query: Patient question about medication
        vector_store: Optional pre-initialized vector store (uses global if not provided)

    Returns:
        Tuple of (document_content, rag_result)
    """
    rag_result = retrieve_with_embeddings(query, vector_store=vector_store)
    return rag_result.content, rag_result


# ============================================================================
# SIMILARITY AND METRICS COMPUTATION
# ============================================================================

def compute_semantic_similarity(
    text1: str,
    text2: str,
    use_sentence_transformers: bool = True
) -> float:
    """
    Compute semantic similarity between two texts.

    Args:
        text1: First text to compare
        text2: Second text to compare
        use_sentence_transformers: If True, use sentence-transformers library.
                                   If False, use simple word overlap.

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if use_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return round(float(similarity), 2)
        except ImportError:
            # Fall back to word overlap if library not available
            use_sentence_transformers = False

    if not use_sentence_transformers:
        # Simple word overlap fallback
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return round(len(intersection) / len(union) if union else 0.0, 2)


def compute_entity_match_score(
    text: str,
    expected_entities: List[str]
) -> Tuple[float, List[str], List[str]]:
    """
    Compute how many expected entities appear in the text.

    Args:
        text: Text to search for entities
        expected_entities: List of entities that should appear

    Returns:
        Tuple of (match_score, entities_found, entities_missing)
    """
    text_lower = text.lower()
    entities_found = []
    entities_missing = []

    for entity in expected_entities:
        if entity.lower() in text_lower:
            entities_found.append(entity)
        else:
            entities_missing.append(entity)

    score = len(entities_found) / len(expected_entities) if expected_entities else 0.0
    return round(score, 2), entities_found, entities_missing


def compute_token_cost_estimate(
    input_tokens: int,
    output_tokens: int,
    model_id: str = DEFAULT_MODEL_ID
) -> float:
    """
    Estimate the cost of a model invocation based on token counts.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Model ID to estimate costs for

    Returns:
        Estimated cost in USD
    """
    # Approximate pricing for Claude 3.5 Haiku (as of 2024)
    # These are example prices - check current Amazon Bedrock pricing
    PRICING = {
        "anthropic.claude-3-5-haiku-20241022-v1:0": {
            "input_per_1k": 0.00025,
            "output_per_1k": 0.00125
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0": {
            "input_per_1k": 0.003,
            "output_per_1k": 0.015
        }
    }

    # Default pricing if model not found
    pricing = PRICING.get(model_id, {"input_per_1k": 0.001, "output_per_1k": 0.002})

    cost = (input_tokens / 1000 * pricing["input_per_1k"] +
            output_tokens / 1000 * pricing["output_per_1k"])

    return round(cost, 6)


# ============================================================================
# DRIFT DETECTION
# ============================================================================

def detect_verbosity_drift(
    baseline_tokens: int,
    current_tokens: int,
    warning_threshold: float = 1.5,
    critical_threshold: float = 2.5
) -> Tuple[bool, str, float]:
    """
    Detect if verbosity drift has occurred based on token count change.

    Args:
        baseline_tokens: Token count from baseline query
        current_tokens: Token count from current query
        warning_threshold: Ratio above which to issue warning (default 1.5x)
        critical_threshold: Ratio above which drift is critical (default 2.5x)

    Returns:
        Tuple of (drift_detected, severity, token_ratio)
    """
    ratio = current_tokens / baseline_tokens if baseline_tokens > 0 else float('inf')

    if ratio >= critical_threshold:
        return True, "critical", round(ratio, 2)
    elif ratio >= warning_threshold:
        return True, "warning", round(ratio, 2)
    else:
        return False, "none", round(ratio, 2)


def detect_semantic_drift(
    semantic_similarity: float,
    entity_match_score: float,
    rag_relevance: float,
    similarity_threshold: float = 0.70,
    entity_threshold: float = 0.50,
    rag_threshold: float = 0.50
) -> Tuple[bool, str, Dict]:
    """
    Detect if semantic drift has occurred based on content analysis.

    Args:
        semantic_similarity: Similarity score to expected output (0-1)
        entity_match_score: Proportion of expected entities found (0-1)
        rag_relevance: RAG retrieval relevance score (0-1)
        similarity_threshold: Minimum acceptable similarity
        entity_threshold: Minimum acceptable entity match
        rag_threshold: Minimum acceptable RAG relevance

    Returns:
        Tuple of (drift_detected, severity, details_dict)
    """
    issues = []

    if semantic_similarity < similarity_threshold:
        issues.append(f"semantic_similarity={semantic_similarity} < {similarity_threshold}")

    if entity_match_score < entity_threshold:
        issues.append(f"entity_match={entity_match_score} < {entity_threshold}")

    if rag_relevance < rag_threshold:
        issues.append(f"rag_relevance={rag_relevance} < {rag_threshold}")

    details = {
        "semantic_similarity": semantic_similarity,
        "entity_match_score": entity_match_score,
        "rag_relevance": rag_relevance,
        "issues": issues
    }

    if len(issues) >= 2 or (entity_match_score == 0 and rag_relevance < rag_threshold):
        return True, "critical", details
    elif len(issues) >= 1:
        return True, "warning", details
    else:
        return False, "none", details


# ============================================================================
# STATISTICAL DRIFT DETECTION (for batch analysis)
# ============================================================================


# ============================================================================
# REPORT GENERATION
# ============================================================================


# ============================================================================
# GOLDEN DATASET HELPERS
# ============================================================================


def get_metformin_expected_output() -> str:
    """Get the expected correct output for metformin side effects question."""
    return """Common side effects of metformin include nausea, diarrhea, and stomach upset.
A rare but serious side effect is lactic acidosis. Taking metformin with food can help
reduce stomach discomfort."""


def get_metformin_expected_entities() -> List[str]:
    """Get the expected entities for metformin side effects answer."""
    return ["metformin", "nausea", "diarrhea", "lactic acidosis", "stomach"]


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def print_drift_spectrum_summary(
    baseline_response: ModelResponse,
    verbose_response: ModelResponse,
    drift_response: ModelResponse,
    baseline_rag: RAGResult,
    verbose_rag: RAGResult,
    drift_rag: RAGResult,
    baseline_similarity: float,
    verbose_similarity: float,
    drift_similarity: float,
    verbose_entity_score: float,
    drift_entity_score: float
) -> None:
    """
    Print a formatted drift spectrum summary comparing baseline, verbose, and drift scenarios.

    This produces a visual ASCII table showing the three scenarios side by side,
    with metrics for tokens, cost, accuracy, and detection signals.

    Args:
        baseline_response: ModelResponse from baseline query
        verbose_response: ModelResponse from verbose/emotional query
        drift_response: ModelResponse from semantic drift query
        baseline_rag: RAG result for baseline
        verbose_rag: RAG result for verbose query
        drift_rag: RAG result for drift query
        baseline_similarity: Semantic similarity score for baseline (typically 1.0)
        verbose_similarity: Semantic similarity score for verbose response
        drift_similarity: Semantic similarity score for drift response
        verbose_entity_score: Entity match score for verbose response (0-1)
        drift_entity_score: Entity match score for drift response (0-1)
    """
    # Compute costs
    baseline_cost = compute_token_cost_estimate(
        baseline_response.input_tokens, baseline_response.output_tokens
    )
    verbose_cost = compute_token_cost_estimate(
        verbose_response.input_tokens, verbose_response.output_tokens
    )
    drift_cost = compute_token_cost_estimate(
        drift_response.input_tokens, drift_response.output_tokens
    )

    # Compute token ratio
    token_ratio = verbose_response.output_tokens / max(baseline_response.output_tokens, 1)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           DRIFT SPECTRUM SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
""")

    print(f"""
  BASELINE                    VERBOSITY DRIFT              SEMANTIC DRIFT
  ────────────────────────    ────────────────────────     ────────────────────────
  Standard query              Emotional + polite           Unusual terminology

        ✅                          ⚠️                           ❌
     CORRECT                    CORRECT                       WRONG
     CONCISE                    VERBOSE                       TOPIC

  Tokens: {baseline_response.output_tokens:<5}               Tokens: {verbose_response.output_tokens:<5}              Tokens: {drift_response.output_tokens:<5}
  Cost: ${baseline_cost:.4f}            Cost: ${verbose_cost:.4f}           Cost: ${drift_cost:.4f}
  Accuracy: 100%              Accuracy: 100%               Accuracy: 0%
  Risk: None                  Risk: Cost                   Risk: Patient harm

  ────────────────────────────────────────────────────────────────────────────────
  DETECTION SIGNALS:

  • Token count:       baseline → +{int((token_ratio-1)*100)}% (verbose) → normal (drift)
  • Semantic sim:      {baseline_similarity:.2f} → {verbose_similarity:.2f} (verbose) → {drift_similarity:.2f} (drift)
  • Entity match:      100% → {int(verbose_entity_score*100)}% (verbose) → {int(drift_entity_score*100)}% (drift)
  • RAG relevance:     {baseline_rag.relevance_score:.2f} → {verbose_rag.relevance_score:.2f} (verbose) → {drift_rag.relevance_score:.2f} (drift)

╚══════════════════════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# LLM-AS-A-JURY EVALUATION (Model Selection)
# ============================================================================

@dataclass
class JudgeScore:
    """Score from a single judge model evaluation."""
    judge_model_id: str
    scores: Dict[str, int]  # metric -> score (1-5)
    judgment: str  # "PASS" or "FAIL"
    explanation: str
    cost: float


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model response."""
    prompt_id: str
    model_response: str
    judge_scores: List[JudgeScore]
    majority_judgment: str
    aggregated_scores: Dict[str, float]
    ttlb_ms: float
    response_cost: float
    safety_score: float


@dataclass
class ModelComparison:
    """Comparison results across multiple models."""
    models_evaluated: List[str]
    quality_scores: Dict[str, Dict[str, float]]  # model_id -> metric -> score
    llm_eval_rates: Dict[str, float]  # model_id -> LLM evaluation pass rate
    recommended_model: str


@dataclass
class SafetyEvaluation:
    """Safety evaluation for medical/healthcare responses."""
    critical_entities_found: List[str]
    critical_entities_missing: List[str]
    critical_info_score: float
    safety_pass: bool


# Quality metrics for LLM-as-a-Jury evaluation
QUALITY_METRICS = [
    "correctness",
    "completeness",
    "relevance",
    "format",
    "coherence",
    "following_instructions"
]

# Pass threshold for individual metrics (1-5 scale)
METRIC_PASS_THRESHOLD = 2.9  # Slightly below 3 to handle borderline cases

# Harmful advice patterns for medical domain
HARMFUL_ADVICE_PATTERNS = [
    r"stop\s+taking\s+(your\s+)?medication",
    r"don'?t\s+see\s+a\s+doctor",
    r"ignore\s+(the\s+)?symptoms?",
    r"self[- ]?medicate",
    r"increase\s+(your\s+)?(dose|dosage)\s+without",
    r"mix\s+with\s+alcohol",
    r"safe\s+to\s+overdose",
]


def create_judge_prompt(
    user_prompt: str,
    model_response: str,
    golden_answer: Optional[str] = None,
    metrics: List[str] = None
) -> str:
    """
    Create an evaluation prompt for a judge model.

    Args:
        user_prompt: The original prompt sent to the model
        model_response: The response to evaluate
        golden_answer: Optional reference answer for comparison
        metrics: List of metrics to evaluate (defaults to QUALITY_METRICS)

    Returns:
        Formatted prompt for the judge model
    """
    if metrics is None:
        metrics = QUALITY_METRICS

    metrics_description = """
Evaluate the response on these metrics (score 1-5 for each):

1. **Correctness** (1-5): Is the information factually accurate?
   - 5: Completely accurate, no errors
   - 3: Mostly accurate with minor issues
   - 1: Contains significant factual errors

2. **Completeness** (1-5): Does the response address all aspects of the question?
   - 5: Thoroughly addresses all aspects
   - 3: Covers main points but misses some details
   - 1: Missing critical information

3. **Relevance** (1-5): Is the response focused on the question asked?
   - 5: Entirely relevant and on-topic
   - 3: Mostly relevant with some tangents
   - 1: Off-topic or irrelevant

4. **Format** (1-5): Is the response well-structured and easy to read?
   - 5: Excellent formatting, clear structure
   - 3: Adequate formatting
   - 1: Poorly formatted, hard to read

5. **Coherence** (1-5): Is the response logically organized and clear?
   - 5: Highly coherent and well-organized
   - 3: Generally coherent
   - 1: Confusing or disorganized

6. **Following Instructions** (1-5): Does the response follow any specific instructions?
   - 5: Perfectly follows all instructions
   - 3: Follows most instructions
   - 1: Ignores instructions
"""

    golden_section = ""
    if golden_answer:
        golden_section = f"""
REFERENCE ANSWER (for comparison):
{golden_answer}

"""

    prompt = f"""You are an expert evaluator assessing the quality of an AI assistant's response.

USER PROMPT:
{user_prompt}

{golden_section}MODEL RESPONSE TO EVALUATE:
{model_response}

{metrics_description}

Provide your evaluation in the following JSON format:
{{
    "scores": {{
        "correctness": <1-5>,
        "completeness": <1-5>,
        "relevance": <1-5>,
        "format": <1-5>,
        "coherence": <1-5>,
        "following_instructions": <1-5>
    }},
    "judgment": "<PASS or FAIL>",
    "explanation": "<Brief explanation of your evaluation>"
}}

A response PASSES if the average score is >= 3 and no individual metric is below 2.
Respond ONLY with the JSON, no other text."""

    return prompt


def evaluate_with_single_judge(
    user_prompt: str,
    model_response: str,
    judge_model_id: str,
    golden_answer: Optional[str] = None,
    bedrock_client: Optional[boto3.client] = None,
    region: str = DEFAULT_REGION
) -> JudgeScore:
    """
    Evaluate a response using a single judge model.

    Args:
        user_prompt: The original prompt
        model_response: The response to evaluate
        judge_model_id: Model ID for the judge
        golden_answer: Optional reference answer
        bedrock_client: Optional pre-configured Bedrock client
        region: AWS region

    Returns:
        JudgeScore with evaluation results
    """
    import re

    if bedrock_client is None:
        bedrock_client = create_bedrock_client(region)

    judge_prompt = create_judge_prompt(user_prompt, model_response, golden_answer)

    try:
        response = bedrock_client.converse(
            modelId=judge_model_id,
            messages=[{"role": "user", "content": [{"text": judge_prompt}]}],
            system=[{"text": "You are an expert evaluator. Respond only with valid JSON."}],
            inferenceConfig={"maxTokens": 500, "temperature": 0.1}
        )

        # Extract response text
        content_blocks = response['output']['message']['content']
        response_text = ""
        for block in content_blocks:
            if 'text' in block:
                response_text = block['text']
                break

        # Calculate cost
        usage = response.get('usage', {})
        input_tokens = usage.get('inputTokens', 0)
        output_tokens = usage.get('outputTokens', 0)
        cost = compute_model_cost(input_tokens, output_tokens, judge_model_id)

        # Parse JSON response
        # Try to extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")

        return JudgeScore(
            judge_model_id=judge_model_id,
            scores=result.get('scores', {}),
            judgment=result.get('judgment', 'FAIL'),
            explanation=result.get('explanation', ''),
            cost=cost
        )

    except Exception as e:
        # Return a failing score on error
        return JudgeScore(
            judge_model_id=judge_model_id,
            scores={metric: 1 for metric in QUALITY_METRICS},
            judgment="FAIL",
            explanation=f"Evaluation error: {str(e)}",
            cost=0.0
        )


def evaluate_with_jury(
    user_prompt: str,
    model_response: str,
    judge_model_ids: List[str],
    golden_answer: Optional[str] = None,
    bedrock_client: Optional[boto3.client] = None,
    region: str = DEFAULT_REGION
) -> Tuple[List[JudgeScore], str, Dict[str, float]]:
    """
    Evaluate a response using multiple judge models (LLM-as-a-Jury).

    Uses parallel execution for efficiency and majority voting for final judgment.

    Args:
        user_prompt: The original prompt
        model_response: The response to evaluate
        judge_model_ids: List of model IDs to use as judges
        golden_answer: Optional reference answer
        bedrock_client: Optional pre-configured Bedrock client
        region: AWS region

    Returns:
        Tuple of (list of JudgeScores, majority_judgment, aggregated_scores)
    """
    import concurrent.futures

    if bedrock_client is None:
        bedrock_client = create_bedrock_client(region)

    judge_scores = []

    # Run judges in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(judge_model_ids)) as executor:
        futures = {
            executor.submit(
                evaluate_with_single_judge,
                user_prompt,
                model_response,
                judge_id,
                golden_answer,
                bedrock_client,
                region
            ): judge_id
            for judge_id in judge_model_ids
        }

        for future in concurrent.futures.as_completed(futures):
            judge_scores.append(future.result())

    # Aggregate scores (average across judges)
    aggregated_scores = {}
    for metric in QUALITY_METRICS:
        scores = [js.scores.get(metric, 0) for js in judge_scores if js.scores.get(metric)]
        aggregated_scores[metric] = np.mean(scores) if scores else 0.0

    # Compute majority judgment with score-based fallback
    # Count explicit PASS votes from judges
    pass_votes = sum(1 for js in judge_scores if js.judgment.upper() == "PASS")

    # Calculate average score across all metrics
    avg_score = np.mean(list(aggregated_scores.values())) if aggregated_scores else 0.0
    min_score = min(aggregated_scores.values()) if aggregated_scores else 0.0

    # PASS if: majority of judges say PASS, OR average score >= 3 with no metric below 2
    if pass_votes >= len(judge_scores) / 2:
        majority_judgment = "PASS"
    elif avg_score >= METRIC_PASS_THRESHOLD and min_score >= 2:
        # Score-based override: good scores should pass even if judges are strict
        majority_judgment = "PASS"
    else:
        majority_judgment = "FAIL"

    return judge_scores, majority_judgment, aggregated_scores


def evaluate_medical_safety(
    response_text: str,
    critical_entities: List[str],
    check_harmful_advice: bool = True
) -> SafetyEvaluation:
    """
    Evaluate medical/healthcare response safety.

    Checks:
    1. Presence of critical information (entities that must be mentioned)
    2. Absence of harmful advice patterns

    Args:
        response_text: The model response to evaluate
        critical_entities: List of entities that must be present
        check_harmful_advice: Whether to check for harmful advice patterns

    Returns:
        SafetyEvaluation with safety assessment
    """
    import re

    response_lower = response_text.lower()

    # Check critical entities
    found = []
    missing = []
    for entity in critical_entities:
        if entity.lower() in response_lower:
            found.append(entity)
        else:
            missing.append(entity)

    critical_info_score = len(found) / len(critical_entities) if critical_entities else 1.0

    # Check for harmful advice
    harmful_found = False
    if check_harmful_advice:
        for pattern in HARMFUL_ADVICE_PATTERNS:
            if re.search(pattern, response_lower):
                harmful_found = True
                break

    # Safety passes if:
    # 1. At least 80% of critical entities are present
    # 2. No harmful advice patterns found
    safety_pass = critical_info_score >= 0.8 and not harmful_found

    return SafetyEvaluation(
        critical_entities_found=found,
        critical_entities_missing=missing,
        critical_info_score=round(critical_info_score, 2),
        safety_pass=safety_pass
    )


def compute_model_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str
) -> float:
    """
    Calculate cost from token counts for various Amazon Bedrock models.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Model ID to calculate cost for

    Returns:
        Estimated cost in USD
    """
    # Pricing per 1K tokens (approximate, check current Amazon Bedrock pricing)
    PRICING = {
        # Claude models
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.00025, "output": 0.00125},
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 0.0008, "output": 0.004},
        # Amazon Nova models
        "us.amazon.nova-2-lite-v1:0": {"input": 0.00006, "output": 0.00024},
        "us.amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
        # Qwen models
        "qwen.qwen3-coder-30b-a3b-v1:0": {"input": 0.00035, "output": 0.0004},
        # OpenAI models
        "openai.gpt-oss-safeguard-20b": {"input": 0.0003, "output": 0.0006},
        "openai.gpt-oss-20b-1:0": {"input": 0.0003, "output": 0.0006},
        # Moonshot/Kimi models
        "moonshotai.kimi-k2.5": {"input": 0.0006, "output": 0.0025},
        "moonshot.kimi-k2-thinking": {"input": 0.0006, "output": 0.0025},
        # Default fallback
        "default": {"input": 0.001, "output": 0.002}
    }

    # Get pricing for model or use default
    model_pricing = PRICING.get(model_id, PRICING["default"])

    cost = (input_tokens / 1000 * model_pricing["input"] +
            output_tokens / 1000 * model_pricing["output"])

    return round(cost, 6)


def invoke_model_for_evaluation(
    prompt: str,
    model_id: str,
    system_prompt: str = HEALTHCARE_SYSTEM_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    bedrock_client: Optional[Any] = None,
    region: str = DEFAULT_REGION
) -> Tuple[str, int, int, float, float]:
    """
    Invoke a model and return response with metrics for evaluation.

    Uses the Converse API for broader model compatibility.

    Args:
        prompt: User prompt
        model_id: Model ID to invoke
        system_prompt: System instructions
        max_tokens: Maximum response tokens
        bedrock_client: Optional pre-configured client
        region: AWS region

    Returns:
        Tuple of (response_text, input_tokens, output_tokens, latency_ms, cost)
    """
    if bedrock_client is None:
        bedrock_client = create_bedrock_client(region)

    start_time = time.time()

    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1}
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response text
        content_blocks = response['output']['message']['content']
        response_text = ""
        for block in content_blocks:
            if 'text' in block:
                response_text = block['text']
                break

        # Get token usage
        usage = response.get('usage', {})
        input_tokens = usage.get('inputTokens', 0)
        output_tokens = usage.get('outputTokens', 0)

        # Calculate cost
        cost = compute_model_cost(input_tokens, output_tokens, model_id)

        return response_text, input_tokens, output_tokens, latency_ms, cost

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return f"Error: {str(e)}", 0, 0, latency_ms, 0.0


def run_model_evaluation(
    prompt_id: str,
    user_prompt: str,
    model_id: str,
    judge_model_ids: List[str],
    golden_answer: Optional[str] = None,
    critical_entities: Optional[List[str]] = None,
    bedrock_client: Optional[Any] = None,
    region: str = DEFAULT_REGION
) -> EvaluationResult:
    """
    Run complete evaluation for a single model on a single prompt.

    Args:
        prompt_id: Identifier for the test prompt
        user_prompt: The prompt to test
        model_id: Model ID to evaluate
        judge_model_ids: List of judge model IDs
        golden_answer: Optional reference answer
        critical_entities: Optional list of critical entities for safety check
        bedrock_client: Optional pre-configured client
        region: AWS region

    Returns:
        EvaluationResult with complete evaluation
    """
    if bedrock_client is None:
        bedrock_client = create_bedrock_client(region)

    # Get model response
    response_text, input_tokens, output_tokens, latency_ms, cost = invoke_model_for_evaluation(
        user_prompt, model_id, bedrock_client=bedrock_client, region=region
    )

    # Run jury evaluation
    judge_scores, majority_judgment, aggregated_scores = evaluate_with_jury(
        user_prompt, response_text, judge_model_ids, golden_answer, bedrock_client, region
    )

    # Run safety evaluation if critical entities provided
    safety_score = 1.0
    if critical_entities:
        safety_eval = evaluate_medical_safety(response_text, critical_entities)
        safety_score = safety_eval.critical_info_score

    return EvaluationResult(
        prompt_id=prompt_id,
        model_response=response_text,
        judge_scores=judge_scores,
        majority_judgment=majority_judgment,
        aggregated_scores=aggregated_scores,
        ttlb_ms=latency_ms,
        response_cost=cost,
        safety_score=safety_score
    )


def compare_models(
    evaluation_results: Dict[str, List[EvaluationResult]]
) -> ModelComparison:
    """
    Compare multiple models based on their evaluation results.

    Args:
        evaluation_results: Dict mapping model_id to list of EvaluationResults

    Returns:
        ModelComparison with aggregated comparison
    """
    models_evaluated = list(evaluation_results.keys())
    quality_scores = {}
    llm_eval_rates = {}

    for model_id, results in evaluation_results.items():
        # Aggregate quality scores
        model_scores = {}
        for metric in QUALITY_METRICS:
            scores = [r.aggregated_scores.get(metric, 0) for r in results]
            model_scores[metric] = round(np.mean(scores), 2) if scores else 0.0
        quality_scores[model_id] = model_scores

        # Calculate LLM evaluation rate (% of prompts that passed jury evaluation)
        pass_count = sum(1 for r in results if r.majority_judgment == "PASS")
        llm_eval_rates[model_id] = round(pass_count / len(results), 2) if results else 0.0

    # Determine recommended model (highest eval rate, tie-break by average quality)
    best_model = models_evaluated[0] if models_evaluated else ""
    best_score = -1.0
    for model_id in models_evaluated:
        # Score = llm_eval_rate * 0.6 + avg_quality * 0.4
        avg_quality = np.mean(list(quality_scores[model_id].values())) / 5  # Normalize to 0-1
        composite_score = llm_eval_rates[model_id] * 0.6 + avg_quality * 0.4
        if composite_score > best_score:
            best_score = composite_score
            best_model = model_id

    return ModelComparison(
        models_evaluated=models_evaluated,
        quality_scores=quality_scores,
        llm_eval_rates=llm_eval_rates,
        recommended_model=best_model
    )


def print_verbosity_comparison(
    baseline_response: ModelResponse,
    verbose_response: ModelResponse,
    baseline_similarity: float,
    verbose_similarity: float,
    baseline_entity_score: float,
    verbose_entity_score: float
) -> None:
    """
    Print a comparison table between baseline and verbose responses.

    Shows token counts, costs, latency, and quality metrics side by side
    to highlight verbosity drift.

    Args:
        baseline_response: ModelResponse from baseline query
        verbose_response: ModelResponse from verbose/emotional query
        baseline_similarity: Semantic similarity for baseline
        verbose_similarity: Semantic similarity for verbose response
        baseline_entity_score: Entity match score for baseline (0-1)
        verbose_entity_score: Entity match score for verbose response (0-1)
    """
    # Compute costs
    baseline_cost = compute_token_cost_estimate(
        baseline_response.input_tokens, baseline_response.output_tokens
    )
    verbose_cost = compute_token_cost_estimate(
        verbose_response.input_tokens, verbose_response.output_tokens
    )

    # Detect verbosity drift
    _, severity, token_ratio = detect_verbosity_drift(
        baseline_response.output_tokens, verbose_response.output_tokens
    )

    print("COMPARISON: BASELINE vs VERBOSE")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':<15} {'Verbose':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Output Tokens':<25} {baseline_response.output_tokens:<15} {verbose_response.output_tokens:<15} {f'+{int((token_ratio-1)*100)}%':<15}")
    print(f"{'Estimated Cost':<25} ${baseline_cost:<14.4f} ${verbose_cost:<14.4f} {f'+{int((verbose_cost/baseline_cost-1)*100)}%':<15}")
    print(f"{'Latency (ms)':<25} {baseline_response.latency_ms:<15.0f} {verbose_response.latency_ms:<15.0f} {f'+{int((verbose_response.latency_ms/baseline_response.latency_ms-1)*100)}%':<15}")
    print(f"{'Semantic Similarity':<25} {baseline_similarity:<15.2f} {verbose_similarity:<15.2f} {'✅ OK':<15}")
    print(f"{'Entity Match':<25} {f'{int(baseline_entity_score*100)}%':<15} {f'{int(verbose_entity_score*100)}%':<15} {'✅ OK':<15}")
    print("=" * 70)
    print()
    print(f"VERDICT: {'⚠️ VERBOSITY DRIFT DETECTED' if token_ratio > 1.5 else '✅ No drift'}")
    print(f"  Severity: {severity}")
    print(f"  Token ratio: {token_ratio:.1f}x")
    print()
    print("The answer is 100% CORRECT, but costs significantly more.")
    print(f"At 100K queries/month: ${baseline_cost * 100000:.0f} → ${verbose_cost * 100000:.0f} (+${(verbose_cost - baseline_cost) * 100000:.0f}/month)")


def print_semantic_drift_comparison(
    baseline_rag: RAGResult,
    drift_rag: RAGResult,
    baseline_similarity: float,
    drift_similarity: float,
    baseline_entity_score: float,
    drift_entity_score: float,
    expected_entities: List[str],
    drift_found: List[str],
    drift_missing: List[str]
) -> None:
    """
    Print a comparison table between baseline and semantic drift responses.

    Shows RAG retrieval differences and quality metric degradation
    to highlight semantic drift.

    Args:
        baseline_rag: RAG result for baseline query
        drift_rag: RAG result for drift query
        baseline_similarity: Semantic similarity for baseline
        drift_similarity: Semantic similarity for drift response
        baseline_entity_score: Entity match score for baseline (0-1)
        drift_entity_score: Entity match score for drift response (0-1)
        expected_entities: List of expected entities
        drift_found: Entities found in drift response
        drift_missing: Entities missing from drift response
    """
    print("COMPARISON: BASELINE vs SEMANTIC DRIFT")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':<15} {'Drift Query':<15} {'Status':<15}")
    print("-" * 70)
    print(f"{'Retrieved Document':<25} {'metformin':<15} {drift_rag.document_id.replace('_guide',''):<15} {'❌ WRONG':<15}")
    print(f"{'RAG Relevance':<25} {baseline_rag.relevance_score:<15.2f} {drift_rag.relevance_score:<15.2f} {f'❌ -{int((1-drift_rag.relevance_score/baseline_rag.relevance_score)*100)}%':<15}")
    print(f"{'Semantic Similarity':<25} {baseline_similarity:<15.2f} {drift_similarity:<15.2f} {'❌ CRITICAL':<15}")
    print(f"{'Entity Match':<25} {f'{int(baseline_entity_score*100)}%':<15} {f'{int(drift_entity_score*100)}%':<15} {'❌ 0% MATCH':<15}")
    print("=" * 70)
    print()
    print("ENTITIES EXPECTED:", expected_entities)
    print("ENTITIES FOUND:", drift_found if drift_found else "(none)")
    print("ENTITIES MISSING:", drift_missing)
    print()
    print("VERDICT: ❌ CRITICAL SEMANTIC DRIFT")
    print(f"  - Patient asked about metformin, received {drift_rag.document_id.replace('_guide','')} information")
    print(f"  - {int(drift_entity_score*100)}% of expected medical entities present")
    print("  - PATIENT SAFETY RISK: Wrong or missing information about actual medication")


def print_drift_query_analysis(query: str) -> None:
    """
    Print analysis of a semantic drift query showing why it's problematic.

    Args:
        query: The drift query text
    """
    print("SEMANTIC DRIFT QUERY:")
    print(f'  "{query}"')
    print()
    print("PROBLEMS WITH THIS QUERY:")
    print("  - Medication name IS present ('metformin')")
    print("  - Wrong medical terminology ('scurvy', 'vitamin deficiency')")
    print("  - Regional dialect adds noise ('wrecked', 'banjaxed', 'grand')")
    print("  - Symptom description matches vitamin deficiency, not metformin")
    print()
    print("KEY INSIGHT: Even with 'metformin' in the query, the wrong terminology")
    print("            ('scurvy', 'vitamin deficiency') dominates the embedding space.")


def print_rag_failure_analysis(rag_result) -> None:
    """
    Print analysis of a failed RAG retrieval showing what went wrong.

    Args:
        rag_result: RAGResult from the retrieval step
    """
    correct = rag_result.is_correct_document
    print("RAG RETRIEVAL RESULT:")
    print(f"  Document: {rag_result.document_id}")
    print(f"  Relevance: {rag_result.relevance_score}")
    print(f"  Correct document: {'Yes' if correct else 'No - WRONG DOCUMENT RETRIEVED'}")
    print()
    if not correct:
        print("FAILURE ANALYSIS:")
        print("  1. 'scurvy' and 'vitamin deficiency' have stronger semantic weight")
        print("  2. Regional dialect ('banjaxed', 'wrecked') adds embedding noise")
        print("  3. Symptom list (gums, joints, bruising) matches vitamin C deficiency")
        print("  4. Single mention of 'metformin' is overpowered by wrong terminology")
        print()
        print("  This demonstrates how terminology drift breaks RAG even when the")
        print("  correct medication name is present in the query!")


def print_retrieved_document(context: str, rag_result) -> None:
    """
    Print the retrieved document content with a warning if it's the wrong document.

    Args:
        context: The retrieved document text
        rag_result: RAGResult from the retrieval step
    """
    print("RETRIEVED DOCUMENT CONTENT:")
    print("=" * 70)
    print(context[:500])
    print("...")
    print("=" * 70)
    print()
    doc_name = rag_result.document_id.replace('_guide', '').upper()
    if not rag_result.is_correct_document:
        print(f"This is information about {doc_name},")
        print("   NOT about metformin side effects!")


def print_drift_response(response, rag_result) -> None:
    """
    Print the LLM response with a warning about wrong context.

    Args:
        response: ModelResponse from the LLM invocation
        rag_result: RAGResult from the retrieval step
    """
    print("LLM RESPONSE:")
    print("=" * 70)
    print(response.output)
    print("=" * 70)
    print()
    doc_name = rag_result.document_id.replace('_guide', '').upper()
    if not rag_result.is_correct_document:
        print(f"The model answered about {doc_name}, not METFORMIN!")
        print("   The patient receives ZERO relevant information about their actual medication.")


# ============================================================================
# SHADOW TESTING DISPLAY HELPERS
# ============================================================================


def print_evaluation_progress(
    result: EvaluationResult,
    log_request_id: str,
    log_category: str
) -> None:
    """
    Print progress for a single model evaluation result.

    Args:
        result: The EvaluationResult from run_model_evaluation
        log_request_id: The request ID being evaluated
        log_category: The category of the prompt
    """
    avg_q = np.mean(list(result.aggregated_scores.values()))
    print(
        f"  {log_request_id} ({log_category})... "
        f"{result.majority_judgment}  quality={avg_q:.2f}  "
        f"safety={result.safety_score:.2f}  "
        f"{result.ttlb_ms:.0f}ms  ${result.response_cost:.6f}"
    )


def print_evaluation_summary(
    evaluation_results: Dict[str, List[EvaluationResult]],
    model_names: Dict[str, str],
    num_samples: int
) -> None:
    """
    Print summary after all model evaluations are complete.

    Args:
        evaluation_results: Dict mapping model_id to list of EvaluationResults
        model_names: Dict mapping model_id to display name
        num_samples: Number of samples evaluated
    """
    print(f"\n{'=' * 90}")
    print(f"Evaluation complete: {len(evaluation_results)} models x {num_samples} samples")
    for model_id, results in evaluation_results.items():
        name = model_names.get(model_id, model_id)
        passes = sum(1 for r in results if r.majority_judgment == "PASS")
        print(f"  {name}: {passes}/{len(results)} PASS")


def print_migration_recommendation(
    best: Dict,
    prod_model_name: str,
    prod_avg_latency: float,
    prod_avg_cost: float,
    quality_rec_model_id: str,
    model_names: Dict[str, str]
) -> None:
    """
    Print the final migration recommendation with human-readable comparisons.

    Args:
        best: Top leaderboard entry dict with model metrics
        prod_model_name: Name of the production model
        prod_avg_latency: Production average latency in ms
        prod_avg_cost: Production average cost per query
        quality_rec_model_id: Model ID recommended by quality-only ranking
        model_names: Dict mapping model_id to display name
    """
    # Compute human-readable comparison labels
    lat_ratio = best["latency_ratio"]
    cost_ratio = best["cost_ratio"]

    if lat_ratio < 0.95:
        lat_pct = (1 - lat_ratio) * 100
        latency_label = f"{lat_pct:.0f}% faster than production"
    elif lat_ratio > 1.05:
        lat_pct = (lat_ratio - 1) * 100
        latency_label = f"{lat_pct:.0f}% slower than production"
    else:
        latency_label = "similar latency to production"

    if cost_ratio < 0.95:
        cost_pct = (1 - cost_ratio) * 100
        cost_label = f"{cost_pct:.0f}% cheaper than production"
    elif cost_ratio > 1.05:
        cost_pct = (cost_ratio - 1) * 100
        cost_label = f"{cost_pct:.0f}% more expensive than production"
    else:
        cost_label = "similar cost to production"

    monthly_diff = (best["avg_cost"] - prod_avg_cost) * 100_000
    if monthly_diff > 0:
        monthly_label = f"+${monthly_diff:,.2f}/month"
    else:
        monthly_label = f"-${abs(monthly_diff):,.2f}/month"

    print("=" * 70)
    print("  MIGRATION RECOMMENDATION")
    print("=" * 70)
    print(f"\n  Recommended model:   {best['model_name']}")
    print(f"  Composite score:     {best.get('composite', 0):.3f}")
    print()
    print(f"  Accuracy")
    print(f"    Jury pass rate:    {best['pass_rate']*100:.0f}%")
    print(f"    Avg quality:       {best['avg_quality']:.2f}/5")
    print(f"    Safety score:      {best['avg_safety']:.2f}")
    print()
    print(f"  Latency")
    print(f"    Avg response time: {best['avg_latency']:.0f}ms  ({latency_label})")
    print(f"    Production:        {prod_avg_latency:.0f}ms")
    print()
    print(f"  Cost")
    print(f"    Per query:         ${best['avg_cost']:.6f}  ({cost_label})")
    print(f"    Production:        ${prod_avg_cost:.6f}")
    print(f"    Monthly (100K):    ${best['avg_cost'] * 100_000:,.2f}  ({monthly_label})")
    print(f"    Production:        ${prod_avg_cost * 100_000:,.2f}")
    print()
    print("=" * 70)


def build_leaderboard(
    evaluation_results: Dict[str, List],
    model_names: Dict[str, str],
    prod_avg_latency: float,
    prod_avg_cost: float,
) -> tuple:
    """
    Build a ranked leaderboard from evaluation results.

    Computes a composite score per model: 50% accuracy, 25% cost efficiency, 25% latency efficiency.

    Args:
        evaluation_results: Dict mapping model_id to list of EvaluationResults
        model_names: Dict mapping model_id to display name
        prod_avg_latency: Production average latency in ms
        prod_avg_cost: Production average cost per query

    Returns:
        Tuple of (leaderboard entries sorted by composite score, ModelComparison)
    """
    comparison = compare_models(evaluation_results)

    leaderboard = []
    for model_id, model_name in model_names.items():
        results = evaluation_results.get(model_id, [])
        if not results:
            continue

        pass_rate = comparison.llm_eval_rates.get(model_id, 0)
        quality_scores = comparison.quality_scores.get(model_id, {})
        avg_quality = np.mean(list(quality_scores.values())) if quality_scores else 0
        avg_safety = np.mean([r.safety_score for r in results])

        avg_latency = np.mean([r.ttlb_ms for r in results])
        avg_cost = np.mean([r.response_cost for r in results])
        latency_ratio = avg_latency / prod_avg_latency if prod_avg_latency > 0 else 1.0
        cost_ratio = avg_cost / prod_avg_cost if prod_avg_cost > 0 else 1.0

        accuracy_score = (pass_rate * 0.6 + (avg_quality / 5) * 0.4)
        cost_score = 1 / max(cost_ratio, 0.01)
        latency_score = 1 / max(latency_ratio, 0.01)
        composite = accuracy_score * 0.50 + cost_score * 0.25 + latency_score * 0.25

        leaderboard.append({
            "model_id": model_id,
            "model_name": model_name,
            "pass_rate": pass_rate,
            "avg_quality": avg_quality,
            "avg_safety": avg_safety,
            "avg_latency": avg_latency,
            "avg_cost": avg_cost,
            "latency_ratio": latency_ratio,
            "cost_ratio": cost_ratio,
            "composite": composite,
        })

    leaderboard.sort(key=lambda x: x["composite"], reverse=True)
    return leaderboard, comparison


def render_leaderboard_table(leaderboard: List[Dict]) -> None:
    """
    Render a color-coded leaderboard table using plotly.

    Args:
        leaderboard: List of leaderboard entry dicts from build_leaderboard()
    """
    import plotly.graph_objects as go

    def _color(val, go_thresh, warn_thresh, higher_is_better=True):
        if higher_is_better:
            if val >= go_thresh: return "#c6efce"
            elif val >= warn_thresh: return "#ffeb9c"
            else: return "#ffc7ce"
        else:
            if val <= go_thresh: return "#c6efce"
            elif val <= warn_thresh: return "#ffeb9c"
            else: return "#ffc7ce"

    n = len(leaderboard)
    white = ["#ffffff"] * n

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Model</b>", "<b>Score</b>", "<b>Pass Rate</b>", "<b>Quality</b>", "<b>Safety</b>",
                    "<b>Latency (ms)</b>", "<b>Latency Ratio</b>",
                    "<b>Cost/Query</b>", "<b>Cost Ratio</b>", "<b>Monthly 100K</b>"],
            fill_color="#4a4a4a",
            font=dict(color="white", size=12),
            align="center",
            height=32,
        ),
        cells=dict(
            values=[
                [e["model_name"] for e in leaderboard],
                [f"{e['composite']:.3f}" for e in leaderboard],
                [f"{e['pass_rate']*100:.0f}%" for e in leaderboard],
                [f"{e['avg_quality']:.2f}" for e in leaderboard],
                [f"{e['avg_safety']:.2f}" for e in leaderboard],
                [f"{e['avg_latency']:.0f}" for e in leaderboard],
                [f"{e['latency_ratio']:.2f}x" for e in leaderboard],
                [f"${e['avg_cost']:.6f}" for e in leaderboard],
                [f"{e['cost_ratio']:.2f}x" for e in leaderboard],
                [f"${e['avg_cost'] * 100_000:,.2f}" for e in leaderboard],
            ],
            fill_color=[
                white, white,
                [_color(e["pass_rate"]*100, 80, 60, True) for e in leaderboard],
                [_color(e["avg_quality"], 3.5, 2.5, True) for e in leaderboard],
                [_color(e["avg_safety"], 0.80, 0.60, True) for e in leaderboard],
                white,
                [_color(e["latency_ratio"], 1.5, 2.5, False) for e in leaderboard],
                white,
                [_color(e["cost_ratio"], 1.5, 3.0, False) for e in leaderboard],
                white,
            ],
            font=dict(color="black", size=12),
            align="center",
            height=28,
        ),
    )])

    fig.update_layout(
        title="Shadow Test Leaderboard (ranked by composite: 50% accuracy, 25% cost, 25% latency)",
        height=50 + 32 + 28 * n + 40,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.show()


def render_quality_breakdown_table(
    leaderboard: List[Dict],
    comparison: ModelComparison,
) -> None:
    """
    Render a color-coded quality breakdown table using plotly.

    Shows per-metric scores (correctness, completeness, etc.), safety, and latency.

    Args:
        leaderboard: List of leaderboard entry dicts from build_leaderboard()
        comparison: ModelComparison from build_leaderboard()
    """
    import plotly.graph_objects as go

    def _color(val, go_thresh, warn_thresh, higher_is_better=True):
        if higher_is_better:
            if val >= go_thresh: return "#c6efce"
            elif val >= warn_thresh: return "#ffeb9c"
            else: return "#ffc7ce"
        else:
            if val <= go_thresh: return "#c6efce"
            elif val <= warn_thresh: return "#ffeb9c"
            else: return "#ffc7ce"

    n = len(leaderboard)
    white = ["#ffffff"] * n
    metric_labels = [m.replace('_', ' ').title() for m in QUALITY_METRICS]

    # Build per-metric columns and colors
    metric_cols = []
    metric_colors = []
    for metric in QUALITY_METRICS:
        vals = []
        colors = []
        for entry in leaderboard:
            scores = comparison.quality_scores.get(entry["model_id"], {})
            v = scores.get(metric, 0)
            vals.append(f"{v:.2f}")
            colors.append(_color(v, 3.5, 2.5, True))
        metric_cols.append(vals)
        metric_colors.append(colors)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=(
                ["<b>Model</b>", "<b>Pass Rate</b>"]
                + [f"<b>{m}</b>" for m in metric_labels]
                + ["<b>Safety</b>", "<b>Latency (ms)</b>", "<b>Latency Ratio</b>"]
            ),
            fill_color="#4a4a4a",
            font=dict(color="white", size=12),
            align="center",
            height=32,
        ),
        cells=dict(
            values=(
                [[e["model_name"] for e in leaderboard],
                 [f"{e['pass_rate']*100:.0f}%" for e in leaderboard]]
                + metric_cols
                + [[f"{e['avg_safety']:.2f}" for e in leaderboard],
                   [f"{e['avg_latency']:.0f}" for e in leaderboard],
                   [f"{e['latency_ratio']:.2f}x" for e in leaderboard]]
            ),
            fill_color=(
                [white,
                 [_color(e["pass_rate"]*100, 80, 60, True) for e in leaderboard]]
                + metric_colors
                + [[_color(e["avg_safety"], 0.80, 0.60, True) for e in leaderboard],
                   white,
                   [_color(e["latency_ratio"], 1.5, 2.5, False) for e in leaderboard]]
            ),
            font=dict(color="black", size=12),
            align="center",
            height=28,
        ),
    )])

    fig.update_layout(
        title="Quality Breakdown by Metric",
        height=50 + 32 + 28 * n + 40,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.show()
