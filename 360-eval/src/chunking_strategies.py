import os
import logging
import importlib.util
from typing import List, Dict, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------
# 1. Recursive Chunking
# ----------------------------------------

def recursive_chunking(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: List[str] = None
) -> List[str]:
    """
    Recursively chunk text using hierarchical separators.

    Uses LangChain's RecursiveCharacterTextSplitter under the hood.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        separators: List of separators to try (if None, uses defaults)

    Returns:
        List of text chunks
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        chunks = splitter.split_text(text)
        logger.debug(f"Recursive chunking: {len(text)} chars → {len(chunks)} chunks")
        return chunks

    except ImportError:
        logger.error("langchain-text-splitters not installed. Install with: pip install langchain-text-splitters")
        raise
    except Exception as e:
        logger.error(f"Error in recursive chunking: {str(e)}", exc_info=True)
        raise


# ----------------------------------------
# 2. Semantic Chunking
# ----------------------------------------

def semantic_chunking(
    text: str,
    embedding_model_id: str,
    embedding_params: Optional[Dict] = None,
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000
) -> List[str]:
    """
    Chunk text based on semantic similarity between sentences.

    Groups consecutive sentences until semantic similarity drops below threshold.

    Args:
        text: Text to chunk
        embedding_model_id: Embedding model for semantic similarity
        embedding_params: Parameters for embedding model
        similarity_threshold: Similarity threshold for grouping (0-1)
        min_chunk_size: Minimum characters per chunk
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    try:
        from embedding_utils import generate_embeddings_litellm

        # Split into sentences
        sentences = split_into_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # Generate embeddings for all sentences
        embeddings, _ = generate_embeddings_litellm(
            texts=sentences,
            model_id=embedding_model_id,
            provider_params=embedding_params or {}
        )

        # Convert to numpy for easier computation
        embeddings_array = np.array(embeddings)

        # Group sentences based on semantic similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_size = len(sentences[0])

        for i in range(1, len(sentences)):
            # Calculate cosine similarity with previous sentence
            similarity = cosine_similarity(embeddings_array[i-1], embeddings_array[i])

            # Check if we should continue current chunk
            would_exceed_max = current_chunk_size + len(sentences[i]) > max_chunk_size
            is_similar = similarity >= similarity_threshold
            meets_min_size = current_chunk_size >= min_chunk_size

            if is_similar and not would_exceed_max:
                # Add to current chunk
                current_chunk.append(sentences[i])
                current_chunk_size += len(sentences[i])
            else:
                # Start new chunk if current chunk meets minimum size
                if meets_min_size or would_exceed_max:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentences[i]]
                    current_chunk_size = len(sentences[i])
                else:
                    # Still add to current chunk if below minimum
                    current_chunk.append(sentences[i])
                    current_chunk_size += len(sentences[i])

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.debug(f"Semantic chunking: {len(sentences)} sentences → {len(chunks)} chunks")
        return chunks

    except ImportError as e:
        logger.error(f"Required library not available: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in semantic chunking: {str(e)}", exc_info=True)
        raise


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple heuristics.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    import re

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (0-1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


# ----------------------------------------
# 3. Fixed-Size Chunking
# ----------------------------------------

def fixed_size_chunking(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Chunk text into fixed-size pieces with overlap.

    Simple sliding window approach.

    Args:
        text: Text to chunk
        chunk_size: Characters per chunk
        chunk_overlap: Overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)

        # Move window forward
        start += (chunk_size - chunk_overlap)

    logger.debug(f"Fixed-size chunking: {text_length} chars → {len(chunks)} chunks")
    return chunks


# ----------------------------------------
# 4. Custom Chunking
# ----------------------------------------

def custom_chunking(
    text: str,
    custom_script_path: str,
    **kwargs
) -> List[str]:
    """
    Chunk text using user-provided custom Python script.

    The custom script must define a function:
        def chunk_text(text: str, **kwargs) -> List[str]

    Args:
        text: Text to chunk
        custom_script_path: Path to custom Python script
        **kwargs: Additional arguments to pass to custom function

    Returns:
        List of text chunks
    """
    try:
        if not os.path.exists(custom_script_path):
            raise FileNotFoundError(f"Custom script not found: {custom_script_path}")

        # Load the custom script as a module
        spec = importlib.util.spec_from_file_location("custom_chunker", custom_script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load custom script: {custom_script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check if chunk_text function exists
        if not hasattr(module, 'chunk_text'):
            raise AttributeError(
                f"Custom script must define a 'chunk_text(text: str, **kwargs) -> List[str]' function"
            )

        # Execute custom chunking
        chunk_text_func = getattr(module, 'chunk_text')
        chunks = chunk_text_func(text, **kwargs)

        # Validate output
        if not isinstance(chunks, list):
            raise TypeError(f"Custom chunk_text() must return a list, got {type(chunks)}")
        if not all(isinstance(c, str) for c in chunks):
            raise TypeError("Custom chunk_text() must return a list of strings")

        logger.info(f"Custom chunking: {len(text)} chars → {len(chunks)} chunks using {custom_script_path}")
        return chunks

    except Exception as e:
        logger.error(f"Error in custom chunking: {str(e)}", exc_info=True)
        raise


# ----------------------------------------
# Unified Chunking Interface
# ----------------------------------------

def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model_id: str = None,
    embedding_params: Dict = None,
    similarity_threshold: float = 0.5,
    custom_script_path: str = None,
    **kwargs
) -> List[str]:
    """
    Chunk text using specified strategy.

    Args:
        text: Text to chunk
        strategy: Chunking strategy ('recursive', 'semantic', 'fixed', 'custom')
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks
        embedding_model_id: Embedding model for semantic chunking
        embedding_params: Parameters for embedding model
        similarity_threshold: Threshold for semantic chunking
        custom_script_path: Path to custom chunking script
        **kwargs: Additional arguments for custom chunking

    Returns:
        List of text chunks
    """
    strategy = strategy.lower()

    logger.info(f"Chunking text using '{strategy}' strategy (size={chunk_size}, overlap={chunk_overlap})")

    if strategy == "recursive":
        return recursive_chunking(text, chunk_size, chunk_overlap)

    elif strategy == "semantic":
        if not embedding_model_id:
            raise ValueError("embedding_model_id required for semantic chunking")
        return semantic_chunking(
            text,
            embedding_model_id,
            embedding_params,
            similarity_threshold,
            min_chunk_size=chunk_size // 2,
            max_chunk_size=chunk_size
        )

    elif strategy == "fixed":
        return fixed_size_chunking(text, chunk_size, chunk_overlap)

    elif strategy == "custom":
        if not custom_script_path:
            raise ValueError("custom_script_path required for custom chunking")
        return custom_chunking(text, custom_script_path, **kwargs)

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Use 'recursive', 'semantic', 'fixed', or 'custom'")


# ----------------------------------------
# Batch Chunking
# ----------------------------------------

def chunk_documents(
    texts: List[str],
    strategy: str = "recursive",
    **kwargs
) -> List[Dict[str, any]]:
    """
    Chunk multiple documents and track source.

    Args:
        texts: List of text documents
        strategy: Chunking strategy
        **kwargs: Arguments for chunking strategy

    Returns:
        List of dicts with 'chunk', 'source_index', 'chunk_index' keys
    """
    all_chunks = []

    for doc_idx, text in enumerate(texts):
        try:
            chunks = chunk_text(text, strategy=strategy, **kwargs)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'chunk': chunk,
                    'source_index': doc_idx,
                    'chunk_index': chunk_idx,
                    'total_chunks_in_doc': len(chunks)
                })

        except Exception as e:
            logger.error(f"Failed to chunk document {doc_idx}: {str(e)}")
            continue

    logger.info(f"Chunked {len(texts)} documents → {len(all_chunks)} total chunks")
    return all_chunks


# ----------------------------------------
# Chunk Statistics
# ----------------------------------------

def analyze_chunks(chunks: List[str]) -> Dict[str, any]:
    """
    Analyze chunk statistics.

    Args:
        chunks: List of text chunks

    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'avg_chunk_size': 0,
            'min_chunk_size': 0,
            'max_chunk_size': 0,
            'total_chars': 0
        }

    chunk_sizes = [len(c) for c in chunks]

    return {
        'num_chunks': len(chunks),
        'avg_chunk_size': np.mean(chunk_sizes),
        'std_chunk_size': np.std(chunk_sizes),
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
        'total_chars': sum(chunk_sizes),
        'median_chunk_size': np.median(chunk_sizes)
    }
