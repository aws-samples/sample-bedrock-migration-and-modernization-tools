import os
import logging
import shutil
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


# ----------------------------------------
# ChromaDB Vector Store Creation
# ----------------------------------------

def create_chroma_vectorstore(
    chunks: List[str],
    embeddings: List[List[float]],
    collection_name: str,
    persist_dir: str,
    metadata: Optional[List[Dict]] = None
) -> chromadb.Collection:
    """
    Create ChromaDB vector store from chunks and embeddings.

    Args:
        chunks: List of text chunks
        embeddings: List of embedding vectors (same length as chunks)
        collection_name: Name for the collection
        persist_dir: Directory to persist the database
        metadata: Optional metadata for each chunk

    Returns:
        ChromaDB Collection object
    """
    try:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length")

        logger.info(f"Creating ChromaDB vector store: {collection_name} in {persist_dir}")

        # Create persist directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)

        # Initialize ChromaDB client with persistence
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Delete collection if it exists (fresh creation)
        try:
            client.delete_collection(name=collection_name)
            logger.debug(f"Deleted existing collection: {collection_name}")
        except:
            pass

        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "RAG evaluation vector store"}
        )

        # Prepare IDs
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Prepare metadata
        if metadata is None:
            metadata = [{"chunk_index": i, "chunk_length": len(chunk)} for i, chunk in enumerate(chunks)]
        else:
            # Ensure metadata is a list of dicts
            if len(metadata) != len(chunks):
                logger.warning(f"Metadata length ({len(metadata)}) doesn't match chunks ({len(chunks)}), using default")
                metadata = [{"chunk_index": i} for i in range(len(chunks))]

        # Add documents to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata
        )

        logger.info(f"Successfully created vector store with {len(chunks)} chunks")
        return collection

    except Exception as e:
        logger.error(f"Error creating ChromaDB vector store: {str(e)}", exc_info=True)
        raise


# ----------------------------------------
# Vector Store Loading
# ----------------------------------------

def load_chroma_vectorstore(
    persist_dir: str,
    collection_name: str
) -> chromadb.Collection:
    """
    Load existing ChromaDB vector store.

    Args:
        persist_dir: Directory where database is persisted
        collection_name: Name of the collection

    Returns:
        ChromaDB Collection object
    """
    try:
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Vector store directory not found: {persist_dir}")

        logger.info(f"Loading ChromaDB vector store: {collection_name} from {persist_dir}")

        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        collection = client.get_collection(name=collection_name)
        count = collection.count()

        logger.info(f"Successfully loaded vector store with {count} chunks")
        return collection

    except Exception as e:
        logger.error(f"Error loading ChromaDB vector store: {str(e)}", exc_info=True)
        raise


# ----------------------------------------
# Retrieval Functions
# ----------------------------------------

def retrieve_from_vectorstore(
    collection: chromadb.Collection,
    query_embedding: List[float],
    top_k: int = 5,
    filter_metadata: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Retrieve top-k most similar chunks from vector store.

    Args:
        collection: ChromaDB collection
        query_embedding: Query embedding vector
        top_k: Number of chunks to retrieve
        filter_metadata: Optional metadata filter

    Returns:
        Dictionary with 'chunks', 'distances', 'metadata', 'ids'
    """
    try:
        logger.debug(f"Retrieving top-{top_k} chunks from vector store")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        # Extract results
        retrieved_chunks = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        ids = results['ids'][0] if results['ids'] else []

        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")

        return {
            'chunks': retrieved_chunks,
            'distances': distances,
            'metadata': metadatas,
            'ids': ids,
            'num_results': len(retrieved_chunks)
        }

    except Exception as e:
        logger.error(f"Error retrieving from vector store: {str(e)}", exc_info=True)
        raise


def retrieve_multiple_queries(
    collection: chromadb.Collection,
    query_embeddings: List[List[float]],
    top_k: int = 5
) -> List[Dict[str, any]]:
    """
    Retrieve chunks for multiple queries.

    Args:
        collection: ChromaDB collection
        query_embeddings: List of query embedding vectors
        top_k: Number of chunks per query

    Returns:
        List of retrieval results (one per query)
    """
    results = []

    for i, query_embedding in enumerate(query_embeddings):
        try:
            result = retrieve_from_vectorstore(collection, query_embedding, top_k)
            results.append(result)
        except Exception as e:
            logger.error(f"Error retrieving for query {i}: {str(e)}")
            results.append({
                'chunks': [],
                'distances': [],
                'metadata': [],
                'ids': [],
                'num_results': 0,
                'error': str(e)
            })

    return results


# ----------------------------------------
# Vector Store Management
# ----------------------------------------

def get_vectorstore_stats(collection: chromadb.Collection) -> Dict[str, any]:
    """
    Get statistics about the vector store.

    Args:
        collection: ChromaDB collection

    Returns:
        Dictionary with statistics
    """
    try:
        count = collection.count()

        # Get a sample to determine embedding dimensions
        sample = collection.get(limit=1, include=['embeddings'])
        dimensions = len(sample['embeddings'][0]) if sample['embeddings'] is not None and len(sample['embeddings']) > 0 else 0

        stats = {
            'num_chunks': count,
            'embedding_dimensions': dimensions,
            'collection_name': collection.name,
            'metadata': collection.metadata
        }

        logger.debug(f"Vector store stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}", exc_info=True)
        return {
            'num_chunks': 0,
            'embedding_dimensions': 0,
            'error': str(e)
        }


def cleanup_vectorstore(persist_dir: str) -> bool:
    """
    Delete vector store directory.

    Args:
        persist_dir: Directory to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(persist_dir):
            logger.info(f"Cleaning up vector store: {persist_dir}")
            shutil.rmtree(persist_dir)
            logger.info(f"Successfully deleted vector store directory")
            return True
        else:
            logger.warning(f"Vector store directory does not exist: {persist_dir}")
            return False

    except Exception as e:
        logger.error(f"Error cleaning up vector store: {str(e)}", exc_info=True)
        return False


# ----------------------------------------
# Batch Operations
# ----------------------------------------

def add_chunks_to_vectorstore(
    collection: chromadb.Collection,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: Optional[List[Dict]] = None,
    start_index: int = 0
) -> int:
    """
    Add additional chunks to existing vector store.

    Args:
        collection: ChromaDB collection
        chunks: List of text chunks to add
        embeddings: List of embedding vectors
        metadata: Optional metadata for each chunk
        start_index: Starting index for chunk IDs

    Returns:
        Number of chunks added
    """
    try:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length")

        # Prepare IDs
        ids = [f"chunk_{start_index + i}" for i in range(len(chunks))]

        # Prepare metadata
        if metadata is None:
            metadata = [{"chunk_index": start_index + i} for i in range(len(chunks))]

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata
        )

        logger.info(f"Added {len(chunks)} chunks to vector store")
        return len(chunks)

    except Exception as e:
        logger.error(f"Error adding chunks to vector store: {str(e)}", exc_info=True)
        raise


# ----------------------------------------
# Helper Functions
# ----------------------------------------

def convert_distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """
    Convert distance metric to similarity score.

    Args:
        distance: Distance value from ChromaDB
        metric: Distance metric used ('cosine', 'euclidean', 'manhattan')

    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if metric == "cosine":
        # ChromaDB cosine distance is 1 - cosine_similarity
        # So similarity = 1 - distance
        return max(0.0, min(1.0, 1.0 - distance))
    elif metric == "euclidean":
        # Convert euclidean distance to similarity (heuristic)
        return 1.0 / (1.0 + distance)
    elif metric == "manhattan":
        # Convert manhattan distance to similarity (heuristic)
        return 1.0 / (1.0 + distance)
    else:
        logger.warning(f"Unknown metric '{metric}', returning raw distance")
        return distance


def format_retrieval_results(
    results: Dict[str, any],
    include_scores: bool = True
) -> List[Dict[str, any]]:
    """
    Format retrieval results into a more readable structure.

    Args:
        results: Raw results from retrieve_from_vectorstore()
        include_scores: Whether to include similarity scores

    Returns:
        List of formatted result dicts
    """
    formatted = []

    chunks = results.get('chunks', [])
    distances = results.get('distances', [])
    metadatas = results.get('metadata', [])
    ids = results.get('ids', [])

    for i in range(len(chunks)):
        item = {
            'rank': i + 1,
            'chunk': chunks[i],
            'id': ids[i] if i < len(ids) else None,
            'metadata': metadatas[i] if i < len(metadatas) else {}
        }

        if include_scores and i < len(distances):
            item['distance'] = distances[i]
            item['similarity'] = convert_distance_to_similarity(distances[i])

        formatted.append(item)

    return formatted
