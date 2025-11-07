"""
Similarity Calculators for RAG Evaluation

Provides multiple methods for calculating text similarity:
- Jaccard (word overlap)
- Cosine (embedding-based)
- Sentence Transformer (semantic similarity)
- LLM Judge (LLM-based evaluation)
"""

import logging
import time
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
import numpy as np
import threading

logger = logging.getLogger(__name__)

# Global lock for thread-safe model loading
_model_load_lock = threading.Lock()


# ----------------------------------------
# Base Similarity Calculator
# ----------------------------------------

class SimilarityCalculator(ABC):
    """
    Abstract base class for similarity calculators.
    """

    def __init__(self, name: str, default_threshold: float = 0.7):
        """
        Initialize similarity calculator.

        Args:
            name: Name of the similarity method
            default_threshold: Default similarity threshold (0-1)
        """
        self.name = name
        self.default_threshold = default_threshold

    @abstractmethod
    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        pass

    def batch_calculate(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Calculate similarities for multiple text pairs.

        Default implementation calls calculate() for each pair.
        Subclasses can override for batch optimization.

        Args:
            texts1: List of first texts
            texts2: List of second texts

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have same length")

        return [self.calculate(t1, t2) for t1, t2 in zip(texts1, texts2)]

    def is_similar(self, text1: str, text2: str, threshold: Optional[float] = None) -> bool:
        """
        Check if two texts are similar above threshold.

        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (uses default if None)

        Returns:
            True if similarity >= threshold
        """
        threshold = threshold if threshold is not None else self.default_threshold
        return self.calculate(text1, text2) >= threshold


# ----------------------------------------
# 1. Jaccard Similarity (Word Overlap)
# ----------------------------------------

class JaccardSimilarity(SimilarityCalculator):
    """
    Jaccard similarity based on word token overlap.
    Fast, simple, lexical matching only.
    """

    def __init__(self, default_threshold: float = 0.7):
        super().__init__("jaccard", default_threshold)

    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity (intersection / union).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity score (0-1)
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


# ----------------------------------------
# 2. Cosine Similarity (Embedding-based)
# ----------------------------------------

class CosineSimilarity(SimilarityCalculator):
    """
    Cosine similarity using pre-computed embeddings.
    Requires embeddings to be generated separately.
    """

    def __init__(self, embedding_cache: Dict[str, List[float]], default_threshold: float = 0.85):
        """
        Initialize with embedding cache.

        Args:
            embedding_cache: Dict mapping text -> embedding vector
            default_threshold: Similarity threshold (cosine uses higher threshold)
        """
        super().__init__("cosine", default_threshold)
        self.embedding_cache = embedding_cache

    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between embedding vectors.

        Args:
            text1: First text (must be in cache)
            text2: Second text (must be in cache)

        Returns:
            Cosine similarity score (0-1)
        """
        # Get embeddings from cache
        emb1 = self.embedding_cache.get(text1)
        emb2 = self.embedding_cache.get(text2)

        if emb1 is None or emb2 is None:
            logger.warning(f"Missing embedding for cosine similarity, falling back to 0.0")
            return 0.0

        # Calculate cosine similarity
        return self._cosine_similarity(emb1, emb2)

    def batch_calculate(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Optimized batch calculation using vectorized operations.

        Args:
            texts1: List of first texts
            texts2: List of second texts

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have same length")

        scores = []
        for t1, t2 in zip(texts1, texts2):
            scores.append(self.calculate(t1, t2))

        return scores

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1, normalized from -1 to 1)
        """
        try:
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Cosine similarity ranges from -1 to 1, normalize to 0-1
            cosine_sim = dot_product / (norm1 * norm2)
            normalized = (cosine_sim + 1) / 2  # Map from [-1,1] to [0,1]

            return float(normalized)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0


# ----------------------------------------
# 3. Sentence Transformer Similarity
# ----------------------------------------

class SentenceTransformerSimilarity(SimilarityCalculator):
    """
    Semantic similarity using Sentence Transformers.
    Uses specialized sentence embedding models.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", default_threshold: float = 0.80):
        """
        Initialize Sentence Transformer model.

        Args:
            model_name: HuggingFace model name
            default_threshold: Similarity threshold
        """
        super().__init__("sentence_transformer", default_threshold)
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model with thread-safety."""
        try:
            from sentence_transformers import SentenceTransformer

            # Use a lock to prevent concurrent model loading issues
            # This is especially important on MPS (Metal Performance Shaders) on macOS
            with _model_load_lock:
                logger.info(f"Loading Sentence Transformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded successfully: {self.model_name}")

        except ImportError:
            logger.error("sentence-transformers library not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer model {self.model_name}: {e}", exc_info=True)
            raise

    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using Sentence Transformers.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        if self.model is None:
            logger.error(f"Sentence Transformer model {self.model_name} not loaded")
            return 0.0

        try:
            from sentence_transformers import util

            # Validate inputs
            if not text1 or not text2:
                logger.warning("Empty text provided to calculate()")
                return 0.0

            # Encode texts
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True, show_progress_bar=False)

            # Calculate cosine similarity
            similarity = util.cos_sim(embeddings[0], embeddings[1])

            # Convert to float and normalize to [0, 1]
            score = float(similarity.item())
            normalized = (score + 1) / 2  # Map from [-1,1] to [0,1]

            return normalized

        except Exception as e:
            logger.error(f"Error calculating sentence transformer similarity with {self.model_name}: {e}", exc_info=True)
            return 0.0

    def batch_calculate(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Optimized batch calculation using batch encoding.

        Args:
            texts1: List of first texts
            texts2: List of second texts

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have same length")

        if self.model is None:
            logger.error(f"Sentence Transformer model {self.model_name} not loaded")
            return [0.0] * len(texts1)

        try:
            from sentence_transformers import util

            # Batch encode all texts
            all_texts = texts1 + texts2
            logger.debug(f"Batch encoding {len(all_texts)} texts with {self.model_name}")
            embeddings = self.model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)

            # Split embeddings
            emb1 = embeddings[:len(texts1)]
            emb2 = embeddings[len(texts1):]

            # Calculate pairwise similarities
            scores = []
            for i in range(len(texts1)):
                similarity = util.cos_sim(emb1[i], emb2[i])
                score = float(similarity.item())
                normalized = (score + 1) / 2
                scores.append(normalized)

            logger.debug(f"Successfully calculated {len(scores)} similarity scores")
            return scores

        except Exception as e:
            logger.error(f"Error in batch sentence transformer similarity with {self.model_name}: {e}", exc_info=True)
            return [0.0] * len(texts1)


# ----------------------------------------
# 4. LLM Judge Similarity
# ----------------------------------------

class LLMJudgeSimilarity(SimilarityCalculator):
    """
    LLM-based similarity evaluation.
    Uses a language model to judge semantic similarity.
    """

    SIMILARITY_PROMPT = """Compare the semantic similarity between these two text chunks.
Rate from 0.0 (completely different) to 1.0 (essentially identical).
Consider meaning, not just exact words.

Chunk 1: {chunk1}

Chunk 2: {chunk2}

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

    def __init__(
        self,
        model_id: str,
        provider_params: Optional[Dict] = None,
        default_threshold: float = 0.7
    ):
        """
        Initialize LLM Judge.

        Args:
            model_id: LLM model identifier (e.g., "bedrock/claude-3-5-haiku")
            provider_params: Provider-specific parameters (API keys, region, etc.)
            default_threshold: Similarity threshold
        """
        super().__init__("llm_judge", default_threshold)
        self.model_id = model_id
        self.provider_params = provider_params or {}

    def calculate(self, text1: str, text2: str) -> float:
        """
        Use LLM to judge similarity between texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1) as judged by LLM
        """
        try:
            from utils import run_inference

            # Create prompt
            prompt = self.SIMILARITY_PROMPT.format(chunk1=text1, chunk2=text2)

            # Set LLM parameters for quick, deterministic response
            params = {
                **self.provider_params,
                "maxTokens": 10,  # Just need a number
                "temperature": 0.0,  # Deterministic
                "topP": 1.0
            }

            # Call LLM
            response = run_inference(
                model_name=self.model_id,
                prompt_text=prompt,
                provider_params=params,
                stream=False
            )

            # Parse response
            score_text = response.get('text', '').strip()
            score = self._parse_score(score_text)

            logger.debug(f"LLM Judge ({self.model_id}): {score:.3f}")
            return score

        except Exception as e:
            logger.error(f"Error in LLM judge similarity: {e}")
            return 0.0

    def batch_calculate(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Calculate similarities for multiple pairs.

        Note: This calls the LLM sequentially. For production use,
        consider implementing parallel batch processing.

        Args:
            texts1: List of first texts
            texts2: List of second texts

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have same length")

        logger.info(f"Running LLM Judge for {len(texts1)} comparisons (sequential)...")

        scores = []
        for i, (t1, t2) in enumerate(zip(texts1, texts2)):
            if i % 10 == 0:
                logger.debug(f"LLM Judge progress: {i}/{len(texts1)}")
            score = self.calculate(t1, t2)
            scores.append(score)

        return scores

    @staticmethod
    def _parse_score(score_text: str) -> float:
        """
        Parse LLM response to extract similarity score.

        Args:
            score_text: LLM response text

        Returns:
            Parsed score (0-1), or 0.0 if parsing fails
        """
        import re

        try:
            # Try to extract first number in range [0, 1]
            # Look for patterns like: 0.85, .85, 1.0, 0
            matches = re.findall(r'\b(0?\.\d+|1\.0|0|1)\b', score_text)

            if matches:
                score = float(matches[0])
                # Clamp to [0, 1]
                return max(0.0, min(1.0, score))

            logger.warning(f"Could not parse score from: {score_text}")
            return 0.0

        except ValueError:
            logger.warning(f"Failed to parse score: {score_text}")
            return 0.0


# ----------------------------------------
# Factory Function
# ----------------------------------------

def create_similarity_calculator(
    method: str,
    config: Dict
) -> SimilarityCalculator:
    """
    Factory function to create similarity calculator based on method.

    Args:
        method: Similarity method name ('jaccard', 'cosine', 'sentence_transformer', 'llm_judge')
        config: Configuration dict with method-specific parameters

    Returns:
        SimilarityCalculator instance

    Raises:
        ValueError: If method is unknown
    """
    threshold = config.get('threshold', 0.7)

    if method == 'jaccard':
        return JaccardSimilarity(default_threshold=threshold)

    elif method == 'cosine':
        embedding_cache = config.get('embedding_cache', {})
        if not embedding_cache:
            logger.warning("No embedding cache provided for cosine similarity")
        return CosineSimilarity(embedding_cache=embedding_cache, default_threshold=threshold)

    elif method == 'sentence_transformer':
        model_id = config.get('model_id', 'all-MiniLM-L6-v2')
        return SentenceTransformerSimilarity(model_name=model_id, default_threshold=threshold)

    elif method == 'llm_judge':
        model_id = config.get('model_id')
        if not model_id:
            raise ValueError("model_id required for llm_judge similarity")
        provider_params = config.get('provider_params', {})
        return LLMJudgeSimilarity(
            model_id=model_id,
            provider_params=provider_params,
            default_threshold=threshold
        )

    else:
        raise ValueError(f"Unknown similarity method: {method}. Must be one of: jaccard, cosine, sentence_transformer, llm_judge")
