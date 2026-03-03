"""Tests for model-merger handler functions.

Tests the get_base_model_id() function which handles deduplication
of model variants by normalizing model IDs to their base form.
"""

import pytest
import re
import sys
import importlib.util
from pathlib import Path

# Load the model_matcher module directly from file (bypasses __init__.py and powertools dependency)
MODEL_MATCHER_PATH = (
    Path(__file__).parent.parent
    / "layers"
    / "common"
    / "python"
    / "shared"
    / "model_matcher.py"
)

spec = importlib.util.spec_from_file_location("model_matcher", MODEL_MATCHER_PATH)
model_matcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_matcher)

get_model_variant_info = model_matcher.get_model_variant_info


# Copy of the function under test (mirrors handler.py implementation)
def get_base_model_id(model_id: str) -> str:
    """
    Extract the base model ID by removing context window and variant suffixes.

    Uses the centralized model_matcher utility for consistent behavior across
    all pipeline components.

    Examples:
        'anthropic.claude-3-5-sonnet-20240620-v1:0:18k' -> 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        'amazon.nova-premier-v1:0:mm' -> 'amazon.nova-premier-v1:0'
        'amazon.nova-reel-v1:1' -> 'amazon.nova-reel-v1:0'
        'amazon.titan-embed-image-v1' -> 'amazon.titan-embed-image-v1:0'
    """
    variant_info = get_model_variant_info(model_id)
    base_id = variant_info.get("base_id", model_id)

    # Normalize version suffix (:1, :2, etc.) to :0
    # Only if the model ends with :N where N is a single digit
    base_id = re.sub(r":([1-9])$", ":0", base_id)

    # Add :0 if model doesn't have a version suffix at all
    # Check if model ends with :\d+ pattern
    if not re.search(r":\d+$", base_id):
        base_id = f"{base_id}:0"

    return base_id


class TestGetBaseModelId:
    """Tests for get_base_model_id function."""

    # Pattern 1: Context window suffixes (:NNNk)
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            (
                "anthropic.claude-3-5-sonnet-20240620-v1:0:18k",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
            ),
            (
                "anthropic.claude-3-5-sonnet-20240620-v1:0:200k",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
            ),
            (
                "meta.llama3-70b-instruct-v1:0:51k",
                "meta.llama3-70b-instruct-v1:0",
            ),
            (
                "anthropic.claude-3-opus-20240229-v1:0:32k",
                "anthropic.claude-3-opus-20240229-v1:0",
            ),
        ],
    )
    def test_removes_context_window_suffix(self, model_id, expected):
        """Should remove :NNNk context window suffixes."""
        assert get_base_model_id(model_id) == expected

    # Pattern 2: Multimodal suffix (:mm)
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            (
                "amazon.nova-premier-v1:0:mm",
                "amazon.nova-premier-v1:0",
            ),
            (
                "amazon.nova-pro-v1:0:mm",
                "amazon.nova-pro-v1:0",
            ),
            (
                "amazon.nova-lite-v1:0:mm",
                "amazon.nova-lite-v1:0",
            ),
        ],
    )
    def test_removes_multimodal_suffix(self, model_id, expected):
        """Should remove :mm multimodal suffixes."""
        assert get_base_model_id(model_id) == expected

    # Pattern 3: Version normalization (:1, :2, etc. -> :0)
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            (
                "amazon.nova-reel-v1:1",
                "amazon.nova-reel-v1:0",
            ),
            (
                "amazon.nova-canvas-v1:1",
                "amazon.nova-canvas-v1:0",
            ),
            (
                "some.model-v1:2",
                "some.model-v1:0",
            ),
            (
                "some.model-v1:9",
                "some.model-v1:0",
            ),
        ],
    )
    def test_normalizes_version_suffix(self, model_id, expected):
        """Should normalize version suffixes (:1, :2, etc.) to :0."""
        assert get_base_model_id(model_id) == expected

    # Pattern 4: Add :0 if missing
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            (
                "amazon.titan-embed-image-v1",
                "amazon.titan-embed-image-v1:0",
            ),
            (
                "amazon.titan-embed-text-v1",
                "amazon.titan-embed-text-v1:0",
            ),
            (
                "cohere.embed-english-v3",
                "cohere.embed-english-v3:0",
            ),
            (
                "cohere.embed-multilingual-v3",
                "cohere.embed-multilingual-v3:0",
            ),
        ],
    )
    def test_adds_version_suffix_if_missing(self, model_id, expected):
        """Should add :0 suffix if model doesn't have a version suffix."""
        assert get_base_model_id(model_id) == expected

    # Base models (should remain unchanged)
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            (
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
            ),
            (
                "amazon.titan-text-express-v1:0",
                "amazon.titan-text-express-v1:0",
            ),
            (
                "meta.llama3-8b-instruct-v1:0",
                "meta.llama3-8b-instruct-v1:0",
            ),
            (
                "mistral.mistral-7b-instruct-v0:2",
                "mistral.mistral-7b-instruct-v0:0",
            ),
        ],
    )
    def test_base_models_normalized(self, model_id, expected):
        """Base models should be normalized to :0."""
        assert get_base_model_id(model_id) == expected

    # Combined patterns (e.g., context window + multimodal)
    def test_handles_combined_suffixes(self):
        """Should handle models that might have multiple suffix types."""
        # Context window variant of a model that also has mm variant
        assert get_base_model_id("amazon.nova-pro-v1:0:200k") == "amazon.nova-pro-v1:0"
        assert get_base_model_id("amazon.nova-pro-v1:0:mm") == "amazon.nova-pro-v1:0"
        # Both should normalize to the same base
        base1 = get_base_model_id("amazon.nova-pro-v1:0:200k")
        base2 = get_base_model_id("amazon.nova-pro-v1:0:mm")
        assert base1 == base2

    # Edge cases
    def test_preserves_model_with_k_in_name(self):
        """Should not incorrectly strip 'k' that's part of the model name."""
        # Model names with 'k' in them should not be affected
        assert get_base_model_id("meta.llama3-8k-v1:0") == "meta.llama3-8k-v1:0"

    def test_handles_double_digit_versions(self):
        """Double-digit versions should not be normalized (only single digit)."""
        # :10, :20, etc. are not version suffixes, they're something else
        # The function only normalizes single-digit versions
        assert get_base_model_id("some.model-v1:10") == "some.model-v1:10"

    def test_deduplication_scenario(self):
        """Test that variants all normalize to the same base for deduplication."""
        variants = [
            "amazon.nova-premier-v1:0",
            "amazon.nova-premier-v1:0:mm",
            "amazon.nova-premier-v1:0:200k",
        ]
        base_ids = [get_base_model_id(v) for v in variants]
        # All should normalize to the same base
        assert len(set(base_ids)) == 1
        assert base_ids[0] == "amazon.nova-premier-v1:0"

    def test_titan_embed_deduplication(self):
        """Test that titan-embed models with/without :0 deduplicate correctly."""
        variants = [
            "amazon.titan-embed-image-v1",
            "amazon.titan-embed-image-v1:0",
        ]
        base_ids = [get_base_model_id(v) for v in variants]
        # Both should normalize to the same base
        assert len(set(base_ids)) == 1
        assert base_ids[0] == "amazon.titan-embed-image-v1:0"

    def test_nova_reel_deduplication(self):
        """Test that nova-reel :0 and :1 variants deduplicate correctly."""
        variants = [
            "amazon.nova-reel-v1:0",
            "amazon.nova-reel-v1:1",
        ]
        base_ids = [get_base_model_id(v) for v in variants]
        # Both should normalize to the same base
        assert len(set(base_ids)) == 1
        assert base_ids[0] == "amazon.nova-reel-v1:0"
