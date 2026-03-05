"""Tests for Mantle Availability Flags in final-aggregator (Task 10).

Tests for Phase 3 - Mantle Availability Flags:
- has_mantle_pricing() function
- build_mantle_inference() with has_pricing flag
- availability.mantle.has_pricing in final output

These tests verify that Mantle pricing availability is correctly detected
and included in the model output.
"""

from __future__ import annotations

import pytest
import re
from pathlib import Path

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"
LAYERS_PATH = Path(__file__).parent.parent / "layers" / "common" / "python" / "shared"


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


def get_model_matcher_source() -> str:
    """Read the model_matcher source code."""
    return (LAYERS_PATH / "model_matcher.py").read_text()


def _extract_function(func_name: str, source_lines: list) -> str:
    """Extract a function definition from source lines."""
    in_function = False
    func_lines = []
    base_indent = 0

    for line in source_lines:
        if f"def {func_name}(" in line:
            in_function = True
            base_indent = len(line) - len(line.lstrip())
            func_lines.append(line)
        elif in_function:
            # Check if we've exited the function (new top-level def or class)
            stripped = line.lstrip()
            current_indent = len(line) - len(line.lstrip()) if stripped else -1

            if stripped and current_indent >= 0 and current_indent <= base_indent:
                if (
                    stripped.startswith("def ")
                    or stripped.startswith("class ")
                    or stripped.startswith("@")
                ):
                    break
            func_lines.append(line)

    return "\n".join(func_lines)


def _extract_constant(const_name: str, source_lines: list) -> str:
    """Extract a constant definition from source lines.

    Handles both simple assignments (NAME = value) and type-annotated
    assignments (NAME: type = value).
    """
    in_constant = False
    const_lines = []
    bracket_depth = 0

    for line in source_lines:
        # Match both "NAME = " and "NAME: type = " patterns
        if line.startswith(f"{const_name} = ") or line.startswith(f"{const_name}: "):
            in_constant = True
            const_lines.append(line)
            # Count brackets to handle multi-line definitions
            bracket_depth += line.count("[") + line.count("{") + line.count("(")
            bracket_depth -= line.count("]") + line.count("}") + line.count(")")
            if bracket_depth <= 0:
                break
        elif in_constant:
            const_lines.append(line)
            bracket_depth += line.count("[") + line.count("{") + line.count("(")
            bracket_depth -= line.count("]") + line.count("}") + line.count(")")
            if bracket_depth <= 0:
                break

    return "\n".join(const_lines)


def _load_mantle_functions():
    """Load Mantle-related functions by extracting them from source.

    Since the handler has dependencies on shared modules that aren't available
    in the test environment, we extract and exec just the functions we need.
    """
    source = get_handler_source("final-aggregator")
    model_matcher_source = get_model_matcher_source()
    lines = source.split("\n")
    model_matcher_lines = model_matcher_source.split("\n")

    # List of all constants needed by get_canonical_model_id and get_provider_from_model_id
    constants_to_extract = [
        "PROVIDER_ALIASES",
        "PROVIDER_DISPLAY_NAMES",
        "DIMENSION_SUFFIX_PATTERN",
        "MULTIMODAL_SUFFIX",
        "API_VERSION_SUFFIXES",
        "REDUNDANT_PROVIDER_PATTERN",
        "API_VERSION_IN_ID_PATTERN",
        "INSTANCE_VERSION_PATTERN",
        "CONTEXT_WINDOW_PATTERN",
        "SEMANTIC_VERSION_PATTERN",
    ]

    constants_source = []
    for const_name in constants_to_extract:
        const_src = _extract_constant(const_name, model_matcher_lines)
        if const_src:
            constants_source.append(const_src)

    # Extract get_canonical_model_id from model_matcher
    canonical_source = _extract_function("get_canonical_model_id", model_matcher_lines)

    # Extract calculate_match_score from model_matcher
    calculate_match_score_source = _extract_function(
        "calculate_match_score", model_matcher_lines
    )

    # Extract has_semantic_conflict from model_matcher
    has_semantic_conflict_source = _extract_function(
        "has_semantic_conflict", model_matcher_lines
    )

    # Extract get_provider_display_name from model_matcher
    get_provider_display_name_source = _extract_function(
        "get_provider_display_name", model_matcher_lines
    )

    # Extract get_provider_from_model_id from model_matcher
    get_provider_source = _extract_function(
        "get_provider_from_model_id", model_matcher_lines
    )

    # Extract functions from final-aggregator handler
    has_mantle_pricing_source = _extract_function("has_mantle_pricing", lines)
    build_mantle_inference_source = _extract_function("build_mantle_inference", lines)
    build_availability_source = _extract_function("build_availability", lines)
    get_size_category_source = _extract_function("get_size_category", lines)
    build_specs_source = _extract_function("build_specs", lines)
    build_pricing_alias_source = _extract_function("build_pricing_alias", lines)
    create_mantle_only_stub_source = _extract_function("create_mantle_only_stub", lines)
    derive_model_name_source = _extract_function("_derive_model_name_from_id", lines)

    # Extract constants from handler
    mantle_provider_names_source = _extract_constant("MANTLE_PROVIDER_NAMES", lines)
    mantle_model_name_overrides_source = _extract_constant(
        "MANTLE_MODEL_NAME_OVERRIDES", lines
    )
    uppercase_words_source = _extract_constant("_UPPERCASE_WORDS", lines)

    # Validate we found the required functions
    if not has_mantle_pricing_source:
        raise RuntimeError("Could not find has_mantle_pricing")
    if not build_mantle_inference_source:
        raise RuntimeError("Could not find build_mantle_inference")
    if not build_availability_source:
        raise RuntimeError("Could not find build_availability")
    if not canonical_source:
        raise RuntimeError("Could not find get_canonical_model_id")

    # Create a namespace with re module and mock dependencies
    class MockLogger:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    class MockConfigLoader:
        def get_documentation_url(self, key):
            return f"https://docs.aws.amazon.com/{key}"

    def mock_get_config_loader():
        return MockConfigLoader()

    namespace = {
        "re": re,
        "logger": MockLogger(),
        "get_config_loader": mock_get_config_loader,
        "Optional": None,  # Not needed for runtime
    }

    # Execute constants first (from model_matcher)
    for const_src in constants_source:
        exec(const_src, namespace)

    # Execute handler constants
    if mantle_provider_names_source:
        exec(mantle_provider_names_source, namespace)
    if mantle_model_name_overrides_source:
        exec(mantle_model_name_overrides_source, namespace)
    if uppercase_words_source:
        exec(uppercase_words_source, namespace)

    # Execute model_matcher functions (dependencies)
    exec(canonical_source, namespace)
    if calculate_match_score_source:
        exec(calculate_match_score_source, namespace)
    if has_semantic_conflict_source:
        exec(has_semantic_conflict_source, namespace)
    if get_provider_display_name_source:
        exec(get_provider_display_name_source, namespace)
    if get_provider_source:
        exec(get_provider_source, namespace)

    # Execute handler functions
    exec(has_mantle_pricing_source, namespace)
    exec(build_mantle_inference_source, namespace)
    if get_size_category_source:
        exec(get_size_category_source, namespace)
    if build_specs_source:
        exec(build_specs_source, namespace)
    if build_pricing_alias_source:
        exec(build_pricing_alias_source, namespace)
    exec(build_availability_source, namespace)
    if derive_model_name_source:
        exec(derive_model_name_source, namespace)
    if create_mantle_only_stub_source:
        exec(create_mantle_only_stub_source, namespace)

    return namespace


# Load functions once at module level
try:
    _funcs = _load_mantle_functions()
    has_mantle_pricing = _funcs["has_mantle_pricing"]
    build_mantle_inference = _funcs["build_mantle_inference"]
    build_availability = _funcs["build_availability"]
    create_mantle_only_stub = _funcs.get("create_mantle_only_stub")
    _load_error = None
except Exception as e:
    has_mantle_pricing = None
    build_mantle_inference = None
    build_availability = None
    create_mantle_only_stub = None
    _load_error = str(e)


@pytest.mark.skipif(
    has_mantle_pricing is None,
    reason=f"Could not load has_mantle_pricing function: {_load_error}",
)
class TestHasMantlePricing:
    """Tests for has_mantle_pricing function."""

    @pytest.fixture
    def pricing_with_mantle_group(self):
        """Pricing data with Mantle pricing group."""
        return {
            "providers": {
                "Anthropic": {
                    "anthropic.claude-3-sonnet": {
                        "regions": {
                            "us-east-1": {
                                "pricing_groups": {
                                    "Mantle": [
                                        {"dimensions": {"inference_mode": "mantle"}}
                                    ],
                                    "On-Demand": [
                                        {"dimensions": {"inference_mode": "on_demand"}}
                                    ],
                                }
                            }
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def pricing_with_mantle_dimension(self):
        """Pricing data with Mantle in dimensions (inference_mode)."""
        return {
            "providers": {
                "Meta": {
                    "meta.llama3-8b": {
                        "regions": {
                            "us-west-2": {
                                "pricing_groups": {
                                    "On-Demand": [
                                        {"dimensions": {"inference_mode": "mantle"}}
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def pricing_with_mantle_source(self):
        """Pricing data with Mantle in dimensions (source)."""
        return {
            "providers": {
                "DeepSeek": {
                    "deepseek.v3": {
                        "regions": {
                            "eu-west-1": {
                                "pricing_groups": {
                                    "On-Demand": [{"dimensions": {"source": "mantle"}}]
                                }
                            }
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def pricing_without_mantle(self):
        """Pricing data without any Mantle pricing."""
        return {
            "providers": {
                "Anthropic": {
                    "anthropic.claude-3-sonnet": {
                        "regions": {
                            "us-east-1": {
                                "pricing_groups": {
                                    "On-Demand": [
                                        {"dimensions": {"inference_mode": "on_demand"}}
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }

    def test_has_mantle_pricing_true_with_group(self, pricing_with_mantle_group):
        """has_mantle_pricing returns True when Mantle group exists."""
        # Arrange
        model_id = "anthropic.claude-3-sonnet"

        # Act
        result = has_mantle_pricing(model_id, pricing_with_mantle_group)

        # Assert
        assert result is True

    def test_has_mantle_pricing_true_with_inference_mode(
        self, pricing_with_mantle_dimension
    ):
        """has_mantle_pricing returns True when inference_mode is mantle."""
        # Arrange
        model_id = "meta.llama3-8b"

        # Act
        result = has_mantle_pricing(model_id, pricing_with_mantle_dimension)

        # Assert
        assert result is True

    def test_has_mantle_pricing_true_with_source(self, pricing_with_mantle_source):
        """has_mantle_pricing returns True when source is mantle."""
        # Arrange
        model_id = "deepseek.v3"

        # Act
        result = has_mantle_pricing(model_id, pricing_with_mantle_source)

        # Assert
        assert result is True

    def test_has_mantle_pricing_false(self, pricing_without_mantle):
        """has_mantle_pricing returns False for non-Mantle pricing."""
        # Arrange
        model_id = "anthropic.claude-3-sonnet"

        # Act
        result = has_mantle_pricing(model_id, pricing_without_mantle)

        # Assert
        assert result is False

    def test_has_mantle_pricing_no_data(self):
        """has_mantle_pricing handles None gracefully."""
        # Arrange
        model_id = "any.model"

        # Act
        result = has_mantle_pricing(model_id, None)

        # Assert
        assert result is False

    def test_has_mantle_pricing_empty_data(self):
        """has_mantle_pricing handles empty dict gracefully."""
        # Arrange
        model_id = "any.model"

        # Act
        result = has_mantle_pricing(model_id, {})

        # Assert
        assert result is False

    def test_has_mantle_pricing_unknown_model(self, pricing_with_mantle_group):
        """has_mantle_pricing returns False for unknown model."""
        # Arrange
        model_id = "unknown.model"

        # Act
        result = has_mantle_pricing(model_id, pricing_with_mantle_group)

        # Assert
        assert result is False


@pytest.mark.skipif(
    build_mantle_inference is None,
    reason=f"Could not load build_mantle_inference function: {_load_error}",
)
class TestBuildMantleInference:
    """Tests for build_mantle_inference with has_pricing flag."""

    @pytest.fixture
    def mantle_by_model(self):
        """Sample Mantle model data."""
        return {
            "anthropic.claude-3-sonnet": {
                "regions": ["us-east-1", "us-west-2"],
                "supports_responses_api": True,
            },
            "meta.llama3-8b": {
                "regions": ["us-east-1"],
                "supports_responses_api": False,
            },
        }

    @pytest.fixture
    def pricing_with_mantle(self):
        """Pricing data with Mantle pricing."""
        return {
            "providers": {
                "Anthropic": {
                    "anthropic.claude-3-sonnet": {
                        "regions": {
                            "us-east-1": {
                                "pricing_groups": {
                                    "Mantle": [
                                        {"dimensions": {"inference_mode": "mantle"}}
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def pricing_without_mantle(self):
        """Pricing data without Mantle pricing."""
        return {
            "providers": {
                "Meta": {
                    "meta.llama3-8b": {
                        "regions": {
                            "us-east-1": {
                                "pricing_groups": {
                                    "On-Demand": [
                                        {"dimensions": {"inference_mode": "on_demand"}}
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }

    def test_build_mantle_inference_with_pricing(
        self, mantle_by_model, pricing_with_mantle
    ):
        """build_mantle_inference includes has_pricing: True when Mantle pricing exists."""
        # Arrange
        model_id = "anthropic.claude-3-sonnet"

        # Act
        result = build_mantle_inference(model_id, mantle_by_model, pricing_with_mantle)

        # Assert
        assert result["supported"] is True
        assert result["has_pricing"] is True
        assert result["mantle_regions"] == ["us-east-1", "us-west-2"]
        assert result["supports_responses_api"] is True

    def test_build_mantle_inference_without_pricing(
        self, mantle_by_model, pricing_without_mantle
    ):
        """build_mantle_inference sets has_pricing: False when no Mantle pricing."""
        # Arrange
        model_id = "meta.llama3-8b"

        # Act
        result = build_mantle_inference(
            model_id, mantle_by_model, pricing_without_mantle
        )

        # Assert
        assert result["supported"] is True
        assert result["has_pricing"] is False
        assert result["mantle_regions"] == ["us-east-1"]
        assert result["supports_responses_api"] is False

    def test_build_mantle_inference_no_pricing_data(self, mantle_by_model):
        """build_mantle_inference sets has_pricing: False when pricing_data is None."""
        # Arrange
        model_id = "anthropic.claude-3-sonnet"

        # Act
        result = build_mantle_inference(model_id, mantle_by_model, None)

        # Assert
        assert result["supported"] is True
        assert result["has_pricing"] is False
        assert result["mantle_regions"] == ["us-east-1", "us-west-2"]

    def test_build_mantle_inference_no_mantle_support(self):
        """build_mantle_inference returns unsupported when model not in Mantle."""
        # Arrange
        model_id = "unknown.model"

        # Act
        result = build_mantle_inference(model_id, {}, None)

        # Assert
        assert result["supported"] is False
        assert result["has_pricing"] is False
        assert result["mantle_regions"] == []
        assert result["supports_responses_api"] is False


@pytest.mark.skipif(
    build_availability is None,
    reason=f"Could not load build_availability function: {_load_error}",
)
class TestBuildAvailabilityMantleHasPricing:
    """Tests for has_pricing in availability.mantle structure."""

    @pytest.fixture
    def mantle_data_with_pricing(self):
        """Mantle data with has_pricing: True."""
        return {
            "supported": True,
            "mantle_regions": ["us-east-1", "us-west-2"],
            "supports_responses_api": True,
            "has_pricing": True,
        }

    @pytest.fixture
    def mantle_data_without_pricing(self):
        """Mantle data with has_pricing: False."""
        return {
            "supported": True,
            "mantle_regions": ["us-east-1"],
            "supports_responses_api": False,
            "has_pricing": False,
        }

    def test_availability_mantle_has_pricing_true(self, mantle_data_with_pricing):
        """availability.mantle includes has_pricing: True."""
        # Arrange
        regional_availability = ["us-east-1", "us-west-2"]
        cross_region_data = {"supported": False, "source_regions": [], "profiles": []}
        batch_inference_data = {"supported": False, "supported_regions": []}
        provisioned_data = {"supported": False, "provisioned_regions": []}

        # Act
        result = build_availability(
            regional_availability=regional_availability,
            cross_region_data=cross_region_data,
            batch_inference_data=batch_inference_data,
            provisioned_data=provisioned_data,
            mantle_data=mantle_data_with_pricing,
            is_mantle_only=False,
        )

        # Assert
        assert result["mantle"]["supported"] is True
        assert result["mantle"]["has_pricing"] is True
        assert result["mantle"]["regions"] == ["us-east-1", "us-west-2"]
        assert result["mantle"]["responses_api"] is True

    def test_availability_mantle_has_pricing_false(self, mantle_data_without_pricing):
        """availability.mantle includes has_pricing: False."""
        # Arrange
        regional_availability = ["us-east-1"]
        cross_region_data = {"supported": False, "source_regions": [], "profiles": []}
        batch_inference_data = {"supported": False, "supported_regions": []}
        provisioned_data = {"supported": False, "provisioned_regions": []}

        # Act
        result = build_availability(
            regional_availability=regional_availability,
            cross_region_data=cross_region_data,
            batch_inference_data=batch_inference_data,
            provisioned_data=provisioned_data,
            mantle_data=mantle_data_without_pricing,
            is_mantle_only=False,
        )

        # Assert
        assert result["mantle"]["supported"] is True
        assert result["mantle"]["has_pricing"] is False
        assert result["mantle"]["regions"] == ["us-east-1"]
        assert result["mantle"]["responses_api"] is False

    def test_availability_mantle_only_model(self, mantle_data_with_pricing):
        """availability.mantle.only is True for Mantle-only models."""
        # Arrange
        regional_availability = []  # No on-demand regions
        cross_region_data = {"supported": False, "source_regions": [], "profiles": []}
        batch_inference_data = {"supported": False, "supported_regions": []}
        provisioned_data = {"supported": False, "provisioned_regions": []}

        # Act
        result = build_availability(
            regional_availability=regional_availability,
            cross_region_data=cross_region_data,
            batch_inference_data=batch_inference_data,
            provisioned_data=provisioned_data,
            mantle_data=mantle_data_with_pricing,
            is_mantle_only=True,
        )

        # Assert
        assert result["mantle"]["only"] is True
        assert result["mantle"]["has_pricing"] is True
        assert result["on_demand"]["supported"] is False
        assert result["on_demand"]["regions"] == []


@pytest.mark.skipif(
    create_mantle_only_stub is None,
    reason=f"Could not load create_mantle_only_stub function: {_load_error}",
)
class TestMantleOnlyStubHasPricing:
    """Tests for has_pricing in Mantle-only stub models."""

    def test_create_mantle_only_stub_default_has_pricing(self):
        """create_mantle_only_stub sets has_pricing: False by default."""
        # Arrange
        mantle_id = "zai.glm-4.6"
        regions = ["us-east-1", "us-west-2"]
        collection_timestamp = "2024-01-01T00:00:00.000000+00:00"

        # Act
        result = create_mantle_only_stub(
            mantle_id, regions, collection_timestamp, supports_responses_api=False
        )

        # Assert
        assert result["availability"]["mantle"]["has_pricing"] is False
        assert result["availability"]["mantle"]["supported"] is True
        assert result["availability"]["mantle"]["only"] is True
        assert result["has_pricing"] is False
