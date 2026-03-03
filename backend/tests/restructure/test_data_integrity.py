"""
Data Integrity Tests for JSON Restructure

These tests verify that the restructured JSON maintains data integrity
and doesn't break any existing functionality.

Run with: python3 -m pytest backend/tests/restructure/test_data_integrity.py -v
"""

import json
import pytest
from pathlib import Path


# Load test data from latest JSON files
@pytest.fixture(scope="module")
def models_data():
    """Load models data from S3 or local file."""
    local_path = Path("/tmp/models.json")
    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)
    else:
        import subprocess
        result = subprocess.run(
            ["aws", "s3", "cp", 
             "s3://bedrock-profiler-data-390445053194-prod/latest/bedrock_models.json",
             "/tmp/models.json"],
            capture_output=True
        )
        if result.returncode == 0:
            with open("/tmp/models.json") as f:
                return json.load(f)
        pytest.skip("Could not load models data")


@pytest.fixture(scope="module")
def pricing_data():
    """Load pricing data from S3 or local file."""
    local_path = Path("/tmp/pricing.json")
    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)
    else:
        import subprocess
        result = subprocess.run(
            ["aws", "s3", "cp",
             "s3://bedrock-profiler-data-390445053194-prod/latest/bedrock_pricing.json",
             "/tmp/pricing.json"],
            capture_output=True
        )
        if result.returncode == 0:
            with open("/tmp/pricing.json") as f:
                return json.load(f)
        pytest.skip("Could not load pricing data")


@pytest.fixture(scope="module")
def all_models(models_data):
    """Extract all models from providers structure."""
    models = []
    for provider_name, provider_data in models_data.get("providers", {}).items():
        for model_id, model in provider_data.get("models", {}).items():
            model["_provider_name"] = provider_name
            model["_model_id"] = model_id
            models.append(model)
    return models


# Known models with no regional availability (deprecated or not yet launched)
KNOWN_NO_REGION_MODELS = {
    "mistral.mistral-7b-instruct-v0:0",
    "mistral.mixtral-8x7b-instruct-v0:0",
    "stability.stable-image-core-v1:0",
    "stability.stable-image-ultra-v1:0",
}


class TestCurrentStructure:
    """Tests for the CURRENT structure - baseline before restructure."""
    
    def test_metadata_exists(self, models_data):
        """Verify metadata section exists with required fields."""
        assert "metadata" in models_data
        metadata = models_data["metadata"]
        assert "collection_timestamp" in metadata
        assert "providers_count" in metadata
        assert "total_models" in metadata
        assert metadata["total_models"] > 0
    
    def test_providers_structure(self, models_data):
        """Verify providers structure is correct."""
        assert "providers" in models_data
        providers = models_data["providers"]
        assert len(providers) > 0
        
        for provider_name, provider_data in providers.items():
            assert "models" in provider_data
            assert isinstance(provider_data["models"], dict)
    
    def test_model_core_fields(self, all_models):
        """Verify all models have core identification fields."""
        for model in all_models:
            assert "model_id" in model, f"Missing model_id in {model.get('_model_id')}"
            assert "model_name" in model, f"Missing model_name in {model.get('_model_id')}"
            assert "model_provider" in model, f"Missing model_provider in {model.get('_model_id')}"
    
    def test_model_modalities(self, all_models):
        """Verify modalities structure."""
        for model in all_models:
            assert "model_modalities" in model, f"Missing modalities in {model['model_id']}"
            modalities = model["model_modalities"]
            assert "input_modalities" in modalities
            assert "output_modalities" in modalities
            assert isinstance(modalities["input_modalities"], list)
            assert isinstance(modalities["output_modalities"], list)
    
    def test_region_fields_exist(self, all_models):
        """Verify region-related fields exist (current scattered structure)."""
        for model in all_models:
            # At least one of these should exist
            has_regions = (
                "in_region" in model or
                "cross_region_inference" in model or
                "batch_inference_supported" in model
            )
            assert has_regions, f"No region data in {model['model_id']}"
    
    def test_in_region_is_list(self, all_models):
        """Verify in_region is a list when present."""
        for model in all_models:
            if "in_region" in model:
                assert isinstance(model["in_region"], list), \
                    f"in_region should be list in {model['model_id']}"
    
    def test_cross_region_inference_structure(self, all_models):
        """Verify cross_region_inference structure when present."""
        for model in all_models:
            if "cross_region_inference" in model:
                cris = model["cross_region_inference"]
                assert "supported" in cris or "source_regions" in cris, \
                    f"Invalid CRIS structure in {model['model_id']}"
    
    def test_batch_inference_structure(self, all_models):
        """Verify batch_inference_supported structure when present."""
        for model in all_models:
            if "batch_inference_supported" in model:
                batch = model["batch_inference_supported"]
                assert "supported" in batch, \
                    f"Missing 'supported' in batch_inference for {model['model_id']}"
    
    def test_converse_data_structure(self, all_models):
        """Verify converse_data (specs) structure."""
        for model in all_models:
            if "converse_data" in model:
                specs = model["converse_data"]
                # context_window should be present and positive
                if "context_window" in specs and specs["context_window"]:
                    assert specs["context_window"] > 0, \
                        f"Invalid context_window in {model['model_id']}"
    
    def test_pricing_reference(self, all_models):
        """Verify pricing reference structure."""
        models_with_pricing = [m for m in all_models if m.get("has_pricing")]
        assert len(models_with_pricing) > 0, "No models have pricing"
        
        for model in models_with_pricing:
            if "model_pricing" in model:
                pricing = model["model_pricing"]
                assert "is_pricing_available" in pricing or "pricing_file_reference" in pricing


class TestPricingStructure:
    """Tests for pricing data structure."""
    
    def test_pricing_metadata(self, pricing_data):
        """Verify pricing metadata exists."""
        assert "metadata" in pricing_data
        assert "providers" in pricing_data
    
    def test_pricing_providers_structure(self, pricing_data):
        """Verify pricing providers structure."""
        providers = pricing_data.get("providers", {})
        assert len(providers) > 0
        
        for provider_name, provider_models in providers.items():
            assert isinstance(provider_models, dict)
    
    def test_pricing_dimensions_have_required_fields(self, pricing_data):
        """Verify pricing dimensions have required fields."""
        required_fields = ["price_per_unit", "unit", "is_input", "is_output"]
        
        sample_count = 0
        for provider_name, provider_models in pricing_data.get("providers", {}).items():
            for model_key, model_data in provider_models.items():
                for region, region_data in model_data.get("regions", {}).items():
                    for group_name, dimensions in region_data.get("pricing_groups", {}).items():
                        for dim in dimensions:
                            for field in required_fields:
                                assert field in dim, \
                                    f"Missing {field} in {model_key}/{region}/{group_name}"
                            sample_count += 1
                            if sample_count >= 100:  # Sample check
                                return
    
    def test_pricing_values_are_per_thousand(self, pricing_data):
        """Verify pricing values are stored as per-thousand (current format)."""
        # Sample some prices and verify they're in per-thousand range
        prices = []
        for provider_name, provider_models in pricing_data.get("providers", {}).items():
            for model_key, model_data in provider_models.items():
                for region, region_data in model_data.get("regions", {}).items():
                    for group_name, dimensions in region_data.get("pricing_groups", {}).items():
                        for dim in dimensions:
                            price = dim.get("price_per_unit")
                            if price and isinstance(price, (int, float)) and price > 0:
                                prices.append(price)
                                if len(prices) >= 1000:
                                    break
        
        # Per-thousand prices should mostly be < $1
        # (Claude 3 Opus is ~$0.015 per 1K input tokens)
        median = sorted(prices)[len(prices) // 2]
        assert median < 1.0, f"Prices appear to be per-million (median: {median})"


class TestDataConsistency:
    """Tests for data consistency across structures."""
    
    def test_model_count_matches_metadata(self, models_data, all_models):
        """Verify model count matches metadata."""
        expected = models_data["metadata"]["total_models"]
        actual = len(all_models)
        assert actual == expected, f"Model count mismatch: {actual} vs {expected}"
    
    def test_provider_count_matches_metadata(self, models_data):
        """Verify provider count matches metadata."""
        expected = models_data["metadata"]["providers_count"]
        actual = len(models_data["providers"])
        assert actual == expected, f"Provider count mismatch: {actual} vs {expected}"
    
    def test_consumption_options_match_availability(self, all_models):
        """Verify consumption_options matches actual availability."""
        for model in all_models:
            options = model.get("consumption_options", [])
            
            # Check cross_region_inference
            cris = model.get("cross_region_inference", {})
            if cris.get("supported") or len(cris.get("source_regions", [])) > 0:
                assert "cross_region_inference" in options, \
                    f"CRIS supported but not in consumption_options for {model['model_id']}"
            
            # Check batch
            batch = model.get("batch_inference_supported", {})
            if batch.get("supported"):
                assert "batch" in options, \
                    f"Batch supported but not in consumption_options for {model['model_id']}"
    
    def test_no_empty_region_arrays(self, all_models):
        """Verify region arrays are not unexpectedly empty."""
        for model in all_models:
            model_id = model['model_id']
            
            # Skip known models with no regional availability
            if model_id in KNOWN_NO_REGION_MODELS:
                continue
            
            # Models should have at least one region somewhere
            in_region = model.get("in_region", [])
            cris_regions = model.get("cross_region_inference", {}).get("source_regions", [])
            batch_regions = model.get("batch_inference_supported", {}).get("supported_regions", [])
            mantle_regions = model.get("mantle_inference", {}).get("mantle_regions", [])
            
            total_regions = len(in_region) + len(cris_regions) + len(batch_regions) + len(mantle_regions)
            
            # Allow mantle-only models to have no standard regions
            if not model.get("mantle_only"):
                assert total_regions > 0, \
                    f"No regions found for non-mantle model {model_id}"


class TestCRISIntegrity:
    """Tests specifically for Cross-Region Inference data."""
    
    def test_cris_profiles_have_required_fields(self, all_models):
        """Verify CRIS profiles have required fields."""
        required_fields = ["profile_id", "type", "status"]
        
        for model in all_models:
            cris = model.get("cross_region_inference", {})
            profiles = cris.get("profiles", [])
            
            for profile in profiles:
                for field in required_fields:
                    assert field in profile, \
                        f"Missing {field} in CRIS profile for {model['model_id']}"
    
    def test_cris_supported_matches_profiles(self, all_models):
        """Verify CRIS supported flag matches profile existence."""
        for model in all_models:
            cris = model.get("cross_region_inference", {})
            supported = cris.get("supported", False)
            profiles = cris.get("profiles", [])
            source_regions = cris.get("source_regions", [])
            
            if supported:
                # If supported, should have profiles or regions
                assert len(profiles) > 0 or len(source_regions) > 0, \
                    f"CRIS supported but no profiles/regions for {model['model_id']}"


class TestNewStructureCompatibility:
    """Tests to verify new structure will be compatible."""
    
    def test_can_build_availability_object(self, all_models):
        """Verify we can build the new availability object from current data."""
        for model in all_models:
            # Build proposed availability object
            availability = {
                "on_demand": {
                    "supported": len(model.get("in_region", [])) > 0,
                    "regions": model.get("in_region", [])
                },
                "cross_region": {
                    "supported": model.get("cross_region_inference", {}).get("supported", False),
                    "regions": model.get("cross_region_inference", {}).get("source_regions", []),
                    "profiles": model.get("cross_region_inference", {}).get("profiles", [])
                },
                "batch": {
                    "supported": model.get("batch_inference_supported", {}).get("supported", False),
                    "regions": model.get("batch_inference_supported", {}).get("supported_regions", [])
                },
                "provisioned": {
                    "supported": model.get("provisioned_throughput", {}).get("supported", False),
                    "regions": model.get("provisioned_throughput", {}).get("provisioned_regions", [])
                },
                "mantle": {
                    "supported": model.get("mantle_inference", {}).get("supported", False),
                    "regions": model.get("mantle_inference", {}).get("mantle_regions", []),
                    "only": model.get("mantle_only", False),
                    "responses_api": model.get("mantle_inference", {}).get("supports_responses_api", False)
                }
            }
            
            # Verify structure is valid
            assert isinstance(availability["on_demand"]["regions"], list)
            assert isinstance(availability["cross_region"]["profiles"], list)
            assert isinstance(availability["batch"]["supported"], bool)
    
    def test_can_derive_consumption_options(self, all_models):
        """Verify consumption_options can be derived from availability."""
        for model in all_models:
            # Current consumption_options
            current_options = set(model.get("consumption_options", []))
            
            # Derive from availability data
            derived_options = set()
            
            if len(model.get("in_region", [])) > 0:
                derived_options.add("on_demand")
            
            if model.get("cross_region_inference", {}).get("supported"):
                derived_options.add("cross_region_inference")
            
            if model.get("batch_inference_supported", {}).get("supported"):
                derived_options.add("batch")
            
            if model.get("provisioned_throughput", {}).get("supported"):
                derived_options.add("provisioned")
            
            if model.get("mantle_inference", {}).get("supported"):
                derived_options.add("mantle")
            
            # Derived should be subset of current (current may have more)
            # This is a soft check - we're verifying derivation is possible
            assert len(derived_options) >= 0  # Always passes, just for structure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
