"""Tests for SAM template structure."""

import yaml
import pytest
from pathlib import Path

SAM_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent / "infra" / "backend-template.yaml"
)


class CloudFormationLoader(yaml.SafeLoader):
    """Custom YAML loader that handles CloudFormation intrinsic functions."""

    pass


# Register CloudFormation intrinsic functions as multi-constructors
def _cfn_intrinsic_constructor(loader, tag_suffix, node):
    """Handle CloudFormation intrinsic functions like !Ref, !Sub, !If, etc."""
    if isinstance(node, yaml.ScalarNode):
        return {tag_suffix: loader.construct_scalar(node)}
    elif isinstance(node, yaml.SequenceNode):
        return {tag_suffix: loader.construct_sequence(node)}
    elif isinstance(node, yaml.MappingNode):
        return {tag_suffix: loader.construct_mapping(node)}
    return {tag_suffix: None}


# Add multi-constructor for all CloudFormation tags
CloudFormationLoader.add_multi_constructor("!", _cfn_intrinsic_constructor)


def load_template():
    """Load and parse the SAM template with CloudFormation intrinsic function support."""
    with open(SAM_TEMPLATE_PATH) as f:
        return yaml.load(f, Loader=CloudFormationLoader)


def test_sam_template_valid_yaml():
    """Template should be valid YAML."""
    template = load_template()
    assert template is not None
    assert "AWSTemplateFormatVersion" in template


def test_sam_template_has_powertools_layer():
    """Globals should include Powertools layer."""
    template = load_template()
    layers = template.get("Globals", {}).get("Function", {}).get("Layers", [])
    layer_strs = [str(layer) for layer in layers]
    assert any("AWSLambdaPowertoolsPythonV3" in layer for layer in layer_strs)


def test_sam_template_has_log_level_param():
    """Template should have LogLevel parameter."""
    template = load_template()
    params = template.get("Parameters", {})
    assert "LogLevel" in params


def test_sam_template_globals_env_vars():
    """Globals should have Powertools environment variables."""
    template = load_template()
    env_vars = (
        template.get("Globals", {})
        .get("Function", {})
        .get("Environment", {})
        .get("Variables", {})
    )
    assert "POWERTOOLS_SERVICE_NAME" in env_vars
