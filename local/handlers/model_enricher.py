"""
Model Enricher - Local Handler

Enriches models with capabilities, use cases, and documentation links.
Derives this information from model metadata (modalities, provider, model name).

This is a standalone version of backend/lambdas/model-enricher/handler.py
"""


def extract_capabilities(model_data: dict) -> list:
    """Extract model capabilities from modalities and model info."""
    capabilities = set()

    modalities = model_data.get('model_modalities', {})
    input_modalities = modalities.get('input_modalities', [])
    output_modalities = modalities.get('output_modalities', [])
    model_name = model_data.get('model_name', '').lower()
    model_id = model_data.get('model_id', '').lower()

    # Text capabilities
    if 'TEXT' in input_modalities and 'TEXT' in output_modalities:
        capabilities.update(['text_generation', 'chat'])

    # Vision capabilities
    if 'IMAGE' in input_modalities:
        capabilities.update(['vision', 'image_understanding', 'multimodal'])

    # Video capabilities
    if 'VIDEO' in input_modalities:
        capabilities.update(['video_understanding', 'multimodal'])

    # Image generation
    if 'IMAGE' in output_modalities:
        capabilities.update(['image_generation', 'text_to_image'])

    # Video generation
    if 'VIDEO' in output_modalities:
        capabilities.update(['video_generation'])

    # Embedding capabilities
    if 'EMBEDDING' in output_modalities or 'embed' in model_name:
        capabilities.update(['embeddings', 'semantic_search'])
        if 'image' in model_name:
            capabilities.add('image_embeddings')
        else:
            capabilities.add('text_embeddings')

    # Code capabilities
    if any(term in model_id for term in ['code', 'codestral', 'devstral']):
        capabilities.add('code_generation')

    # Reasoning capability
    if any(term in model_id for term in ['claude', 'sonnet', 'opus', 'nova', 'llama', 'mistral', 'command']):
        capabilities.add('reasoning')

    # Function calling
    if any(term in model_id for term in ['claude', 'nova', 'llama-3', 'mistral', 'command-r']):
        capabilities.add('function_calling')

    # Streaming support
    if model_data.get('streaming_supported', False):
        capabilities.add('streaming')

    return sorted(list(capabilities))


def extract_use_cases(capabilities: list, model_data: dict) -> list:
    """Extract use cases based on capabilities."""
    use_cases = set()

    capability_to_use_cases = {
        'chat': ['conversational_ai', 'customer_support', 'virtual_assistants'],
        'text_generation': ['content_creation', 'creative_writing', 'summarization', 'translation'],
        'reasoning': ['complex_analysis', 'problem_solving', 'decision_support'],
        'code_generation': ['software_development', 'code_review', 'debugging', 'documentation'],
        'vision': ['image_analysis', 'visual_qa', 'document_analysis', 'ocr'],
        'image_understanding': ['content_moderation', 'product_cataloging'],
        'video_understanding': ['video_analysis', 'content_moderation'],
        'image_generation': ['creative_design', 'marketing_assets', 'prototyping'],
        'video_generation': ['video_production', 'animation'],
        'embeddings': ['semantic_search', 'recommendation_systems', 'clustering'],
        'text_embeddings': ['document_retrieval', 'similarity_matching'],
        'image_embeddings': ['image_search', 'visual_similarity'],
        'function_calling': ['api_integration', 'workflow_automation', 'tool_use'],
    }

    for capability in capabilities:
        if capability in capability_to_use_cases:
            use_cases.update(capability_to_use_cases[capability])

    return sorted(list(use_cases))


def get_documentation_links(model_data: dict) -> dict:
    """Get documentation links based on provider and model."""
    provider = model_data.get('model_provider', '')
    model_id = model_data.get('model_id', '').lower()

    # Default Bedrock documentation
    default_links = {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html',
        'model_documentation': 'https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    }

    # Provider-specific documentation
    provider_docs = {
        'Anthropic': {
            'provider_documentation': 'https://docs.anthropic.com/',
            'api_reference': 'https://docs.anthropic.com/claude/reference/'
        },
        'Amazon': {
            'provider_documentation': 'https://docs.aws.amazon.com/bedrock/latest/userguide/titan-models.html'
        },
        'Meta': {
            'provider_documentation': 'https://llama.meta.com/docs/'
        },
        'Mistral AI': {
            'provider_documentation': 'https://docs.mistral.ai/'
        },
        'Cohere': {
            'provider_documentation': 'https://docs.cohere.com/'
        },
        'AI21 Labs': {
            'provider_documentation': 'https://docs.ai21.com/'
        },
        'Stability AI': {
            'provider_documentation': 'https://platform.stability.ai/docs/'
        }
    }

    links = default_links.copy()

    # Check for Nova models
    if 'nova' in model_id:
        links['provider_documentation'] = 'https://docs.aws.amazon.com/bedrock/latest/userguide/nova.html'
        return links

    if provider in provider_docs:
        links.update(provider_docs[provider])

    return links


def enrich_model(model_data: dict) -> dict:
    """Enrich a single model with capabilities, use cases, and documentation."""
    enriched = model_data.copy()

    capabilities = extract_capabilities(model_data)
    enriched['model_capabilities'] = capabilities

    use_cases = extract_use_cases(capabilities, model_data)
    enriched['model_use_cases'] = use_cases

    enriched['documentation_links'] = get_documentation_links(model_data)

    return enriched


def enrich_providers(providers: dict) -> dict:
    """Enrich all models in the providers structure."""
    enriched_providers = {}

    for provider_name, provider_data in providers.items():
        enriched_providers[provider_name] = {'models': {}}

        for model_id, model_data in provider_data.get('models', {}).items():
            enriched_providers[provider_name]['models'][model_id] = enrich_model(model_data)

    return enriched_providers
