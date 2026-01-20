"""
Direct API Collector for Amazon Bedrock
Uses direct REST API calls to /foundation-models endpoint
"""

import boto3
import requests
import json
import logging
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from typing import Dict, List, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

class DirectBedrockAPICollector:
    """Direct API collector using /foundation-models endpoint"""

    def __init__(self, profile_name: str = None):
        self.profile_name = profile_name or Config.AWS_PROFILE_NAME
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://bedrock.us-east-1.amazonaws.com"

    def collect_models(self, provider_filter: str = None) -> Dict[str, Any]:
        """Collect models using direct API approach"""
        self.logger.info(f"🚀 Using direct API collector for Bedrock models")

        try:
            # Get AWS credentials
            session = boto3.Session(profile_name=self.profile_name)
            credentials = session.get_credentials()

            # Build endpoint URL with optional provider filter
            endpoint = "/foundation-models"
            if provider_filter:
                endpoint += f"?byProvider={provider_filter}"

            url = f"{self.base_url}{endpoint}"
            self.logger.info(f"📡 Calling: {url}")

            # Create and sign request
            request = AWSRequest(method='GET', url=url)
            SigV4Auth(credentials, 'bedrock', 'us-east-1').add_auth(request)
            prepared_request = request.prepare()

            # Make request
            response = requests.get(
                prepared_request.url,
                headers=dict(prepared_request.headers),
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"API returned {response.status_code}: {response.text}")

            data = response.json()
            models = data.get('modelSummaries', [])

            self.logger.info(f"✅ Retrieved {len(models)} models via direct API")

            return self._process_models(models)

        except Exception as e:
            self.logger.error(f"❌ Direct API collection failed: {e}")
            raise

    def _process_models(self, models: List[Dict]) -> Dict[str, Any]:
        """Process models from direct API response"""
        processed_models = {}

        for model in models:
            model_id = model.get('modelId', '')
            if not model_id:
                continue

            processed_models[model_id] = {
                'model_id': model_id,
                'model_name': model.get('modelName', ''),
                'model_provider': model.get('providerName', ''),
                'model_family': model.get('modelFamily', ''),
                'description': model.get('description', {}),
                'input_modalities': model.get('inputModalities', []),
                'output_modalities': model.get('outputModalities', []),
                'inference_types_supported': model.get('inferenceTypesSupported', []),
                'response_streaming_supported': model.get('responseStreamingSupported', False),
                'customizations_supported': model.get('customizationsSupported', []),
                'guardrails_supported': model.get('guardrailsSupported', False),
                'model_lifecycle': model.get('modelLifecycle', {}),
                'explicit_prompt_caching': model.get('explicitPromptCaching', {}),
                'features_supported': model.get('featuresSupported', {}),
                'converse': model.get('converse', {}),
                'intelligent_prompt_routing': model.get('intelligentPromptRouting', {}),
                'latency_optimization_supported': model.get('latencyOptimizationSupported', False)
            }

        return processed_models

    def get_providers(self) -> List[str]:
        """Get list of available providers"""
        try:
            all_models = self.collect_models()
            providers = set()

            for model_data in all_models.values():
                provider = model_data.get('model_provider')
                if provider:
                    providers.add(provider)

            return sorted(list(providers))

        except Exception as e:
            self.logger.error(f"Failed to get providers: {e}")
            return []