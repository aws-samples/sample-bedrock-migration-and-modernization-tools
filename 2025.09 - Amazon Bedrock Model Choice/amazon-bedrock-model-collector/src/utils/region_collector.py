"""
Bedrock Region Collector
Discovers all AWS regions where Amazon Bedrock is available
"""

import boto3
import logging
from typing import List, Optional
from botocore.exceptions import ClientError, NoCredentialsError

from config import Config


class BedrockRegionCollector:
    """Collector for discovering Bedrock-enabled AWS regions"""

    def __init__(self, profile_name: Optional[str] = None, default_region: str = 'us-east-1'):
        """
        Initialize the region collector

        Args:
            profile_name: AWS profile name to use
            default_region: Default AWS region for initial client setup
        """
        self.profile_name = profile_name
        self.default_region = default_region
        self.session = None
        self.bedrock_regions = []

        self.logger = logging.getLogger(__name__)
        self._initialize_session()

    def _initialize_session(self):
        """Initialize AWS session with credentials"""
        try:
            if self.profile_name:
                self.session = boto3.Session(
                    profile_name=self.profile_name,
                    region_name=self.default_region
                )
            else:
                self.session = boto3.Session(region_name=self.default_region)

            # Test session by making a simple call
            ec2 = self.session.client('ec2', region_name=self.default_region)
            ec2.describe_regions()

            self.logger.info(f"✅ AWS session initialized successfully with profile: {self.profile_name or 'default'}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AWS session: {e}")
            raise

    def discover_bedrock_regions(self) -> List[str]:
        """
        Discover all regions where Bedrock is available

        Returns:
            List of region names where Bedrock is available
        """
        self.logger.info("Starting Bedrock region discovery...")

        # Get all AWS regions first
        all_regions = self._get_all_aws_regions()
        if not all_regions:
            self.logger.warning("Could not retrieve AWS regions, using known Bedrock regions")
            return self._verify_known_regions()

        self.logger.info(f"Testing Bedrock availability across {len(all_regions)} AWS regions...")

        bedrock_regions = []

        # Test each region for Bedrock availability
        for i, region in enumerate(all_regions):
            self.logger.debug(f"[{i+1}/{len(all_regions)}] Testing region: {region}")

            try:
                bedrock = self.session.client('bedrock', region_name=region)
                # Try to list foundation models - if this works, Bedrock is available
                response = bedrock.list_foundation_models()

                # If we get here without exception, Bedrock is available
                bedrock_regions.append(region)
                self.logger.info(f"✅ Bedrock available in: {region}")

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = str(e).lower()

                # Check for specific "not available" indicators
                if any(term in error_message for term in [
                    'not supported', 'not available', 'invalid region',
                    'unsupported region', 'region not supported', 'service not available'
                ]) or error_code in ['InvalidRegion', 'UnsupportedRegion', 'OptInRequired']:
                    self.logger.debug(f"✗ Bedrock not available in: {region}")
                else:
                    self.logger.warning(f"⚠️  Unexpected error in {region}: {error_code} - {e}")

            except Exception as e:
                self.logger.debug(f"✗ Error testing {region}: {e}")

        # If no regions found through testing, use known regions
        if not bedrock_regions:
            self.logger.warning("No regions detected through API testing, using known Bedrock regions")
            bedrock_regions = self._verify_known_regions()

        self.bedrock_regions = sorted(bedrock_regions)
        self.logger.info(f"🎯 Bedrock region discovery complete: {len(self.bedrock_regions)} regions")

        for region in self.bedrock_regions:
            self.logger.info(f"   - {region}")

        return self.bedrock_regions

    def _get_all_aws_regions(self) -> List[str]:
        """
        Get list of all AWS regions

        Returns:
            List of all AWS region names
        """
        try:
            ec2 = self.session.client('ec2', region_name=self.default_region)
            response = ec2.describe_regions()
            regions = [region['RegionName'] for region in response['Regions']]
            self.logger.info(f"Retrieved {len(regions)} total AWS regions")
            return sorted(regions)

        except Exception as e:
            self.logger.error(f"Failed to get AWS regions: {e}")
            return []

    def _verify_known_regions(self) -> List[str]:
        """
        Verify known Bedrock regions as fallback

        Returns:
            List of verified Bedrock regions
        """
        self.logger.info("Verifying known Bedrock regions as fallback...")

        verified_regions = []

        for region in Config.KNOWN_BEDROCK_REGIONS:
            try:
                bedrock = self.session.client('bedrock', region_name=region)
                bedrock.list_foundation_models()
                verified_regions.append(region)
                self.logger.info(f"✅ Verified Bedrock in: {region}")

            except Exception as e:
                self.logger.debug(f"✗ Could not verify Bedrock in: {region}")

        if not verified_regions:
            # Absolute fallback - use a basic set of known working regions
            verified_regions = ['us-east-1', 'us-west-2', 'eu-west-1']
            self.logger.warning(f"Using absolute fallback regions: {verified_regions}")

        return verified_regions

    def get_region_statistics(self) -> dict:
        """
        Get statistics about region discovery

        Returns:
            Dictionary with region discovery statistics
        """
        return {
            'total_regions_discovered': len(self.bedrock_regions),
            'regions': self.bedrock_regions,
            'discovery_method': 'api_testing' if self.bedrock_regions else 'known_regions_fallback'
        }