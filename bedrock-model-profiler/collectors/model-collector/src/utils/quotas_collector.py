"""
Service Quotas Collector - CLEAN VERSION
Only collects quotas, no filtering. All filtering is done in data_processor.py
"""

import boto3
import logging
import time
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed


class ServiceQuotasCollector:
    """Clean collector for Bedrock service quotas - collection ONLY, no filtering"""

    def __init__(self, profile_name: Optional[str] = None, regions: List[str] = None,
                 use_parallel: bool = True, max_workers: int = 10):
        self.profile_name = profile_name
        self.regions = regions or []
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.session = None
        self.logger = logging.getLogger(__name__)
        self._initialize_session()

    def _initialize_session(self):
        """Initialize AWS session"""
        try:
            if self.profile_name:
                self.session = boto3.Session(profile_name=self.profile_name)
            else:
                self.session = boto3.Session()
            self.logger.info("✅ Service quotas collector initialized (clean version)")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize quotas collector: {e}")
            raise

    def collect_service_quotas(self) -> Dict[str, Any]:
        """
        Collect ALL service quotas from all regions - NO FILTERING
        Returns: Raw quota data for data_processor.py to filter
        """
        if self.use_parallel:
            return self._collect_service_quotas_parallel()
        else:
            return self._collect_service_quotas_sequential()

    def _collect_service_quotas_parallel(self) -> Dict[str, Any]:
        """Collect service quotas from all regions IN PARALLEL"""
        self.logger.info(f"🚀 Collecting ALL Bedrock service quotas from {len(self.regions)} regions (parallel, workers={self.max_workers})...")

        all_quotas = {}
        start_time = time.time()

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all region tasks
            future_to_region = {
                executor.submit(self._collect_region_quotas, region): region
                for region in self.regions
            }

            # Collect results as they complete
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    region_data = future.result()
                    if region_data:
                        all_quotas[region] = region_data
                        quota_count = len(region_data.get('quotas', {}))
                        self.logger.info(f"✅ [{region}] Collected {quota_count} quotas")
                except Exception as e:
                    self.logger.error(f"❌ [{region}] Collection failed: {e}")

        elapsed = time.time() - start_time
        total_quotas = sum(len(region_data.get('quotas', {})) for region_data in all_quotas.values())
        self.logger.info(f"✅ Parallel quota collection complete from {len(self.regions)} regions in {elapsed:.2f}s")
        self.logger.info(f"📊 Total quotas collected: {total_quotas}")

        return all_quotas

    def _collect_service_quotas_sequential(self) -> Dict[str, Any]:
        """Collect service quotas from all regions SEQUENTIALLY"""
        self.logger.info(f"🐌 Collecting ALL Bedrock service quotas from {len(self.regions)} regions (sequential)...")

        all_quotas = {}
        start_time = time.time()

        for region in self.regions:
            region_data = self._collect_region_quotas(region)
            if region_data:
                all_quotas[region] = region_data
                quota_count = len(region_data.get('quotas', {}))
                self.logger.info(f"✅ [{region}] Collected {quota_count} quotas")

        elapsed = time.time() - start_time
        total_quotas = sum(len(region_data.get('quotas', {})) for region_data in all_quotas.values())
        self.logger.info(f"✅ Sequential quota collection complete from {len(self.regions)} regions in {elapsed:.2f}s")
        self.logger.info(f"📊 Total quotas collected: {total_quotas}")

        return all_quotas

    def _collect_region_quotas(self, region: str) -> Dict[str, Any]:
        """Collect ALL Bedrock quotas from a specific region - NO FILTERING"""
        try:
            quotas_client = self.session.client('service-quotas', region_name=region)

            # Get all Bedrock service quotas in one call
            paginator = quotas_client.get_paginator('list_service_quotas')

            region_quotas = {}
            total_quotas = 0

            for page in paginator.paginate(ServiceCode='bedrock'):
                for quota in page.get('Quotas', []):
                    quota_code = quota.get('QuotaCode', '')
                    quota_name = quota.get('QuotaName', '')

                    if quota_code:
                        region_quotas[quota_code] = {
                            'quota_code': quota_code,
                            'quota_name': quota_name,
                            'quota_arn': quota.get('QuotaArn', ''),
                            'description': quota.get('Description', ''),
                            'quota_applied_at_level': quota.get('QuotaAppliedAtLevel', ''),
                            'value': quota.get('Value', 0),
                            'unit': quota.get('Unit', ''),
                            'adjustable': quota.get('Adjustable', False),
                            'global_quota': quota.get('GlobalQuota', False),
                            'usage_metric': quota.get('UsageMetric', {}),
                            'period': quota.get('Period', {})
                        }
                        total_quotas += 1

            return {
                'quotas': region_quotas,
                'region': region,
                'collection_timestamp': self._get_timestamp(),
                'total_quotas': total_quotas
            }

        except ClientError as e:
            if 'AccessDenied' in str(e) or 'UnauthorizedOperation' in str(e):
                self.logger.warning(f"⚠️  [{region}] Access denied for service quotas")
            else:
                self.logger.error(f"❌ [{region}] AWS API error: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"❌ [{region}] Unexpected error: {e}")
            return {}

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp"""
        return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())