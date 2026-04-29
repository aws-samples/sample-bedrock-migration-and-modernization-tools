// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { backendFetch } from './client';
import type { Model, ModelListResponse, PricingInfo } from '../types';

export async function listModels(region: string): Promise<Model[]> {
  const data = await backendFetch(`/discovery/models?region=${encodeURIComponent(region)}`) as ModelListResponse;
  return data.models;
}

export function getPricing(modelId: string, region: string): Promise<PricingInfo> {
  return backendFetch(
    `/discovery/pricing?model_id=${encodeURIComponent(modelId)}&region=${encodeURIComponent(region)}`
  ) as Promise<PricingInfo>;
}
