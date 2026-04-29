// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { backendFetch } from './client';
import type { MetricQueryResponse } from '../types';

export interface QueryMetricsParams {
  tenant_id: string;
  metric_name: string;
  stat?: string;
  period?: number;
  hours?: number;
}

export async function queryMetrics(params: QueryMetricsParams): Promise<MetricQueryResponse> {
  const qs = new URLSearchParams();
  qs.set('tenant_id', params.tenant_id);
  qs.set('metric_name', params.metric_name);
  if (params.stat) qs.set('stat', params.stat);
  if (params.period) qs.set('period', String(params.period));
  if (params.hours) qs.set('hours', String(params.hours));
  return backendFetch(`/metrics/query?${qs}`) as Promise<MetricQueryResponse>;
}
