// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { backendFetch } from './client';
import type { Alert, CreateAlertRequest } from '../types';

export async function listAlerts(tenantId?: string, tagFilters?: Record<string, string>): Promise<Alert[]> {
  const params = new URLSearchParams();
  if (tenantId) params.set('tenant_id', tenantId);
  if (tagFilters && Object.keys(tagFilters).length > 0) {
    params.set('tag_filters', Object.entries(tagFilters).map(([k,v]) => `${k}:${v}`).join(','));
  }
  const path = params.toString() ? `/alerts?${params}` : '/alerts';
  const data = await backendFetch(path) as { alerts: Alert[] };
  return data.alerts;
}

export function getAlert(id: string): Promise<Alert> {
  return backendFetch(`/alerts/${id}`) as Promise<Alert>;
}

export function createAlert(req: CreateAlertRequest): Promise<Alert> {
  return backendFetch('/alerts', {
    method: 'POST',
    body: JSON.stringify(req),
  }) as Promise<Alert>;
}

export function updateAlert(id: string, body: Partial<CreateAlertRequest>): Promise<Alert> {
  return backendFetch(`/alerts/${id}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  }) as Promise<Alert>;
}

export function deleteAlert(id: string): Promise<void> {
  return backendFetch(`/alerts/${id}`, { method: 'DELETE' }) as Promise<void>;
}
