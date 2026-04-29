// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { backendFetch } from './client';
import type { Dashboard, CreateDashboardRequest, UpdateDashboardRequest, DashboardWidget } from '../types';

export async function listDashboards(tenantId?: string, tagFilters?: Record<string, string>): Promise<Dashboard[]> {
  const params = new URLSearchParams();
  if (tenantId) params.set('tenant_id', tenantId);
  if (tagFilters && Object.keys(tagFilters).length > 0) {
    params.set('tag_filters', Object.entries(tagFilters).map(([k,v]) => `${k}:${v}`).join(','));
  }
  const path = params.toString() ? `/dashboards?${params}` : '/dashboards';
  const data = await backendFetch(path) as { dashboards: Dashboard[] };
  return data.dashboards;
}

export function getDashboard(id: string): Promise<Dashboard> {
  return backendFetch(`/dashboards/${id}`) as Promise<Dashboard>;
}

export function createDashboard(req: CreateDashboardRequest): Promise<Dashboard> {
  return backendFetch('/dashboards', {
    method: 'POST',
    body: JSON.stringify(req),
  }) as Promise<Dashboard>;
}

export function updateDashboard(id: string, body: UpdateDashboardRequest): Promise<Dashboard> {
  return backendFetch(`/dashboards/${id}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  }) as Promise<Dashboard>;
}

export async function getDashboardWidgets(id: string): Promise<DashboardWidget[]> {
  const dashboard = await getDashboard(id);
  return dashboard.widgets ?? [];
}

export function deleteDashboard(id: string): Promise<void> {
  return backendFetch(`/dashboards/${id}`, { method: 'DELETE' }) as Promise<void>;
}
