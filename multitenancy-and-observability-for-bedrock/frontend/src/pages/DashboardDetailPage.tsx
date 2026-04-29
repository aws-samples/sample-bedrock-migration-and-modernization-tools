// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getDashboard, updateDashboard } from '../api/dashboards';
import { listProfiles } from '../api/profiles';
import { queryMetrics } from '../api/metrics';
import type { Dashboard, DashboardWidget, Profile, UpdateDashboardRequest } from '../types';
import SuccessBanner from '../components/SuccessBanner';
import PieChartCard from '../components/PieChart';

const PERIOD_LABELS: Record<number, string> = {
  60: '1 min',
  300: '5 min',
  3600: '1 hour',
  86400: '1 day',
};

const ANALYSIS_LABELS: Record<string, string> = {
  none: 'None',
  trend: 'Trend (Linear Regression)',
  anomaly_detection: 'Anomaly Detection',
  arima: 'ARIMA Seasonality',
};

const WIDGET_TYPE_LABELS: Record<string, string> = {
  timeseries: 'Time Series',
  bar: 'Bar Chart',
  number: 'Number',
  single_value: 'Single Value',
  pie: 'Pie Chart',
};

const DashboardDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [dashboardName, setDashboardName] = useState('');
  const [overrides, setOverrides] = useState<Record<string, Partial<DashboardWidget>>>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [actualRequestsPerMin, setActualRequestsPerMin] = useState<number | null>(null);
  const [capacityLimit, setCapacityLimit] = useState<number>(1000);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [d, t] = await Promise.all([getDashboard(id), listProfiles().catch(() => [])]);
        setDashboard(d);
        setDashboardName(d.dashboard_name);
        setOverrides(d.widget_overrides ?? {});
        setProfiles(t);
        // Set capacity limit from the profile if available
        const profile = t.find((p) => p.tenant_id === d.tenant_id);
        if (profile?.capacity_limit) {
          setCapacityLimit(profile.capacity_limit);
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [id]);

  // Fetch metrics for capacity utilization pie chart
  useEffect(() => {
    if (!dashboard) return;
    const tenantId = dashboard.tenant_id;
    setMetricsLoading(true);
    setMetricsError(null);

    queryMetrics({ tenant_id: tenantId, metric_name: 'InvocationSuccess', stat: 'Sum', period: 60, hours: 24 })
      .then((res) => {
        const total = res.datapoints.reduce((s, d) => s + d.value, 0);
        const points = res.datapoints.length;
        setActualRequestsPerMin(points > 0 ? total / points : 0);
      })
      .catch((err) => {
        setMetricsError(err instanceof Error ? err.message : 'Failed to load metrics');
      })
      .finally(() => setMetricsLoading(false));
  }, [dashboard]);

  const profileName = (tenantId: string) => {
    const t = profiles.find((t) => t.tenant_id === tenantId);
    return t ? t.tenant_name : tenantId;
  };

  const getEffectiveValue = (widget: DashboardWidget, field: keyof DashboardWidget) => {
    const override = overrides[widget.widget_id];
    if (override && override[field] !== undefined) return override[field];
    return widget[field];
  };

  const setWidgetOverride = (widgetId: string, field: string, value: string | number | null) => {
    setOverrides((prev) => {
      const current = prev[widgetId] ?? {};
      return { ...prev, [widgetId]: { ...current, [field]: value } };
    });
    setHasChanges(true);
  };

  const handleNameChange = (name: string) => {
    setDashboardName(name);
    setHasChanges(true);
  };

  const handleSave = async () => {
    if (!id || !dashboard) return;
    setSaving(true);
    setError(null);
    try {
      const body: UpdateDashboardRequest = {};
      if (dashboardName !== dashboard.dashboard_name) {
        body.dashboard_name = dashboardName;
      }
      // Only send overrides that differ from original
      const changedOverrides: Record<string, Partial<DashboardWidget>> = {};
      for (const [wid, ov] of Object.entries(overrides)) {
        const orig = dashboard.widget_overrides?.[wid] ?? {};
        if (JSON.stringify(ov) !== JSON.stringify(orig)) {
          changedOverrides[wid] = ov;
        }
      }
      if (Object.keys(changedOverrides).length > 0) {
        body.widget_overrides = changedOverrides;
      }
      const updated = await updateDashboard(id, body);
      setDashboard(updated);
      setOverrides(updated.widget_overrides ?? {});
      setHasChanges(false);
      setSuccessMsg('Dashboard saved successfully');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to save dashboard');
    } finally {
      setSaving(false);
    }
  };

  const inputStyle: React.CSSProperties = {
    padding: '8px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    width: '100%',
    boxSizing: 'border-box',
  };

  if (loading) {
    return (
      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>
        <p style={{ color: '#6b7280' }}>Loading dashboard...</p>
      </div>
    );
  }

  if (error && !dashboard) {
    return (
      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>
        <div style={{ padding: '12px 16px', backgroundColor: '#fef2f2', color: '#dc2626', borderRadius: '6px', marginBottom: '16px', fontSize: '14px' }}>
          {error}
        </div>
        <button onClick={() => navigate('/dashboards')} style={{ color: '#2563eb', background: 'none', border: 'none', cursor: 'pointer', fontSize: '14px' }}>
          Back to Dashboards
        </button>
      </div>
    );
  }

  if (!dashboard) return null;

  const widgets = dashboard.widgets ?? [];

  const usedCapacity = actualRequestsPerMin ?? 0;
  const capacityData = actualRequestsPerMin !== null
    ? [
        { name: 'Used', value: Math.min(usedCapacity, capacityLimit), color: usedCapacity > capacityLimit * 0.9 ? '#ef4444' : '#3b82f6' },
        { name: 'Available', value: Math.max(0, capacityLimit - usedCapacity), color: '#e5e7eb' },
      ]
    : [];

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>
      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <button
          onClick={() => navigate('/dashboards')}
          style={{ color: '#2563eb', background: 'none', border: 'none', cursor: 'pointer', fontSize: '14px', padding: 0, marginBottom: '12px' }}
        >
          &larr; Back to Dashboards
        </button>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px' }}>
          <div style={{ flex: 1 }}>
            <input
              type="text"
              value={dashboardName}
              onChange={(e) => handleNameChange(e.target.value)}
              style={{
                ...inputStyle,
                fontSize: '20px',
                fontWeight: 600,
                border: '1px solid transparent',
                padding: '4px 8px',
                width: 'auto',
                minWidth: '300px',
              }}
              onFocus={(e) => { e.target.style.borderColor = '#d1d5db'; }}
              onBlur={(e) => { e.target.style.borderColor = 'transparent'; }}
            />
            <div style={{ display: 'flex', gap: '16px', marginTop: '8px', fontSize: '13px', color: '#6b7280' }}>
              <span>Profile: <strong>{profileName(dashboard.tenant_id)}</strong></span>
              <span>Template: <strong>{dashboard.template_id}</strong></span>
              <span>Region: <strong>{dashboard.region}</strong></span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
            {dashboard.console_url && (
              <a
                href={dashboard.console_url}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#fff',
                  border: '1px solid #93c5fd',
                  borderRadius: '6px',
                  color: '#2563eb',
                  cursor: 'pointer',
                  fontSize: '14px',
                  textDecoration: 'none',
                  fontWeight: 500,
                }}
              >
                Open in CloudWatch
              </a>
            )}
            <button
              onClick={handleSave}
              disabled={!hasChanges || saving}
              style={{
                padding: '8px 18px',
                backgroundColor: hasChanges ? '#2563eb' : '#9ca3af',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: hasChanges ? 'pointer' : 'default',
                fontWeight: 600,
                fontSize: '14px',
              }}
            >
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ padding: '12px 16px', backgroundColor: '#fef2f2', color: '#dc2626', borderRadius: '6px', marginBottom: '16px', fontSize: '14px' }}>
          {error}
        </div>
      )}

      {/* Interactive Capacity Utilization */}
      <div style={{ marginBottom: '24px', maxWidth: '500px' }}>
        <PieChartCard
          title={`Capacity Utilization (${Math.round(usedCapacity)} / ${capacityLimit} req/min)`}
          data={capacityData}
          loading={metricsLoading}
          error={metricsError}
        />
        <div style={{ marginTop: '8px' }}>
          <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', fontWeight: 500, color: '#374151' }}>
            Capacity Limit (req/min)
          </label>
          <input
            type="number"
            value={capacityLimit}
            onChange={(e) => setCapacityLimit(Math.max(1, Number(e.target.value)))}
            min={1}
            style={{ ...inputStyle, fontSize: '13px', padding: '6px 10px', width: '160px' }}
          />
        </div>
      </div>

      {/* Widget Grid */}
      {widgets.length === 0 ? (
        <p style={{ color: '#6b7280' }}>No widgets in this dashboard.</p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          {widgets.map((w) => {
            const effectiveStat = getEffectiveValue(w, 'stat') as string;
            const effectivePeriod = getEffectiveValue(w, 'period') as number;
            const effectiveAnalysis = getEffectiveValue(w, 'analysis') as string | null;
            const availableStats = w.available_stats?.length ? w.available_stats : ['Sum', 'Average', 'Minimum', 'Maximum', 'p50', 'p90', 'p99'];
            const availablePeriods = w.available_periods?.length ? w.available_periods : [60, 300, 3600, 86400];
            const availableAnalysis = w.available_analysis?.length ? w.available_analysis : ['none', 'trend', 'anomaly_detection', 'arima'];

            return (
              <div
                key={w.widget_id}
                style={{
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  padding: '16px',
                  backgroundColor: '#fff',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 600 }}>{w.title}</h3>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '2px 10px',
                      borderRadius: '9999px',
                      fontSize: '11px',
                      fontWeight: 600,
                      backgroundColor: '#e0e7ff',
                      color: '#3730a3',
                    }}
                  >
                    {WIDGET_TYPE_LABELS[w.type] ?? w.type}
                  </span>
                </div>

                <div style={{ fontSize: '13px', color: '#6b7280', marginBottom: '12px' }}>
                  <div>Metrics: {w.metrics?.join(', ') || '-'}</div>
                  {w.dimensions?.length > 0 && <div>Dimensions: {w.dimensions.join(', ')}</div>}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: '2px', fontSize: '12px', fontWeight: 500, color: '#374151' }}>Stat</label>
                    <select
                      value={effectiveStat}
                      onChange={(e) => setWidgetOverride(w.widget_id, 'stat', e.target.value)}
                      style={{ ...inputStyle, fontSize: '13px', padding: '6px 10px' }}
                    >
                      {availableStats.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: '2px', fontSize: '12px', fontWeight: 500, color: '#374151' }}>Period</label>
                    <select
                      value={effectivePeriod}
                      onChange={(e) => setWidgetOverride(w.widget_id, 'period', Number(e.target.value))}
                      style={{ ...inputStyle, fontSize: '13px', padding: '6px 10px' }}
                    >
                      {availablePeriods.map((p) => (
                        <option key={p} value={p}>{PERIOD_LABELS[p] ?? `${p}s`}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: '2px', fontSize: '12px', fontWeight: 500, color: '#374151' }}>Analysis</label>
                    <select
                      value={effectiveAnalysis ?? 'none'}
                      onChange={(e) => setWidgetOverride(w.widget_id, 'analysis', e.target.value === 'none' ? null : e.target.value)}
                      style={{ ...inputStyle, fontSize: '13px', padding: '6px 10px' }}
                    >
                      {availableAnalysis.map((a) => (
                        <option key={a} value={a}>{ANALYSIS_LABELS[a] ?? a}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
      <SuccessBanner message={successMsg} onDismiss={() => setSuccessMsg(null)} />
    </div>
  );
};

export default DashboardDetailPage;
