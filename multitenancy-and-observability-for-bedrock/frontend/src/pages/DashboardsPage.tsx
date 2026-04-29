// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listDashboards, createDashboard, deleteDashboard, getDashboardWidgets } from '../api/dashboards';
import { listProfiles } from '../api/profiles';
import type { Dashboard, CreateDashboardRequest, DashboardWidget, Profile } from '../types';
import SuccessBanner from '../components/SuccessBanner';

const TEMPLATE_OPTIONS: { value: string; label: string }[] = [
  { value: 'cost_overview', label: 'Cost Overview' },
  { value: 'performance', label: 'Performance' },
  { value: 'capacity', label: 'Capacity' },
  { value: 'executive_summary', label: 'Executive Summary' },
  { value: 'latency', label: 'Latency' },
  { value: 'efficiency', label: 'Efficiency' },
  { value: 'custom', label: 'Custom' },
];

const CHART_CATEGORIES = [
  { name: 'Cost', charts: [
    { id: 'cost_timeseries', title: 'Token Costs', type: 'timeseries' },
    { id: 'cost_comparison_bar', title: 'All Profiles Cost Comparison', type: 'bar' },
  ]},
  { name: 'Performance', charts: [
    { id: 'tokens_per_minute', title: 'Tokens per Minute', type: 'timeseries' },
    { id: 'requests_per_minute', title: 'Requests per Minute', type: 'timeseries' },
    { id: 'all_profiles_tokens_bar', title: 'All Profiles Tokens/Min', type: 'bar' },
  ]},
  { name: 'Latency', charts: [
    { id: 'avg_latency', title: 'Average Latency (ms)', type: 'timeseries' },
    { id: 'p90_latency', title: 'P90 Latency (ms)', type: 'timeseries' },
    { id: 'p99_latency', title: 'P99 Latency (ms)', type: 'timeseries' },
    { id: 'latency_comparison_bar', title: 'All Profiles Latency Comparison', type: 'bar' },
    { id: 'avg_latency_value', title: 'Avg Latency (ms)', type: 'number' },
    { id: 'p99_latency_value', title: 'P99 Latency (ms)', type: 'number' },
  ]},
  { name: 'Efficiency', charts: [
    { id: 'cost_per_request', title: 'Cost per Request ($)', type: 'timeseries' },
    { id: 'tokens_per_request', title: 'Tokens per Request', type: 'timeseries' },
    { id: 'input_output_ratio', title: 'Output / Input Token Ratio', type: 'timeseries' },
    { id: 'output_tokens_per_dollar', title: 'Output Tokens per Dollar', type: 'timeseries' },
  ]},
  { name: 'Capacity', charts: [
    { id: 'error_rate_pct', title: 'Error Rate %', type: 'timeseries' },
    { id: 'all_profiles_requests_bar', title: 'All Profiles Requests/Min', type: 'bar' },
    { id: 'success_failure_pie', title: 'Success vs Failure', type: 'pie' },
    { id: 'capacity_utilization_pie', title: 'Capacity Utilization', type: 'pie' },
  ]},
  { name: 'Executive', charts: [
    { id: 'total_cost_value', title: 'Total Cost ($)', type: 'number' },
    { id: 'total_requests_value', title: 'Total Requests', type: 'number' },
    { id: 'top_tenants_cost_bar', title: 'Top Profiles by Cost', type: 'bar' },
  ]},
];

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

const DashboardsPage: React.FC = () => {
  const navigate = useNavigate();
  const [dashboards, setDashboards] = useState<Dashboard[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [selectedProfileIds, setSelectedProfileIds] = useState<string[]>([]);
  const [templateId, setTemplateId] = useState('');
  const [dashboardName, setDashboardName] = useState('');
  const [creating, setCreating] = useState(false);
  const [templateWidgets, setTemplateWidgets] = useState<DashboardWidget[]>([]);
  const [widgetOverrides, setWidgetOverrides] = useState<Record<string, Partial<DashboardWidget>>>({});
  const [loadingWidgets, setLoadingWidgets] = useState(false);
  const [selectedTagDimensions, setSelectedTagDimensions] = useState<string[]>([]);
  const [tagFilter, setTagFilter] = useState<{ key: string; value: string }>({ key: '', value: '' });
  const [selectionMode, setSelectionMode] = useState<'manual' | 'tag'>('manual');
  const [groupTagKey, setGroupTagKey] = useState('');
  const [groupTagValue, setGroupTagValue] = useState('');
  const [selectedWidgetIds, setSelectedWidgetIds] = useState<string[]>([]);
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({});

  const fetchDashboards = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listDashboards();
      setDashboards(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load dashboards';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const fetchProfiles = async () => {
    try {
      const data = await listProfiles();
      setProfiles(data);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    fetchDashboards();
    fetchProfiles();
  }, []);

  // When template changes, load sample widgets from an existing dashboard of that template
  useEffect(() => {
    if (!templateId) {
      setTemplateWidgets([]);
      setWidgetOverrides({});
      return;
    }
    const existing = dashboards.find((d) => d.template_id === templateId);
    if (existing && existing.widgets?.length) {
      setTemplateWidgets(existing.widgets);
    } else if (existing) {
      setLoadingWidgets(true);
      getDashboardWidgets(existing.dashboard_id)
        .then((w) => setTemplateWidgets(w))
        .catch(() => setTemplateWidgets([]))
        .finally(() => setLoadingWidgets(false));
    } else {
      setTemplateWidgets([]);
    }
    setWidgetOverrides({});
  }, [templateId, dashboards]);

  const handleWidgetOverride = (widgetId: string, field: string, value: string | number | null) => {
    setWidgetOverrides((prev) => {
      const current = prev[widgetId] ?? {};
      return { ...prev, [widgetId]: { ...current, [field]: value } };
    });
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setCreating(true);
    try {
      const cleanOverrides: Record<string, Partial<DashboardWidget>> = {};
      for (const [wid, ov] of Object.entries(widgetOverrides)) {
        if (Object.keys(ov).length > 0) cleanOverrides[wid] = ov;
      }

      // Resolve profile IDs from tag grouping if needed
      const resolvedProfileIds = selectionMode === 'tag' ? matchingProfileIds : selectedProfileIds;

      const req: CreateDashboardRequest = {
        tenant_id: resolvedProfileIds[0],
        template_id: templateId,
        tenant_ids: resolvedProfileIds,
      };
      if (dashboardName) {
        req.dashboard_name = dashboardName;
      }
      if (Object.keys(cleanOverrides).length > 0) {
        req.widget_overrides = cleanOverrides;
      }
      if (selectedTagDimensions.length > 0) {
        req.tag_dimensions = selectedTagDimensions;
      }
      if (templateId === 'custom' && selectedWidgetIds.length > 0) {
        req.widget_ids = selectedWidgetIds;
      }
      await createDashboard(req);

      setShowForm(false);
      setSelectedProfileIds([]);
      setTemplateId('');
      setDashboardName('');
      setWidgetOverrides({});
      setSelectedTagDimensions([]);
      setTagFilter({ key: '', value: '' });
      setSelectionMode('manual');
      setGroupTagKey('');
      setGroupTagValue('');
      setSelectedWidgetIds([]);
      setExpandedCategories({});
      await fetchDashboards();
      const profileNames = resolvedProfileIds.map((id) => profiles.find((t) => t.tenant_id === id)?.tenant_name ?? id).join(', ');
      setSuccessMsg(`Dashboard created for ${profileNames}`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to create dashboard';
      setError(message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, dashboardId: string) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this dashboard?')) return;
    try {
      await deleteDashboard(dashboardId);
      await fetchDashboards();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete dashboard';
      setError(message);
    }
  };

  const profileName = (tenantId: string) => {
    const t = profiles.find((t) => t.tenant_id === tenantId);
    return t ? t.tenant_name : tenantId;
  };

  const templateLabel = (templateId: string) => {
    const t = TEMPLATE_OPTIONS.find((o) => o.value === templateId);
    return t ? t.label : templateId;
  };

  // Compute all tag keys and their values across all profiles
  const allTagKeys = Array.from(new Set(profiles.flatMap((p) => Object.keys(p.tags || {})))).sort();
  const tagValuesForKey = (key: string): string[] => {
    const values = new Set<string>();
    for (const p of profiles) {
      const v = p.tags?.[key];
      if (Array.isArray(v)) v.forEach((x) => values.add(x));
      else if (v) values.add(v as string);
    }
    return [...values].sort();
  };

  // Compute matching profiles for tag grouping mode
  const matchingProfileIds: string[] = selectionMode === 'tag' && groupTagKey && groupTagValue
    ? profiles.filter((p) => {
        const v = p.tags?.[groupTagKey];
        if (Array.isArray(v)) return v.includes(groupTagValue);
        return v === groupTagValue;
      }).map((p) => p.tenant_id)
    : [];

  const effectiveProfileIds = selectionMode === 'tag' ? matchingProfileIds : selectedProfileIds;

  const tableStyle: React.CSSProperties = { width: '100%', borderCollapse: 'collapse' };
  const thStyle: React.CSSProperties = { textAlign: 'left', padding: '10px 12px', borderBottom: '2px solid #e5e7eb', fontWeight: 600, fontSize: '14px', color: '#374151' };
  const tdStyle: React.CSSProperties = { padding: '10px 12px', borderBottom: '1px solid #f3f4f6', fontSize: '14px' };
  const inputStyle: React.CSSProperties = { padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: '6px', fontSize: '14px', width: '100%', boxSizing: 'border-box' };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h1 style={{ margin: 0 }}>Dashboards</h1>
        <button
          onClick={() => setShowForm((prev) => !prev)}
          style={{
            padding: '8px 18px',
            backgroundColor: showForm ? '#6b7280' : '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: 600,
          }}
        >
          {showForm ? 'Cancel' : 'Create Dashboard'}
        </button>
      </div>

      {showForm && (
        <form
          onSubmit={handleCreate}
          style={{
            padding: '20px',
            backgroundColor: '#f9fafb',
            borderRadius: '8px',
            marginBottom: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
          }}
        >
          {/* Profile Selection Mode Toggle */}
          <div>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '14px', fontWeight: 500 }}>Profile Selection</label>
            <div style={{ display: 'flex', gap: '16px', marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', cursor: 'pointer' }}>
                <input
                  type="radio"
                  name="selectionMode"
                  checked={selectionMode === 'manual'}
                  onChange={() => { setSelectionMode('manual'); setGroupTagKey(''); setGroupTagValue(''); }}
                />
                Select profiles
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', cursor: 'pointer' }}>
                <input
                  type="radio"
                  name="selectionMode"
                  checked={selectionMode === 'tag'}
                  onChange={() => { setSelectionMode('tag'); setSelectedProfileIds([]); }}
                />
                Group by tag
              </label>
            </div>

            {selectionMode === 'manual' && (
              <div>
                <div style={{ border: '1px solid #d1d5db', borderRadius: '6px', padding: '8px 12px', maxHeight: '150px', overflowY: 'auto', backgroundColor: '#fff' }}>
                  {profiles.map((t) => (
                    <label key={t.tenant_id} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 0', fontSize: '14px', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={selectedProfileIds.includes(t.tenant_id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedProfileIds((prev) => [...prev, t.tenant_id]);
                          } else {
                            setSelectedProfileIds((prev) => prev.filter((id) => id !== t.tenant_id));
                          }
                        }}
                      />
                      {t.tenant_name} <span style={{ color: '#9ca3af', fontSize: '12px' }}>({t.region})</span>
                    </label>
                  ))}
                  {profiles.length === 0 && <p style={{ color: '#9ca3af', fontSize: '13px', margin: 0 }}>No profiles available</p>}
                </div>
                {selectedProfileIds.length > 0 && (
                  <span style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px', display: 'block' }}>{selectedProfileIds.length} selected</span>
                )}
              </div>
            )}

            {selectionMode === 'tag' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <div style={{ flex: 1 }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#374151', marginBottom: '2px' }}>Tag</label>
                    <select
                      value={groupTagKey}
                      onChange={(e) => { setGroupTagKey(e.target.value); setGroupTagValue(''); }}
                      style={inputStyle}
                    >
                      <option value="">Select tag...</option>
                      {allTagKeys.map((k) => (
                        <option key={k} value={k}>{k}</option>
                      ))}
                    </select>
                  </div>
                  {groupTagKey && (
                    <div style={{ flex: 1 }}>
                      <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#374151', marginBottom: '2px' }}>Value</label>
                      <select
                        value={groupTagValue}
                        onChange={(e) => setGroupTagValue(e.target.value)}
                        style={inputStyle}
                      >
                        <option value="">Select value...</option>
                        {tagValuesForKey(groupTagKey).map((v) => (
                          <option key={v} value={v}>{v}</option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>
                {matchingProfileIds.length > 0 && (
                  <div style={{ fontSize: '13px', color: '#374151', backgroundColor: '#fff', border: '1px solid #d1d5db', borderRadius: '6px', padding: '8px 12px' }}>
                    <span style={{ fontWeight: 500 }}>Matching ({matchingProfileIds.length}):</span>{' '}
                    {matchingProfileIds.map((id) => profileName(id)).join(', ')}
                  </div>
                )}
                {groupTagKey && groupTagValue && matchingProfileIds.length === 0 && (
                  <p style={{ color: '#9ca3af', fontSize: '13px', margin: 0 }}>No profiles match this tag</p>
                )}
              </div>
            )}
          </div>

          {/* Tag Filter — only show tags common to ALL selected profiles */}
          {selectedProfileIds.length > 1 && (() => {
            const selected = profiles.filter((p) => selectedProfileIds.includes(p.tenant_id));

            // Find tags common to ALL selected profiles (intersection)
            const commonTags: Record<string, string[]> = {};
            if (selected.length > 0) {
              // Start with keys from the first profile
              const firstTags = selected[0].tags || {};
              for (const key of Object.keys(firstTags)) {
                // Check if ALL other profiles also have this key
                const allHaveKey = selected.every((p) => p.tags && key in p.tags);
                if (allHaveKey) {
                  // Collect unique values across all profiles for this key
                  const values = new Set<string>();
                  for (const p of selected) {
                    const v = p.tags[key];
                    if (Array.isArray(v)) v.forEach((x) => values.add(x));
                    else if (v) values.add(v as string);
                  }
                  commonTags[key] = [...values].sort();
                }
              }
            }

            const tagKeys = Object.keys(commonTags);
            if (tagKeys.length === 0) return null;

            // Apply current tag filter to show which profiles match
            const filteredIds = tagFilter.key && tagFilter.value
              ? selected.filter((p) => {
                  const v = p.tags?.[tagFilter.key];
                  if (Array.isArray(v)) return v.includes(tagFilter.value);
                  return v === tagFilter.value;
                }).map((p) => p.tenant_id)
              : null;

            return (
              <div>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '14px', fontWeight: 500 }}>
                  Filter by Tag <span style={{ color: '#6b7280', fontWeight: 400 }}>(common tags only)</span>
                </label>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <select
                    value={tagFilter.key}
                    onChange={(e) => setTagFilter({ key: e.target.value, value: '' })}
                    style={inputStyle}
                  >
                    <option value="">No filter</option>
                    {tagKeys.map((k) => (
                      <option key={k} value={k}>{k}</option>
                    ))}
                  </select>
                  {tagFilter.key && (
                    <select
                      value={tagFilter.value}
                      onChange={(e) => {
                        const val = e.target.value;
                        setTagFilter((prev) => ({ ...prev, value: val }));
                        // Auto-filter the selected profiles
                        if (val) {
                          const matching = selected.filter((p) => {
                            const v = p.tags?.[tagFilter.key];
                            if (Array.isArray(v)) return v.includes(val);
                            return v === val;
                          }).map((p) => p.tenant_id);
                          setSelectedProfileIds(matching);
                        }
                      }}
                      style={inputStyle}
                    >
                      <option value="">Select value...</option>
                      {commonTags[tagFilter.key]?.map((v) => (
                        <option key={v} value={v}>{v}</option>
                      ))}
                    </select>
                  )}
                  {filteredIds && (
                    <span style={{ fontSize: '12px', color: '#6b7280' }}>
                      {filteredIds.length} of {selected.length} profiles match
                    </span>
                  )}
                </div>
              </div>
            );
          })()}

          {/* Template */}
          <div>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '14px', fontWeight: 500 }}>Template</label>
            <select
              value={templateId}
              onChange={(e) => setTemplateId(e.target.value)}
              required
              style={inputStyle}
            >
              <option value="">Select a template</option>
              {TEMPLATE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          {/* Custom Chart Picker */}
          {templateId === 'custom' && (
            <div>
              <label style={{ display: 'block', marginBottom: '6px', fontSize: '14px', fontWeight: 500 }}>
                Select Charts {selectedWidgetIds.length > 0 && <span style={{ color: '#6b7280', fontWeight: 400 }}>({selectedWidgetIds.length} selected)</span>}
              </label>
              <div style={{ border: '1px solid #d1d5db', borderRadius: '6px', backgroundColor: '#fff', overflow: 'hidden' }}>
                {CHART_CATEGORIES.map((cat) => {
                  const isExpanded = expandedCategories[cat.name] !== false; // default expanded
                  const selectedInCat = cat.charts.filter((c) => selectedWidgetIds.includes(c.id)).length;
                  const allInCatSelected = selectedInCat === cat.charts.length;
                  return (
                    <div key={cat.name}>
                      <div
                        onClick={() => setExpandedCategories((prev) => ({ ...prev, [cat.name]: !isExpanded }))}
                        style={{
                          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                          padding: '8px 12px', cursor: 'pointer', backgroundColor: '#f9fafb',
                          borderBottom: '1px solid #e5e7eb', userSelect: 'none',
                        }}
                      >
                        <span style={{ fontSize: '14px', fontWeight: 600 }}>
                          {isExpanded ? '\u25BC' : '\u25B6'} {cat.name}
                          {selectedInCat > 0 && <span style={{ color: '#6b7280', fontWeight: 400, marginLeft: '8px' }}>({selectedInCat}/{cat.charts.length})</span>}
                        </span>
                        <label
                          onClick={(e) => e.stopPropagation()}
                          style={{ fontSize: '12px', color: '#6b7280', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
                        >
                          <input
                            type="checkbox"
                            checked={allInCatSelected}
                            onChange={(e) => {
                              const catIds = cat.charts.map((c) => c.id);
                              if (e.target.checked) {
                                setSelectedWidgetIds((prev) => [...new Set([...prev, ...catIds])]);
                              } else {
                                setSelectedWidgetIds((prev) => prev.filter((id) => !catIds.includes(id)));
                              }
                            }}
                          />
                          All
                        </label>
                      </div>
                      {isExpanded && (
                        <div style={{ padding: '4px 12px 8px' }}>
                          {cat.charts.map((chart) => (
                            <label key={chart.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 0', fontSize: '14px', cursor: 'pointer' }}>
                              <input
                                type="checkbox"
                                checked={selectedWidgetIds.includes(chart.id)}
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    setSelectedWidgetIds((prev) => [...prev, chart.id]);
                                  } else {
                                    setSelectedWidgetIds((prev) => prev.filter((id) => id !== chart.id));
                                  }
                                }}
                              />
                              {chart.title}
                              <span style={{ fontSize: '11px', padding: '1px 6px', borderRadius: '4px', backgroundColor: '#eff6ff', color: '#2563eb' }}>
                                {chart.type}
                              </span>
                            </label>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Dashboard Name */}
          <div>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '14px', fontWeight: 500 }}>Dashboard Name (optional)</label>
            <input
              type="text"
              value={dashboardName}
              onChange={(e) => setDashboardName(e.target.value)}
              placeholder={effectiveProfileIds.length > 1 ? 'Name — profile name will be appended' : 'Custom dashboard name'}
              style={inputStyle}
            />
          </div>

          {/* Widget Preview & Customization — shown for preset templates */}
          {templateId && templateId !== 'custom' && templateWidgets.length > 0 && (
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: 500 }}>
                Widgets ({templateWidgets.length} charts)
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '12px' }}>
                {loadingWidgets ? (
                  <p style={{ color: '#6b7280', fontSize: '13px' }}>Loading widgets...</p>
                ) : (
                  templateWidgets.map((w) => {
                    const overStat = widgetOverrides[w.widget_id]?.stat;
                    const overPeriod = widgetOverrides[w.widget_id]?.period;
                    const overAnalysis = widgetOverrides[w.widget_id]?.analysis;
                    const availableStats = w.available_stats?.length ? w.available_stats : ['Sum', 'Average', 'Minimum', 'Maximum', 'p50', 'p90', 'p99'];
                    const availablePeriods = w.available_periods?.length ? w.available_periods : [60, 300, 3600, 86400];
                    const availableAnalysis = w.available_analysis?.length ? w.available_analysis : ['none', 'trend', 'anomaly_detection', 'arima'];

                    const typeLabel: Record<string, string> = { timeseries: 'Time Series', bar: 'Bar Chart', number: 'Number', single_value: 'Single Value' };

                    return (
                      <div key={w.widget_id} style={{ border: '1px solid #e5e7eb', borderRadius: '8px', padding: '14px', backgroundColor: '#fff' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                          <span style={{ fontSize: '14px', fontWeight: 600 }}>{w.title}</span>
                          <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '4px', backgroundColor: '#eff6ff', color: '#2563eb' }}>
                            {typeLabel[w.type] ?? w.type}
                          </span>
                        </div>
                        <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '10px' }}>
                          Metrics: {w.metrics?.join(', ')}
                        </div>
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                          <div style={{ flex: '1 1 90px' }}>
                            <label style={{ display: 'block', fontSize: '11px', fontWeight: 500, color: '#374151', marginBottom: '2px' }}>Stat</label>
                            <select
                              value={overStat ?? w.stat}
                              onChange={(e) => handleWidgetOverride(w.widget_id, 'stat', e.target.value)}
                              style={{ ...inputStyle, fontSize: '12px', padding: '4px 8px' }}
                            >
                              {availableStats.map((s) => (
                                <option key={s} value={s}>{s}</option>
                              ))}
                            </select>
                          </div>
                          <div style={{ flex: '1 1 90px' }}>
                            <label style={{ display: 'block', fontSize: '11px', fontWeight: 500, color: '#374151', marginBottom: '2px' }}>Period</label>
                            <select
                              value={overPeriod ?? w.period}
                              onChange={(e) => handleWidgetOverride(w.widget_id, 'period', Number(e.target.value))}
                              style={{ ...inputStyle, fontSize: '12px', padding: '4px 8px' }}
                            >
                              {availablePeriods.map((p) => (
                                <option key={p} value={p}>{PERIOD_LABELS[p] ?? `${p}s`}</option>
                              ))}
                            </select>
                          </div>
                          <div style={{ flex: '1 1 120px' }}>
                            <label style={{ display: 'block', fontSize: '11px', fontWeight: 500, color: '#374151', marginBottom: '2px' }}>Analysis</label>
                            <select
                              value={overAnalysis ?? w.analysis ?? 'none'}
                              onChange={(e) => handleWidgetOverride(w.widget_id, 'analysis', e.target.value === 'none' ? null : e.target.value)}
                              style={{ ...inputStyle, fontSize: '12px', padding: '4px 8px' }}
                            >
                              {availableAnalysis.map((a) => (
                                <option key={a} value={a}>{ANALYSIS_LABELS[a] ?? a}</option>
                              ))}
                            </select>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={effectiveProfileIds.length === 0 || !templateId || creating || (templateId === 'custom' && selectedWidgetIds.length === 0)}
              style={{
                padding: '8px 18px',
                backgroundColor: effectiveProfileIds.length === 0 || !templateId || creating || (templateId === 'custom' && selectedWidgetIds.length === 0) ? '#93c5fd' : '#2563eb',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: effectiveProfileIds.length === 0 || !templateId || creating || (templateId === 'custom' && selectedWidgetIds.length === 0) ? 'not-allowed' : 'pointer',
                fontWeight: 600,
              }}
            >
              {creating
                ? 'Creating...'
                : effectiveProfileIds.length > 1
                  ? `Create Dashboard (${effectiveProfileIds.length} profiles)`
                  : 'Create Dashboard'}
            </button>
          </div>
        </form>
      )}

      {error && (
        <div style={{ padding: '12px 16px', backgroundColor: '#fef2f2', color: '#dc2626', borderRadius: '6px', marginBottom: '16px', fontSize: '14px' }}>
          {error}
        </div>
      )}

      {loading ? (
        <p style={{ color: '#6b7280' }}>Loading dashboards...</p>
      ) : dashboards.length === 0 ? (
        <p style={{ color: '#6b7280' }}>No dashboards found. Create one to get started.</p>
      ) : (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Profile</th>
              <th style={thStyle}>Template</th>
              <th style={thStyle}>Region</th>
              <th style={thStyle}>Created</th>
              <th style={thStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {dashboards.map((d) => (
              <tr
                key={d.dashboard_id}
                onClick={() => navigate(`/dashboards/${d.dashboard_id}`)}
                style={{ cursor: 'pointer' }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = '#f9fafb'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = ''; }}
              >
                <td style={{ ...tdStyle, fontWeight: 500, color: '#2563eb' }}>{d.dashboard_name}</td>
                <td style={tdStyle}>
                  {(d.tenant_ids && d.tenant_ids.length > 1)
                    ? d.tenant_ids.map((tid) => profileName(tid)).join(', ')
                    : profileName(d.tenant_id)}
                </td>
                <td style={tdStyle}>{templateLabel(d.template_id)}</td>
                <td style={tdStyle}>{d.region}</td>
                <td style={tdStyle}>
                  {d.created_at ? new Date(d.created_at).toLocaleDateString() : '-'}
                </td>
                <td style={{ ...tdStyle, display: 'flex', gap: '8px' }}>
                  {d.console_url && (
                    <a
                      href={d.console_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      style={{
                        padding: '4px 10px',
                        backgroundColor: '#fff',
                        border: '1px solid #93c5fd',
                        borderRadius: '4px',
                        color: '#2563eb',
                        cursor: 'pointer',
                        fontSize: '13px',
                        textDecoration: 'none',
                      }}
                    >
                      Open in CloudWatch
                    </a>
                  )}
                  <button
                    onClick={(e) => handleDelete(e, d.dashboard_id)}
                    style={{
                      padding: '4px 10px',
                      backgroundColor: '#fff',
                      border: '1px solid #fca5a5',
                      borderRadius: '4px',
                      color: '#dc2626',
                      cursor: 'pointer',
                      fontSize: '13px',
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <SuccessBanner message={successMsg} onDismiss={() => setSuccessMsg(null)} />
    </div>
  );
};

export default DashboardsPage;
