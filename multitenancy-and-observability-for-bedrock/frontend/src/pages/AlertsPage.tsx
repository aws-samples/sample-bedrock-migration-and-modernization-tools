// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { listAlerts, createAlert, deleteAlert } from '../api/alerts';
import { listProfiles } from '../api/profiles';
import type { Alert, CreateAlertRequest, Profile, AlertAction, ThresholdMode, AlertActionConfig } from '../types';
import SuccessBanner from '../components/SuccessBanner';

const METRIC_GROUPS: { group: string; options: { value: string; label: string }[] }[] = [
  {
    group: 'Cost',
    options: [
      { value: 'InputTokensCost', label: 'Input Token Cost ($)' },
      { value: 'OutputTokensCost', label: 'Output Token Cost ($)' },
    ],
  },
  {
    group: 'Throughput',
    options: [
      { value: 'InputTokens', label: 'Input Tokens' },
      { value: 'OutputTokens', label: 'Output Tokens' },
    ],
  },
  {
    group: 'Requests',
    options: [
      { value: 'InvocationSuccess', label: 'Successful Invocations' },
      { value: 'InvocationFailure', label: 'Failed Invocations' },
    ],
  },
  {
    group: 'Latency',
    options: [
      { value: 'InvocationLatencyMs', label: 'Invocation Latency (ms)' },
    ],
  },
  {
    group: 'Analysis',
    options: [
      { value: 'anomaly_detection', label: 'Anomaly Detection (breach anomaly band)' },
      { value: 'trend_slope', label: 'Trend Slope (rate of change)' },
      { value: 'arima_deviation', label: 'ARIMA Seasonal Deviation (outside seasonal band)' },
    ],
  },
];

const COMPARISON_OPTIONS = [
  { value: 'GreaterThanOrEqualToThreshold', label: '>=' },
  { value: 'LessThanOrEqualToThreshold', label: '<=' },
  { value: 'GreaterThanThreshold', label: '>' },
  { value: 'LessThanThreshold', label: '<' },
];

const PERIOD_OPTIONS: { value: number; label: string }[] = [
  { value: 60, label: '1 minute' },
  { value: 300, label: '5 minutes' },
  { value: 3600, label: '1 hour' },
  { value: 86400, label: '1 day' },
];

// Analysis metrics need a base metric to analyze
const ANALYSIS_METRICS = ['anomaly_detection', 'trend_slope', 'arima_deviation'];
const BASE_METRICS = METRIC_GROUPS.flatMap((g) => g.options).filter((o) => !ANALYSIS_METRICS.includes(o.value));

function isAnalysisMetric(metric: string): boolean {
  return ANALYSIS_METRICS.includes(metric);
}

function statusBadge(state: string | undefined, action: AlertAction) {
  if (state === 'OK') {
    return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: '9999px', fontSize: '12px', fontWeight: 600, backgroundColor: '#d1fae5', color: '#059669' }}>OK</span>;
  }
  if (state === 'ALARM') {
    let label = 'ALARM';
    if (action === 'throttle') label = 'THROTTLED';
    if (action === 'suspend') label = 'SUSPENDED';
    return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: '9999px', fontSize: '12px', fontWeight: 600, backgroundColor: '#fee2e2', color: '#dc2626' }}>{label}</span>;
  }
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: '9999px', fontSize: '12px', fontWeight: 600, backgroundColor: '#e5e7eb', color: '#6b7280' }}>{state ?? 'INSUFFICIENT_DATA'}</span>;
}

const AlertsPage: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  // Form state
  const [alertName, setAlertName] = useState('');
  const [tenantId, setTenantId] = useState('');
  const [metricName, setMetricName] = useState('');
  const [baseMetric, setBaseMetric] = useState('');  // for analysis metrics
  const [thresholdMode, setThresholdMode] = useState<ThresholdMode>('absolute');
  const [thresholdValue, setThresholdValue] = useState('');
  const [comparison, setComparison] = useState('GreaterThanOrEqualToThreshold');
  const [period, setPeriod] = useState(300);
  const [action, setAction] = useState<AlertAction>('notify');
  const [email, setEmail] = useState('');
  const [autoRecoverMinutes, setAutoRecoverMinutes] = useState('');

  const fetchAlerts = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listAlerts();
      setAlerts(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  };

  const fetchProfiles = async () => {
    try {
      const data = await listProfiles();
      setProfiles(data);
    } catch { /* ignore */ }
  };

  useEffect(() => {
    fetchAlerts();
    fetchProfiles();
  }, []);

  const resetForm = () => {
    setAlertName('');
    setTenantId('');
    setMetricName('');
    setBaseMetric('');
    setThresholdMode('absolute');
    setThresholdValue('');
    setComparison('GreaterThanOrEqualToThreshold');
    setPeriod(300);
    setAction('notify');
    setEmail('');
    setAutoRecoverMinutes('');
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const actionConfig: AlertActionConfig = {};
      if (email) actionConfig.email = email;
      if (autoRecoverMinutes) actionConfig.auto_recover_minutes = Number(autoRecoverMinutes);

      // For analysis metrics, send both the analysis type and the base metric
      const effectiveMetric = isAnalysisMetric(metricName)
        ? `${metricName}:${baseMetric}`
        : metricName;

      const req: CreateAlertRequest = {
        alert_name: alertName,
        tenant_id: tenantId,
        alert_type: isAnalysisMetric(metricName) ? metricName : 'metric_threshold',
        metric_name: isAnalysisMetric(metricName) ? baseMetric : effectiveMetric,
        threshold_mode: thresholdMode,
        threshold_value: Number(thresholdValue),
        action,
        action_config: actionConfig,
        period,
      };
      if (thresholdMode === 'absolute') {
        req.comparison = comparison;
      }
      await createAlert(req);
      setShowForm(false);
      resetForm();
      await fetchAlerts();
      setSuccessMsg('Alert created successfully');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create alert');
    }
  };

  const handleDelete = async (alertId: string) => {
    if (!window.confirm('Are you sure you want to delete this alert?')) return;
    try {
      await deleteAlert(alertId);
      await fetchAlerts();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete alert');
    }
  };

  const profileName = (tid: string) => {
    const t = profiles.find((t) => t.tenant_id === tid);
    return t ? t.tenant_name : tid;
  };

  const thresholdModeLabel = (mode: ThresholdMode) => {
    if (mode === 'percentage_increase') return '% Increase';
    if (mode === 'percentage_decrease') return '% Decrease';
    return 'Absolute';
  };

  const actionLabel = (a: AlertAction) => {
    if (a === 'throttle') return 'Throttle';
    if (a === 'suspend') return 'Suspend';
    return 'Notify';
  };

  const metricLabel = (metric: string) => {
    for (const g of METRIC_GROUPS) {
      const opt = g.options.find((o) => o.value === metric);
      if (opt) return opt.label;
    }
    return metric;
  };

  const tableStyle: React.CSSProperties = { width: '100%', borderCollapse: 'collapse' };
  const thStyle: React.CSSProperties = { textAlign: 'left', padding: '10px 12px', borderBottom: '2px solid #e5e7eb', fontWeight: 600, fontSize: '14px', color: '#374151' };
  const tdStyle: React.CSSProperties = { padding: '10px 12px', borderBottom: '1px solid #f3f4f6', fontSize: '14px' };
  const inputStyle: React.CSSProperties = { padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: '6px', fontSize: '14px', width: '100%', boxSizing: 'border-box' };
  const labelStyle: React.CSSProperties = { display: 'block', marginBottom: '4px', fontSize: '14px', fontWeight: 500 };
  const radioGroupStyle: React.CSSProperties = { display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' };
  const radioLabelStyle: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: '4px', fontSize: '14px', cursor: 'pointer' };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h1 style={{ margin: 0 }}>Alerts</h1>
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
          {showForm ? 'Cancel' : 'Create Alert'}
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
            gap: '14px',
          }}
        >
          {/* Alert Name */}
          <div>
            <label style={labelStyle}>Alert Name</label>
            <input type="text" value={alertName} onChange={(e) => setAlertName(e.target.value)} required placeholder="e.g. High cost warning" style={inputStyle} />
          </div>

          {/* Tenant */}
          <div>
            <label style={labelStyle}>Profile</label>
            <select value={tenantId} onChange={(e) => setTenantId(e.target.value)} required style={inputStyle}>
              <option value="">Select a profile</option>
              {profiles.map((t) => (
                <option key={t.tenant_id} value={t.tenant_id}>{t.tenant_name}</option>
              ))}
            </select>
          </div>

          {/* Metric — grouped with optgroup */}
          <div>
            <label style={labelStyle}>What to Monitor</label>
            <select value={metricName} onChange={(e) => { setMetricName(e.target.value); if (!isAnalysisMetric(e.target.value)) setBaseMetric(''); }} required style={inputStyle}>
              <option value="">Select a metric or analysis</option>
              {METRIC_GROUPS.map((g) => (
                <optgroup key={g.group} label={g.group}>
                  {g.options.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          {/* If analysis metric selected, show base metric picker */}
          {isAnalysisMetric(metricName) && (
            <div>
              <label style={labelStyle}>Apply Analysis To</label>
              <select value={baseMetric} onChange={(e) => setBaseMetric(e.target.value)} required style={inputStyle}>
                <option value="">Select the metric to analyze</option>
                {BASE_METRICS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
              <p style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px', marginBottom: 0 }}>
                {metricName === 'anomaly_detection' && 'Alert fires when the metric breaches its anomaly detection band (2 standard deviations).'}
                {metricName === 'trend_slope' && 'Alert fires when the linear trend slope exceeds the threshold (rate of change per period).'}
                {metricName === 'arima_deviation' && 'Alert fires when the metric deviates beyond the seasonal prediction band (3 standard deviations).'}
              </p>
            </div>
          )}

          {/* Threshold Mode */}
          <div>
            <label style={labelStyle}>Threshold Mode</label>
            <div style={radioGroupStyle}>
              <label style={radioLabelStyle}>
                <input type="radio" name="thresholdMode" value="absolute" checked={thresholdMode === 'absolute'} onChange={() => setThresholdMode('absolute')} />
                Absolute Value
              </label>
              <label style={radioLabelStyle}>
                <input type="radio" name="thresholdMode" value="percentage_increase" checked={thresholdMode === 'percentage_increase'} onChange={() => setThresholdMode('percentage_increase')} />
                % Increase
              </label>
              <label style={radioLabelStyle}>
                <input type="radio" name="thresholdMode" value="percentage_decrease" checked={thresholdMode === 'percentage_decrease'} onChange={() => setThresholdMode('percentage_decrease')} />
                % Decrease
              </label>
            </div>
          </div>

          {/* Threshold Value */}
          <div>
            <label style={labelStyle}>
              Threshold {thresholdMode === 'absolute' ? 'Value' : '(%)'}
            </label>
            <input
              type="number"
              step="any"
              value={thresholdValue}
              onChange={(e) => setThresholdValue(e.target.value)}
              required
              placeholder={thresholdMode === 'absolute' ? 'e.g. 100' : 'e.g. 20'}
              style={inputStyle}
            />
          </div>

          {/* Comparison (only for absolute) */}
          {thresholdMode === 'absolute' && (
            <div>
              <label style={labelStyle}>Comparison</label>
              <select value={comparison} onChange={(e) => setComparison(e.target.value)} style={inputStyle}>
                {COMPARISON_OPTIONS.map((c) => (
                  <option key={c.value} value={c.value}>{c.label}</option>
                ))}
              </select>
            </div>
          )}

          {/* Period */}
          <div>
            <label style={labelStyle}>Evaluation Period</label>
            <select value={period} onChange={(e) => setPeriod(Number(e.target.value))} style={inputStyle}>
              {PERIOD_OPTIONS.map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>

          {/* Action */}
          <div>
            <label style={labelStyle}>Action When Triggered</label>
            <div style={radioGroupStyle}>
              <label style={radioLabelStyle}>
                <input type="radio" name="action" value="notify" checked={action === 'notify'} onChange={() => setAction('notify')} />
                Notify (email)
              </label>
              <label style={radioLabelStyle}>
                <input type="radio" name="action" value="throttle" checked={action === 'throttle'} onChange={() => setAction('throttle')} />
                Throttle Profile
              </label>
              <label style={radioLabelStyle}>
                <input type="radio" name="action" value="suspend" checked={action === 'suspend'} onChange={() => setAction('suspend')} />
                Suspend Profile
              </label>
            </div>
          </div>

          {/* Email */}
          <div>
            <label style={labelStyle}>
              Notification Email {action === 'notify' ? '' : '(optional)'}
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required={action === 'notify'}
              placeholder="user@example.com"
              style={inputStyle}
            />
          </div>

          {/* Auto-recover (throttle/suspend only) */}
          {(action === 'throttle' || action === 'suspend') && (
            <div>
              <label style={labelStyle}>Auto-Recover After (minutes, optional)</label>
              <input
                type="number"
                value={autoRecoverMinutes}
                onChange={(e) => setAutoRecoverMinutes(e.target.value)}
                placeholder="Leave blank for manual recovery only"
                style={inputStyle}
              />
            </div>
          )}

          <div>
            <button
              type="submit"
              style={{
                padding: '10px 24px',
                backgroundColor: '#2563eb',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '14px',
              }}
            >
              Create Alert
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
        <p style={{ color: '#6b7280' }}>Loading alerts...</p>
      ) : alerts.length === 0 ? (
        <p style={{ color: '#6b7280' }}>No alerts found. Create one to get started.</p>
      ) : (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Profile</th>
              <th style={thStyle}>Metric</th>
              <th style={thStyle}>Mode</th>
              <th style={thStyle}>Threshold</th>
              <th style={thStyle}>Action</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Created</th>
              <th style={thStyle}></th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((a) => (
              <tr key={a.alert_id}>
                <td style={{ ...tdStyle, fontWeight: 500 }}>{a.alert_name}</td>
                <td style={tdStyle}>{profileName(a.tenant_id)}</td>
                <td style={tdStyle}>
                  {metricLabel(a.metric_name)}
                  {a.alert_type && ANALYSIS_METRICS.includes(a.alert_type) && (
                    <span style={{ display: 'block', fontSize: '11px', color: '#6b7280' }}>
                      via {metricLabel(a.alert_type)}
                    </span>
                  )}
                </td>
                <td style={tdStyle}>{thresholdModeLabel(a.threshold_mode)}</td>
                <td style={tdStyle}>
                  {a.threshold_mode === 'absolute'
                    ? `${COMPARISON_OPTIONS.find((c) => c.value === a.comparison)?.label ?? '>='} ${a.threshold_value}`
                    : `${a.threshold_value}%`}
                </td>
                <td style={tdStyle}>{actionLabel(a.action)}</td>
                <td style={tdStyle}>{statusBadge(a.alarm_state, a.action)}</td>
                <td style={tdStyle}>
                  {a.created_at ? new Date(a.created_at).toLocaleDateString() : '-'}
                </td>
                <td style={tdStyle}>
                  <button
                    onClick={() => handleDelete(a.alert_id)}
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

export default AlertsPage;
