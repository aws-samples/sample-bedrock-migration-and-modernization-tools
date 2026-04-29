// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useMemo } from 'react';
import { listModels, getPricing } from '../api/discovery';
import type { Model, PricingInfo } from '../types';

const REGIONS = [
  'us-east-1',
  'us-west-2',
  'eu-west-1',
  'eu-west-2',
  'eu-central-1',
  'ap-northeast-1',
  'ap-southeast-1',
  'ap-southeast-2',
  'ca-central-1',
];

const DiscoveryPage: React.FC = () => {
  const [region, setRegion] = useState('us-east-1');
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState('');

  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);
  const [pricing, setPricing] = useState<PricingInfo | null>(null);
  const [pricingLoading, setPricingLoading] = useState(false);
  const [pricingError, setPricingError] = useState<string | null>(null);

  const fetchModels = async (selectedRegion: string) => {
    setLoading(true);
    setError(null);
    setExpandedModelId(null);
    setPricing(null);
    try {
      const data = await listModels(selectedRegion);
      setModels(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load models';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleRegionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newRegion = e.target.value;
    setRegion(newRegion);
    fetchModels(newRegion);
  };

  const handleRowClick = async (modelId: string) => {
    if (expandedModelId === modelId) {
      setExpandedModelId(null);
      setPricing(null);
      return;
    }

    setExpandedModelId(modelId);
    setPricing(null);
    setPricingError(null);
    setPricingLoading(true);
    try {
      const data = await getPricing(modelId, region);
      setPricing(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load pricing';
      setPricingError(message);
    } finally {
      setPricingLoading(false);
    }
  };

  const filteredModels = useMemo(() => {
    if (!filter.trim()) return models;
    const q = filter.toLowerCase();
    return models.filter(
      (m) =>
        (m.model_name ?? '').toLowerCase().includes(q) ||
        (m.provider_name ?? '').toLowerCase().includes(q) ||
        m.model_id.toLowerCase().includes(q)
    );
  }, [models, filter]);

  const thStyle: React.CSSProperties = {
    textAlign: 'left',
    padding: '10px 12px',
    borderBottom: '2px solid #e5e7eb',
    fontWeight: 600,
    fontSize: '14px',
    color: '#374151',
  };

  const tdStyle: React.CSSProperties = {
    padding: '10px 12px',
    borderBottom: '1px solid #f3f4f6',
    fontSize: '14px',
  };

  const formatCost = (input?: number, output?: number): string => {
    if (input == null || output == null) return 'N/A';
    const cost = input + output;
    return `$${cost.toFixed(4)}`;
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      <h1 style={{ marginBottom: '20px' }}>Model Discovery</h1>

      <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
        <div>
          <label style={{ fontWeight: 600, fontSize: '14px', marginRight: '8px' }}>Region:</label>
          <select
            value={region}
            onChange={handleRegionChange}
            style={{
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '14px',
            }}
          >
            {REGIONS.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        <input
          type="text"
          placeholder="Filter by model name or provider..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{
            padding: '8px 12px',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            fontSize: '14px',
            flex: 1,
            minWidth: '200px',
          }}
        />
      </div>

      {error && (
        <div
          style={{
            padding: '12px 16px',
            backgroundColor: '#fef2f2',
            color: '#dc2626',
            borderRadius: '6px',
            marginBottom: '16px',
            fontSize: '14px',
          }}
        >
          {error}
        </div>
      )}

      {loading ? (
        <p style={{ color: '#6b7280' }}>Loading models...</p>
      ) : models.length === 0 ? (
        <p style={{ color: '#6b7280' }}>
          Select a region to discover available models.
        </p>
      ) : filteredModels.length === 0 ? (
        <p style={{ color: '#6b7280' }}>No models match your filter.</p>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Model Name</th>
              <th style={thStyle}>Provider</th>
              <th style={thStyle}>Model ID</th>
              <th style={thStyle}>Modalities</th>
              <th style={thStyle}>Streaming</th>
              <th style={thStyle}>Status</th>
            </tr>
          </thead>
          <tbody>
            {filteredModels.map((m) => (
              <React.Fragment key={m.model_id}>
                <tr
                  onClick={() => handleRowClick(m.model_id)}
                  style={{
                    cursor: 'pointer',
                    backgroundColor: expandedModelId === m.model_id ? '#eff6ff' : undefined,
                  }}
                >
                  <td style={tdStyle}>{m.model_name ?? '-'}</td>
                  <td style={tdStyle}>{m.provider_name ?? '-'}</td>
                  <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '13px' }}>
                    {m.model_id}
                  </td>
                  <td style={tdStyle}>
                    {m.input_modalities?.join(', ') ?? '-'} &rarr; {m.output_modalities?.join(', ') ?? '-'}
                  </td>
                  <td style={tdStyle}>{m.response_streaming_supported ? 'Yes' : 'No'}</td>
                  <td style={tdStyle}>{m.model_lifecycle_status ?? '-'}</td>
                </tr>
                {expandedModelId === m.model_id && (
                  <tr>
                    <td colSpan={6} style={{ padding: '16px 24px', backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                      <h4 style={{ marginTop: 0, marginBottom: '12px' }}>Pricing Details</h4>
                      {pricingLoading ? (
                        <p style={{ color: '#6b7280', margin: 0 }}>Loading pricing...</p>
                      ) : pricingError ? (
                        <p style={{ color: '#dc2626', margin: 0 }}>{pricingError}</p>
                      ) : pricing ? (
                        <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 16px', fontSize: '14px' }}>
                          <span style={{ fontWeight: 600 }}>Input Cost (per 1M tokens):</span>
                          <span>${pricing.input_cost?.toFixed(6) ?? 'N/A'}</span>
                          <span style={{ fontWeight: 600 }}>Output Cost (per 1M tokens):</span>
                          <span>${pricing.output_cost?.toFixed(6) ?? 'N/A'}</span>
                          <span style={{ fontWeight: 600 }}>Pricing Source:</span>
                          <span>{pricing.pricing_source ?? '-'}</span>
                          <span style={{ fontWeight: 600 }}>From Cache:</span>
                          <span>{pricing.from_cache ? 'Yes' : 'No'}</span>
                          <span style={{ fontWeight: 600 }}>1M input + 1M output:</span>
                          <span>
                            {formatCost(pricing.input_cost, pricing.output_cost)}
                          </span>
                        </div>
                      ) : (
                        <p style={{ color: '#6b7280', margin: 0 }}>No pricing data available.</p>
                      )}
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default DiscoveryPage;
