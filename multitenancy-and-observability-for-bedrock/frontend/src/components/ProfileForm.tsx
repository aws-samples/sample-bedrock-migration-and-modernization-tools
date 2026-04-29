// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';
import type { Profile, CreateProfileRequest } from '../types';
import { PREDEFINED_TAG_CATEGORIES } from '../types';

interface ProfileFormProps {
  onSubmit: (data: CreateProfileRequest) => Promise<void>;
  onCancel: () => void;
  initialValues?: Partial<Profile>;
  isEdit?: boolean;
}

const ProfileForm: React.FC<ProfileFormProps> = ({
  onSubmit,
  onCancel,
  initialValues,
  isEdit = false,
}) => {
  const [profileName, setProfileName] = useState(initialValues?.tenant_name ?? '');
  const [modelId, setModelId] = useState(initialValues?.model_id ?? '');
  const [region, setRegion] = useState(initialValues?.region ?? 'us-east-1');
  const [capacityLimit, setCapacityLimit] = useState<string>(
    initialValues?.capacity_limit ? String(initialValues.capacity_limit) : ''
  );

  // Predefined tag category values
  const [predefinedTags, setPredefinedTags] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    for (const cat of PREDEFINED_TAG_CATEGORIES) {
      const existing = initialValues?.tags?.[cat];
      initial[cat] = Array.isArray(existing) ? existing.join(' / ') : (existing as string) ?? '';
    }
    return initial;
  });

  // Custom (non-predefined) tags
  const [customTags, setCustomTags] = useState<Array<{ key: string; value: string }>>(() => {
    if (!initialValues?.tags) return [];
    const rows: Array<{ key: string; value: string }> = [];
    const predefinedSet = new Set<string>(PREDEFINED_TAG_CATEGORIES);
    for (const [key, value] of Object.entries(initialValues.tags)) {
      if (predefinedSet.has(key)) continue;
      if (Array.isArray(value)) {
        for (const v of value) {
          rows.push({ key, value: v });
        }
      } else {
        rows.push({ key, value: value as string });
      }
    }
    return rows;
  });

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addCustomTag = () => {
    setCustomTags((prev) => [...prev, { key: '', value: '' }]);
  };

  const removeCustomTag = (index: number) => {
    setCustomTags((prev) => prev.filter((_, i) => i !== index));
  };

  const updateCustomTag = (index: number, field: 'key' | 'value', val: string) => {
    setCustomTags((prev) =>
      prev.map((tag, i) => (i === index ? { ...tag, [field]: val } : tag))
    );
  };

  const updatePredefinedTag = (category: string, value: string) => {
    setPredefinedTags((prev) => ({ ...prev, [category]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!profileName.trim()) {
      setError('Profile name is required.');
      return;
    }
    if (!isEdit && !modelId.trim()) {
      setError('Model ID is required.');
      return;
    }

    // Build tags record from predefined + custom
    const tagsRecord: Record<string, string | string[]> = {};

    // Add predefined tags (skip empty values — backend will auto-fill Tenant)
    for (const cat of PREDEFINED_TAG_CATEGORIES) {
      const val = predefinedTags[cat]?.trim();
      if (val) {
        tagsRecord[cat] = val;
      }
    }

    // Add custom tags (group duplicate keys into arrays)
    for (const tag of customTags) {
      const k = tag.key.trim();
      if (!k) continue;
      const v = tag.value.trim();
      if (k in tagsRecord) {
        const existing = tagsRecord[k];
        if (Array.isArray(existing)) {
          existing.push(v);
        } else {
          tagsRecord[k] = [existing, v];
        }
      } else {
        tagsRecord[k] = v;
      }
    }

    const data: CreateProfileRequest = {
      tenant_name: profileName.trim(),
      model_id: modelId.trim(),
      region: region.trim(),
      tags: tagsRecord,
    };
    const parsedLimit = Number(capacityLimit);
    if (parsedLimit > 0) {
      data.capacity_limit = parsedLimit;
    }

    setSubmitting(true);
    try {
      await onSubmit(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Submission failed';
      setError(message);
    } finally {
      setSubmitting(false);
    }
  };

  const fieldStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    marginBottom: '12px',
  };

  const labelStyle: React.CSSProperties = {
    fontWeight: 600,
    marginBottom: '4px',
    fontSize: '14px',
  };

  const inputStyle: React.CSSProperties = {
    padding: '8px 10px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
  };

  const sectionLabelStyle: React.CSSProperties = {
    fontWeight: 600,
    fontSize: '14px',
    marginBottom: '8px',
    color: '#374151',
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '20px',
        marginBottom: '20px',
        backgroundColor: '#fafafa',
      }}
    >
      <h3 style={{ marginTop: 0 }}>{isEdit ? 'Edit Profile' : 'Create Profile'}</h3>

      {error && (
        <div style={{ color: '#dc2626', marginBottom: '12px', fontSize: '14px' }}>
          {error}
        </div>
      )}

      <div style={fieldStyle}>
        <label style={labelStyle}>Profile Name *</label>
        <input
          type="text"
          value={profileName}
          onChange={(e) => setProfileName(e.target.value)}
          placeholder="My Profile"
          style={inputStyle}
          required
        />
      </div>

      <div style={fieldStyle}>
        <label style={labelStyle}>Model ID *</label>
        <input
          type="text"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          placeholder="anthropic.claude-3-5-sonnet-20241022-v2:0"
          style={inputStyle}
          disabled={isEdit}
          required={!isEdit}
        />
      </div>

      <div style={fieldStyle}>
        <label style={labelStyle}>Region</label>
        <input
          type="text"
          value={region}
          onChange={(e) => setRegion(e.target.value)}
          style={inputStyle}
          disabled={isEdit}
        />
      </div>

      <div style={fieldStyle}>
        <label style={labelStyle}>Capacity Limit (req/min)</label>
        <input
          type="number"
          value={capacityLimit}
          onChange={(e) => setCapacityLimit(e.target.value)}
          placeholder="e.g. 1000 — used for capacity utilization charts"
          style={inputStyle}
          min={1}
        />
        <span style={{ fontSize: '12px', color: '#6b7280', marginTop: '2px' }}>
          Max requests per minute for this profile. Used in capacity utilization pie charts.
        </span>
      </div>

      {/* Predefined Tag Categories */}
      <div style={{ marginBottom: '16px' }}>
        <div style={sectionLabelStyle}>Tag Categories</div>
        {PREDEFINED_TAG_CATEGORIES.map((cat) => (
          <div
            key={cat}
            style={{ display: 'flex', gap: '8px', marginBottom: '6px', alignItems: 'center' }}
          >
            <label
              style={{
                width: '120px',
                fontSize: '13px',
                fontWeight: 500,
                color: '#1d4ed8',
                flexShrink: 0,
              }}
            >
              {cat}
            </label>
            <input
              type="text"
              placeholder={cat === 'Tenant' ? 'Auto-filled from profile name if empty' : `Enter ${cat.toLowerCase()}`}
              value={predefinedTags[cat] || ''}
              onChange={(e) => updatePredefinedTag(cat, e.target.value)}
              style={{ ...inputStyle, flex: 1 }}
            />
          </div>
        ))}
      </div>

      {/* Custom Tags */}
      <div style={{ marginBottom: '12px' }}>
        <div style={sectionLabelStyle}>Custom Tags</div>
        {customTags.map((tag, idx) => (
          <div
            key={idx}
            style={{ display: 'flex', gap: '8px', marginBottom: '6px', alignItems: 'center' }}
          >
            <input
              type="text"
              placeholder="Key"
              value={tag.key}
              onChange={(e) => updateCustomTag(idx, 'key', e.target.value)}
              style={{ ...inputStyle, flex: 1 }}
            />
            <input
              type="text"
              placeholder="Value"
              value={tag.value}
              onChange={(e) => updateCustomTag(idx, 'value', e.target.value)}
              style={{ ...inputStyle, flex: 1 }}
            />
            <button
              type="button"
              onClick={() => removeCustomTag(idx)}
              style={{
                padding: '6px 10px',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                cursor: 'pointer',
                backgroundColor: '#fff',
              }}
            >
              Remove
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={addCustomTag}
          style={{
            padding: '6px 12px',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            cursor: 'pointer',
            backgroundColor: '#fff',
            fontSize: '13px',
          }}
        >
          + Add Custom Tag
        </button>
      </div>

      <div style={{ display: 'flex', gap: '10px', marginTop: '16px' }}>
        <button
          type="submit"
          disabled={submitting}
          style={{
            padding: '8px 18px',
            backgroundColor: '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            cursor: submitting ? 'not-allowed' : 'pointer',
            fontWeight: 600,
          }}
        >
          {submitting ? 'Saving...' : isEdit ? 'Save Changes' : 'Create Profile'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          style={{
            padding: '8px 18px',
            backgroundColor: '#fff',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            cursor: 'pointer',
          }}
        >
          Cancel
        </button>
      </div>
    </form>
  );
};

export default ProfileForm;
