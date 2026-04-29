// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { listProfileTags } from '../api/profiles';
import { PREDEFINED_TAG_CATEGORIES } from '../types';

interface TagFilterBarProps {
  onFilterChange: (filters: Record<string, string>) => void;
}

const predefinedSet = new Set<string>(PREDEFINED_TAG_CATEGORIES);

const TagFilterBar: React.FC<TagFilterBarProps> = ({ onFilterChange }) => {
  const [availableTags, setAvailableTags] = useState<Record<string, string[]>>({});
  const [selectedKey, setSelectedKey] = useState('');
  const [selectedValue, setSelectedValue] = useState('');
  const [activeFilters, setActiveFilters] = useState<Record<string, string>>({});
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    listProfileTags()
      .then((resp) => setAvailableTags(resp.tags))
      .catch(() => setLoadError('Failed to load tags'));
  }, []);

  const allKeys = Object.keys(availableTags);
  const predefinedKeys = allKeys.filter((k) => predefinedSet.has(k));
  const customKeys = allKeys.filter((k) => !predefinedSet.has(k));
  const tagKeys = [...predefinedKeys, ...customKeys];
  const tagValues = selectedKey ? (availableTags[selectedKey] ?? []) : [];

  const handleAdd = () => {
    if (!selectedKey || !selectedValue) return;
    const next = { ...activeFilters, [selectedKey]: selectedValue };
    setActiveFilters(next);
    onFilterChange(next);
    setSelectedKey('');
    setSelectedValue('');
  };

  const handleRemove = (key: string) => {
    const next = { ...activeFilters };
    delete next[key];
    setActiveFilters(next);
    onFilterChange(next);
  };

  const selectStyle: React.CSSProperties = {
    padding: '6px 10px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '13px',
    backgroundColor: '#fff',
  };

  const addBtnStyle: React.CSSProperties = {
    padding: '6px 14px',
    backgroundColor: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: selectedKey && selectedValue ? 'pointer' : 'not-allowed',
    fontWeight: 600,
    fontSize: '13px',
    opacity: selectedKey && selectedValue ? 1 : 0.5,
  };

  const chipStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 10px',
    backgroundColor: '#eff6ff',
    color: '#2563eb',
    borderRadius: '9999px',
    fontSize: '12px',
    fontWeight: 500,
  };

  const chipRemoveStyle: React.CSSProperties = {
    background: 'none',
    border: 'none',
    color: '#2563eb',
    cursor: 'pointer',
    fontWeight: 700,
    fontSize: '14px',
    padding: 0,
    lineHeight: 1,
  };

  if (loadError) {
    return null; // silently skip if tags endpoint not available
  }

  if (tagKeys.length === 0) {
    return null; // no tags to filter on
  }

  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '13px', fontWeight: 500, color: '#374151' }}>Filter by tag:</span>
        <select
          value={selectedKey}
          onChange={(e) => { setSelectedKey(e.target.value); setSelectedValue(''); }}
          style={selectStyle}
        >
          <option value="">Key</option>
          {predefinedKeys.length > 0 && (
            <optgroup label="Categories">
              {predefinedKeys.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </optgroup>
          )}
          {customKeys.length > 0 && (
            <optgroup label="Custom">
              {customKeys.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </optgroup>
          )}
          {predefinedKeys.length === 0 && customKeys.length === 0 && tagKeys.map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
        <select
          value={selectedValue}
          onChange={(e) => setSelectedValue(e.target.value)}
          disabled={!selectedKey}
          style={{ ...selectStyle, opacity: selectedKey ? 1 : 0.5 }}
        >
          <option value="">Value</option>
          {tagValues.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
        <button onClick={handleAdd} disabled={!selectedKey || !selectedValue} style={addBtnStyle}>
          Add
        </button>
      </div>

      {Object.keys(activeFilters).length > 0 && (
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px' }}>
          {Object.entries(activeFilters).map(([key, value]) => (
            <span key={key} style={chipStyle}>
              {key}: {value}
              <button onClick={() => handleRemove(key)} style={chipRemoveStyle} title="Remove filter">
                x
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

export default TagFilterBar;
