// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listProfiles, createProfile, deleteProfile } from '../api/profiles';
import type { Profile, CreateProfileRequest } from '../types';
import { PREDEFINED_TAG_CATEGORIES } from '../types';
import StatusBadge from '../components/StatusBadge';
import ProfileForm from '../components/ProfileForm';
import SuccessBanner from '../components/SuccessBanner';
import TagFilterBar from '../components/TagFilterBar';

const ProfilesPage: React.FC = () => {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [tagFilters, setTagFilters] = useState<Record<string, string>>({});

  const fetchProfiles = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listProfiles(tagFilters);
      setProfiles(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load profiles';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProfiles();
  }, [tagFilters]);

  const handleCreate = async (data: CreateProfileRequest) => {
    await createProfile(data);
    setShowForm(false);
    await fetchProfiles();
    setSuccessMsg('Profile created successfully');
  };

  const handleDelete = async (profileId: string) => {
    if (!window.confirm('Are you sure you want to delete this profile?')) return;
    try {
      await deleteProfile(profileId);
      await fetchProfiles();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete profile';
      setError(message);
    }
  };

  const tableStyle: React.CSSProperties = {
    width: '100%',
    borderCollapse: 'collapse',
  };

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

  const tagBadgeStyle: React.CSSProperties = {
    display: 'inline-block',
    padding: '2px 8px',
    backgroundColor: '#f3f4f6',
    color: '#374151',
    borderRadius: '4px',
    fontSize: '11px',
    marginRight: '4px',
    marginBottom: '2px',
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h1 style={{ margin: 0 }}>Profiles</h1>
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
          {showForm ? 'Cancel' : 'Create Profile'}
        </button>
      </div>

      {showForm && (
        <ProfileForm
          onSubmit={handleCreate}
          onCancel={() => setShowForm(false)}
        />
      )}

      <TagFilterBar onFilterChange={setTagFilters} />

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
        <p style={{ color: '#6b7280' }}>Loading profiles...</p>
      ) : profiles.length === 0 ? (
        <p style={{ color: '#6b7280' }}>No profiles found. Create one to get started.</p>
      ) : (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Model</th>
              <th style={thStyle}>Region</th>
              <th style={thStyle}>Strategy</th>
              <th style={thStyle}>Tags</th>
              <th style={thStyle}>Created</th>
              <th style={thStyle}></th>
            </tr>
          </thead>
          <tbody>
            {profiles.map((t) => (
              <tr key={t.tenant_id} style={{ cursor: 'pointer' }}>
                <td style={tdStyle}>
                  <Link
                    to={`/profiles/${t.tenant_id}`}
                    style={{ color: '#2563eb', textDecoration: 'none', fontWeight: 500 }}
                  >
                    {t.tenant_name}
                  </Link>
                </td>
                <td style={tdStyle}>
                  <StatusBadge status={t.status} />
                </td>
                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '13px' }}>
                  {t.model_id}
                </td>
                <td style={tdStyle}>{t.region}</td>
                <td style={tdStyle}>{t.profile_strategy ?? '-'}</td>
                <td style={tdStyle}>
                  {t.tags && Object.keys(t.tags).length > 0
                    ? Object.entries(t.tags).map(([k, v]) => {
                        const isPredefined = (PREDEFINED_TAG_CATEGORIES as readonly string[]).includes(k);
                        const displayValue = Array.isArray(v) ? v.join(', ') : v;
                        return (
                          <span
                            key={k}
                            style={{
                              ...tagBadgeStyle,
                              backgroundColor: isPredefined ? '#dbeafe' : '#f3f4f6',
                              color: isPredefined ? '#1d4ed8' : '#374151',
                            }}
                          >
                            {k}: {displayValue}
                          </span>
                        );
                      })
                    : '-'}
                </td>
                <td style={tdStyle}>
                  {t.created_at ? new Date(t.created_at).toLocaleDateString() : '-'}
                </td>
                <td style={tdStyle}>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(t.tenant_id);
                    }}
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

export default ProfilesPage;
