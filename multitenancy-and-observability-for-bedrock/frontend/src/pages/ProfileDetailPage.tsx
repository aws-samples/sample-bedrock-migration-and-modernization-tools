// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getProfile, updateProfile, activateProfile, suspendProfile, deleteProfile } from '../api/profiles';
import type { Profile, CreateProfileRequest } from '../types';
import StatusBadge from '../components/StatusBadge';
import ProfileForm from '../components/ProfileForm';

const ProfileDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const fetchProfile = async () => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getProfile(id);
      setProfile(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load profile';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProfile();
  }, [id]);

  const handleActivate = async () => {
    if (!id) return;
    setActionLoading(true);
    try {
      await activateProfile(id);
      await fetchProfile();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to activate profile';
      setError(message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleSuspend = async () => {
    if (!id) return;
    setActionLoading(true);
    try {
      await suspendProfile(id);
      await fetchProfile();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to suspend profile';
      setError(message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!id) return;
    if (!window.confirm('Are you sure you want to delete this profile? This action cannot be undone.')) {
      return;
    }
    setActionLoading(true);
    try {
      await deleteProfile(id);
      navigate('/profiles');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete profile';
      setError(message);
      setActionLoading(false);
    }
  };

  const handleEdit = async (data: CreateProfileRequest) => {
    if (!id) return;
    await updateProfile(id, {
      tenant_name: data.tenant_name,
      tags: data.tags,
    });
    setEditing(false);
    await fetchProfile();
  };

  const cardStyle: React.CSSProperties = {
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    padding: '24px',
    backgroundColor: '#fff',
    marginBottom: '20px',
  };

  const fieldRow: React.CSSProperties = {
    display: 'flex',
    marginBottom: '12px',
  };

  const fieldLabel: React.CSSProperties = {
    width: '180px',
    fontWeight: 600,
    color: '#374151',
    fontSize: '14px',
    flexShrink: 0,
  };

  const fieldValue: React.CSSProperties = {
    fontSize: '14px',
    color: '#1f2937',
    fontFamily: 'inherit',
  };

  const btnStyle = (bg: string, color: string, border?: string): React.CSSProperties => ({
    padding: '8px 16px',
    backgroundColor: bg,
    color,
    border: border ?? 'none',
    borderRadius: '6px',
    cursor: actionLoading ? 'not-allowed' : 'pointer',
    fontWeight: 600,
    fontSize: '14px',
  });

  if (loading) {
    return (
      <div style={{ maxWidth: '800px', margin: '0 auto', padding: '24px' }}>
        <p style={{ color: '#6b7280' }}>Loading profile...</p>
      </div>
    );
  }

  if (error && !profile) {
    return (
      <div style={{ maxWidth: '800px', margin: '0 auto', padding: '24px' }}>
        <Link to="/profiles" style={{ color: '#2563eb', textDecoration: 'none', fontSize: '14px' }}>
          &larr; Back to Profiles
        </Link>
        <div
          style={{
            padding: '12px 16px',
            backgroundColor: '#fef2f2',
            color: '#dc2626',
            borderRadius: '6px',
            marginTop: '16px',
            fontSize: '14px',
          }}
        >
          {error}
        </div>
      </div>
    );
  }

  if (!profile) return null;

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '24px' }}>
      <Link to="/profiles" style={{ color: '#2563eb', textDecoration: 'none', fontSize: '14px' }}>
        &larr; Back to Profiles
      </Link>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '16px 0' }}>
        <h1 style={{ margin: 0 }}>{profile.tenant_name}</h1>
        <StatusBadge status={profile.status} />
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

      {editing ? (
        <ProfileForm
          onSubmit={handleEdit}
          onCancel={() => setEditing(false)}
          initialValues={profile}
          isEdit
        />
      ) : (
        <>
          <div style={cardStyle}>
            <div style={fieldRow}>
              <span style={fieldLabel}>Profile ID</span>
              <span style={{ ...fieldValue, fontFamily: 'monospace', fontSize: '13px' }}>
                {profile.tenant_id}
              </span>
            </div>
            <div style={fieldRow}>
              <span style={fieldLabel}>Name</span>
              <span style={fieldValue}>{profile.tenant_name}</span>
            </div>
            <div style={fieldRow}>
              <span style={fieldLabel}>Status</span>
              <StatusBadge status={profile.status} />
            </div>
            <div style={fieldRow}>
              <span style={fieldLabel}>Model ID</span>
              <span style={{ ...fieldValue, fontFamily: 'monospace', fontSize: '13px' }}>
                {profile.model_id}
              </span>
            </div>
            <div style={fieldRow}>
              <span style={fieldLabel}>Region</span>
              <span style={fieldValue}>{profile.region}</span>
            </div>
            {profile.inference_profile_id && (
              <div style={fieldRow}>
                <span style={fieldLabel}>Inference Profile ID</span>
                <span style={{ ...fieldValue, fontFamily: 'monospace', fontSize: '13px' }}>
                  {profile.inference_profile_id}
                </span>
              </div>
            )}
            {profile.inference_profile_arn && (
              <div style={fieldRow}>
                <span style={fieldLabel}>ARN</span>
                <span style={{ ...fieldValue, fontFamily: 'monospace', fontSize: '12px', wordBreak: 'break-all' }}>
                  {profile.inference_profile_arn}
                </span>
              </div>
            )}
            {profile.profile_strategy && (
              <div style={fieldRow}>
                <span style={fieldLabel}>Strategy</span>
                <span style={fieldValue}>{profile.profile_strategy}</span>
              </div>
            )}
            {profile.tags && Object.keys(profile.tags).length > 0 && (
              <div style={fieldRow}>
                <span style={fieldLabel}>Tags</span>
                <div>
                  {Object.entries(profile.tags).map(([key, val]) => {
                    const display = Array.isArray(val) ? val.join(', ') : String(val);
                    return (
                      <span key={key} style={{ display: 'inline-block', padding: '2px 8px', backgroundColor: '#f3f4f6', color: '#374151', borderRadius: '4px', fontSize: '12px', marginRight: '6px', marginBottom: '4px' }}>
                        <strong>{key}:</strong> {display}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
            {profile.created_at && (
              <div style={fieldRow}>
                <span style={fieldLabel}>Created</span>
                <span style={fieldValue}>{new Date(profile.created_at).toLocaleString()}</span>
              </div>
            )}
            {profile.updated_at && (
              <div style={fieldRow}>
                <span style={fieldLabel}>Updated</span>
                <span style={fieldValue}>{new Date(profile.updated_at).toLocaleString()}</span>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {profile.status !== 'active' && (
              <button
                onClick={handleActivate}
                disabled={actionLoading}
                style={btnStyle('#16a34a', '#fff')}
              >
                Activate
              </button>
            )}
            {profile.status === 'active' && (
              <button
                onClick={handleSuspend}
                disabled={actionLoading}
                style={btnStyle('#f59e0b', '#fff')}
              >
                Suspend
              </button>
            )}
            <button
              onClick={() => setEditing(true)}
              disabled={actionLoading}
              style={btnStyle('#fff', '#374151', '1px solid #d1d5db')}
            >
              Edit
            </button>
            <button
              onClick={handleDelete}
              disabled={actionLoading}
              style={btnStyle('#dc2626', '#fff')}
            >
              Delete
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default ProfileDetailPage;
