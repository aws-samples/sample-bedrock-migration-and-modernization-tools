// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { backendFetch } from './client';
import type { Profile, ProfileListResponse, CreateProfileRequest } from '../types';

export async function listProfiles(tagFilters?: Record<string, string>): Promise<Profile[]> {
  const params = new URLSearchParams();
  if (tagFilters && Object.keys(tagFilters).length > 0) {
    params.set('tag_filters', Object.entries(tagFilters).map(([k,v]) => `${k}:${v}`).join(','));
  }
  const path = params.toString() ? `/profiles?${params}` : '/profiles';
  const data = await backendFetch(path) as ProfileListResponse;
  return data.profiles;
}

export interface ProfileTagsResponse {
  tags: Record<string, string[]>;
  predefined_categories: string[];
}

export async function listProfileTags(): Promise<ProfileTagsResponse> {
  return backendFetch('/profiles/tags') as Promise<ProfileTagsResponse>;
}

export function getProfile(id: string): Promise<Profile> {
  return backendFetch(`/profiles/${id}`) as Promise<Profile>;
}

export function createProfile(req: CreateProfileRequest): Promise<Profile> {
  return backendFetch('/profiles', {
    method: 'POST',
    body: JSON.stringify(req),
  }) as Promise<Profile>;
}

export function updateProfile(id: string, body: Partial<CreateProfileRequest>): Promise<Profile> {
  return backendFetch(`/profiles/${id}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  }) as Promise<Profile>;
}

export function deleteProfile(id: string): Promise<void> {
  return backendFetch(`/profiles/${id}`, { method: 'DELETE' }) as Promise<void>;
}

export function activateProfile(id: string): Promise<Profile> {
  return backendFetch(`/profiles/${id}/activate`, { method: 'POST' }) as Promise<Profile>;
}

export function suspendProfile(id: string): Promise<Profile> {
  return backendFetch(`/profiles/${id}/suspend`, { method: 'POST' }) as Promise<Profile>;
}
