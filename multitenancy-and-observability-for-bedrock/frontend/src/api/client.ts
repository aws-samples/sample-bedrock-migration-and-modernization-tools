// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

const BACKEND_API_URL = import.meta.env.VITE_BACKEND_API_URL as string;
const GATEWAY_API_URL = import.meta.env.VITE_GATEWAY_API_URL as string;

export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, body: unknown) {
    const detail = (body && typeof body === 'object' && 'error' in body)
      ? (body as { error: string }).error
      : `status ${status}`;
    super(`API error: ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

async function baseFetch(baseUrl: string, path: string, options: RequestInit = {}): Promise<unknown> {
  const url = `${baseUrl}${path}`;
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> | undefined),
  };

  const response = await fetch(url, { ...options, headers });

  if (response.status === 204) {
    return undefined;
  }

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError(response.status, body);
  }

  return body;
}

export function backendFetch(path: string, options?: RequestInit): Promise<unknown> {
  return baseFetch(BACKEND_API_URL, path, options);
}

export function gatewayFetch(path: string, options?: RequestInit): Promise<unknown> {
  return baseFetch(GATEWAY_API_URL, path, options);
}
