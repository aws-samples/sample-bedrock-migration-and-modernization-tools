// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { gatewayFetch } from './client';
import type { Message, InvokeResponse } from '../types';

export function invoke(tenantId: string, messages: Message[]): Promise<InvokeResponse> {
  return gatewayFetch('/invoke', {
    method: 'POST',
    headers: { 'Tenant-Id': tenantId },
    body: JSON.stringify({ messages }),
  }) as Promise<InvokeResponse>;
}
