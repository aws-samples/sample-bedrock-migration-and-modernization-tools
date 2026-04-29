// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

interface StatusBadgeProps {
  status: string;
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  const normalized = status.toLowerCase();
  let className = 'status-badge';

  if (normalized === 'active') {
    className += ' status-active';
  } else if (normalized === 'suspended') {
    className += ' status-suspended';
  } else if (normalized === 'throttled') {
    className += ' status-throttled';
  }

  return <span className={className}>{status}</span>;
}
