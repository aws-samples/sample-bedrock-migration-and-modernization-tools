// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import { PieChart as RechartsPieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { PieLabelRenderProps } from 'recharts';

interface PieChartDataItem {
  name: string;
  value: number;
  color: string;
}

interface PieChartProps {
  title: string;
  data: PieChartDataItem[];
  loading?: boolean;
  error?: string | null;
}

const PieChartCard: React.FC<PieChartProps> = ({ title, data, loading, error }) => {
  const total = data.reduce((s, d) => s + d.value, 0);

  return (
    <div style={{
      border: '1px solid #e5e7eb',
      borderRadius: '8px',
      padding: '16px',
      backgroundColor: '#fff',
    }}>
      <h3 style={{ margin: '0 0 12px', fontSize: '15px', fontWeight: 600 }}>{title}</h3>
      {loading && <p style={{ color: '#6b7280', fontSize: '13px' }}>Loading metrics...</p>}
      {error && <p style={{ color: '#dc2626', fontSize: '13px' }}>{error}</p>}
      {!loading && !error && total === 0 && (
        <p style={{ color: '#9ca3af', fontSize: '13px' }}>No data available</p>
      )}
      {!loading && !error && total > 0 && (
        <ResponsiveContainer width="100%" height={220}>
          <RechartsPieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={75}
              label={(props: PieLabelRenderProps) => `${props.name ?? ''} ${(((props.percent as number) ?? 0) * 100).toFixed(1)}%`}
              labelLine={false}
            >
              {data.map((entry, index) => (
                <Cell key={index} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value) => Number(value).toLocaleString()}
            />
            <Legend />
          </RechartsPieChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default PieChartCard;
