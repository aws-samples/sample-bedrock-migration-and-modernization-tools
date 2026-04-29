// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

export const PREDEFINED_TAG_CATEGORIES = [
  'Tenant', 'Environment', 'Region', 'User', 'Model', 'Application',
] as const;
export type PredefinedTagCategory = typeof PREDEFINED_TAG_CATEGORIES[number];

export interface Profile {
  tenant_id: string;
  tenant_name: string;
  status: string;
  model_id: string;
  region: string;
  inference_profile_id: string;
  inference_profile_arn: string;
  profile_strategy: string;
  tags: Record<string, string | string[]>;
  capacity_limit?: number;
  created_at: string;
  updated_at: string;
}

export interface ProfileListResponse {
  profiles: Profile[];
  count: number;
}

export interface CreateProfileRequest {
  tenant_name: string;
  model_id: string;
  region: string;
  tags?: Record<string, string | string[]>;
  capacity_limit?: number;
}

export interface Model {
  model_id: string;
  model_name: string;
  provider_name: string;
  input_modalities: string[];
  output_modalities: string[];
  response_streaming_supported: boolean;
  model_lifecycle_status: string;
  inference_types_supported: string[];
}

export interface ModelListResponse {
  models: Model[];
  count: number;
  region: string;
}

export interface PricingInfo {
  cache_key: string;
  model_id: string;
  region: string;
  input_cost: number;
  output_cost: number;
  pricing_source: string;
  from_cache: boolean;
}

export interface Message {
  role: string;
  content: MessageContent[];
}

export interface MessageContent {
  text: string;
}

export interface InvokeRequest {
  messages: Message[];
}

export interface InvokeResponse {
  output: {
    role: string;
    content: { text: string }[];
  };
  usage: {
    inputTokens: number;
    outputTokens: number;
  };
  cost: {
    inputCost: number;
    outputCost: number;
    totalCost: number;
  };
  latencyMs: number;
}

// --- Dashboard Types ---
export interface DashboardWidget {
  widget_id: string;
  type: 'timeseries' | 'bar' | 'number' | 'single_value' | 'pie';
  title: string;
  metrics: string[];
  dimensions: string[];
  stat: string;
  available_stats: string[];
  period: number;
  available_periods: number[];
  analysis: string | null;
  available_analysis: string[];
  position: { x: number; y: number; w: number; h: number };
  use_search?: boolean;
  stacked?: boolean;
  expression?: string;
}

export interface Dashboard {
  dashboard_id: string;
  tenant_id: string;
  tenant_ids?: string[];
  template_id: string;
  dashboard_name: string;
  cw_dashboard_name: string;
  console_url: string;
  region: string;
  widgets: DashboardWidget[];
  widget_overrides: Record<string, Partial<DashboardWidget>>;
  tag_dimensions?: string[];
  created_at: string;
  updated_at: string;
}

export interface CreateDashboardRequest {
  tenant_id: string;
  template_id: string;
  dashboard_name?: string;
  widget_overrides?: Record<string, Partial<DashboardWidget>>;
  tag_dimensions?: string[];
  tenant_ids?: string[];
  widget_ids?: string[];
}

export interface UpdateDashboardRequest {
  dashboard_name?: string;
  widget_overrides?: Record<string, Partial<DashboardWidget>>;
  tag_dimensions?: string[];
}

// --- Alert Types ---
export type AlertAction = 'notify' | 'throttle' | 'suspend';
export type ThresholdMode = 'absolute' | 'percentage_increase' | 'percentage_decrease';

export interface AlertActionConfig {
  email?: string;
  auto_recover_minutes?: number;
}

export interface Alert {
  alert_id: string;
  alert_name: string;
  tenant_id: string;
  alert_type: string;
  metric_name: string;
  threshold_mode: ThresholdMode;
  threshold_value: number;
  comparison: string;
  action: AlertAction;
  action_config: AlertActionConfig;
  period: number;
  dashboard_id?: string;
  widget_id?: string;
  tag_dimensions?: string[];
  alarm_name: string;
  topic_arn?: string;
  region: string;
  alarm_state?: string;
  created_at: string;
  updated_at: string;
}

export interface CreateAlertRequest {
  alert_name: string;
  tenant_id: string;
  alert_type: string;
  metric_name: string;
  threshold_mode: ThresholdMode;
  threshold_value: number;
  comparison?: string;
  action: AlertAction;
  action_config: AlertActionConfig;
  period?: number;
  dashboard_id?: string;
  widget_id?: string;
  tag_dimensions?: string[];
}

// --- Cost Explorer Types ---
export interface ProfileCost {
  tenant_id: string;
  period_start: string;
  period_end: string;
  cost: number;
  usage_quantity: number;
  currency: string;
}

// --- Metric Query Types ---
export interface MetricDatapoint {
  timestamp: string;
  value: number;
}

export interface MetricQueryResponse {
  profile_id: string;
  metric_name: string;
  stat: string;
  period: number;
  datapoints: MetricDatapoint[];
  count: number;
}
