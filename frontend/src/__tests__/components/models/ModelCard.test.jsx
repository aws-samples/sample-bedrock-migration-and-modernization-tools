import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ModelCard } from '@/components/models/ModelCard'

// Mock useTheme
vi.mock('@/components/layout/ThemeProvider', () => ({
  useTheme: () => ({ theme: 'light' })
}))

// Mock useComparisonStore
vi.mock('@/stores/comparisonStore', () => ({
  useComparisonStore: () => ({
    toggleModel: vi.fn(),
    isModelSelected: () => false
  })
}))

// Mock analytics
vi.mock('@/services/analytics', () => ({
  trackEvent: vi.fn()
}))

// Mock filters utility
vi.mock('@/utils/filters', () => ({
  getCrisGeoScopes: () => ['US', 'EU']
}))

// Sample model data with consumption options
const mockModelWithConsumptionOptions = {
  model_id: 'anthropic.claude-3-sonnet',
  model_name: 'Claude 3 Sonnet',
  model_provider: 'Anthropic',
  model_modalities: {
    input_modalities: ['TEXT', 'IMAGE'],
    output_modalities: ['TEXT']
  },
  model_lifecycle: { status: 'ACTIVE' },
  model_capabilities: ['chat', 'analysis'],
  streaming_supported: true,
  in_region: ['us-east-1', 'us-west-2'],
  cross_region_inference: {
    supported: true,
    source_regions: ['us-east-1']
  },
  mantle_inference: {
    supported: false
  },
  provisioned_throughput: {
    supported: true
  },
  consumption_options: ['on_demand', 'batch', 'cross_region_inference', 'provisioned_throughput'],
  converse_data: {
    context_window: 200000,
    max_output_tokens: 4096
  },
  model_pricing: {
    is_pricing_available: true
  }
}

describe('ModelCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('does not render consumption option tags', () => {
    // Arrange & Act
    render(
      <ModelCard 
        model={mockModelWithConsumptionOptions}
        onViewDetails={vi.fn()}
        onCompare={vi.fn()}
        onToggleFavorite={vi.fn()}
        isFavorite={false}
        preferredRegion="us-east-1"
      />
    )
    
    // Assert - Consumption tags should NOT be rendered in ModelCard
    // The card should not display "On Demand", "Batch", "Cross-Region", "Provisioned" as tags
    const onDemandTag = screen.queryByText('On Demand')
    const batchTag = screen.queryByText('Batch')
    const crossRegionTag = screen.queryByText('Cross-Region')
    const provisionedTag = screen.queryByText('Provisioned')
    
    // These consumption option tags should not appear in the card view
    // (they are shown in the expanded view's AvailabilitySummary instead)
    expect(onDemandTag).not.toBeInTheDocument()
    expect(batchTag).not.toBeInTheDocument()
    expect(crossRegionTag).not.toBeInTheDocument()
    expect(provisionedTag).not.toBeInTheDocument()
    
    // Verify the card still renders basic model info
    expect(screen.getByText('Claude 3 Sonnet')).toBeInTheDocument()
    expect(screen.getByText('Anthropic')).toBeInTheDocument()
  })
})
