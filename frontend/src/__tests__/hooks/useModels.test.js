/**
 * Tests for useModels hook pricing dimension functionality (Task 05).
 *
 * Tests the pricing dimension filtering and extraction:
 * - getFilteredPricing filters by dimension options
 * - getAvailableDimensions returns available dimension values
 * - Mantle pricing is excluded by default
 * - Legacy data without dimensions is handled gracefully
 */

import { describe, it, expect } from 'vitest'
import { getFilteredPricing, getAvailableDimensions } from '../../hooks/useModels.js'

// ============================================================================
// Mock Data
// ============================================================================

const mockPricingDataWithDimensions = {
  providers: {
    Anthropic: {
      'anthropic.claude-3-sonnet': {
        model_name: 'Claude 3 Sonnet',
        model_provider: 'Anthropic',
        primary_pricing_type: 'token',
        available_dimensions: {
          sources: ['standard', 'mantle'],
          geos: ['global'],
          tiers: ['flex', 'priority'],
          contexts: ['standard', 'long'],
        },
        has_mantle_pricing: true,
        regions: {
          'us-east-1': {
            pricing_groups: {
              'On-Demand': [
                {
                  price_per_thousand: 0.003,
                  is_input: true,
                  is_output: false,
                  dimension: 'USE1-Claude3Sonnet-input',
                  dimensions: { source: 'standard', geo: null, tier: null, context: 'standard' },
                },
                {
                  price_per_thousand: 0.015,
                  is_input: false,
                  is_output: true,
                  dimension: 'USE1-Claude3Sonnet-output',
                  dimensions: { source: 'standard', geo: null, tier: null, context: 'standard' },
                },
                {
                  price_per_thousand: 0.005,
                  is_input: true,
                  is_output: false,
                  dimension: 'USE1-Mantle-Claude3Sonnet-input',
                  dimensions: { source: 'mantle', geo: null, tier: null, context: 'standard' },
                },
                {
                  price_per_thousand: 0.020,
                  is_input: false,
                  is_output: true,
                  dimension: 'USE1-Mantle-Claude3Sonnet-output',
                  dimensions: { source: 'mantle', geo: null, tier: null, context: 'standard' },
                },
                {
                  price_per_thousand: 0.004,
                  is_input: true,
                  is_output: false,
                  dimension: 'USE1-Claude3Sonnet-Global-input',
                  dimensions: { source: 'standard', geo: 'global', tier: null, context: 'standard' },
                },
              ],
            },
          },
        },
      },
    },
  },
}

const mockLegacyPricingData = {
  providers: {
    Anthropic: {
      'anthropic.claude-2': {
        model_name: 'Claude 2',
        model_provider: 'Anthropic',
        primary_pricing_type: 'token',
        // No available_dimensions or has_mantle_pricing fields (legacy)
        regions: {
          'us-east-1': {
            pricing_groups: {
              'On-Demand': [
                {
                  price_per_thousand: 0.008,
                  is_input: true,
                  is_output: false,
                  dimension: 'USE1-Claude2-input',
                  // No dimensions field (legacy)
                },
                {
                  price_per_thousand: 0.024,
                  is_input: false,
                  is_output: true,
                  dimension: 'USE1-Claude2-output',
                  // No dimensions field (legacy)
                },
              ],
            },
          },
        },
      },
    },
  },
}

const mockModelWithPricingRef = {
  model_id: 'anthropic.claude-3-sonnet-20240229-v1:0',
  model_name: 'Claude 3 Sonnet',
  model_provider: 'Anthropic',
  pricing: {
    reference: {
      provider: 'Anthropic',
      model_key: 'anthropic.claude-3-sonnet',
    },
  },
}

const mockLegacyModel = {
  model_id: 'anthropic.claude-v2:1',
  model_name: 'Claude 2',
  model_provider: 'Anthropic',
  pricing: {
    reference: {
      provider: 'Anthropic',
      model_key: 'anthropic.claude-2',
    },
  },
}

// ============================================================================
// Tests for getFilteredPricing
// ============================================================================

describe('getFilteredPricing', () => {
  describe('dimension filtering', () => {
    it('should exclude mantle by default', () => {
      const result = getFilteredPricing(
        mockModelWithPricingRef,
        mockPricingDataWithDimensions,
        'us-east-1'
      )

      // Default should use standard pricing (0.003), not mantle (0.005)
      // Note: prices are converted to per-1M in extractSummaryPricing
      expect(result.inputPrice).toBe(3) // 0.003 * 1000
      expect(result.outputPrice).toBe(15) // 0.015 * 1000
    })

    it('should filter by source dimension', () => {
      const result = getFilteredPricing(
        mockModelWithPricingRef,
        mockPricingDataWithDimensions,
        'us-east-1',
        { source: 'mantle' }
      )

      // Should use mantle pricing (0.005)
      expect(result.inputPrice).toBe(5) // 0.005 * 1000
      expect(result.outputPrice).toBe(20) // 0.020 * 1000
    })

    it('should filter by geo dimension', () => {
      const result = getFilteredPricing(
        mockModelWithPricingRef,
        mockPricingDataWithDimensions,
        'us-east-1',
        { geo: 'global' }
      )

      // Should use global pricing (0.004)
      expect(result.inputPrice).toBe(4) // 0.004 * 1000
    })

    it('should return pricingSource from cascade', () => {
      const result = getFilteredPricing(
        mockModelWithPricingRef,
        mockPricingDataWithDimensions,
        'us-east-1'
      )

      expect(result.pricingSource).toBeDefined()
    })

    it('should return availableDimensions', () => {
      const result = getFilteredPricing(
        mockModelWithPricingRef,
        mockPricingDataWithDimensions,
        'us-east-1'
      )

      expect(result.availableDimensions).toBeDefined()
      expect(result.availableDimensions.sources).toContain('standard')
      expect(result.availableDimensions.sources).toContain('mantle')
    })
  })

  describe('backward compatibility', () => {
    it('should handle legacy data without dimensions', () => {
      const result = getFilteredPricing(
        mockLegacyModel,
        mockLegacyPricingData,
        'us-east-1'
      )

      // Should still return pricing
      expect(result.inputPrice).toBe(8) // 0.008 * 1000
      expect(result.outputPrice).toBe(24) // 0.024 * 1000
    })

    it('should return default availableDimensions for legacy data', () => {
      const result = getFilteredPricing(
        mockLegacyModel,
        mockLegacyPricingData,
        'us-east-1'
      )

      expect(result.availableDimensions).toBeDefined()
      expect(result.availableDimensions.sources).toEqual(['standard'])
      expect(result.availableDimensions.contexts).toEqual(['standard'])
    })

    it('should return pricingSource for legacy data', () => {
      const result = getFilteredPricing(
        mockLegacyModel,
        mockLegacyPricingData,
        'us-east-1'
      )

      // Legacy data with On-Demand group should get "In-region" source
      expect(result.pricingSource === null || result.pricingSource.startsWith('In-region')).toBe(true)
    })
  })

  describe('edge cases', () => {
    it('should return null for missing model', () => {
      const result = getFilteredPricing(
        { model_id: 'nonexistent' },
        mockPricingDataWithDimensions,
        'us-east-1'
      )

      expect(result).toBeNull()
    })

    it('should return null for null pricing data', () => {
      const result = getFilteredPricing(
        mockModelWithPricingRef,
        null,
        'us-east-1'
      )

      expect(result).toBeNull()
    })

    it('should fall back to us-east-1 for unknown region', () => {
      const result = getFilteredPricing(
        mockModelWithPricingRef,
        mockPricingDataWithDimensions,
        'unknown-region'
      )

      // Should fall back to us-east-1 pricing
      expect(result.inputPrice).toBe(3)
    })
  })
})

// ============================================================================
// Tests for getAvailableDimensions
// ============================================================================

describe('getAvailableDimensions', () => {
  it('should return available dimensions from model pricing', () => {
    const result = getAvailableDimensions(
      mockModelWithPricingRef,
      mockPricingDataWithDimensions
    )

    expect(result).toBeDefined()
    expect(result.sources).toContain('standard')
    expect(result.sources).toContain('mantle')
    expect(result.geos).toContain('global')
    expect(result.tiers).toContain('flex')
    expect(result.tiers).toContain('priority')
    expect(result.contexts).toContain('standard')
    expect(result.contexts).toContain('long')
  })

  it('should return default dimensions for legacy data', () => {
    const result = getAvailableDimensions(
      mockLegacyModel,
      mockLegacyPricingData
    )

    expect(result).toBeDefined()
    expect(result.sources).toEqual(['standard'])
    expect(result.geos).toEqual([])
    expect(result.tiers).toEqual([])
    expect(result.contexts).toEqual(['standard'])
  })

  it('should return null for missing model', () => {
    const result = getAvailableDimensions(
      { model_id: 'nonexistent' },
      mockPricingDataWithDimensions
    )

    expect(result).toBeNull()
  })

  it('should return null for null pricing data', () => {
    const result = getAvailableDimensions(
      mockModelWithPricingRef,
      null
    )

    expect(result).toBeNull()
  })
})

// ============================================================================
// Tests for Pricing Type Detection
// ============================================================================

describe('Pricing Type Detection', () => {
  it('should return correct pricing type for token models', () => {
    const result = getFilteredPricing(
      mockModelWithPricingRef,
      mockPricingDataWithDimensions,
      'us-east-1'
    )

    expect(result.pricingType).toBe('token')
    expect(result.unitLabel).toBe('per 1M tokens')
  })
})
