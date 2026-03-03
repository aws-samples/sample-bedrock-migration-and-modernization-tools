/**
 * Tests for frontend configuration externalization (Task 06).
 *
 * Tests the frontend config system:
 * - dataSource.js environment variable usage
 * - generated-constants.js structure and exports
 * - Helper functions (getProviderColor, getRegionInfo, getContextSizeCategory)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// ============================================================================
// Tests for Generated Constants Structure
// ============================================================================

describe('Generated Constants', () => {
  it('should have correct structure with all required exports', async () => {
    const constants = await import('../../config/generated-constants.js')

    expect(constants.providerColors).toBeDefined()
    expect(constants.regionCoordinates).toBeDefined()
    expect(constants.awsRegions).toBeDefined()
    expect(constants.geoRegionOptions).toBeDefined()
    expect(constants.geoPrefixMap).toBeDefined()
    expect(constants.contextWindowThresholds).toBeDefined()
    expect(constants.configMetadata).toBeDefined()
  })

  it('should export helper functions', async () => {
    const { getProviderColor, getRegionInfo, getContextSizeCategory } = await import(
      '../../config/generated-constants.js'
    )

    expect(typeof getProviderColor).toBe('function')
    expect(typeof getRegionInfo).toBe('function')
    expect(typeof getContextSizeCategory).toBe('function')
  })

  it('providerColors should have expected providers', async () => {
    const { providerColors } = await import('../../config/generated-constants.js')

    // Check for common providers
    expect(providerColors.Amazon).toBeDefined()
    expect(providerColors.Anthropic).toBeDefined()
    expect(providerColors.Meta).toBeDefined()
    expect(providerColors.default).toBeDefined()
  })

  it('regionCoordinates should have expected regions', async () => {
    const { regionCoordinates } = await import('../../config/generated-constants.js')

    // Check for common regions
    expect(regionCoordinates['us-east-1']).toBeDefined()
    expect(regionCoordinates['us-west-2']).toBeDefined()
    expect(regionCoordinates['eu-west-1']).toBeDefined()

    // Check structure
    const usEast1 = regionCoordinates['us-east-1']
    expect(usEast1).toHaveProperty('lat')
    expect(usEast1).toHaveProperty('lng')
    expect(usEast1).toHaveProperty('name')
    expect(usEast1).toHaveProperty('geo')
  })

  it('awsRegions should be an array with correct structure', async () => {
    const { awsRegions } = await import('../../config/generated-constants.js')

    expect(Array.isArray(awsRegions)).toBe(true)
    expect(awsRegions.length).toBeGreaterThan(0)

    // Check structure of first region
    const firstRegion = awsRegions[0]
    expect(firstRegion).toHaveProperty('value')
    expect(firstRegion).toHaveProperty('label')
    expect(firstRegion).toHaveProperty('geo')
  })

  it('geoRegionOptions should include All Regions option', async () => {
    const { geoRegionOptions } = await import('../../config/generated-constants.js')

    expect(Array.isArray(geoRegionOptions)).toBe(true)
    const allRegions = geoRegionOptions.find((opt) => opt.value === 'All Regions')
    expect(allRegions).toBeDefined()
  })

  it('configMetadata should have version and generatedAt', async () => {
    const { configMetadata } = await import('../../config/generated-constants.js')

    expect(configMetadata).toHaveProperty('version')
    expect(configMetadata).toHaveProperty('generatedAt')
    expect(configMetadata).toHaveProperty('source')
  })
})

// ============================================================================
// Tests for Helper Functions
// ============================================================================

describe('getProviderColor', () => {
  it('should return color for known provider', async () => {
    const { getProviderColor } = await import('../../config/generated-constants.js')

    const color = getProviderColor('Amazon')

    expect(color).toMatch(/^#[0-9A-Fa-f]{6}$/)
    expect(color).toBe('#FF9900') // Amazon's brand color
  })

  it('should return default color for unknown provider', async () => {
    const { getProviderColor, providerColors } = await import(
      '../../config/generated-constants.js'
    )

    const color = getProviderColor('UnknownProvider')

    expect(color).toBe(providerColors.default)
  })

  it('should return default color for null/undefined provider', async () => {
    const { getProviderColor, providerColors } = await import(
      '../../config/generated-constants.js'
    )

    expect(getProviderColor(null)).toBe(providerColors.default)
    expect(getProviderColor(undefined)).toBe(providerColors.default)
  })
})

describe('getRegionInfo', () => {
  it('should return info for known region', async () => {
    const { getRegionInfo } = await import('../../config/generated-constants.js')

    const info = getRegionInfo('us-east-1')

    expect(info).toHaveProperty('name')
    expect(info).toHaveProperty('geo')
    expect(info.geo).toBe('US')
  })

  it('should return fallback for unknown region', async () => {
    const { getRegionInfo } = await import('../../config/generated-constants.js')

    const info = getRegionInfo('unknown-region-1')

    expect(info).toHaveProperty('name')
    expect(info.name).toBe('unknown-region-1')
    expect(info.geo).toBe('Unknown')
  })
})

describe('getContextSizeCategory', () => {
  it('should return Small for context < 32000', async () => {
    const { getContextSizeCategory } = await import('../../config/generated-constants.js')

    const result = getContextSizeCategory(16000)

    expect(result.label).toBe('Small')
    expect(result.tier).toBe(1)
  })

  it('should return Medium for context between 32000 and 128000', async () => {
    const { getContextSizeCategory } = await import('../../config/generated-constants.js')

    const result = getContextSizeCategory(64000)

    expect(result.label).toBe('Medium')
    expect(result.tier).toBe(2)
  })

  it('should return Large for context between 128000 and 500000', async () => {
    const { getContextSizeCategory } = await import('../../config/generated-constants.js')

    const result = getContextSizeCategory(200000)

    expect(result.label).toBe('Large')
    expect(result.tier).toBe(3)
  })

  it('should return XL for context >= 500000', async () => {
    const { getContextSizeCategory } = await import('../../config/generated-constants.js')

    const result = getContextSizeCategory(1000000)

    expect(result.label).toBe('XL')
    expect(result.tier).toBe(4)
  })

  it('should return Unknown for null/undefined context', async () => {
    const { getContextSizeCategory } = await import('../../config/generated-constants.js')

    expect(getContextSizeCategory(null).label).toBe('Unknown')
    expect(getContextSizeCategory(undefined).label).toBe('Unknown')
  })

  it('should return Unknown for non-number context', async () => {
    const { getContextSizeCategory } = await import('../../config/generated-constants.js')

    expect(getContextSizeCategory('not a number').label).toBe('Unknown')
  })
})

// ============================================================================
// Tests for Data Source Configuration
// ============================================================================

describe('Data Source Configuration', () => {
  it('should export DATA_URLS with required paths', async () => {
    const { DATA_URLS } = await import('../../config/dataSource.js')

    expect(DATA_URLS).toHaveProperty('models')
    expect(DATA_URLS).toHaveProperty('pricing')
    expect(DATA_URLS).toHaveProperty('frontendConfig')
  })

  it('should export DATA_SOURCE_CONFIG with bucket info', async () => {
    const { DATA_SOURCE_CONFIG } = await import('../../config/dataSource.js')

    expect(DATA_SOURCE_CONFIG).toHaveProperty('bucket')
    expect(DATA_SOURCE_CONFIG).toHaveProperty('region')
    expect(DATA_SOURCE_CONFIG).toHaveProperty('prefix')
    expect(DATA_SOURCE_CONFIG).toHaveProperty('isDevelopment')
  })

  it('DATA_URLS should have correct path structure', async () => {
    const { DATA_URLS, DATA_SOURCE_CONFIG } = await import('../../config/dataSource.js')

    // In test environment (development), should use s3-data proxy
    if (DATA_SOURCE_CONFIG.isDevelopment) {
      expect(DATA_URLS.models).toContain('/s3-data/')
      expect(DATA_URLS.pricing).toContain('/s3-data/')
    } else {
      // In production, should use direct paths
      expect(DATA_URLS.models).toMatch(/^\/latest\//)
      expect(DATA_URLS.pricing).toMatch(/^\/latest\//)
    }
  })

  it('DATA_URLS should include bedrock_models.json and bedrock_pricing.json', async () => {
    const { DATA_URLS } = await import('../../config/dataSource.js')

    expect(DATA_URLS.models).toContain('bedrock_models.json')
    expect(DATA_URLS.pricing).toContain('bedrock_pricing.json')
  })

  it('DATA_URLS frontendConfig should point to config directory', async () => {
    const { DATA_URLS } = await import('../../config/dataSource.js')

    expect(DATA_URLS.frontendConfig).toContain('config/frontend-config.json')
  })
})
