/**
 * Tests for region utilities (Task 11).
 *
 * Tests the centralized region metadata access:
 * - getRegionName returns correct display names
 * - getRegionInfo returns full region info
 * - getRegionGeo returns geographic group
 * - getRegionsInGeo filters regions by geo
 * - formatRegion formats region for display
 * - getRegionCoordinates returns map coordinates
 * - No duplicate region definitions (refactoring verification)
 */

import { describe, it, expect } from 'vitest'
import {
  getRegionName,
  getRegionInfo,
  getRegionGeo,
  getRegionsInGeo,
  formatRegion,
  getRegionCoordinates,
  buildRegionMeta,
  buildGeoPrefixMap,
} from '../../utils/regionUtils.js'

describe('Region Utilities', () => {
  describe('getRegionName', () => {
    it('should return correct name for known region', () => {
      expect(getRegionName('us-east-1')).toBe('N. Virginia')
    })

    it('should return correct name for EU region', () => {
      expect(getRegionName('eu-west-1')).toBe('Ireland')
    })

    it('should return correct name for AP region', () => {
      expect(getRegionName('ap-northeast-1')).toBe('Tokyo')
    })

    it('should return code for unknown region', () => {
      expect(getRegionName('unknown-region')).toBe('unknown-region')
    })

    it('should return code for null/undefined', () => {
      expect(getRegionName(null)).toBe(null)
      expect(getRegionName(undefined)).toBe(undefined)
    })
  })

  describe('getRegionInfo', () => {
    it('should return full info for known region', () => {
      const info = getRegionInfo('us-east-1')

      expect(info).toHaveProperty('name')
      expect(info).toHaveProperty('geo')
      expect(info.name).toBe('N. Virginia')
      expect(info.geo).toBe('US')
    })

    it('should return info with coordinates for known region', () => {
      const info = getRegionInfo('us-east-1')

      expect(info).toHaveProperty('lat')
      expect(info).toHaveProperty('lng')
      expect(typeof info.lat).toBe('number')
      expect(typeof info.lng).toBe('number')
    })

    it('should return default info for unknown region', () => {
      const info = getRegionInfo('unknown-region')

      expect(info.name).toBe('unknown-region')
      expect(info.geo).toBe('Unknown')
    })
  })

  describe('getRegionGeo', () => {
    it('should return geo group for US region', () => {
      expect(getRegionGeo('us-east-1')).toBe('US')
      expect(getRegionGeo('us-west-2')).toBe('US')
    })

    it('should return geo group for EU region', () => {
      expect(getRegionGeo('eu-west-1')).toBe('EU')
      expect(getRegionGeo('eu-central-1')).toBe('EU')
    })

    it('should return geo group for AP region', () => {
      expect(getRegionGeo('ap-northeast-1')).toBe('AP')
      expect(getRegionGeo('ap-southeast-1')).toBe('AP')
    })

    it('should return geo group for other regions', () => {
      expect(getRegionGeo('ca-central-1')).toBe('CA')
      expect(getRegionGeo('sa-east-1')).toBe('SA')
      expect(getRegionGeo('me-south-1')).toBe('ME')
      expect(getRegionGeo('af-south-1')).toBe('AF')
    })

    it('should return Unknown for unknown region', () => {
      expect(getRegionGeo('unknown-region')).toBe('Unknown')
    })
  })

  describe('getRegionsInGeo', () => {
    it('should return all US regions', () => {
      const usRegions = getRegionsInGeo('US')

      expect(usRegions.length).toBeGreaterThan(0)
      expect(usRegions.every((r) => r.geo === 'US')).toBe(true)
      // Should include known US regions
      expect(usRegions.some((r) => r.value === 'us-east-1')).toBe(true)
      expect(usRegions.some((r) => r.value === 'us-west-2')).toBe(true)
    })

    it('should return all EU regions', () => {
      const euRegions = getRegionsInGeo('EU')

      expect(euRegions.length).toBeGreaterThan(0)
      expect(euRegions.every((r) => r.geo === 'EU')).toBe(true)
      expect(euRegions.some((r) => r.value === 'eu-west-1')).toBe(true)
    })

    it('should return all AP regions', () => {
      const apRegions = getRegionsInGeo('AP')

      expect(apRegions.length).toBeGreaterThan(0)
      expect(apRegions.every((r) => r.geo === 'AP')).toBe(true)
      expect(apRegions.some((r) => r.value === 'ap-northeast-1')).toBe(true)
    })

    it('should return empty array for unknown geo', () => {
      const unknownRegions = getRegionsInGeo('UNKNOWN')

      expect(unknownRegions).toEqual([])
    })
  })

  describe('formatRegion', () => {
    it('should format region correctly', () => {
      expect(formatRegion('us-east-1')).toBe('N. Virginia (us-east-1)')
    })

    it('should format EU region correctly', () => {
      expect(formatRegion('eu-west-1')).toBe('Ireland (eu-west-1)')
    })

    it('should format AP region correctly', () => {
      expect(formatRegion('ap-northeast-1')).toBe('Tokyo (ap-northeast-1)')
    })

    it('should handle unknown region gracefully', () => {
      expect(formatRegion('unknown-region')).toBe('unknown-region (unknown-region)')
    })
  })

  describe('getRegionCoordinates', () => {
    it('should return coordinates for known region', () => {
      const coords = getRegionCoordinates('us-east-1')

      expect(coords).toHaveProperty('lat')
      expect(coords).toHaveProperty('lng')
      expect(typeof coords.lat).toBe('number')
      expect(typeof coords.lng).toBe('number')
    })

    it('should return valid lat/lng values', () => {
      const coords = getRegionCoordinates('us-east-1')

      // N. Virginia coordinates should be approximately correct
      expect(coords.lat).toBeGreaterThan(30)
      expect(coords.lat).toBeLessThan(45)
      expect(coords.lng).toBeGreaterThan(-85)
      expect(coords.lng).toBeLessThan(-70)
    })

    it('should return null for unknown region', () => {
      const coords = getRegionCoordinates('unknown-region')

      expect(coords).toBeNull()
    })
  })

  describe('buildRegionMeta', () => {
    it('should build region metadata object', () => {
      const meta = buildRegionMeta()

      expect(meta).toHaveProperty('us-east-1')
      expect(meta['us-east-1']).toHaveProperty('label')
      expect(meta['us-east-1']).toHaveProperty('geo')
      expect(meta['us-east-1'].label).toBe('N. Virginia')
      expect(meta['us-east-1'].geo).toBe('US')
    })

    it('should include all major regions', () => {
      const meta = buildRegionMeta()

      expect(meta).toHaveProperty('us-east-1')
      expect(meta).toHaveProperty('us-west-2')
      expect(meta).toHaveProperty('eu-west-1')
      expect(meta).toHaveProperty('ap-northeast-1')
    })
  })

  describe('buildGeoPrefixMap', () => {
    it('should build geo prefix map', () => {
      const prefixMap = buildGeoPrefixMap()

      expect(prefixMap).toHaveProperty('us')
      expect(prefixMap).toHaveProperty('eu')
      expect(prefixMap).toHaveProperty('ap')
      expect(prefixMap['us']).toBe('US')
      expect(prefixMap['eu']).toBe('EU')
      expect(prefixMap['ap']).toBe('AP')
    })
  })
})

describe('No Duplicate Region Definitions', () => {
  it('should not have hard-coded REGION_META object literal in filters.js', async () => {
    // This test verifies the refactoring was successful
    // filters.js should import from generated-constants, not define its own REGION_META
    const fs = await import('fs')
    const path = await import('path')

    const filtersPath = path.resolve(import.meta.dirname, '../../utils/filters.js')
    const content = fs.readFileSync(filtersPath, 'utf8')

    // Should NOT have a hard-coded object literal like:
    // const REGION_META = {
    //   'us-east-1': { label: 'N. Virginia', geo: 'US' },
    //   ...
    // }
    // Instead, it should build REGION_META from regionCoordinates

    // Check that it imports regionCoordinates from generated-constants
    // The import is multi-line, so we check for both parts
    expect(content).toContain('regionCoordinates')
    expect(content).toContain('generated-constants')

    // Check that REGION_META is built from regionCoordinates, not hard-coded
    expect(content).toMatch(/const REGION_META = Object\.fromEntries/)

    // Should NOT have hard-coded region entries like 'us-east-1': { label:
    expect(content).not.toMatch(/'us-east-1':\s*\{\s*label:\s*['"]N\. Virginia['"]/)
  })

  it('should use regionUtils for region operations', async () => {
    // Verify regionUtils.js exists and exports the expected functions
    const regionUtils = await import('../../utils/regionUtils.js')

    expect(typeof regionUtils.getRegionName).toBe('function')
    expect(typeof regionUtils.getRegionInfo).toBe('function')
    expect(typeof regionUtils.getRegionGeo).toBe('function')
    expect(typeof regionUtils.getRegionsInGeo).toBe('function')
    expect(typeof regionUtils.formatRegion).toBe('function')
    expect(typeof regionUtils.getRegionCoordinates).toBe('function')
  })

  it('should have consistent region data between regionUtils and filters', async () => {
    // Verify that regionUtils and filters use the same underlying data
    const { getRegionName, getRegionGeo } = await import('../../utils/regionUtils.js')
    const { DEFAULT_AWS_REGIONS } = await import('../../utils/filters.js')

    // Check that us-east-1 has consistent data
    const usEast1FromUtils = {
      name: getRegionName('us-east-1'),
      geo: getRegionGeo('us-east-1'),
    }

    const usEast1FromFilters = DEFAULT_AWS_REGIONS.find((r) => r.value === 'us-east-1')

    expect(usEast1FromFilters).toBeDefined()
    expect(usEast1FromFilters.label).toContain(usEast1FromUtils.name)
    expect(usEast1FromFilters.geo).toBe(usEast1FromUtils.geo)
  })
})
