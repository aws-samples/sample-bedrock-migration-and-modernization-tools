/**
 * Region utilities - centralized region metadata access.
 * All region data comes from generated constants (synced from backend).
 */

import { 
  regionCoordinates, 
  awsRegions,
  geoRegionOptions,
  geoPrefixMap,
  getRegionInfo as generatedGetRegionInfo
} from '../config/constants.js'

import { geoGroups } from '../config/constants.js'

/**
 * Get region display name
 */
export function getRegionName(regionCode) {
  return regionCoordinates[regionCode]?.name || regionCode
}

/**
 * Get region geo group
 */
export function getRegionGeo(regionCode) {
  return regionCoordinates[regionCode]?.geo || 'Unknown'
}

/**
 * Get region IATA airport code
 */
export function getAirportCode(regionCode) {
  return regionCoordinates[regionCode]?.iata || null
}

/**
 * Get region coordinates for map display
 */
export function getRegionCoordinates(regionCode) {
  const data = regionCoordinates[regionCode]
  if (!data || data.lat === undefined) return null
  return { lat: data.lat, lng: data.lng }
}

/**
 * Get full region info
 */
export function getRegionInfo(regionCode) {
  return generatedGetRegionInfo(regionCode)
}

/**
 * Get geo group info (name and color)
 */
export function getGeoGroupInfo(geo) {
  return geoGroups[geo] || { name: geo, color: '#64748b' }
}

/**
 * Get all regions for a geo group
 */
export function getRegionsInGeo(geo) {
  return awsRegions.filter(r => r.geo === geo)
}

/**
 * Format region for display (e.g., "N. Virginia (us-east-1)")
 */
export function formatRegion(regionCode) {
  const name = getRegionName(regionCode)
  return `${name} (${regionCode})`
}

/**
 * Build region metadata object from generated constants.
 * Returns { [regionCode]: { label, geo } } format for backward compatibility.
 */
export function buildRegionMeta() {
  return Object.fromEntries(
    Object.entries(regionCoordinates).map(([code, data]) => [
      code,
      { label: data.name, geo: data.geo }
    ])
  )
}

/**
 * Build geo prefix map from generated constants.
 * Returns { prefix: geo } format (e.g., { us: 'US', eu: 'EU' })
 */
export function buildGeoPrefixMap() {
  return Object.fromEntries(
    Object.entries(geoPrefixMap).map(([geo, prefix]) => [
      prefix.replace('-', ''),
      geo
    ])
  )
}

// Re-export for convenience
export { regionCoordinates, awsRegions, geoRegionOptions, geoPrefixMap, geoGroups }
