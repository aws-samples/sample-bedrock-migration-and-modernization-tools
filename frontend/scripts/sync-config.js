#!/usr/bin/env node
/**
 * Sync frontend configuration from backend profiler-config.json
 * 
 * This script reads the backend config and generates frontend constants.
 * Run this during development or CI/CD to keep frontend in sync with backend.
 * 
 * Usage:
 *   node scripts/sync-config.js
 *   node scripts/sync-config.js --output src/config/generated-constants.js
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Parse command line arguments
const args = process.argv.slice(2)
const outputIndex = args.indexOf('--output')
const outputPath = outputIndex !== -1 
  ? args[outputIndex + 1] 
  : path.join(__dirname, '../src/config/generated-constants.js')

// Path to backend config
const backendConfigPath = path.join(__dirname, '../../backend/config/profiler-config.json')

function main() {
  console.log('Syncing frontend config from backend...')
  
  // Read backend config
  if (!fs.existsSync(backendConfigPath)) {
    console.error(`Backend config not found: ${backendConfigPath}`)
    process.exit(1)
  }
  
  const backendConfig = JSON.parse(fs.readFileSync(backendConfigPath, 'utf8'))
  
  // Extract frontend-relevant config
  const regionConfig = backendConfig.region_configuration || {}
  const providerConfig = backendConfig.provider_configuration || {}
  const modelConfig = backendConfig.model_configuration || {}
  
  // Build region coordinates
  const regionCoordinates = {}
  const coordinates = regionConfig.region_coordinates || {}
  
  for (const [code, coords] of Object.entries(coordinates)) {
    regionCoordinates[code] = {
      lat: coords.lat,
      lng: coords.lng,
      name: coords.name,
      geo: coords.geo,
      iata: coords.iata
    }
  }
  
  // Build AWS regions list
  const awsRegions = regionConfig.aws_regions || []
  
  // Build geo region options
  const geoRegionOptions = regionConfig.geo_region_options || []
  
  // Build provider colors
  const providerColors = providerConfig.provider_colors || {}
  
  // Build context thresholds
  const contextThresholds = modelConfig.context_window_thresholds || {
    small: 32000,
    medium: 128000,
    large: 500000
  }
  
  // Generate JavaScript content
  const jsContent = `/**
 * Auto-generated constants from backend profiler-config.json
 * Generated at: ${new Date().toISOString()}
 * Source version: ${backendConfig.version || 'unknown'}
 * 
 * DO NOT EDIT MANUALLY - Run 'npm run sync-config' to regenerate
 */

// Provider Colors - Brand colors for each model provider
export const providerColors = ${JSON.stringify(providerColors, null, 2)};

// Region Coordinates - Complete list with coordinates for map display
export const regionCoordinates = ${JSON.stringify(regionCoordinates, null, 2)};

// AWS Regions for dropdown selectors
export const awsRegions = ${JSON.stringify(awsRegions, null, 2)};

// Geographic region options
export const geoRegionOptions = ${JSON.stringify(geoRegionOptions, null, 2)};

// Geo prefix map
export const geoPrefixMap = ${JSON.stringify(regionConfig.geo_prefix_map || {}, null, 2)};

// Context window size categories
export const contextWindowThresholds = ${JSON.stringify(contextThresholds, null, 2)};

// Config metadata
export const configMetadata = {
  version: "${backendConfig.version || 'unknown'}",
  generatedAt: "${new Date().toISOString()}",
  source: "profiler-config.json"
};

// Helper function to get provider color
export function getProviderColor(provider) {
  return providerColors[provider] || providerColors.default || '#64748b';
}

// Helper function to get region info
export function getRegionInfo(regionCode) {
  return regionCoordinates[regionCode] || { name: regionCode, geo: 'Unknown' };
}

// Helper function to get context size category
export function getContextSizeCategory(contextWindow) {
  if (!contextWindow || typeof contextWindow !== 'number') {
    return { label: 'Unknown', color: 'bg-slate-500', tier: 0 };
  }
  if (contextWindow < contextWindowThresholds.small) {
    return { label: 'Small', color: 'bg-slate-500', tier: 1 };
  }
  if (contextWindow < contextWindowThresholds.medium) {
    return { label: 'Medium', color: 'bg-blue-500', tier: 2 };
  }
  if (contextWindow < contextWindowThresholds.large) {
    return { label: 'Large', color: 'bg-emerald-500', tier: 3 };
  }
  return { label: 'XL', color: 'bg-purple-500', tier: 4 };
}
`

  // Write output file
  const outputDir = path.dirname(outputPath)
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true })
  }
  
  fs.writeFileSync(outputPath, jsContent)
  console.log(`Generated: ${outputPath}`)
  console.log(`  - ${Object.keys(regionCoordinates).length} regions`)
  console.log(`  - ${Object.keys(providerColors).length} provider colors`)
  console.log(`  - ${awsRegions.length} AWS region options`)
}

main()
