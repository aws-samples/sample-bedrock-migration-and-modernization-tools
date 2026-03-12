/**
 * Import region metadata from generated constants (synced from backend)
 */
import { 
  regionCoordinates, 
  awsRegions as generatedAwsRegions,
  geoRegionOptions as generatedGeoRegionOptions,
  geoPrefixMap 
} from '../config/generated-constants.js'

/**
 * Import filter options configuration for normalization
 */
import {
  CUSTOMIZATION_OPTIONS,
  LANGUAGE_OPTIONS,
  LANGUAGE_PRIORITY,
  CAPABILITY_OPTIONS,
  USE_CASE_OPTIONS,
  extractNormalizedValues,
  getDisplayLabel,
  normalizeValue,
  CAPABILITY_REVERSE_MAP,
  USE_CASE_REVERSE_MAP,
  LANGUAGE_REVERSE_MAP,
  CUSTOMIZATION_REVERSE_MAP,
} from '../config/filterOptions.js'

// Re-export for convenience
export { getDisplayLabel }

/**
 * Check if a value is a GEO-level selection
 */
export function isGeoSelection(value) {
  return value?.startsWith('geo:')
}

/**
 * Get regions for a GEO selection
 */
export function getRegionsForGeo(geoValue, awsRegionsList) {
  if (!isGeoSelection(geoValue)) return []
  const geo = geoValue.replace('geo:', '')
  return awsRegionsList.filter(r => r.geo === geo).map(r => r.value)
}

/**
 * Build region metadata from generated constants.
 * This replaces the hard-coded REGION_META.
 */
const REGION_META = Object.fromEntries(
  Object.entries(regionCoordinates).map(([code, data]) => [
    code,
    { label: data.name, geo: data.geo }
  ])
)

/**
 * Auto-detect geo from region code prefix for unknown regions.
 * Uses geoPrefixMap from generated constants.
 */
const GEO_PREFIX_MAP = Object.fromEntries(
  Object.entries(geoPrefixMap).map(([geo, prefix]) => [
    prefix.replace('-', ''),
    geo
  ])
)
// Add additional prefix mappings not in geoPrefixMap
GEO_PREFIX_MAP.il = 'ME'
GEO_PREFIX_MAP.mx = 'SA'
GEO_PREFIX_MAP.in = 'AP'

/**
 * Geo sort order for consistent region ordering.
 */
const GEO_SORT_ORDER = { US: 0, EU: 1, AP: 2, CA: 3, SA: 4, ME: 5, AF: 6 }

/**
 * Build a region entry { value, label, geo } from a region code.
 * Uses REGION_META for known regions, auto-generates for unknown ones.
 */
function buildRegionEntry(code) {
  const meta = REGION_META[code]
  if (meta) {
    return { value: code, label: `${meta.label} (${code})`, geo: meta.geo }
  }
  // Unknown region — auto-detect geo from prefix
  const prefix = code.split('-')[0]
  const geo = GEO_PREFIX_MAP[prefix] || 'US'
  // Generate a human-readable label from the code (e.g. "ap-southeast-8" → "Ap Southeast 8")
  const label = code
    .split('-')
    .map(s => s.charAt(0).toUpperCase() + s.slice(1))
    .join(' ')
  return { value: code, label: `${label} (${code})`, geo }
}

/**
 * Default AWS regions — fallback when no model data is available.
 * Comprehensive list derived from REGION_META.
 */
export const DEFAULT_AWS_REGIONS = Object.keys(REGION_META)
  .map(buildRegionEntry)
  .sort((a, b) => {
    const geoA = GEO_SORT_ORDER[a.geo] ?? 99
    const geoB = GEO_SORT_ORDER[b.geo] ?? 99
    if (geoA !== geoB) return geoA - geoB
    return a.value.localeCompare(b.value)
  })

/**
 * Backward-compatible alias — static list for consumers that don't have model data.
 */
export const awsRegions = DEFAULT_AWS_REGIONS

/**
 * Build a dynamic AWS regions list from model data.
 * Extracts all unique regions from models[].availability.on_demand.regions and
 * models[].availability.cross_region.regions, then returns sorted
 * { value, label, geo } entries.
 *
 * Falls back to DEFAULT_AWS_REGIONS if models is empty.
 */
export function buildAwsRegionsFromModels(models) {
  if (!models || !models.length) return DEFAULT_AWS_REGIONS

  const regionCodes = new Set()
  models.forEach(m => {
    if (m.availability?.on_demand?.regions) {
      m.availability.on_demand.regions.forEach(r => regionCodes.add(r))
    }
    if (m.availability?.cross_region?.regions) {
      m.availability.cross_region.regions.forEach(r => regionCodes.add(r))
    }
    if (m.availability?.mantle?.regions) {
      m.availability.mantle.regions.forEach(r => regionCodes.add(r))
    }
  })

  if (!regionCodes.size) return DEFAULT_AWS_REGIONS

  return [...regionCodes]
    .map(buildRegionEntry)
    .sort((a, b) => {
      const geoA = GEO_SORT_ORDER[a.geo] ?? 99
      const geoB = GEO_SORT_ORDER[b.geo] ?? 99
      if (geoA !== geoB) return geoA - geoB
      return a.value.localeCompare(b.value)
    })
}

/**
 * Geographic region options - imported from generated constants
 */
export const geoRegionOptions = generatedGeoRegionOptions

/**
 * Model status options
 */
export const modelStatusOptions = [
  { value: 'All Status', label: 'All Status' },
  { value: 'ACTIVE', label: 'Active' },
  { value: 'LEGACY', label: 'Legacy' },
  { value: 'EOL', label: 'End of Life' },
  { value: 'MIXED', label: 'Mixed Status' },
]

/**
 * CRIS support options - includes all geographic scopes
 * Each scope matches exactly (JP, AU, APAC are separate)
 */
export const crisSupportOptions = [
  { value: 'All Models', label: 'All Models' },
  { value: 'GLOBAL', label: 'Global' },
  { value: 'US', label: 'US' },
  { value: 'EU', label: 'EU' },
  { value: 'APAC', label: 'APAC' },
  { value: 'JP', label: 'Japan' },
  { value: 'AU', label: 'Australia' },
  { value: 'CA', label: 'Canada' },
  { value: 'SA', label: 'South America' },
  { value: 'ME', label: 'Middle East' },
  { value: 'AF', label: 'Africa' },
  { value: 'CRIS Not Supported', label: 'Not Supported' },
]

/**
 * Helper to extract CRIS geographic scopes from a model
 * Extracts scope dynamically from profile_id prefix (e.g., "us.anthropic..." -> "US")
 * Returns exact scopes (uppercased) - JP, AU, APAC are separate
 */
export function getCrisGeoScopes(model) {
  const profiles = model?.availability?.cross_region?.profiles || []
  return [...new Set(profiles.map(p => {
    const profileId = p.profile_id || p.inference_profile_id
    const prefix = profileId?.split('.')[0]
    return prefix?.toUpperCase() || null
  }).filter(Boolean))]
}

/**
 * Streaming support options
 */
export const streamingSupportOptions = [
  { value: 'All Models', label: 'All Models' },
  { value: 'Streaming Supported', label: 'Streaming Supported' },
  { value: 'Streaming Not Supported', label: 'Not Supported' },
]

/**
 * Mantle support options
 */
export const mantleSupportOptions = [
  { value: 'All Models', label: 'All Models' },
  { value: 'Mantle Supported', label: 'Mantle Supported' },
  { value: 'Mantle Only', label: 'Mantle Only' },
  { value: 'No Mantle', label: 'No Mantle' },
]

/**
 * Context window filter options
 */
export const contextFilterOptions = [
  { value: 'All Models', label: 'All Context Sizes' },
  { value: 'Small (< 32K)', label: 'Small (< 32K)' },
  { value: 'Medium (32K-128K)', label: 'Medium (32K-128K)' },
  { value: 'Large (128K-500K)', label: 'Large (128K-500K)' },
  { value: 'XL (> 500K)', label: 'XL (> 500K)' },
]

/**
 * Modality options
 */
export const modalityOptions = [
  { value: 'All Modalities', label: 'All Modalities' },
  { value: 'TEXT', label: 'Text' },
  { value: 'IMAGE', label: 'Image' },
  { value: 'DOCUMENT', label: 'Document' },
  { value: 'VIDEO', label: 'Video' },
  { value: 'AUDIO', label: 'Audio' },
  { value: 'SPEECH', label: 'Speech' },
]

/**
 * Added date filter options
 */
export const addedFilterOptions = [
  { value: 'all', label: 'All Models' },
  { value: 'last_update', label: 'Last Update' },
  { value: 'last_month', label: 'Last Month' },
]

/**
 * Sort options for model explorer
 */
export const sortOptions = [
  { value: 'newest', label: 'Newest First' },
  { value: 'name-asc', label: 'Name A-Z' },
  { value: 'name-desc', label: 'Name Z-A' },
  { value: 'provider-asc', label: 'Provider A-Z' },
  { value: 'context-desc', label: 'Context Window (Largest)' },
  { value: 'context-asc', label: 'Context Window (Smallest)' },
  { value: 'price-input-asc', label: 'Price: Input (Low-High)' },
  { value: 'price-output-asc', label: 'Price: Output (Low-High)' },
]

/**
 * Sort models by the specified sort option
 * @param {Array} models - Array of models to sort
 * @param {string} sortBy - Sort option value
 * @param {Function} getPricingForModel - Function to get pricing for a model
 * @param {string} preferredRegion - Preferred region for pricing lookup
 * @returns {Array} Sorted models array
 */
export function sortModels(models, sortBy, getPricingForModel, preferredRegion) {
  if (!sortBy || sortBy === 'default') return models

  const sorted = [...models]

  const getPrice = (model, type) => {
    if (!getPricingForModel) return null
    const result = getPricingForModel(model, preferredRegion)
    const summary = result?.summary
    if (!summary) return null
    return type === 'input' ? summary.inputPrice : summary.outputPrice
  }

  // Helper to get effective context window (max of base and extended)
  const getEffectiveContext = (model) => {
    const base = model.specs?.context_window || 0
    const extended = model.specs?.extended_context_window || model.specs?.extended_context || 0
    return Math.max(base, extended)
  }
  
  // Helper to parse release date (handles ISO string or timestamp)
  const parseReleaseDate = (model) => {
    // Try lifecycle.release_date first (ISO string like "2025-09-29T00:00:00Z")
    const releaseDate = model.lifecycle?.release_date
    if (releaseDate) {
      if (typeof releaseDate === 'string') {
        return new Date(releaseDate).getTime() || 0
      }
      if (typeof releaseDate === 'number') {
        return releaseDate
      }
    }
    // Fallback to collection_metadata.first_discovered_at
    const discovered = model.collection_metadata?.first_discovered_at
    if (discovered) {
      return new Date(discovered).getTime() || 0
    }
    return 0
  }
  
  sorted.sort((a, b) => {
    switch (sortBy) {
      case 'newest': {
        const dateA = parseReleaseDate(a)
        const dateB = parseReleaseDate(b)
        return dateB - dateA // Newest first (higher timestamp first)
      }
      case 'name-asc':
        return (a.model_name || '').localeCompare(b.model_name || '')
      case 'name-desc':
        return (b.model_name || '').localeCompare(a.model_name || '')
      case 'provider-asc':
        return (a.model_provider || '').localeCompare(b.model_provider || '')
      case 'context-desc': {
        const ctxA = getEffectiveContext(a)
        const ctxB = getEffectiveContext(b)
        return ctxB - ctxA // Largest first
      }
      case 'context-asc': {
        const ctxA = getEffectiveContext(a)
        const ctxB = getEffectiveContext(b)
        return ctxA - ctxB // Smallest first
      }
      case 'price-input-asc': {
        const priceA = getPrice(a, 'input') ?? Infinity
        const priceB = getPrice(b, 'input') ?? Infinity
        return priceA - priceB
      }
      case 'price-output-asc': {
        const priceA = getPrice(a, 'output') ?? Infinity
        const priceB = getPrice(b, 'output') ?? Infinity
        return priceA - priceB
      }
      default:
        return 0
    }
  })

  return sorted
}

/**
 * Pricing availability filter options
 */
export const pricingFilterOptions = [
  { value: 'All Models', label: 'All' },
  { value: 'Has Pricing', label: 'Yes' },
  { value: 'No Pricing', label: 'No' },
]

/**
 * Feature filter options for multi-select
 */
export const featureFilterOptions = [
  'Streaming',
  'Flex Pricing',
  'Priority Pricing',
]

/**
 * Routing strategy options for orthogonal filter
 * Determines how requests are routed (in-region vs cross-region)
 */
export const routingStrategyOptions = [
  { value: 'in_region', label: 'In-Region', description: 'Data stays in one region' },
  { value: 'cris', label: 'CRIS', description: 'Cross-region inference' },
]

/**
 * Inference mode options for orthogonal filter
 * Determines how inference is executed (on-demand, batch, provisioned)
 */
export const inferenceModeOptions = [
  { value: 'on_demand', label: 'On-Demand', description: 'Real-time, pay-per-token' },
  { value: 'batch', label: 'Batch', description: 'Async, 50% cost savings' },
  { value: 'provisioned', label: 'Provisioned', description: 'Dedicated capacity', privileged: true },
]

/**
 * @deprecated Use routingStrategyOptions and inferenceModeOptions instead.
 * Consumption type definitions for the legacy cascading filter.
 * Maps internal keys to display labels and availability paths.
 */
export const consumptionTypeDefinitions = {
  runtime_api: {
    label: 'Runtime API',
    availabilityPath: 'on_demand',
    description: 'Standard on-demand inference API'
  },
  mantle: {
    label: 'Mantle',
    availabilityPath: 'mantle',
    description: 'Mantle-based access'
  },
  batch: {
    label: 'Batch',
    availabilityPath: 'batch',
    description: 'Batch inference'
  },
  cris: {
    label: 'CRIS',
    availabilityPath: 'cross_region',
    description: 'Cross-region inference'
  },
  provisioned: {
    label: 'Provisioned',
    availabilityPath: 'provisioned',
    description: 'Provisioned throughput'
  },
  reserved: {
    label: 'Reserved',
    availabilityPath: 'reserved',
    description: 'Reserved capacity'
  },
}

/**
 * CRIS endpoint options (geographic scopes)
 */
export const crisEndpointOptions = [
  { value: 'GLOBAL', label: 'Global' },
  { value: 'US', label: 'US' },
  { value: 'EU', label: 'EU' },
  { value: 'APAC', label: 'APAC' },
  { value: 'JP', label: 'JP' },
  { value: 'AU', label: 'AU' },
  { value: 'CA', label: 'CA' },
  { value: 'SA', label: 'SA' },
  { value: 'ME', label: 'ME' },
  { value: 'AF', label: 'AF' },
]

/**
 * Initial filter state
 */
export const initialFilterState = {
  searchQuery: '',
  primaryRegion: 'all',
  providers: [],
  geoRegion: 'All Regions',
  modelStatus: 'All Status',
  streamingSupport: 'All Models',
  pricingFilter: 'All Models',
  useCases: [],
  modality: 'All Modalities',
  capabilities: [],
  customizations: [],
  languages: [],
  contextFilter: 'All Models',
  // New nested routing filter state (matches RegionalAvailability)
  selectedRouting: null,      // null | 'in_region' | 'cris'
  selectedApi: null,          // null | 'runtime_api' | 'mantle' (only when selectedRouting === 'in_region')
  selectedCrisScopes: [],     // ['Global', 'US', 'EU', 'APAC', 'JP', 'AU', 'CA'] (only when selectedRouting === 'cris')
  selectedGeos: [],           // ['NAMER', 'EMEA', 'APAC', 'LATAM', 'GOVCLOUD'] (when selectedRouting !== 'cris')
  selectedRegions: [],        // Specific regions (overrides geo if set) - e.g., ['us-east-1', 'eu-west-1']
  // Features filters
  batchSupport: 'All Models',      // 'All Models' | 'Batch Supported' | 'No Batch'
  reservedSupport: 'All Models',   // 'All Models' | 'Reserved Supported' | 'No Reserved'
  flexPricing: 'All Models',       // 'All Models' | 'Has Flex' | 'No Flex'
  priorityPricing: 'All Models',   // 'All Models' | 'Has Priority' | 'No Priority'
}

/**
 * @deprecated Use getDisplayLabel from filterOptions.js instead
 * Format raw option values into readable labels
 * @param {string} value - Raw value from data
 * @param {string} type - Type of option: 'customization', 'language', 'capability', 'useCase'
 * @returns {string} Formatted label
 */
export function formatOptionLabel(value, type = 'default') {
  // Delegate to the new getDisplayLabel function
  return getDisplayLabel(value, type)
}

/**
 * Extract unique providers from models
 */
export function extractProviders(models) {
  const providers = new Set()
  models.forEach(m => {
    if (m.model_provider) {
      providers.add(m.model_provider)
    }
  })
  return Array.from(providers).sort()
}

/**
 * Extract unique capabilities from models - normalized
 */
export function extractCapabilities(models) {
  return extractNormalizedValues(models, 'capabilities', CAPABILITY_OPTIONS)
}

/**
 * Extract unique use cases from models - normalized
 */
export function extractUseCases(models) {
  return extractNormalizedValues(models, 'use_cases', USE_CASE_OPTIONS)
}

/**
 * Extract unique customizations from models - normalized
 */
export function extractCustomizations(models) {
  const normalizedSet = new Set()
  
  models.forEach(m => {
    const customization = m.customization?.customization_supported
    if (Array.isArray(customization)) {
      customization.forEach(c => {
        const normalized = CUSTOMIZATION_OPTIONS[c]
        if (normalized) {
          normalizedSet.add(normalized)
        }
      })
    }
  })
  
  return Array.from(normalizedSet).sort()
}

/**
 * Extract unique languages from models - normalized
 */
export function extractLanguages(models) {
  return extractNormalizedValues(models, 'languages', LANGUAGE_OPTIONS, LANGUAGE_PRIORITY)
}

/**
 * Extract unique modalities from models
 */
export function extractModalities(models) {
  const modalities = new Set()
  models.forEach(m => {
    if (m.modalities?.input_modalities) {
      m.modalities.input_modalities.forEach(mod => modalities.add(mod))
    }
    if (m.modalities?.output_modalities) {
      m.modalities.output_modalities.forEach(mod => modalities.add(mod))
    }
  })
  return Array.from(modalities).sort()
}

/**
 * Apply all filters to models
 * @param {Array} models - Array of models to filter
 * @param {Object} filters - Filter state object
 * @param {Function} getPricingForModel - Optional function to get pricing for a model
 * @returns {Array} Filtered models array
 */
export function applyFilters(models, filters, getPricingForModel = null) {
  let filtered = [...models]

  // Search query
  if (filters.searchQuery) {
    const query = filters.searchQuery.toLowerCase()
    filtered = filtered.filter(m =>
      m.model_name?.toLowerCase().includes(query) ||
      m.model_id?.toLowerCase().includes(query) ||
      m.model_provider?.toLowerCase().includes(query) ||
      m.capabilities?.some(c => c.toLowerCase().includes(query))
    )
  }

  // Provider filter
  if (filters.providers && filters.providers.length > 0) {
    filtered = filtered.filter(m => filters.providers.includes(m.model_provider))
  }

  // Geographic region filter — checks in-region, CRIS, and Mantle
  if (filters.geoRegion && filters.geoRegion !== 'All Regions') {
    const prefixMap = { 'US': 'us-', 'EU': 'eu-', 'AP': 'ap-', 'CA': 'ca-', 'SA': 'sa-', 'ME': 'me-', 'AF': 'af-' }
    const prefix = prefixMap[filters.geoRegion]
    if (prefix) {
      const regionMatchesGeo = (r) => {
        if (r.startsWith(prefix)) return true
        // Special cases: il- regions belong to ME geo, mx- regions belong to SA geo
        if (filters.geoRegion === 'ME' && r.startsWith('il-')) return true
        if (filters.geoRegion === 'SA' && r.startsWith('mx-')) return true
        return false
      }
      filtered = filtered.filter(m =>
        m.availability?.on_demand?.regions?.some(regionMatchesGeo) ||
        m.availability?.cross_region?.regions?.some(regionMatchesGeo) ||
        m.availability?.mantle?.regions?.some(regionMatchesGeo)
      )
    }
  }

  // Model status filter - handles MIXED status models appearing in applicable filters
  if (filters.modelStatus && filters.modelStatus !== 'All Status') {
    filtered = filtered.filter(m => {
      const status = m.lifecycle?.status || m.model_status
      const globalStatus = m.lifecycle?.global_status
      const statusSummary = m.lifecycle?.status_summary
      
      // If filtering for MIXED, only include models with global_status === 'MIXED'
      if (filters.modelStatus === 'MIXED') {
        return globalStatus === 'MIXED'
      }
      
      // For ACTIVE, LEGACY, EOL filters:
      // Include if direct status matches OR if MIXED model has regions with that status
      if (status === filters.modelStatus) {
        return true
      }
      
      // Check if this is a MIXED model that has regions with the filtered status
      if (globalStatus === 'MIXED' && statusSummary) {
        const regionsWithStatus = statusSummary[filters.modelStatus]
        return regionsWithStatus && regionsWithStatus.length > 0
      }
      
      return false
    })
  }

  // New nested routing filter (matches RegionalAvailability structure)
  if (filters.selectedRouting) {
    if (filters.selectedRouting === 'in_region') {
      // Filter by In-Region availability
      filtered = filtered.filter(m => {
        const hideInRegion = m.availability?.hide_in_region ?? false
        const hasMantle = m.availability?.mantle?.supported === true ||
                         (m.availability?.mantle?.regions?.length > 0)
        const hasRuntime = !hideInRegion && (
          m.availability?.on_demand?.supported === true ||
          (m.availability?.on_demand?.regions?.length > 0)
        )
        
        // API sub-filter (single-select)
        if (filters.selectedApi === 'runtime_api') {
          return hasRuntime
        }
        if (filters.selectedApi === 'mantle') {
          return hasMantle
        }
        // No API filter (All) - show models with runtime OR mantle
        return hasRuntime || hasMantle
      })
    } else if (filters.selectedRouting === 'cris') {
      // Filter by CRIS availability
      filtered = filtered.filter(m => {
        const crisSupported = m.availability?.cross_region?.supported === true
        if (!crisSupported) return false
        
        // CRIS scope sub-filter (multi-select)
        if (filters.selectedCrisScopes && filters.selectedCrisScopes.length > 0) {
          const modelScopes = getCrisGeoScopes(m)
          return filters.selectedCrisScopes.some(scope => modelScopes.includes(scope))
        }
        return true
      })
    }
  }

  // Geo and Region filter (applies when routing is not 'cris')
  // selectedRegions takes precedence over selectedGeos for specific region filtering
  if (filters.selectedRouting !== 'cris') {
    const hasSelectedRegions = filters.selectedRegions && filters.selectedRegions.length > 0
    const hasSelectedGeos = filters.selectedGeos && filters.selectedGeos.length > 0
    
    if (hasSelectedRegions || hasSelectedGeos) {
      // Map business geos to AWS region prefixes
      const geoToPrefixes = {
        'NAMER': ['us-', 'ca-'],
        'EMEA': ['eu-', 'me-', 'af-', 'il-'],
        'APAC': ['ap-'],
        'LATAM': ['sa-', 'mx-'],
        'GOVCLOUD': ['us-gov-'],
      }
      
      // Get prefixes from selected geos
      const selectedPrefixes = hasSelectedGeos 
        ? filters.selectedGeos.flatMap(geo => geoToPrefixes[geo] || [])
        : []
      
      // Get specific region codes
      const selectedRegionCodes = hasSelectedRegions ? filters.selectedRegions : []
      
      filtered = filtered.filter(m => {
        const allRegions = [
          ...(m.availability?.on_demand?.regions || []),
          ...(m.availability?.cross_region?.regions || []),
          ...(m.availability?.mantle?.regions || []),
        ]
        
        // Check if model is available in any selected specific region
        const matchesSpecificRegion = selectedRegionCodes.length > 0 &&
          allRegions.some(region => selectedRegionCodes.includes(region))
        
        // Check if model is available in any selected geo (via prefix matching)
        const matchesGeoPrefix = selectedPrefixes.length > 0 &&
          allRegions.some(region => 
            selectedPrefixes.some(prefix => region.startsWith(prefix))
          )
        
        // Model passes if it matches either specific regions OR geo prefixes
        return matchesSpecificRegion || matchesGeoPrefix
      })
    }
  }

  // Streaming support filter
  if (filters.streamingSupport && filters.streamingSupport !== 'All Models') {
    const supported = filters.streamingSupport === 'Streaming Supported'
    filtered = filtered.filter(m => m.streaming === supported)
  }

  // Pricing availability filter - use getPricingForModel if available, fallback to has_pricing flag
  if (filters.pricingFilter && filters.pricingFilter !== 'All Models') {
    filtered = filtered.filter(m => {
      // Use getPricingForModel for accurate pricing check if available
      let hasPricing = false
      if (getPricingForModel) {
        const pricingResult = getPricingForModel(m, filters.primaryRegion || 'us-east-1')
        const summary = pricingResult?.summary
        hasPricing = summary?.inputPrice != null || 
                     summary?.outputPrice != null || 
                     summary?.imagePrice != null || 
                     summary?.videoPrice != null
      } else {
        // Fallback to has_pricing flag
        hasPricing = m.has_pricing === true
      }
      
      if (filters.pricingFilter === 'Has Pricing') {
        return hasPricing
      } else if (filters.pricingFilter === 'No Pricing') {
        return !hasPricing
      }
      return true
    })
  }

  // Context window filter - use the largest available context window
  if (filters.contextFilter && filters.contextFilter !== 'All Models') {
    filtered = filtered.filter(m => {
      // Get all possible context window values
      const defaultCtx = m.specs?.context_window
      const extendedCtx = m.specs?.extended_context_window || m.specs?.extended_context
      const contextWindows = m.specs?.context_windows // Array of context window objects
      
      // Find the maximum context window
      let maxCtx = defaultCtx || 0
      if (extendedCtx && extendedCtx > maxCtx) maxCtx = extendedCtx
      if (contextWindows && Array.isArray(contextWindows)) {
        contextWindows.forEach(cw => {
          const size = cw.context_window || cw.size || 0
          if (size > maxCtx) maxCtx = size
        })
      }
      
      if (typeof maxCtx !== 'number' || maxCtx === 0) return false
      
      switch (filters.contextFilter) {
        case 'Small (< 32K)': return maxCtx < 32000
        case 'Medium (32K-128K)': return maxCtx >= 32000 && maxCtx < 128000
        case 'Large (128K-500K)': return maxCtx >= 128000 && maxCtx < 500000
        case 'XL (> 500K)': return maxCtx >= 500000
        default: return true
      }
    })
  }

  // Modality filter
  if (filters.modality && filters.modality !== 'All Modalities') {
    filtered = filtered.filter(m =>
      m.modalities?.input_modalities?.includes(filters.modality) ||
      m.modalities?.output_modalities?.includes(filters.modality)
    )
  }

  // Capabilities filter - handle normalized values
  // Selected values are normalized (e.g., "Text Generation"), model data has raw values
  if (filters.capabilities && filters.capabilities.length > 0) {
    filtered = filtered.filter(m => {
      if (!m.capabilities) return false
      // Check if any model capability maps to any selected normalized value
      return m.capabilities.some(cap => {
        const normalized = normalizeValue(cap, CAPABILITY_OPTIONS, false)
        return normalized && filters.capabilities.includes(normalized)
      })
    })
  }

  // Use cases filter - handle normalized values
  if (filters.useCases && filters.useCases.length > 0) {
    filtered = filtered.filter(m => {
      if (!m.use_cases) return false
      return m.use_cases.some(uc => {
        const normalized = normalizeValue(uc, USE_CASE_OPTIONS, false)
        return normalized && filters.useCases.includes(normalized)
      })
    })
  }

  // Customizations filter - handle normalized values
  if (filters.customizations && filters.customizations.length > 0) {
    filtered = filtered.filter(m => {
      const supported = m.customization?.customization_supported
      if (!supported) return false
      return supported.some(c => {
        const normalized = CUSTOMIZATION_OPTIONS[c]
        return normalized && filters.customizations.includes(normalized)
      })
    })
  }

  // Languages filter - handle normalized values
  if (filters.languages && filters.languages.length > 0) {
    filtered = filtered.filter(m => {
      if (!m.languages) return false
      return m.languages.some(lang => {
        const normalized = normalizeValue(lang, LANGUAGE_OPTIONS, false)
        return normalized && filters.languages.includes(normalized)
      })
    })
  }

  // Batch support filter
  if (filters.batchSupport && filters.batchSupport !== 'All Models') {
    const supported = filters.batchSupport === 'Batch Supported'
    filtered = filtered.filter(m => {
      const hasBatch = m.availability?.batch?.supported === true || 
                       m.consumption_options?.includes('batch')
      return supported ? hasBatch : !hasBatch
    })
  }

  // Reserved support filter
  if (filters.reservedSupport && filters.reservedSupport !== 'All Models') {
    const supported = filters.reservedSupport === 'Reserved Supported'
    filtered = filtered.filter(m => {
      const hasReserved = m.availability?.reserved?.supported === true ||
                          m.consumption_options?.includes('reserved')
      return supported ? hasReserved : !hasReserved
    })
  }

  // Flex pricing filter (requires getPricingForModel)
  if (filters.flexPricing && filters.flexPricing !== 'All Models' && getPricingForModel) {
    const hasFlex = filters.flexPricing === 'Has Flex'
    filtered = filtered.filter(m => {
      const pricingResult = getPricingForModel(m, filters.primaryRegion || 'us-east-1')
      const tiers = pricingResult?.fullPricing?.available_dimensions?.tiers || []
      const hasFlexTier = tiers.includes('flex')
      return hasFlex ? hasFlexTier : !hasFlexTier
    })
  }

  // Priority pricing filter (requires getPricingForModel)
  if (filters.priorityPricing && filters.priorityPricing !== 'All Models' && getPricingForModel) {
    const hasPriority = filters.priorityPricing === 'Has Priority'
    filtered = filtered.filter(m => {
      const pricingResult = getPricingForModel(m, filters.primaryRegion || 'us-east-1')
      const tiers = pricingResult?.fullPricing?.available_dimensions?.tiers || []
      const hasPriorityTier = tiers.includes('priority')
      return hasPriority ? hasPriorityTier : !hasPriorityTier
    })
  }

  // Primary region availability filter (skip if 'all' is selected)
  // Checks in-region, CRIS source regions, and Mantle regions
  if (filters.primaryRegion && filters.primaryRegion !== 'all') {
    const modelAvailableInRegion = (m, region) =>
      m.availability?.on_demand?.regions?.includes(region) ||
      m.availability?.cross_region?.regions?.includes(region) ||
      m.availability?.mantle?.regions?.includes(region)

    if (isGeoSelection(filters.primaryRegion)) {
      // GEO selection - filter models available in ANY region within that geo
      const geoRegions = getRegionsForGeo(filters.primaryRegion, awsRegions)
      filtered = filtered.filter(m =>
        geoRegions.some(region => modelAvailableInRegion(m, region))
      )
    } else {
      // Single region selection
      filtered = filtered.filter(m =>
        modelAvailableInRegion(m, filters.primaryRegion)
      )
    }
  }

  return filtered
}

/**
 * Count active filters (excluding defaults)
 */
export function countActiveFilters(filters) {
  let count = 0

  if (filters.searchQuery) count++
  if (filters.providers?.length > 0) count++
  if (filters.geoRegion && filters.geoRegion !== 'All Regions') count++
  if (filters.modelStatus && filters.modelStatus !== 'All Status') count++
  if (filters.streamingSupport && filters.streamingSupport !== 'All Models') count++
  if (filters.pricingFilter && filters.pricingFilter !== 'All Models') count++
  if (filters.contextFilter && filters.contextFilter !== 'All Models') count++
  if (filters.modality && filters.modality !== 'All Modalities') count++
  if (filters.capabilities?.length > 0) count++
  if (filters.useCases?.length > 0) count++
  if (filters.customizations?.length > 0) count++
  if (filters.languages?.length > 0) count++
  // New nested routing filters
  if (filters.selectedRouting) count++
  if (filters.selectedApi) count++
  if (filters.selectedCrisScopes?.length > 0) count++
  if (filters.selectedGeos?.length > 0) count++
  if (filters.selectedRegions?.length > 0) count++
  // Features filters
  if (filters.batchSupport && filters.batchSupport !== 'All Models') count++
  if (filters.reservedSupport && filters.reservedSupport !== 'All Models') count++
  if (filters.flexPricing && filters.flexPricing !== 'All Models') count++
  if (filters.priorityPricing && filters.priorityPricing !== 'All Models') count++

  return count
}

/**
 * @deprecated Use countModelsByRouting and countModelsByInferenceMode instead.
 * Count models by consumption type
 * @param {Array} models - Array of models
 * @returns {Object} Counts by consumption type key
 */
export function countModelsByConsumptionType(models) {
  const counts = {}
  
  Object.keys(consumptionTypeDefinitions).forEach(type => {
    const def = consumptionTypeDefinitions[type]
    counts[type] = models.filter(m => {
      if (type === 'cris') {
        return m.availability?.cross_region?.supported === true
      }
      if (type === 'runtime_api') {
        if (m.availability?.hide_in_region) return false
        return m.availability?.on_demand?.supported === true ||
               (m.availability?.on_demand?.regions?.length > 0)
      }
      const avail = m.availability?.[def.availabilityPath]
      return avail?.supported === true || (avail?.regions?.length > 0)
    }).length
  })
  
  return counts
}

/**
 * @deprecated Use getLocationOptionsForRouting instead.
 * Get available locations based on consumption type selection
 * @param {Array} consumptionTypes - Selected consumption types
 * @param {Array} models - Array of models (for dynamic region extraction)
 * @returns {Object} Location options grouped by type
 */
export function getLocationOptionsForConsumption(consumptionTypes, models = []) {
  const hasCrisOrReserved = consumptionTypes?.some(t => t === 'cris' || t === 'reserved')
  
  // Build dynamic regions from models if available
  const dynamicRegions = models.length > 0 ? buildAwsRegionsFromModels(models) : awsRegions
  
  // Group regions by geo
  const regionsByGeo = {}
  dynamicRegions.forEach(r => {
    if (!regionsByGeo[r.geo]) regionsByGeo[r.geo] = []
    regionsByGeo[r.geo].push(r)
  })
  
  const geoLabels = {
    US: 'US',
    EU: 'EU',
    AP: 'Asia Pacific',
    CA: 'Canada',
    SA: 'South America',
    ME: 'Middle East',
    AF: 'Africa',
    GOV: 'GovCloud',
  }
  
  const result = {
    endpoints: hasCrisOrReserved ? crisEndpointOptions : [],
    geos: Object.entries(regionsByGeo)
      .filter(([geo]) => geo !== 'GOV') // Exclude GovCloud from geo options
      .map(([geo, regions]) => ({
        value: `geo:${geo}`,
        label: geoLabels[geo] || geo,
        count: regions.length
      })),
    regions: dynamicRegions.filter(r => r.geo !== 'GOV'), // Exclude GovCloud regions
  }
  
  return result
}

// =============================================================================
// NEW ORTHOGONAL FILTER HELPERS
// =============================================================================

/**
 * Check if model supports a routing strategy
 * @param {Object} model - Model object
 * @param {string} strategy - 'in_region' or 'cris'
 * @returns {boolean} Whether model supports the routing strategy
 */
export function modelSupportsRouting(model, strategy) {
  if (strategy === 'in_region') {
    // Respect hide_in_region flag (model has Mantle, so In-Region is hidden)
    if (model.availability?.hide_in_region) return false
    const hasOnDemand = model.availability?.on_demand?.supported === true ||
                        (model.availability?.on_demand?.regions?.length > 0)
    const hasMantle = model.availability?.mantle?.supported === true ||
                      (model.availability?.mantle?.regions?.length > 0)
    return hasOnDemand || hasMantle
  }
  if (strategy === 'cris') {
    return model.availability?.cross_region?.supported === true
  }
  return false
}

/**
 * Check if model supports an inference mode
 * @param {Object} model - Model object
 * @param {string} mode - 'on_demand', 'batch', or 'provisioned'
 * @returns {boolean} Whether model supports the inference mode
 */
export function modelSupportsInferenceMode(model, mode) {
  if (mode === 'on_demand') {
    // On-demand is supported if model has any availability
    return model.availability?.on_demand?.supported === true ||
           model.availability?.cross_region?.supported === true ||
           model.availability?.mantle?.supported === true
  }
  if (mode === 'batch') {
    return model.availability?.batch?.supported === true
  }
  if (mode === 'provisioned') {
    return model.availability?.provisioned?.supported === true
  }
  return false
}

/**
 * Count models by routing strategy
 * @param {Array} models - Array of models
 * @returns {Object} Counts by routing strategy { in_region: N, cris: N }
 */
export function countModelsByRouting(models) {
  return {
    in_region: models.filter(m => modelSupportsRouting(m, 'in_region')).length,
    cris: models.filter(m => modelSupportsRouting(m, 'cris')).length,
  }
}

/**
 * Count models by inference mode
 * @param {Array} models - Array of models
 * @returns {Object} Counts by inference mode { on_demand: N, batch: N, provisioned: N }
 */
export function countModelsByInferenceMode(models) {
  return {
    on_demand: models.filter(m => modelSupportsInferenceMode(m, 'on_demand')).length,
    batch: models.filter(m => modelSupportsInferenceMode(m, 'batch')).length,
    provisioned: models.filter(m => modelSupportsInferenceMode(m, 'provisioned')).length,
  }
}

/**
 * Get available locations based on routing strategy selection
 * @param {Array} routingStrategies - Selected routing strategies ['in_region', 'cris']
 * @param {Array} models - Array of models (for dynamic region extraction)
 * @returns {Object} Location options grouped by type { endpoints, geos, regions }
 */
export function getLocationOptionsForRouting(routingStrategies, models = []) {
  const hasCris = routingStrategies?.includes('cris')
  
  // Build dynamic regions from models if available
  const dynamicRegions = models.length > 0 ? buildAwsRegionsFromModels(models) : awsRegions
  
  // Group regions by geo
  const regionsByGeo = {}
  dynamicRegions.forEach(r => {
    if (!regionsByGeo[r.geo]) regionsByGeo[r.geo] = []
    regionsByGeo[r.geo].push(r)
  })
  
  // Business geo groupings (mapping AWS geos to business regions)
  const businessGeos = [
    { value: 'geo:NAMER', label: 'NAMER', geos: ['US', 'CA'] },
    { value: 'geo:EMEA', label: 'EMEA', geos: ['EU', 'ME', 'AF'] },
    { value: 'geo:APAC', label: 'APAC', geos: ['AP'] },
    { value: 'geo:LATAM', label: 'LATAM', geos: ['SA'] },
    { value: 'geo:GOVCLOUD', label: 'GOVCLOUD', geos: ['GOV'] },
  ]
  
  // Count regions per business geo
  businessGeos.forEach(bg => {
    bg.count = bg.geos.reduce((sum, geo) => sum + (regionsByGeo[geo]?.length || 0), 0)
  })
  
  return {
    // CRIS endpoints only shown when CRIS is selected
    endpoints: hasCris ? crisEndpointOptions : [],
    // Business geos always available (filter out empty ones)
    geos: businessGeos.filter(g => g.count > 0),
    // Individual regions always available
    regions: dynamicRegions,
  }
}
