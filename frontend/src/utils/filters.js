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

  sorted.sort((a, b) => {
    switch (sortBy) {
      case 'newest': {
        const dateA = a.lifecycle?.release_date || 0
        const dateB = b.lifecycle?.release_date || 0
        return dateB - dateA // Newest first (higher timestamp first)
      }
      case 'name-asc':
        return (a.model_name || '').localeCompare(b.model_name || '')
      case 'name-desc':
        return (b.model_name || '').localeCompare(a.model_name || '')
      case 'provider-asc':
        return (a.model_provider || '').localeCompare(b.model_provider || '')
      case 'context-desc': {
        const ctxA = a.specs?.context_window || 0
        const ctxB = b.specs?.context_window || 0
        return ctxB - ctxA // Largest first
      }
      case 'context-asc': {
        const ctxA = a.specs?.context_window || 0
        const ctxB = b.specs?.context_window || 0
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
 * Initial filter state
 */
export const initialFilterState = {
  searchQuery: '',
  primaryRegion: 'all',
  providers: [],
  geoRegion: 'All Regions',
  modelStatus: 'All Status',
  crisSupport: 'All Models',
  streamingSupport: 'All Models',
  mantleSupport: 'All Models',
  pricingFilter: 'All Models',
  consumptionOptions: [],
  useCases: [],
  modality: 'All Modalities',
  capabilities: [],
  customizations: [],
  languages: [],
  contextFilter: 'All Models',
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
 * Extract unique capabilities from models
 */
export function extractCapabilities(models) {
  const capabilities = new Set()
  models.forEach(m => {
    if (m.capabilities && Array.isArray(m.capabilities)) {
      m.capabilities.forEach(cap => capabilities.add(cap))
    }
  })
  return Array.from(capabilities).sort()
}

/**
 * Extract unique use cases from models
 */
export function extractUseCases(models) {
  const useCases = new Set()
  models.forEach(m => {
    if (m.use_cases && Array.isArray(m.use_cases)) {
      m.use_cases.forEach(uc => useCases.add(uc))
    }
  })
  return Array.from(useCases).sort()
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

  // CRIS support filter - supports geographic scope filtering
  if (filters.crisSupport && filters.crisSupport !== 'All Models') {
    if (filters.crisSupport === 'CRIS Not Supported') {
      filtered = filtered.filter(m => !m.availability?.cross_region?.supported)
    } else {
      // Filter by geographic scope (GLOBAL, US, EU, APAC)
      filtered = filtered.filter(m => {
        if (!m.availability?.cross_region?.supported) return false
        const scopes = getCrisGeoScopes(m)
        return scopes.includes(filters.crisSupport)
      })
    }
  }

  // Streaming support filter
  if (filters.streamingSupport && filters.streamingSupport !== 'All Models') {
    const supported = filters.streamingSupport === 'Streaming Supported'
    filtered = filtered.filter(m => m.streaming === supported)
  }

  // Mantle support filter
  if (filters.mantleSupport && filters.mantleSupport !== 'All Models') {
    filtered = filtered.filter(m => {
      const mantleSupported = m.availability?.mantle?.supported
      const mantleOnly = m.availability?.mantle?.only
      
      switch (filters.mantleSupport) {
        case 'Mantle Supported':
          return mantleSupported
        case 'Mantle Only':
          return mantleOnly
        case 'No Mantle':
          return !mantleSupported
        default:
          return true
      }
    })
  }

  // Pricing availability filter - use has_pricing flag from model data
  if (filters.pricingFilter && filters.pricingFilter !== 'All Models') {
    filtered = filtered.filter(m => {
      const hasPricing = m.has_pricing === true
      
      if (filters.pricingFilter === 'Has Pricing') {
        return hasPricing
      } else if (filters.pricingFilter === 'No Pricing') {
        return !hasPricing
      }
      return true
    })
  }

  // Context window filter
  if (filters.contextFilter && filters.contextFilter !== 'All Models') {
    filtered = filtered.filter(m => {
      const ctx = m.specs?.context_window
      if (typeof ctx !== 'number') return false
      switch (filters.contextFilter) {
        case 'Small (< 32K)': return ctx < 32000
        case 'Medium (32K-128K)': return ctx >= 32000 && ctx < 128000
        case 'Large (128K-500K)': return ctx >= 128000 && ctx < 500000
        case 'XL (> 500K)': return ctx >= 500000
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

  // Capabilities filter
  if (filters.capabilities && filters.capabilities.length > 0) {
    filtered = filtered.filter(m =>
      filters.capabilities.some(cap => m.capabilities?.includes(cap))
    )
  }

  // Use cases filter
  if (filters.useCases && filters.useCases.length > 0) {
    filtered = filtered.filter(m =>
      filters.useCases.some(uc => m.use_cases?.includes(uc))
    )
  }

  // Customizations filter
  if (filters.customizations && filters.customizations.length > 0) {
    filtered = filtered.filter(m =>
      filters.customizations.some(c => m.customization?.customization_supported?.includes(c))
    )
  }

  // Languages filter
  if (filters.languages && filters.languages.length > 0) {
    filtered = filtered.filter(m =>
      filters.languages.some(lang => m.languages?.includes(lang))
    )
  }

  // Consumption options filter
  if (filters.consumptionOptions && filters.consumptionOptions.length > 0) {
    filtered = filtered.filter(m =>
      filters.consumptionOptions.some(opt => m.consumption_options?.includes(opt))
    )
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
  if (filters.crisSupport && filters.crisSupport !== 'All Models') count++
  if (filters.streamingSupport && filters.streamingSupport !== 'All Models') count++
  if (filters.mantleSupport && filters.mantleSupport !== 'All Models') count++
  if (filters.pricingFilter && filters.pricingFilter !== 'All Models') count++
  if (filters.consumptionOptions?.length > 0) count++
  if (filters.contextFilter && filters.contextFilter !== 'All Models') count++
  if (filters.modality && filters.modality !== 'All Modalities') count++
  if (filters.capabilities?.length > 0) count++
  if (filters.useCases?.length > 0) count++
  if (filters.customizations?.length > 0) count++
  if (filters.languages?.length > 0) count++

  return count
}
