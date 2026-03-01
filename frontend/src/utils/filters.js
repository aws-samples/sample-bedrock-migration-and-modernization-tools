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
 * Comprehensive region metadata derived from backend profiler-config.json region_locations.
 * Maps region code → { label, geo } for all known AWS regions.
 */
const REGION_META = {
  'us-east-1': { label: 'N. Virginia', geo: 'US' },
  'us-east-2': { label: 'Ohio', geo: 'US' },
  'us-west-1': { label: 'N. California', geo: 'US' },
  'us-west-2': { label: 'Oregon', geo: 'US' },
  'eu-west-1': { label: 'Ireland', geo: 'EU' },
  'eu-west-2': { label: 'London', geo: 'EU' },
  'eu-west-3': { label: 'Paris', geo: 'EU' },
  'eu-central-1': { label: 'Frankfurt', geo: 'EU' },
  'eu-central-2': { label: 'Zurich', geo: 'EU' },
  'eu-north-1': { label: 'Stockholm', geo: 'EU' },
  'eu-south-1': { label: 'Milan', geo: 'EU' },
  'eu-south-2': { label: 'Spain', geo: 'EU' },
  'ap-northeast-1': { label: 'Tokyo', geo: 'AP' },
  'ap-northeast-2': { label: 'Seoul', geo: 'AP' },
  'ap-northeast-3': { label: 'Osaka', geo: 'AP' },
  'ap-southeast-1': { label: 'Singapore', geo: 'AP' },
  'ap-southeast-2': { label: 'Sydney', geo: 'AP' },
  'ap-southeast-3': { label: 'Jakarta', geo: 'AP' },
  'ap-southeast-4': { label: 'Melbourne', geo: 'AP' },
  'ap-southeast-5': { label: 'Malaysia', geo: 'AP' },
  'ap-southeast-6': { label: 'Auckland', geo: 'AP' },
  'ap-southeast-7': { label: 'Thailand', geo: 'AP' },
  'ap-south-1': { label: 'Mumbai', geo: 'AP' },
  'ap-south-2': { label: 'Hyderabad', geo: 'AP' },
  'ap-east-1': { label: 'Hong Kong', geo: 'AP' },
  'ap-east-2': { label: 'Taipei', geo: 'AP' },
  'ca-central-1': { label: 'Montreal', geo: 'CA' },
  'ca-west-1': { label: 'Calgary', geo: 'CA' },
  'sa-east-1': { label: 'Sao Paulo', geo: 'SA' },
  'me-south-1': { label: 'Bahrain', geo: 'ME' },
  'me-central-1': { label: 'UAE', geo: 'ME' },
  'il-central-1': { label: 'Tel Aviv', geo: 'ME' },
  'af-south-1': { label: 'Cape Town', geo: 'AF' },
  'mx-central-1': { label: 'Mexico City', geo: 'SA' },
}

/**
 * Auto-detect geo from region code prefix for unknown regions.
 */
const GEO_PREFIX_MAP = {
  us: 'US', ca: 'CA', eu: 'EU', ap: 'AP', sa: 'SA',
  me: 'ME', af: 'AF', il: 'ME', mx: 'SA', in: 'AP',
}

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
 * Extracts all unique regions from models[].in_region and
 * models[].cross_region_inference.source_regions, then returns sorted
 * { value, label, geo } entries.
 *
 * Falls back to DEFAULT_AWS_REGIONS if models is empty.
 */
export function buildAwsRegionsFromModels(models) {
  if (!models || !models.length) return DEFAULT_AWS_REGIONS

  const regionCodes = new Set()
  models.forEach(m => {
    if (m.in_region) {
      m.in_region.forEach(r => regionCodes.add(r))
    }
    if (m.cross_region_inference?.source_regions) {
      m.cross_region_inference.source_regions.forEach(r => regionCodes.add(r))
    }
    if (m.mantle_inference?.mantle_regions) {
      m.mantle_inference.mantle_regions.forEach(r => regionCodes.add(r))
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
 * Geographic region options
 */
export const geoRegionOptions = [
  { value: 'All Regions', label: 'All Regions' },
  { value: 'US', label: 'US Regions' },
  { value: 'EU', label: 'EU Regions' },
  { value: 'AP', label: 'Asia Pacific' },
  { value: 'CA', label: 'Canada' },
  { value: 'SA', label: 'South America' },
  { value: 'ME', label: 'Middle East' },
  { value: 'AF', label: 'Africa' },
]

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
  const profiles = model?.cross_region_inference?.profiles || []
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
    const pricing = getPricingForModel(model, preferredRegion)
    if (!pricing) return null
    return type === 'input' ? pricing.input_price : pricing.output_price
  }

  sorted.sort((a, b) => {
    switch (sortBy) {
      case 'newest': {
        const dateA = a.model_lifecycle?.release_date || 0
        const dateB = b.model_lifecycle?.release_date || 0
        return dateB - dateA // Newest first (higher timestamp first)
      }
      case 'name-asc':
        return (a.model_name || '').localeCompare(b.model_name || '')
      case 'name-desc':
        return (b.model_name || '').localeCompare(a.model_name || '')
      case 'provider-asc':
        return (a.model_provider || '').localeCompare(b.model_provider || '')
      case 'context-desc': {
        const ctxA = a.converse_data?.context_window || 0
        const ctxB = b.converse_data?.context_window || 0
        return ctxB - ctxA // Largest first
      }
      case 'context-asc': {
        const ctxA = a.converse_data?.context_window || 0
        const ctxB = b.converse_data?.context_window || 0
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
    if (m.model_capabilities && Array.isArray(m.model_capabilities)) {
      m.model_capabilities.forEach(cap => capabilities.add(cap))
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
    if (m.model_use_cases && Array.isArray(m.model_use_cases)) {
      m.model_use_cases.forEach(uc => useCases.add(uc))
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
    if (m.model_modalities?.input_modalities) {
      m.model_modalities.input_modalities.forEach(mod => modalities.add(mod))
    }
    if (m.model_modalities?.output_modalities) {
      m.model_modalities.output_modalities.forEach(mod => modalities.add(mod))
    }
  })
  return Array.from(modalities).sort()
}

/**
 * Apply all filters to models
 */
export function applyFilters(models, filters) {
  let filtered = [...models]

  // Search query
  if (filters.searchQuery) {
    const query = filters.searchQuery.toLowerCase()
    filtered = filtered.filter(m =>
      m.model_name?.toLowerCase().includes(query) ||
      m.model_id?.toLowerCase().includes(query) ||
      m.model_provider?.toLowerCase().includes(query) ||
      m.model_capabilities?.some(c => c.toLowerCase().includes(query))
    )
  }

  // Provider filter
  if (filters.providers && filters.providers.length > 0) {
    filtered = filtered.filter(m => filters.providers.includes(m.model_provider))
  }

  // Geographic region filter
  if (filters.geoRegion && filters.geoRegion !== 'All Regions') {
    const prefixMap = { 'US': 'us-', 'EU': 'eu-', 'AP': 'ap-', 'CA': 'ca-', 'SA': 'sa-', 'ME': 'me-', 'AF': 'af-' }
    const prefix = prefixMap[filters.geoRegion]
    if (prefix) {
      filtered = filtered.filter(m =>
        m.in_region?.some(r => {
          if (r.startsWith(prefix)) return true
          // Special cases: il- regions belong to ME geo, mx- regions belong to SA geo
          if (filters.geoRegion === 'ME' && r.startsWith('il-')) return true
          if (filters.geoRegion === 'SA' && r.startsWith('mx-')) return true
          return false
        })
      )
    }
  }

  // Model status filter - handles MIXED status models appearing in applicable filters
  if (filters.modelStatus && filters.modelStatus !== 'All Status') {
    filtered = filtered.filter(m => {
      const status = m.model_lifecycle?.status || m.model_status
      const globalStatus = m.model_lifecycle?.global_status
      const statusSummary = m.model_lifecycle?.status_summary
      
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
      filtered = filtered.filter(m => !m.cross_region_inference?.supported)
    } else {
      // Filter by geographic scope (GLOBAL, US, EU, APAC)
      filtered = filtered.filter(m => {
        if (!m.cross_region_inference?.supported) return false
        const scopes = getCrisGeoScopes(m)
        return scopes.includes(filters.crisSupport)
      })
    }
  }

  // Streaming support filter
  if (filters.streamingSupport && filters.streamingSupport !== 'All Models') {
    const supported = filters.streamingSupport === 'Streaming Supported'
    filtered = filtered.filter(m => m.streaming_supported === supported)
  }

  // Context window filter
  if (filters.contextFilter && filters.contextFilter !== 'All Models') {
    filtered = filtered.filter(m => {
      const ctx = m.converse_data?.context_window
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
      m.model_modalities?.input_modalities?.includes(filters.modality) ||
      m.model_modalities?.output_modalities?.includes(filters.modality)
    )
  }

  // Capabilities filter
  if (filters.capabilities && filters.capabilities.length > 0) {
    filtered = filtered.filter(m =>
      filters.capabilities.some(cap => m.model_capabilities?.includes(cap))
    )
  }

  // Use cases filter
  if (filters.useCases && filters.useCases.length > 0) {
    filtered = filtered.filter(m =>
      filters.useCases.some(uc => m.model_use_cases?.includes(uc))
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
      filters.languages.some(lang => m.languages_supported?.includes(lang))
    )
  }

  // Consumption options filter
  if (filters.consumptionOptions && filters.consumptionOptions.length > 0) {
    filtered = filtered.filter(m =>
      filters.consumptionOptions.some(opt => m.consumption_options?.includes(opt))
    )
  }

  // Primary region availability filter (skip if 'all' is selected)
  if (filters.primaryRegion && filters.primaryRegion !== 'all') {
    if (isGeoSelection(filters.primaryRegion)) {
      // GEO selection - filter models available in ANY region within that geo
      const geoRegions = getRegionsForGeo(filters.primaryRegion, awsRegions)
      filtered = filtered.filter(m =>
        geoRegions.some(region => m.in_region?.includes(region))
      )
    } else {
      // Single region selection
      filtered = filtered.filter(m =>
        m.in_region?.includes(filters.primaryRegion)
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
  if (filters.consumptionOptions?.length > 0) count++
  if (filters.contextFilter && filters.contextFilter !== 'All Models') count++
  if (filters.modality && filters.modality !== 'All Modalities') count++
  if (filters.capabilities?.length > 0) count++
  if (filters.useCases?.length > 0) count++
  if (filters.customizations?.length > 0) count++
  if (filters.languages?.length > 0) count++

  return count
}
