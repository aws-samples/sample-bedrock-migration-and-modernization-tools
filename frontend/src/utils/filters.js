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
 * AWS Regions configuration
 */
export const awsRegions = [
  { value: 'us-east-1', label: 'N. Virginia (us-east-1)', geo: 'US' },
  { value: 'us-east-2', label: 'Ohio (us-east-2)', geo: 'US' },
  { value: 'us-west-1', label: 'N. California (us-west-1)', geo: 'US' },
  { value: 'us-west-2', label: 'Oregon (us-west-2)', geo: 'US' },
  { value: 'eu-west-1', label: 'Ireland (eu-west-1)', geo: 'EU' },
  { value: 'eu-west-2', label: 'London (eu-west-2)', geo: 'EU' },
  { value: 'eu-west-3', label: 'Paris (eu-west-3)', geo: 'EU' },
  { value: 'eu-central-1', label: 'Frankfurt (eu-central-1)', geo: 'EU' },
  { value: 'eu-north-1', label: 'Stockholm (eu-north-1)', geo: 'EU' },
  { value: 'ap-northeast-1', label: 'Tokyo (ap-northeast-1)', geo: 'AP' },
  { value: 'ap-northeast-2', label: 'Seoul (ap-northeast-2)', geo: 'AP' },
  { value: 'ap-southeast-1', label: 'Singapore (ap-southeast-1)', geo: 'AP' },
  { value: 'ap-southeast-2', label: 'Sydney (ap-southeast-2)', geo: 'AP' },
  { value: 'ap-south-1', label: 'Mumbai (ap-south-1)', geo: 'AP' },
  { value: 'ca-central-1', label: 'Canada (ca-central-1)', geo: 'CA' },
  { value: 'sa-east-1', label: 'Sao Paulo (sa-east-1)', geo: 'SA' },
]

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
]

/**
 * Model status options
 */
export const modelStatusOptions = [
  { value: 'All Status', label: 'All Status' },
  { value: 'ACTIVE', label: 'Active' },
  { value: 'LEGACY', label: 'Legacy' },
]

/**
 * CRIS support options
 */
export const crisSupportOptions = [
  { value: 'All Models', label: 'All Models' },
  { value: 'CRIS Supported', label: 'CRIS Supported' },
  { value: 'CRIS Not Supported', label: 'Not Supported' },
]

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
    const prefixMap = { 'US': 'us-', 'EU': 'eu-', 'AP': 'ap-', 'CA': 'ca-', 'SA': 'sa-' }
    const prefix = prefixMap[filters.geoRegion]
    if (prefix) {
      filtered = filtered.filter(m =>
        m.regions_available?.some(r => r.startsWith(prefix))
      )
    }
  }

  // Model status filter
  if (filters.modelStatus && filters.modelStatus !== 'All Status') {
    filtered = filtered.filter(m =>
      m.model_lifecycle?.status === filters.modelStatus ||
      m.model_status === filters.modelStatus
    )
  }

  // CRIS support filter
  if (filters.crisSupport && filters.crisSupport !== 'All Models') {
    const supported = filters.crisSupport === 'CRIS Supported'
    filtered = filtered.filter(m => m.cross_region_inference?.supported === supported)
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
        geoRegions.some(region => m.regions_available?.includes(region))
      )
    } else {
      // Single region selection
      filtered = filtered.filter(m =>
        m.regions_available?.includes(filters.primaryRegion)
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
