import { useState, useEffect, useMemo } from 'react'
import { DATA_URLS } from '../config/dataSource'
import { 
  extractCapabilities, 
  extractUseCases, 
  extractCustomizations, 
  extractLanguages 
} from '../utils/filters'

// Default region from environment variable
const DEFAULT_REGION = import.meta.env.VITE_DEFAULT_REGION || 'us-east-1'

/**
 * Flattens the hierarchical model data into a flat array
 * @param {Object} data - Raw JSON data from bedrock_models.json
 * @returns {Array} Flattened array of model objects
 */
function flattenModels(data) {
  const models = []

  if (!data?.providers) return models

  for (const [providerName, providerData] of Object.entries(data.providers)) {
    if (!providerData?.models) continue

    for (const [modelId, modelData] of Object.entries(providerData.models)) {
      // Skip models flagged as hidden by the backend
      if (modelData.show_model === false) continue

      models.push({
        ...modelData,
        model_id: modelId,
        model_provider: providerName,
      })
    }
  }

  return models
}

/**
 * Extracts unique providers from model data
 * @param {Array} models - Flattened array of models
 * @returns {Array} Unique provider names
 */
function extractProviders(models) {
  return [...new Set(models.map(m => m.model_provider))].sort()
}

/**
 * Extracts unique consumption options from model data
 * @param {Array} models - Flattened array of models
 * @returns {Array} Unique consumption options
 */
function extractConsumptionOptions(models) {
  const options = new Set()
  models.forEach(m => {
    if (m.consumption_options && Array.isArray(m.consumption_options)) {
      m.consumption_options.forEach(opt => options.add(opt))
    }
  })
  return [...options].sort()
}

/**
 * Extracts unique lifecycle statuses from model data
 * @param {Array} models - Flattened array of models
 * @returns {Array} Unique lifecycle statuses
 */
function extractLifecycleStatuses(models) {
  const statuses = new Set()
  models.forEach(m => {
    const status = m.lifecycle?.status || m.model_status
    if (status) {
      statuses.add(status)
    }
  })
  return [...statuses].sort()
}

/**
 * Get pricing for a model from pricing data
 * Uses pricing_file_reference from the model if available, falls back to model_id
 * @param {Object} model - The model object
 * @param {Object} pricingData - The pricing data object
 * @returns {Object|null} Pricing data for the model or null
 */
function isVersionSuffix(modelKey, pricingKey) {
  // After stripping the pricingKey prefix, the remaining suffix must start with
  // a version separator (: or -) to be a legitimate version match.
  // This prevents "minimax-m2" from matching "minimax-m2.5" (suffix ".5").
  if (modelKey.length === pricingKey.length) return true
  const nextChar = modelKey[pricingKey.length]
  return nextChar === ':' || nextChar === '-'
}

function getModelPricing(model, pricingData) {
  if (!pricingData?.providers) return null

  // Respect backend's pricing.available flag — if explicitly false, no pricing
  if (model.pricing?.available === false) return null

  // First try using reference (new) or pricing_file_reference (legacy)
  const pricingRef = model.pricing?.reference ?? model.pricing?.pricing_file_reference
  if (pricingRef?.provider && pricingRef?.model_key) {
    const providerData = pricingData.providers[pricingRef.provider]
    if (providerData?.[pricingRef.model_key]) {
      return providerData[pricingRef.model_key]
    }
    // Fallback: reference model_key may include version suffix (e.g. "-20240307-v1:0")
    // but pricing keys use the base form (e.g. "anthropic.claude-3-haiku")
    // Pick the longest matching key to avoid false positives (e.g. "anthropic.claude" vs "anthropic.claude-3-haiku")
    if (providerData) {
      let bestKey = null
      for (const pricingKey of Object.keys(providerData)) {
        if (pricingRef.model_key.startsWith(pricingKey) && isVersionSuffix(pricingRef.model_key, pricingKey) && (!bestKey || pricingKey.length > bestKey.length)) {
          bestKey = pricingKey
        }
      }
      if (bestKey) return providerData[bestKey]
    }
  }

  // Fallback: try matching by model_id directly
  const modelId = model.model_id
  for (const provider of Object.values(pricingData.providers)) {
    if (provider[modelId]) {
      return provider[modelId]
    }
    // Try prefix match (model_id with version suffix → base pricing key)
    // Pick the longest match to avoid false positives
    let bestKey = null
    for (const pricingKey of Object.keys(provider)) {
      if (modelId.startsWith(pricingKey) && isVersionSuffix(modelId, pricingKey) && (!bestKey || pricingKey.length > bestKey.length)) {
        bestKey = pricingKey
      }
    }
    if (bestKey) return provider[bestKey]
  }

  return null
}

/**
 * Get pricing with specific dimension filters
 * @param {Object} model - The model object
 * @param {Object} pricingData - The pricing data object
 * @param {string} region - The region
 * @param {Object} dimensionFilters - Filters for source, geo, tier, context
 * @returns {Object} Filtered pricing data
 */
export function getFilteredPricing(model, pricingData, region, dimensionFilters = {}) {
  const modelPricing = getModelPricing(model, pricingData)
  if (!modelPricing) return null
  
  return extractSummaryPricing(modelPricing, region, dimensionFilters)
}

/**
 * Get all available dimension options for a model
 * @param {Object} model - The model object
 * @param {Object} pricingData - The pricing data object
 * @returns {Object} Available dimensions
 */
export function getAvailableDimensions(model, pricingData) {
  const modelPricing = getModelPricing(model, pricingData)
  if (!modelPricing) return null
  
  return modelPricing.available_dimensions || {
    sources: ['standard'],
    geos: [],
    tiers: [],
    contexts: ['standard']
  }
}

/**
 * Extract token prices (input/output) from a list of pricing entries.
 * Applies standard filters: skip cache, long-context, reserved, latency, priority; flex fallback.
 * @param {Array} entries - Filtered pricing entries
 * @returns {{ inputPrice: number|null, outputPrice: number|null }}
 */
function extractTokenPricesFromEntries(entries) {
  let inputPrice = null
  let outputPrice = null
  let flexInputPrice = null
  let flexOutputPrice = null

  for (const item of entries) {
    const dim = (item.dimension || '').toLowerCase()
    const desc = (item.description || '').toLowerCase()

    if (dim.includes('cache') || desc.includes('cache')) continue
    if (dim.includes('lctx') || dim.includes('long-context') || dim.includes('longcontext') ||
        desc.includes('long context') || desc.includes('long-context')) continue
    if (dim.includes('reserved') || dim.includes('_tpm_') ||
        desc.includes('reserved') || desc.includes('per minute')) continue
    if (dim.includes('latency') || desc.includes('latency')) continue
    if (dim.includes('-priority')) continue

    const isInput = item.is_input || dim.includes('input') || desc.includes('input')
    const isOutput = item.is_output || dim.includes('output') || desc.includes('output')
    const price = item.price_per_thousand ?? item.price_per_unit

    if (price === 0 || price === null || price === undefined) continue

    const isFlex = dim.includes('-flex')
    if (isFlex) {
      if (isInput && flexInputPrice === null) flexInputPrice = price
      if (isOutput && flexOutputPrice === null) flexOutputPrice = price
    } else {
      if (isInput && inputPrice === null) inputPrice = price
      if (isOutput && outputPrice === null) outputPrice = price
    }
  }

  if (inputPrice === null) inputPrice = flexInputPrice
  if (outputPrice === null) outputPrice = flexOutputPrice

  return { inputPrice, outputPrice }
}

/**
 * Select the best pricing group across all regions using a priority cascade:
 *   1. "On-Demand Global" (CRIS Global - same price everywhere)
 *   2. "On-Demand Geo" (CRIS Geo - find cheapest region)
 *   3. "Mantle" (Mantle pricing)
 *   4. "On-Demand" (In-region - find cheapest region)
 *
 * @param {Object} regions - modelPricing.regions
 * @param {Function} filterByDimensions - dimension filter function
 * @returns {{ entries: Array, pricingSource: string|null }}
 */
function selectBestPricingGroup(regions, filterByDimensions, hideInRegion = false) {
  const CASCADE = [
    { group: 'On-Demand Global', sourcePrefix: 'CRIS Global' },
    { group: 'On-Demand Geo',    sourcePrefix: 'CRIS Geo' },
    { group: 'Mantle',           sourcePrefix: 'Mantle' },
    { group: 'On-Demand',        sourcePrefix: 'In-region' },
  ]

  for (const { group, sourcePrefix } of CASCADE) {
    // Skip In-region pricing when hide_in_region is true
    if (hideInRegion && group === 'On-Demand') continue
    let bestEntries = null
    let bestInputPrice = Infinity
    let bestRegion = null

    for (const [regionKey, regionData] of Object.entries(regions)) {
      if (!regionData?.pricing_groups?.[group]) continue
      const filtered = filterByDimensions(regionData.pricing_groups[group])
      if (filtered.length === 0) continue

      // For Global, price is the same everywhere — just take the first hit
      if (sourcePrefix === 'CRIS Global') {
        return { entries: filtered, pricingSource: sourcePrefix }
      }

      // For Geo and On-Demand, find cheapest by input token price OR unit price
      const { inputPrice } = extractTokenPricesFromEntries(filtered)

      // For non-token pricing (image, video, search_unit), use price_per_unit
      let effectivePrice = inputPrice
      if (effectivePrice === null) {
        const unitPriceEntry = filtered.find(e => e.price_per_unit != null && e.price_per_unit !== 0)
        effectivePrice = unitPriceEntry?.price_per_unit ?? null
      }

      if (effectivePrice !== null && effectivePrice < bestInputPrice) {
        bestInputPrice = effectivePrice
        bestEntries = filtered
        bestRegion = regionKey
      }
    }

    if (bestEntries) {
      const label = `${sourcePrefix} · ${bestRegion}`
      return { entries: bestEntries, pricingSource: label }
    }
  }

  return { entries: [], pricingSource: null }
}

/**
 * Extract summary pricing for a model in a given region
 * Uses a Global → Geo → On-Demand cascade to find best available pricing
 * @param {Object} modelPricing - Pricing data for a model
 * @param {string} region - Preferred region
 * @param {Object} options - Filter options for dimensions
 * @returns {Object} Pricing summary with type and dimension information
 */
function extractSummaryPricing(modelPricing, region = DEFAULT_REGION, options = {}, hideInRegion = false) {
  const nullResult = {
    inputPrice: null,
    outputPrice: null,
    pricingType: null,
    unitLabel: null,
    imagePrice: null,
    imagePrices: null,
    videoPrice: null,
    videoPrices: null,
    dimensions: null,
    availableDimensions: null,
    pricingSource: null,
  }

  if (!modelPricing?.regions) return nullResult

  // Get available dimensions from model-level data
  const availableDimensions = modelPricing.available_dimensions || {
    sources: ['standard'],
    geos: [],
    tiers: [],
    contexts: ['standard']
  }

  // Get model-level pricing type info
  const primaryPricingType = modelPricing.primary_pricing_type || 'token'

  /**
   * Filter pricing entries by dimension options
   * Supports both legacy dimensions (source, geo, tier, context) and new dimensions
   * (inference_mode, geographic_scope, commitment, cache_type)
   * @param {Array} entries - Pricing entries to filter
   * @returns {Array} Filtered entries
   */
  const filterByDimensions = (entries) => {
    if (!entries || entries.length === 0) return []

    // Default filter: exclude Mantle and Reserved from summary
    if (!options || Object.keys(options).length === 0) {
      return entries.filter(e => {
        const dims = e.dimensions || {}
        // Exclude Mantle (both source and inference_mode)
        if (dims.source === 'mantle' || dims.inference_mode === 'mantle') return false
        // Exclude Reserved
        if (dims.inference_mode === 'reserved') return false
        return true
      })
    }

    // Custom filter: support all dimensions (legacy and new)
    return entries.filter(e => {
      const dims = e.dimensions || {}
      // Legacy dimensions
      if (options.source && dims.source !== options.source) return false
      if (options.geo && dims.geo !== options.geo) return false
      if (options.tier && dims.tier !== options.tier) return false
      if (options.context && dims.context !== options.context) return false
      // New dimensions
      if (options.geographic_scope && dims.geographic_scope !== options.geographic_scope) return false
      if (options.inference_mode && dims.inference_mode !== options.inference_mode) return false
      if (options.commitment && dims.commitment !== options.commitment) return false
      if (options.cache_type && dims.cache_type !== options.cache_type) return false
      return true
    })
  }

  // Use cascade to find best pricing group across all regions
  let { entries: onDemand, pricingSource } = selectBestPricingGroup(modelPricing.regions, filterByDimensions, hideInRegion)

  // If no pricing found with default filter (excludes Mantle), retry including Mantle
  // This handles Mantle-only models like MiniMax M2.5
  if (onDemand.length === 0 && (!options || Object.keys(options).length === 0)) {
    const filterIncludingMantle = (entries) => {
      if (!entries || entries.length === 0) return []
      return entries.filter(e => {
        const dims = e.dimensions || {}
        // Only exclude Reserved, allow Mantle
        if (dims.inference_mode === 'reserved') return false
        return true
      })
    }
    const retryResult = selectBestPricingGroup(modelPricing.regions, filterIncludingMantle, hideInRegion)
    onDemand = retryResult.entries
    pricingSource = retryResult.pricingSource
  }

  // Handle image generation models (per-image pricing)
  if (primaryPricingType === 'image_generation') {
    const imagePrices = {}

    for (const item of onDemand) {
      // Check pricing_type or fallback to dimension/unit patterns (T2I, I2I, unit=image)
      const dim = (item.dimension || '').toLowerCase()
      const isImagePricing = item.pricing_type === 'image_generation' ||
                             (item.unit === 'image' && (dim.includes('t2i') || dim.includes('i2i')))
      if (isImagePricing) {
        // Extract resolution and tier from dimension
        // e.g., "NovaCanvas-T2I-1024-Standard" or "TitanImageGeneratorG1-I2I-512-Premium"
        const desc = item.description || ''

        // Try to extract resolution
        const resMatch = dim.match(/(\d{3,4})/)
        const resolution = resMatch ? resMatch[1] : 'standard'

        // Try to extract tier (Standard/Premium)
        const tier = dim.includes('premium') ? 'premium' : 'standard'

        // Try to extract type (T2I = text-to-image, I2I = image-to-image)
        const type = dim.includes('t2i') ? 'text_to_image' :
                     dim.includes('i2i') ? 'image_to_image' : 'generation'

        const key = `${type}_${resolution}_${tier}`
        imagePrices[key] = {
          price: item.price_per_unit,
          resolution,
          tier,
          type,
          description: desc,
        }
      }
    }

    // Get the most common/default price (standard resolution, standard tier, text-to-image)
    const defaultPrice = imagePrices['text_to_image_1024_standard']?.price ||
                        imagePrices['generation_1024_standard']?.price ||
                        Object.values(imagePrices)[0]?.price ||
                        null

    return {
      inputPrice: null,
      outputPrice: null,
      pricingType: 'image_generation',
      unitLabel: 'per image',
      imagePrice: defaultPrice,
      imagePrices: Object.keys(imagePrices).length > 0 ? imagePrices : null,
      videoPrice: null,
      videoPrices: null,
      availableDimensions,
      pricingSource,
    }
  }

  // Handle video generation models (per-video pricing)
  // Also detect video generation from dimensions as fallback (I2V = image-to-video, T2V = text-to-video)
  const hasVideoGenerationDimensions = onDemand.some(item => {
    const dim = (item.dimension || '').toLowerCase()
    return dim.includes('i2v') || dim.includes('t2v') ||
           (dim.includes('video') && item.unit?.toLowerCase() === 'video')
  })

  if (primaryPricingType === 'video_generation' || hasVideoGenerationDimensions) {
    const videoPrices = {}

    for (const item of onDemand) {
      const dimLower = (item.dimension || '').toLowerCase()
      const isVideoGen = item.pricing_type === 'video_generation' ||
                         dimLower.includes('i2v') || dimLower.includes('t2v') ||
                         (dimLower.includes('video') && item.unit?.toLowerCase() === 'video')
      if (isVideoGen) {
        // Extract resolution and fps from dimension
        // e.g., "NovaReel-I2V-Medfps-HDRes", "NovaReel-T2V-Lowfps-SDRes"
        const dim = item.dimension || ''
        const desc = item.description || ''

        // Try to extract type (I2V = image-to-video, T2V = text-to-video)
        const type = dim.includes('I2V') ? 'image_to_video' :
                     dim.includes('T2V') ? 'text_to_video' : 'generation'

        // Try to extract fps tier (Lowfps, Medfps, Highfps)
        const fpsMatch = dim.match(/(Low|Med|High)fps/i)
        const fps = fpsMatch ? fpsMatch[1].toLowerCase() : 'standard'

        // Try to extract resolution (SDRes, HDRes, FHDRes)
        const resMatch = dim.match(/(SD|HD|FHD)Res/i)
        const resolution = resMatch ? resMatch[1].toUpperCase() : 'standard'

        const key = `${type}_${fps}_${resolution}`
        videoPrices[key] = {
          price: item.price_per_unit,
          fps,
          resolution,
          type,
          description: desc,
        }
      }
    }

    // Get the most common/default price (text-to-video, medium fps, HD resolution)
    const defaultPrice = videoPrices['text_to_video_med_HD']?.price ||
                        videoPrices['image_to_video_med_HD']?.price ||
                        videoPrices['generation_standard_standard']?.price ||
                        Object.values(videoPrices)[0]?.price ||
                        null

    return {
      inputPrice: null,
      outputPrice: null,
      pricingType: 'video_generation',
      unitLabel: 'per video',
      imagePrice: null,
      imagePrices: null,
      videoPrice: defaultPrice,
      videoPrices: Object.keys(videoPrices).length > 0 ? videoPrices : null,
      availableDimensions,
      pricingSource,
    }
  }

  // Handle search unit pricing (rerank models like Cohere Rerank, Amazon Rerank)
  if (primaryPricingType === 'search_unit') {
    // Find first search unit pricing entry
    for (const item of onDemand) {
      const price = item.price_per_thousand ?? item.price_per_unit
      if (price !== null && price !== undefined && price !== 0) {
        return {
          inputPrice: price,
          outputPrice: null,
          pricingType: 'search_unit',
          unitLabel: item.unit_label || 'per 1K search units',
          imagePrice: null,
          imagePrices: null,
          videoPrice: null,
          videoPrices: null,
          availableDimensions,
          pricingSource,
        }
      }
    }
  }

  // Handle video per-second pricing (Luma AI Ray)
  if (primaryPricingType === 'video_second') {
    // Find first per-second pricing entry
    for (const item of onDemand) {
      const price = item.price_per_unit
      if (price !== null && price !== undefined && price !== 0) {
        return {
          inputPrice: null,
          outputPrice: null,
          pricingType: 'video_second',
          unitLabel: 'per second',
          imagePrice: null,
          imagePrices: null,
          videoPrice: price,
          videoPrices: null,
          availableDimensions,
          pricingSource,
        }
      }
    }
  }

  // Handle token-based pricing (most common) — use the shared helper
  let { inputPrice, outputPrice } = extractTokenPricesFromEntries(onDemand)

  // Convert from per-1K to per-1M for display
  if (inputPrice !== null) inputPrice = inputPrice * 1000
  if (outputPrice !== null) outputPrice = outputPrice * 1000

  return {
    inputPrice,
    outputPrice,
    pricingType: primaryPricingType,
    unitLabel: primaryPricingType === 'token' ? 'per 1M tokens' : 'per unit',
    imagePrice: null,
    imagePrices: null,
    videoPrice: null,
    videoPrices: null,
    availableDimensions,
    pricingSource,
  }
}

/**
 * Custom hook to load and manage Bedrock model data
 */
export function useModels() {
  const [rawData, setRawData] = useState(null)
  const [pricingData, setPricingData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Load both models and pricing in parallel
    Promise.all([
      fetch(DATA_URLS.models).then(r => {
        if (!r.ok) return Promise.reject('Failed to load models')
        const ct = r.headers.get('content-type') || ''
        if (ct.includes('text/html')) return Promise.reject('Data pipeline has not completed yet. Please refresh in a few minutes.')
        return r.json()
      }),
      fetch(DATA_URLS.pricing).then(r => {
        if (!r.ok) return null
        const ct = r.headers.get('content-type') || ''
        if (ct.includes('text/html')) return null
        return r.json()
      }).catch(() => null)
    ])
      .then(([modelsData, pricing]) => {
        setRawData(modelsData)
        setPricingData(pricing)
        setLoading(false)
      })
      .catch(err => {
        setError(err)
        setLoading(false)
      })
  }, [])

  // Memoize flattened models
  const models = useMemo(() => {
    if (!rawData) return []
    return flattenModels(rawData)
  }, [rawData])

  // Memoize metadata
  const metadata = useMemo(() => {
    if (!rawData?.metadata) return null
    return rawData.metadata
  }, [rawData])

  // Memoize providers list
  const providers = useMemo(() => {
    return extractProviders(models)
  }, [models])

  // Memoize capabilities list
  const capabilities = useMemo(() => {
    return extractCapabilities(models)
  }, [models])

  // Memoize use cases list
  const useCases = useMemo(() => {
    return extractUseCases(models)
  }, [models])

  // Memoize customizations list
  const customizations = useMemo(() => {
    return extractCustomizations(models)
  }, [models])

  // Memoize languages list
  const languages = useMemo(() => {
    return extractLanguages(models)
  }, [models])

  // Memoize consumption options list
  const consumptionOptionsList = useMemo(() => {
    return extractConsumptionOptions(models)
  }, [models])

  // Memoize lifecycle statuses list
  const lifecycleStatusesList = useMemo(() => {
    return extractLifecycleStatuses(models)
  }, [models])

  // Statistics
  const stats = useMemo(() => {
    if (!models.length) return null

    const getStatus = (m) => m.lifecycle?.status || m.model_status
    const activeCount = models.filter(m => getStatus(m) === 'ACTIVE').length
    const legacyCount = models.filter(m => getStatus(m) === 'LEGACY').length

    // Count unique regions (all region types)
    const regions = new Set()
    models.forEach(m => {
      // Deployed pipeline format: availability.on_demand.regions, etc.
      if (m.availability?.on_demand?.regions) {
        m.availability.on_demand.regions.forEach(r => regions.add(r))
      }
      if (m.availability?.cross_region?.regions) {
        m.availability.cross_region.regions.forEach(r => regions.add(r))
      }
      if (m.availability?.batch?.regions) {
        m.availability.batch.regions.forEach(r => regions.add(r))
      }
      if (m.availability?.provisioned?.regions) {
        m.availability.provisioned.regions.forEach(r => regions.add(r))
      }
      if (m.availability?.mantle?.regions) {
        m.availability.mantle.regions.forEach(r => regions.add(r))
      }
    })

    // Count multimodal models (supports both field names)
    const multimodalCount = models.filter(m => {
      const modalities = m.modalities || {}
      const inputs = modalities.input_modalities || []
      const outputs = modalities.output_modalities || []
      return inputs.length > 1 || outputs.length > 1 ||
             inputs.some(i => i !== 'TEXT') ||
             outputs.some(o => o !== 'TEXT')
    }).length

    return {
      totalModels: models.length,
      activeModels: activeCount,
      legacyModels: legacyCount,
      totalProviders: providers.length,
      totalRegions: regions.size,
      multimodalModels: multimodalCount,
    }
  }, [models, providers])

  // Helper to get pricing for a specific model
  // Accepts full model object to use pricing_file_reference for matching
  const getPricingForModel = useMemo(() => {
    return (model, preferredRegion = DEFAULT_REGION) => {
      if (!model) return { fullPricing: null, summary: { inputPrice: null, outputPrice: null } }
      const modelPricing = getModelPricing(model, pricingData)
      // Only hide In-Region pricing if Mantle has its own pricing
      // If Mantle doesn't have pricing, In-Region pricing applies to Mantle access
      const hideInRegion = model.availability?.hide_in_region ?? false
      const mantleHasPricing = model.availability?.mantle?.has_pricing ?? false
      const shouldHideInRegion = hideInRegion && mantleHasPricing
      return {
        fullPricing: modelPricing,
        summary: extractSummaryPricing(modelPricing, preferredRegion, {}, shouldHideInRegion)
      }
    }
  }, [pricingData])

  return {
    models,
    metadata,
    providers,
    capabilities,
    useCases,
    customizations,
    languages,
    consumptionOptionsList,
    lifecycleStatusesList,
    stats,
    loading,
    error,
    pricingData,
    getPricingForModel,
  }
}
