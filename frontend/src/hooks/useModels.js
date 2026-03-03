import { useState, useEffect, useMemo } from 'react'
import { DATA_URLS } from '../config/dataSource'

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
 * Extracts unique capabilities from model data
 * @param {Array} models - Flattened array of models
 * @returns {Array} Unique capabilities
 */
function extractCapabilities(models) {
  const caps = new Set()
  models.forEach(m => {
    if (m.model_capabilities) {
      m.model_capabilities.forEach(cap => caps.add(cap))
    }
  })
  return [...caps].sort()
}

/**
 * Extracts unique use cases from model data
 * @param {Array} models - Flattened array of models
 * @returns {Array} Unique use cases
 */
function extractUseCases(models) {
  const useCases = new Set()
  models.forEach(m => {
    if (m.model_use_cases && Array.isArray(m.model_use_cases)) {
      m.model_use_cases.forEach(uc => useCases.add(uc))
    }
  })
  return [...useCases].sort()
}

/**
 * Extracts unique customization options from model data
 * @param {Array} models - Flattened array of models
 * @returns {Array} Unique customization options
 */
function extractCustomizations(models) {
  const customizations = new Set()
  models.forEach(m => {
    if (m.customization?.customization_supported && Array.isArray(m.customization.customization_supported)) {
      m.customization.customization_supported.forEach(c => customizations.add(c))
    }
  })
  return [...customizations].sort()
}

/**
 * Extracts unique languages from model data
 * @param {Array} models - Flattened array of models
 * @returns {Array} Unique languages
 */
function extractLanguages(models) {
  const languages = new Set()
  models.forEach(m => {
    if (m.languages_supported && Array.isArray(m.languages_supported)) {
      m.languages_supported.forEach(lang => languages.add(lang))
    }
  })
  return [...languages].sort()
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
    const status = m.model_lifecycle?.status || m.model_status
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
function getModelPricing(model, pricingData) {
  if (!pricingData?.providers) return null

  // First try using pricing_file_reference (preferred method)
  const pricingRef = model.model_pricing?.pricing_file_reference
  if (pricingRef?.provider && pricingRef?.model_key) {
    const providerData = pricingData.providers[pricingRef.provider]
    if (providerData?.[pricingRef.model_key]) {
      return providerData[pricingRef.model_key]
    }
  }

  // Fallback: try matching by model_id directly
  const modelId = model.model_id
  for (const provider of Object.values(pricingData.providers)) {
    if (provider[modelId]) {
      return provider[modelId]
    }
  }

  return null
}

/**
 * Extract summary pricing for a model in a given region
 * Handles different pricing types: token, image_generation, image, video, etc.
 * @param {Object} modelPricing - Pricing data for a model
 * @param {string} region - Preferred region
 * @returns {Object} Pricing summary with type information
 */
function extractSummaryPricing(modelPricing, region = DEFAULT_REGION) {
  const nullResult = {
    inputPrice: null,
    outputPrice: null,
    pricingType: null,
    unitLabel: null,
    imagePrice: null,
    imagePrices: null,
    videoPrice: null,
    videoPrices: null,
  }

  if (!modelPricing?.regions) return nullResult

  const regionData = modelPricing.regions[region] ||
                     modelPricing.regions['us-east-1'] ||
                     modelPricing.regions['us-west-2'] ||
                     Object.values(modelPricing.regions)[0]

  if (!regionData?.pricing_groups) return nullResult

  // Get model-level pricing type info
  const primaryPricingType = modelPricing.primary_pricing_type || 'token'

  // Look for On-Demand pricing first
  const onDemand = regionData.pricing_groups['On-Demand'] || []

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
        }
      }
    }
  }

  // Handle token-based pricing (most common)
  // First pass: look for standard pricing (no flex/priority suffix)
  // Second pass: fall back to flex pricing if no standard found
  let inputPrice = null
  let outputPrice = null
  let flexInputPrice = null
  let flexOutputPrice = null

  for (const item of onDemand) {
    const dim = (item.dimension || '').toLowerCase()
    const desc = (item.description || '').toLowerCase()

    // Skip cache-related entries (cache-read, cache-write) - these are special pricing tiers
    if (dim.includes('cache') || desc.includes('cache')) {
      continue
    }

    // Skip long context entries - these have different pricing
    if (dim.includes('lctx') || dim.includes('long-context') || dim.includes('longcontext') ||
        desc.includes('long context') || desc.includes('long-context')) {
      continue
    }

    // Skip reserved/committed capacity entries - these are commitment-based pricing
    if (dim.includes('reserved') || dim.includes('_tpm_') ||
        desc.includes('reserved') || desc.includes('per minute')) {
      continue
    }

    // Skip latency optimized entries - these have premium pricing
    if (dim.includes('latency') || desc.includes('latency')) {
      continue
    }

    // Skip priority tier (most expensive)
    if (dim.includes('-priority')) {
      continue
    }

    const isInput = item.is_input || dim.includes('input') || desc.includes('input')
    const isOutput = item.is_output || dim.includes('output') || desc.includes('output')

    // Use price_per_thousand for tokens, price_per_unit for others
    const price = item.price_per_thousand ?? item.price_per_unit

    // Skip zero prices (usually promotional or placeholder)
    if (price === 0 || price === null || price === undefined) {
      continue
    }

    // Check if this is flex pricing
    const isFlex = dim.includes('-flex')

    if (isFlex) {
      // Store flex prices as fallback
      if (isInput && flexInputPrice === null) {
        flexInputPrice = price
      }
      if (isOutput && flexOutputPrice === null) {
        flexOutputPrice = price
      }
    } else {
      // Standard pricing (preferred)
      if (isInput && inputPrice === null) {
        inputPrice = price
      }
      if (isOutput && outputPrice === null) {
        outputPrice = price
      }
    }
  }

  // Fall back to flex pricing if no standard pricing found
  if (inputPrice === null) inputPrice = flexInputPrice
  if (outputPrice === null) outputPrice = flexOutputPrice

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
      fetch(DATA_URLS.models).then(r => r.ok ? r.json() : Promise.reject('Failed to load models')),
      fetch(DATA_URLS.pricing).then(r => r.ok ? r.json() : null).catch(() => null)
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

    const activeCount = models.filter(m => (m.model_lifecycle?.status || m.model_status) === 'ACTIVE').length
    const legacyCount = models.filter(m => (m.model_lifecycle?.status || m.model_status) === 'LEGACY').length

    // Count unique regions (all region types)
    const regions = new Set()
    models.forEach(m => {
      if (m.in_region) {
        m.in_region.forEach(r => regions.add(r))
      }
      if (m.cross_region_inference?.source_regions) {
        m.cross_region_inference.source_regions.forEach(r => regions.add(r))
      }
      if (m.batch_inference_supported?.supported_regions) {
        m.batch_inference_supported.supported_regions.forEach(r => regions.add(r))
      }
      if (m.provisioned_throughput?.provisioned_regions) {
        m.provisioned_throughput.provisioned_regions.forEach(r => regions.add(r))
      }
      if (m.mantle_inference?.mantle_regions) {
        m.mantle_inference.mantle_regions.forEach(r => regions.add(r))
      }
    })

    // Count multimodal models
    const multimodalCount = models.filter(m => {
      const inputs = m.model_modalities?.input_modalities || []
      const outputs = m.model_modalities?.output_modalities || []
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
      return {
        fullPricing: modelPricing,
        summary: extractSummaryPricing(modelPricing, preferredRegion)
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
