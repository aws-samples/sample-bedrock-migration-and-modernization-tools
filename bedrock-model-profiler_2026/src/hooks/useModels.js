import { useState, useEffect, useMemo } from 'react'
import { DATA_URLS } from '../config/dataSource'

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
 * Extract summary pricing (input/output per 1K) for a model in a given region
 * @param {Object} modelPricing - Pricing data for a model
 * @param {string} region - Preferred region
 * @returns {Object} { inputPrice, outputPrice }
 */
function extractSummaryPricing(modelPricing, region = 'us-east-1') {
  if (!modelPricing?.regions) return { inputPrice: null, outputPrice: null }

  const regionData = modelPricing.regions[region] ||
                     modelPricing.regions['us-east-1'] ||
                     modelPricing.regions['us-west-2'] ||
                     Object.values(modelPricing.regions)[0]

  if (!regionData?.pricing_groups) return { inputPrice: null, outputPrice: null }

  // Look for On-Demand pricing first
  const onDemand = regionData.pricing_groups['On-Demand'] || []

  let inputPrice = null
  let outputPrice = null

  for (const item of onDemand) {
    const dim = (item.dimension || '').toLowerCase()
    const desc = (item.description || '').toLowerCase()

    if ((dim.includes('input') || desc.includes('input')) && !dim.includes('cache')) {
      inputPrice = item.price_per_thousand
    }
    if (dim.includes('output') || desc.includes('output')) {
      outputPrice = item.price_per_thousand
    }
  }

  return { inputPrice, outputPrice }
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

  // Statistics
  const stats = useMemo(() => {
    if (!models.length) return null

    const activeCount = models.filter(m => (m.model_lifecycle?.status || m.model_status) === 'ACTIVE').length
    const legacyCount = models.filter(m => (m.model_lifecycle?.status || m.model_status) === 'LEGACY').length

    // Count unique regions
    const regions = new Set()
    models.forEach(m => {
      if (m.regions_available) {
        m.regions_available.forEach(r => regions.add(r))
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
    return (model, preferredRegion = 'us-east-1') => {
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
    stats,
    loading,
    error,
    pricingData,
    getPricingForModel,
  }
}
