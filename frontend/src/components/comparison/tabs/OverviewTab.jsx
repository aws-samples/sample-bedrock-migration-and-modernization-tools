import { useMemo, useState } from 'react'
import { Check, X, MessageSquare, Image, FileText, Video, Mic, Trophy, DollarSign, Globe, ChevronDown, ChevronRight, Cpu } from 'lucide-react'
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { providerColorClasses } from '@/config/constants'

const modalityIcons = {
  TEXT: MessageSquare,
  IMAGE: Image,
  DOCUMENT: FileText,
  VIDEO: Video,
  AUDIO: Mic,
  SPEECH: Mic,
}

const modalityLabels = {
  TEXT: 'Text',
  IMAGE: 'Image',
  DOCUMENT: 'Doc',
  VIDEO: 'Video',
  AUDIO: 'Audio',
  SPEECH: 'Speech',
}

// Distinct colors for up to 10 models
const radarColors = [
  '#1A9E7A', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6',
  '#EC4899', '#14B8A6', '#F97316', '#6366F1', '#84CC16',
]

function formatNumber(num) {
  if (typeof num !== 'number' || isNaN(num)) return 'N/A'
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(0)}K`
  return num.toString()
}

function formatPrice(price) {
  if (price === null || price === undefined) return 'N/A'
  if (price < 0.01) return `$${price.toFixed(4)}`
  return `$${price.toFixed(2)}`
}

// Detect long-context support from pricing data
function detectLongContext(pricing, region) {
  const fullPricing = pricing?.fullPricing
  const regionData = fullPricing?.regions?.[region] || fullPricing?.regions?.['us-east-1']
  const groups = regionData?.pricing_groups || {}
  return !!(groups['On-Demand Long Context'] && groups['On-Demand Long Context'].length > 0)
}

// Extract extended context window from quota names (e.g., "1M Context Length", "200K Context Length")
function getExtendedContextWindow(model) {
  const quotas = model.quotas ?? model.model_service_quotas ?? {}
  let maxContext = 0
  const pattern = /(\d+)(K|M)\s*Context\s*Length/i
  for (const regionQuotas of Object.values(quotas)) {
    for (const q of regionQuotas) {
      const match = q.quota_name?.match(pattern)
      if (match) {
        const num = parseInt(match[1], 10)
        const multiplier = match[2].toUpperCase() === 'M' ? 1000000 : 1000
        maxContext = Math.max(maxContext, num * multiplier)
      }
    }
  }
  return maxContext > 0 ? maxContext : null
}

// Radar scoring (0-10 scale, benchmarked against relative set)
function computeRadarScores(modelData, benchmarks) {
  const { maxContext, maxRegions, maxCost } = benchmarks
  
  return modelData.map(d => {
    // Cost Efficiency: lower price = higher score (0-10)
    // Score = 10 × (1 - modelCost / maxCost). Cheapest = 10, most expensive = 0
    const totalCost = (d.inputPrice || 0) + (d.outputPrice || 0)
    const costScore = totalCost > 0 && maxCost > 0
      ? 10 * (1 - totalCost / maxCost)
      : 0  // No pricing data = 0 (unknown, not mid-range)
    
    // Context Window: relative to the max context window
    const effectiveCtx = d.effectiveContextWindow || d.contextWindow
    const contextScore = effectiveCtx > 0 && maxContext > 0
      ? Math.min(10 * (effectiveCtx / maxContext), 10)
      : 0
    
    // Availability: relative to the model with most regions
    const regionScore = maxRegions > 0
      ? Math.min(10 * (d.regions.length / maxRegions), 10)
      : 0
    
    return {
      name: d.model.model_name || d.model.model_id,
      costScore: Math.round(costScore * 10) / 10,
      contextScore: Math.round(contextScore * 10) / 10,
      regionScore: Math.round(regionScore * 10) / 10,
    }
  })
}

const providerColors = providerColorClasses

function MetricRow({ label, values, isLight, bestIndices = null }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        {label}
      </td>
      {values.map((value, idx) => {
        const isBest = bestIndices?.has(idx)
        return (
          <td
            key={idx}
            className={cn(
              'px-3 py-2.5 text-center text-sm font-medium',
              isBest
                ? 'text-emerald-600'
                : isLight ? 'text-stone-900' : 'text-white'
            )}
          >
            <div className="flex items-center justify-center gap-1">
              {value}
              {isBest && <Trophy className="h-3 w-3 text-emerald-500 flex-shrink-0" />}
            </div>
          </td>
        )
      })}
    </tr>
  )
}

function ContextRow({ label, values, isLight, bestIndices = null }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        {label}
      </td>
      {values.map((data, idx) => {
        const isBest = bestIndices?.has(idx)
        return (
          <td
            key={idx}
            className={cn(
              'px-3 py-2.5 text-center',
              isBest
                ? 'text-emerald-600'
                : isLight ? 'text-stone-900' : 'text-white'
            )}
          >
            <div className="flex flex-col items-center gap-0.5">
              <div className="flex items-center gap-1">
                <span className="text-sm font-medium">{data.formatted}</span>
                {isBest && <Trophy className="h-3 w-3 text-emerald-500 flex-shrink-0" />}
              </div>
              {data.baseFormatted && (
                <span className={cn(
                  'text-[10px]',
                  isLight ? 'text-stone-400' : 'text-slate-500'
                )}>
                  base: {data.baseFormatted}
                </span>
              )}
              {data.hasLongContext && (
                <span className={cn(
                  'text-[10px] px-1.5 py-0.5 rounded',
                  isLight ? 'bg-purple-100 text-purple-700' : 'bg-purple-500/15 text-purple-400'
                )}>
                  Long context
                </span>
              )}
            </div>
          </td>
        )
      })}
    </tr>
  )
}

function BooleanRow({ label, values, isLight }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        {label}
      </td>
      {values.map((value, idx) => (
        <td key={idx} className="px-3 py-2.5 text-center">
          {value ? (
            <Check className="h-4 w-4 text-emerald-500 mx-auto" />
          ) : (
            <X className="h-4 w-4 text-red-400/40 mx-auto" />
          )}
        </td>
      ))}
    </tr>
  )
}

function ModalitiesRow({ label, values, isLight, isOutput = false }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        {label}
      </td>
      {values.map((modalities, idx) => (
        <td key={idx} className="px-3 py-2.5">
          <div className="flex justify-center gap-1 flex-wrap">
            {modalities.map(mod => {
              const Icon = modalityIcons[mod] || MessageSquare
              return (
                <span
                  key={mod}
                  className={cn(
                    'inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium',
                    isOutput
                      ? isLight ? 'bg-blue-50 text-blue-600' : 'bg-blue-500/10 text-blue-400'
                      : isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/5 text-slate-400'
                  )}
                >
                  <Icon className="h-3 w-3" />
                  {modalityLabels[mod] || mod}
                </span>
              )
            })}
          </div>
        </td>
      ))}
    </tr>
  )
}

// Winner row component for the Winners panel
function WinnerRow({ icon, label, winners, value, isLight, highlight = false, modelData }) {
  if (!winners || winners.length === 0) {
    return (
      <div className={cn(
        'flex items-center justify-between py-1.5 px-2 rounded',
        isLight ? 'bg-white/50' : 'bg-white/[0.02]'
      )}>
        <div className="flex items-center gap-1.5">
          <span className={cn(isLight ? 'text-stone-400' : 'text-slate-500')}>{icon}</span>
          <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>{label}</span>
        </div>
        <span className={cn('text-[10px]', isLight ? 'text-stone-300' : 'text-slate-600')}>—</span>
      </div>
    )
  }

  return (
    <div className={cn(
      'py-1.5 px-2 rounded',
      highlight
        ? isLight ? 'bg-emerald-50/80' : 'bg-emerald-500/10'
        : isLight ? 'bg-white/50' : 'bg-white/[0.02]'
    )}>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-1.5">
          <span className={cn(
            highlight ? 'text-emerald-600' : isLight ? 'text-stone-500' : 'text-slate-400'
          )}>{icon}</span>
          <span className={cn(
            'text-[10px] font-medium',
            highlight ? 'text-emerald-700' : isLight ? 'text-stone-600' : 'text-slate-400'
          )}>{label}</span>
        </div>
        <span className={cn(
          'text-[10px] font-semibold',
          highlight ? 'text-emerald-600' : isLight ? 'text-stone-900' : 'text-white'
        )}>{value}</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {winners.slice(0, 3).map((w, i) => {
          const modelIndex = modelData.findIndex(m => m === w)
          return (
            <span
              key={i}
              className={cn(
                'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium',
                isLight ? 'bg-stone-100 text-stone-700' : 'bg-white/10 text-slate-300'
              )}
            >
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ backgroundColor: radarColors[modelIndex % radarColors.length] }}
              />
              {w.model.model_name?.split(' ').slice(-2).join(' ') || w.model.model_id.split('.').pop()}
            </span>
          )
        })}
        {winners.length > 3 && (
          <span className={cn(
            'text-[9px] px-1',
            isLight ? 'text-stone-400' : 'text-slate-500'
          )}>
            +{winners.length - 3}
          </span>
        )}
      </div>
    </div>
  )
}

// Custom tooltip for the radar chart
function RadarTooltip({ active, payload, label, isLight }) {
  if (!active || !payload?.length) return null
  const sorted = [...payload].sort((a, b) => (b.value || 0) - (a.value || 0))
  return (
    <div className={cn(
      'px-3 py-2 rounded-lg border text-xs shadow-lg',
      isLight
        ? 'bg-white/90 border-stone-200/60 text-stone-900 backdrop-blur-xl'
        : 'bg-slate-900/90 border-white/[0.08] text-white backdrop-blur-xl'
    )}>
      <p className="font-semibold mb-1">{label}</p>
      {sorted.map((entry, idx) => (
        <p key={idx} style={{ color: entry.color }}>
          {entry.name}: <span className="font-medium">{entry.value}/10</span>
        </p>
      ))}
    </div>
  )
}

// Helper to get all regions for a model (on-demand + CRIS + Mantle + Batch)
function getAllModelRegions(model) {
  const onDemand = model.availability?.on_demand?.regions ?? model.in_region ?? model.regions_available ?? []
  const cris = model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions ?? []
  const mantle = model.availability?.mantle?.regions ?? []
  const batch = model.availability?.batch?.regions ?? model.batch_inference_supported?.supported_regions ?? []
  return [...new Set([...onDemand, ...cris, ...mantle, ...batch])]
}

// Helper to get pricing for a specific consumption type with dimension filtering
function getPricesByType(pricing, region, type, crisSubType = 'global', dimensionFilters = {}) {
  const fullPricing = pricing?.fullPricing
  if (!fullPricing?.regions) return { inputPrice: null, outputPrice: null, availableRegions: [], hasData: false }
  
  // Try the specified region first, then us-east-1, then any available region
  const regionData = fullPricing.regions[region] || 
                     fullPricing.regions['us-east-1'] || 
                     Object.values(fullPricing.regions)[0]
  
  if (!regionData?.pricing_groups) return { inputPrice: null, outputPrice: null, availableRegions: [], hasData: false }
  
  const pricingGroups = regionData.pricing_groups
  
  // Map pricing type to group names
  let groupNames
  if (type === 'cris') {
    if (crisSubType === 'geo') {
      groupNames = ['On-Demand Geo', 'On-Demand Long Context Geo']
    } else {
      groupNames = ['On-Demand Global', 'On-Demand Long Context Global']
    }
  } else {
    groupNames = ['On-Demand', 'On-Demand Long Context']
  }
  
  let inputPrice = null
  let outputPrice = null
  
  // Check each group for pricing
  for (const groupName of groupNames) {
    let items = pricingGroups[groupName]
    if (!items || items.length === 0) continue
    
    // Apply dimension filters if provided
    if (Object.keys(dimensionFilters).length > 0) {
      items = items.filter(item => {
        const dims = item.dimensions || {}
        if (dimensionFilters.source && dims.source !== dimensionFilters.source) return false
        if (dimensionFilters.geo && dims.geo !== dimensionFilters.geo) return false
        if (dimensionFilters.tier && dims.tier !== dimensionFilters.tier) return false
        if (dimensionFilters.context && dims.context !== dimensionFilters.context) return false
        return true
      })
    } else {
      // Default: exclude Mantle pricing
      items = items.filter(item => {
        const dims = item.dimensions || {}
        return dims.source !== 'mantle'
      })
    }
    
    for (const item of items) {
      // Get price - multiply by 1000 for per-million display
      const price = item.price_per_thousand != null 
        ? item.price_per_thousand * 1000 
        : item.price_per_unit
      
      if (price == null) continue
      
      const dim = (item.dimension || '').toLowerCase()
      const desc = (item.description || '').toLowerCase()
      
      // Check for input pricing - prefer is_input/is_output flags from new structure
      const isInput = item.is_input || dim.includes('input') || desc.includes('input')
      const isOutput = item.is_output || dim.includes('output') || desc.includes('output')
      
      // Skip cache pricing for the main comparison
      const isCache = dim.includes('cache') || desc.includes('cache')
      if (isCache) continue
      
      if (isInput && inputPrice === null) {
        inputPrice = price
      }
      if (isOutput && outputPrice === null) {
        outputPrice = price
      }
      
      // If we found both, we can stop
      if (inputPrice !== null && outputPrice !== null) break
    }
    
    // If we found pricing in this group, don't check the next one
    if (inputPrice !== null || outputPrice !== null) break
  }
  
  // Get regions that have this pricing type
  const availableRegions = Object.keys(fullPricing.regions).filter(r => {
    const rData = fullPricing.regions[r]
    if (!rData?.pricing_groups) return false
    return groupNames.some(g => rData.pricing_groups[g]?.length > 0)
  })
  
  return { 
    inputPrice, 
    outputPrice, 
    availableRegions,
    hasData: inputPrice !== null || outputPrice !== null
  }
}

function PriceRow({ label, values, isLight, priceRegions = [], pricingType, crisSupported = [] }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        {label}
      </td>
      {values.map((data, idx) => {
        const regions = priceRegions[idx] || []
        const isNA = data.value === 'N/A'
        const hasCris = crisSupported[idx]
        
        // Determine tooltip text based on pricing type and availability
        let tooltipText
        if (isNA) {
          if (pricingType === 'cris') {
            tooltipText = hasCris 
              ? 'CRIS pricing not yet available for this model'
              : 'This model does not support Cross-Region Inference'
          } else {
            tooltipText = 'In-region pricing not available for this model'
          }
        } else {
          tooltipText = pricingType === 'cris'
            ? `CRIS: Available from ${regions.length} source regions`
            : `In Region: Available in ${regions.length} regions`
        }
        
        return (
          <td
            key={idx}
            className={cn(
              'px-3 py-2.5 text-center text-sm font-medium',
              isNA
                ? isLight ? 'text-stone-400' : 'text-slate-500'
                : isLight ? 'text-stone-900' : 'text-white'
            )}
            title={tooltipText}
          >
            <div className="flex items-center justify-center gap-1 cursor-help">
              {data.value}
              {isNA && pricingType === 'cris' && !hasCris && (
                <span className={cn(
                  'text-[9px] px-1 py-0.5 rounded',
                  isLight ? 'bg-stone-100 text-stone-400' : 'bg-white/5 text-slate-500'
                )}>
                  No CRIS
                </span>
              )}
            </div>
          </td>
        )
      })}
    </tr>
  )
}

export function OverviewTab({ selectedModels, getPricingForModel, allModels, isLight }) {
  const [radarCollapsed, setRadarCollapsed] = useState(false)
  const [pricingType, setPricingType] = useState('in_region') // 'in_region' or 'cris'
  const [crisType, setCrisType] = useState('global') // 'global' or 'geo'

  const modelData = useMemo(() => selectedModels.map(({ model, region }) => {
    const pricing = getPricingForModel?.(model, region)
    const priceData = getPricesByType(pricing, region, pricingType, crisType)
    const contextWindow = model.specs?.context_window ?? model.converse_data?.context_window ?? 0
    const maxOutput = model.specs?.max_output ?? model.specs?.max_output_tokens ?? model.converse_data?.max_output_tokens ?? 0
    const inputModalities = model.modalities?.input_modalities ?? model.model_modalities?.input_modalities ?? []
    const outputModalities = model.modalities?.output_modalities ?? model.model_modalities?.output_modalities ?? []
    const regions = getAllModelRegions(model)
    const isActive = (model.lifecycle?.status ?? model.model_lifecycle?.status) === 'ACTIVE' || model.model_status === 'ACTIVE'
    const streamingSupported = model.streaming ?? model.streaming_supported ?? false
    
    // Get CRIS data with proper fallbacks
    const crisData = model.availability?.cross_region ?? model.cross_region_inference ?? {}
    const crisSupported = crisData.supported ?? (crisData.source_regions?.length > 0) ?? (crisData.profiles?.length > 0) ?? false
    
    // Get batch support with proper fallbacks
    const batchData = model.availability?.batch ?? model.batch_inference_supported ?? {}
    const batchSupported = batchData.supported ?? (batchData.supported_regions?.length > 0) ?? (model.consumption_options || []).includes('batch')
    
    const mantleSupported = model.availability?.mantle?.supported ?? false
    const hasLongContext = detectLongContext(pricing, region)
    const extendedContext = getExtendedContextWindow(model)
    const effectiveContextWindow = Math.max(contextWindow, extendedContext || 0)
    const useCasesCount = (model.use_cases ?? model.model_use_cases ?? []).length
    const capabilitiesCount = (model.capabilities ?? model.model_capabilities ?? []).length

    return {
      model,
      region,
      contextWindow,
      effectiveContextWindow,
      maxOutput,
      inputModalities,
      outputModalities,
      regions,
      isActive,
      streamingSupported,
      crisSupported,
      batchSupported,
      mantleSupported,
      hasLongContext: hasLongContext || (extendedContext != null && extendedContext > contextWindow),
      useCasesCount,
      capabilitiesCount,
      inputPrice: priceData.inputPrice,
      outputPrice: priceData.outputPrice,
      priceRegions: priceData.availableRegions,
    }
  }), [selectedModels, getPricingForModel, pricingType, crisType])

  // Relative benchmarks from only the compared models
  const relativeBenchmarks = useMemo(() => {
    const allCosts = modelData.map(d => ((d.inputPrice || 0) + (d.outputPrice || 0))).filter(c => c > 0)
    const maxCost = allCosts.length > 0 ? Math.max(...allCosts) : 1
    const maxContext = Math.max(...modelData.map(d => d.effectiveContextWindow || d.contextWindow), 1)
    const maxRegions = Math.max(...modelData.map(d => d.regions.length), 1)
    return { maxContext, maxRegions, maxCost }
  }, [modelData])

  // Radar chart data
  const radarScores = useMemo(() => computeRadarScores(modelData, relativeBenchmarks), [modelData, relativeBenchmarks])

  const radarChartData = useMemo(() => {
    const axes = ['Context Window', 'Cost Efficiency', 'Availability']
    const scoreKeys = ['contextScore', 'costScore', 'regionScore']
    return axes.map((axis, i) => {
      const point = { axis }
      radarScores.forEach((scores) => {
        point[scores.name] = scores[scoreKeys[i]]
      })
      return point
    })
  }, [radarScores])

  // Find best values
  const maxEffectiveContext = Math.max(...modelData.map(d => d.effectiveContextWindow))
  const maxOutputTokens = Math.max(...modelData.map(d => d.maxOutput))
  const maxRegions = Math.max(...modelData.map(d => d.regions.length))

  const validInputPrices = modelData.filter(d => d.inputPrice !== null && d.inputPrice !== undefined)
  const minInputPrice = validInputPrices.length > 0 ? Math.min(...validInputPrices.map(d => d.inputPrice)) : null

  const validOutputPrices = modelData.filter(d => d.outputPrice !== null && d.outputPrice !== undefined)
  const minOutputPrice = validOutputPrices.length > 0 ? Math.min(...validOutputPrices.map(d => d.outputPrice)) : null

  // Build Sets of all indices that match the best value
  const contextBestSet = new Set(maxEffectiveContext > 0 ? modelData.map((d, i) => d.effectiveContextWindow === maxEffectiveContext ? i : -1).filter(i => i >= 0) : [])
  const outputBestSet = new Set(maxOutputTokens > 0 ? modelData.map((d, i) => d.maxOutput === maxOutputTokens ? i : -1).filter(i => i >= 0) : [])
  const regionsBestSet = new Set(modelData.map((d, i) => d.regions.length === maxRegions ? i : -1).filter(i => i >= 0))
  const inputPriceBestSet = new Set(minInputPrice !== null ? modelData.map((d, i) => d.inputPrice === minInputPrice ? i : -1).filter(i => i >= 0) : [])
  const outputPriceBestSet = new Set(minOutputPrice !== null ? modelData.map((d, i) => d.outputPrice === minOutputPrice ? i : -1).filter(i => i >= 0) : [])

  return (
    <div className="mt-4 space-y-4">
      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-2">
        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>Best Context</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {formatNumber(maxEffectiveContext)}
          </p>
          {contextBestSet.size > 0 && (
            <p className={cn('text-[10px] truncate', isLight ? 'text-stone-400' : 'text-slate-500')}>
              {[...contextBestSet].map(i => modelData[i].model.model_name || modelData[i].model.model_id).join(', ')}
            </p>
          )}
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-emerald-50/50 border-emerald-200' : 'bg-emerald-500/10 border-emerald-500/30'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <DollarSign className={cn('h-3.5 w-3.5 text-emerald-500')} />
            <span className={cn('text-[10px]', isLight ? 'text-emerald-700' : 'text-emerald-400')}>
              Cheapest Input ({pricingType === 'cris' ? 'CRIS' : 'In Region'})
            </span>
          </div>
          <p className="text-lg font-bold text-emerald-600">
            {minInputPrice !== null ? `$${minInputPrice < 0.01 ? minInputPrice.toFixed(4) : minInputPrice.toFixed(2)}` : '—'}
          </p>
          {inputPriceBestSet.size > 0 && (
            <p className={cn('text-[10px] truncate', isLight ? 'text-stone-400' : 'text-slate-500')}>
              {[...inputPriceBestSet].map(i => modelData[i].model.model_name || modelData[i].model.model_id).join(', ')}
            </p>
          )}
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>Most Regions</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {maxRegions}
          </p>
          {regionsBestSet.size > 0 && (
            <p className={cn('text-[10px] truncate', isLight ? 'text-stone-400' : 'text-slate-500')}>
              {[...regionsBestSet].map(i => modelData[i].model.model_name || modelData[i].model.model_id).join(', ')}
            </p>
          )}
        </div>
      </div>

      {/* Radar Chart + Winners Panel */}
      {modelData.length >= 2 && (
        <div className={cn(
          'rounded-lg border overflow-hidden',
          isLight
            ? 'bg-white/70 border-stone-200/60 backdrop-blur-xl'
            : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
        )}>
          {/* Header with collapse button and pricing selector */}
          <div className={cn(
            'border-b flex items-center justify-between',
            isLight ? 'bg-stone-50/60 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
          )}>
            <button
              onClick={() => setRadarCollapsed(!radarCollapsed)}
              className={cn(
                'flex-1 px-4 py-2.5 flex items-center gap-2 transition-colors text-left',
                isLight ? 'hover:bg-stone-100/60' : 'hover:bg-white/[0.04]'
              )}
            >
              {radarCollapsed ? (
                <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
              ) : (
                <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
              )}
              <h3 className={cn('font-semibold text-xs', isLight ? 'text-stone-900' : 'text-white')}>
                Model Comparison Radar
              </h3>
            </button>
            
            {/* Pricing Selector */}
            <div className="flex items-center gap-2 px-4 py-2">
              <span className={cn('text-[10px] font-medium', isLight ? 'text-stone-500' : 'text-slate-500')}>
                Pricing:
              </span>
              <div className={cn(
                'inline-flex rounded-md border overflow-hidden h-6',
                isLight ? 'border-stone-300' : 'border-[#373a40]'
              )}>
                <button
                  onClick={() => setPricingType('in_region')}
                  title="In Region: Standard on-demand pricing. Price varies by AWS region where you deploy."
                  className={cn(
                    'px-2.5 py-0.5 text-[10px] font-medium transition-colors',
                    pricingType === 'in_region'
                      ? isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
                      : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-[#1a1b1e] text-[#9a9b9f] hover:bg-[#2c2d32]'
                  )}
                >
                  In Region
                </button>
                <button
                  onClick={() => setPricingType('cris')}
                  title="CRIS: Cross-Region Inference Service. Same price from any source region, routes to optimal endpoint."
                  className={cn(
                    'px-2.5 py-0.5 text-[10px] font-medium transition-colors border-l',
                    isLight ? 'border-stone-300' : 'border-[#373a40]',
                    pricingType === 'cris'
                      ? isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
                      : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-[#1a1b1e] text-[#9a9b9f] hover:bg-[#2c2d32]'
                  )}
                >
                  CRIS
                </button>
              </div>
              {pricingType === 'cris' && (
                <div className={cn(
                  'inline-flex rounded-md border overflow-hidden h-6',
                  isLight ? 'border-stone-300' : 'border-[#373a40]'
                )}>
                  <button
                    onClick={() => setCrisType('global')}
                    title="Global CRIS: Routes to any region worldwide for optimal latency"
                    className={cn(
                      'px-2 py-0.5 text-[10px] font-medium transition-colors',
                      crisType === 'global'
                        ? isLight ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'
                        : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-[#1a1b1e] text-[#9a9b9f] hover:bg-[#2c2d32]'
                    )}
                  >
                    Global
                  </button>
                  <button
                    onClick={() => setCrisType('geo')}
                    title="Geo CRIS: Routes within a geographic region (e.g., EU, APAC)"
                    className={cn(
                      'px-2 py-0.5 text-[10px] font-medium transition-colors border-l',
                      isLight ? 'border-stone-300' : 'border-[#373a40]',
                      crisType === 'geo'
                        ? isLight ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'
                        : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-[#1a1b1e] text-[#9a9b9f] hover:bg-[#2c2d32]'
                    )}
                  >
                    Geo
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Explanatory text for radar chart */}
          {!radarCollapsed && (
            <div className={cn(
              'px-4 py-2 text-[11px] border-b',
              isLight ? 'text-stone-500 bg-stone-50/30 border-stone-200/60' : 'text-slate-400 bg-white/[0.01] border-white/[0.04]'
            )}>
              <strong>How to read:</strong> Each axis shows a score from 0-10 comparing the selected models. 
              <strong> Context Window</strong> = larger is better. 
              <strong> Cost Efficiency</strong> = cheaper is better (10 = cheapest). 
              <strong> Availability</strong> = more regions is better.
            </div>
          )}

          {!radarCollapsed && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-0">
              {/* Radar Chart - takes 2 columns on large screens */}
              <div className={cn(
                'lg:col-span-2 lg:border-r',
                isLight ? 'border-stone-200/60' : 'border-white/[0.06]'
              )} style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarChartData} cx="50%" cy="50%" outerRadius="65%">
                    <PolarGrid
                      stroke={isLight ? '#d6d3d1' : 'rgba(255,255,255,0.08)'}
                      strokeDasharray="3 3"
                    />
                    <PolarAngleAxis
                      dataKey="axis"
                      tick={{
                        fill: isLight ? '#57534e' : '#94a3b8',
                        fontSize: 12,
                        fontWeight: 500,
                      }}
                    />
                    <PolarRadiusAxis
                      angle={90}
                      domain={[0, 10]}
                      tick={{
                        fill: isLight ? '#a8a29e' : '#475569',
                        fontSize: 9,
                      }}
                      tickCount={6}
                    />
                    {radarScores.map((scores, idx) => (
                      <Radar
                        key={scores.name}
                        name={scores.name}
                        dataKey={scores.name}
                        stroke={radarColors[idx % radarColors.length]}
                        fill={radarColors[idx % radarColors.length]}
                        fillOpacity={0.15}
                        strokeWidth={2}
                      />
                    ))}
                    <Tooltip content={<RadarTooltip isLight={isLight} />} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Winners Panel - takes 1 column */}
              <div className={cn(
                'p-3 space-y-2',
                isLight ? 'bg-stone-50/30' : 'bg-white/[0.01]'
              )}>
                <h4 className={cn(
                  'text-[10px] font-semibold uppercase tracking-wide mb-2',
                  isLight ? 'text-stone-500' : 'text-slate-500'
                )}>
                  Category Winners
                </h4>
                
                {/* Context Window Winner */}
                <WinnerRow
                  icon={<Cpu className="h-3 w-3" />}
                  label="Context Window"
                  winners={[...contextBestSet].map(i => modelData[i])}
                  value={formatNumber(maxEffectiveContext)}
                  isLight={isLight}
                  modelData={modelData}
                />
                
                {/* Cost Efficiency Winner (cheapest input) */}
                <WinnerRow
                  icon={<DollarSign className="h-3 w-3" />}
                  label={`Cheapest (${pricingType === 'cris' ? 'CRIS' : 'In Region'})`}
                  winners={[...inputPriceBestSet].map(i => modelData[i])}
                  value={minInputPrice !== null ? `$${minInputPrice < 0.01 ? minInputPrice.toFixed(4) : minInputPrice.toFixed(2)}` : '—'}
                  isLight={isLight}
                  highlight
                  modelData={modelData}
                />
                
                {/* Availability Winner */}
                <WinnerRow
                  icon={<Globe className="h-3 w-3" />}
                  label="Most Regions"
                  winners={[...regionsBestSet].map(i => modelData[i])}
                  value={`${maxRegions} regions`}
                  isLight={isLight}
                  modelData={modelData}
                />
                
                {/* Max Output Winner */}
                <WinnerRow
                  icon={<FileText className="h-3 w-3" />}
                  label="Max Output"
                  winners={[...outputBestSet].map(i => modelData[i])}
                  value={formatNumber(maxOutputTokens)}
                  isLight={isLight}
                  modelData={modelData}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Comparison Table */}
      <div className={cn(
        'rounded-lg border overflow-hidden',
        isLight
          ? 'bg-white/70 border-stone-200/60 backdrop-blur-xl'
          : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
      )}>
        <div className="overflow-auto max-h-[600px]">
          <table className="w-full">
            <thead className="sticky top-0 z-20">
              <tr className={cn(
                'border-b-2',
                isLight ? 'border-stone-200 bg-stone-50' : 'border-white/[0.06] bg-[#1a1b1e]'
              )}>
                <th className={cn(
                  'px-4 py-3 text-left text-xs font-semibold w-40 min-w-[130px] sticky left-0 z-30',
                  isLight ? 'text-stone-900 bg-stone-50' : 'text-white bg-[#1a1b1e]'
                )}>
                  Feature
                </th>
                {modelData.map(({ model }) => (
                  <th key={model.model_id} className="px-3 py-3 text-center min-w-[100px]">
                    <Badge className={cn(
                      'text-[9px] mb-1',
                      isLight ? 'text-[#faf9f5]' : 'text-white',
                      providerColors[model.model_provider] || providerColors.default
                    )}>
                      {model.model_provider}
                    </Badge>
                    <p className={cn(
                      'text-xs font-semibold line-clamp-2',
                      isLight ? 'text-stone-900' : 'text-white'
                    )}>
                      {model.model_name || model.model_id}
                    </p>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <ContextRow
                label="Context Window"
                values={modelData.map(d => ({
                  formatted: formatNumber(d.effectiveContextWindow),
                  baseFormatted: d.effectiveContextWindow > d.contextWindow ? formatNumber(d.contextWindow) : null,
                  hasLongContext: d.hasLongContext,
                }))}
                isLight={isLight}
                bestIndices={contextBestSet}
              />
              <MetricRow
                label="Max Output"
                values={modelData.map(d => formatNumber(d.maxOutput))}
                isLight={isLight}
                bestIndices={outputBestSet}
              />
              <ModalitiesRow
                label="Input Modalities"
                values={modelData.map(d => d.inputModalities)}
                isLight={isLight}
              />
              <ModalitiesRow
                label="Output Modalities"
                values={modelData.map(d => d.outputModalities)}
                isLight={isLight}
                isOutput={true}
              />
              <BooleanRow
                label="Streaming"
                values={modelData.map(d => d.streamingSupported)}
                isLight={isLight}
              />
              <BooleanRow
                label="Cross-Region (CRIS)"
                values={modelData.map(d => d.crisSupported)}
                isLight={isLight}
              />
              <BooleanRow
                label="Mantle"
                values={modelData.map(d => d.mantleSupported)}
                isLight={isLight}
              />
              <BooleanRow
                label="Batch Inference"
                values={modelData.map(d => d.batchSupported)}
                isLight={isLight}
              />
              <MetricRow
                label="Regions"
                values={modelData.map(d => `${d.regions.length}`)}
                isLight={isLight}
                bestIndices={regionsBestSet}
              />
              <MetricRow
                label="Capabilities"
                values={modelData.map(d => `${d.capabilitiesCount}`)}
                isLight={isLight}
              />
              <MetricRow
                label="Use Cases"
                values={modelData.map(d => `${d.useCasesCount}`)}
                isLight={isLight}
              />
              <PriceRow
                label={`Input Price (1M) - ${pricingType === 'cris' ? 'CRIS' : 'In Region'}`}
                values={modelData.map(d => ({ value: formatPrice(d.inputPrice) }))}
                isLight={isLight}
                priceRegions={modelData.map(d => d.priceRegions)}
                pricingType={pricingType}
                crisSupported={modelData.map(d => d.crisSupported)}
              />
              <PriceRow
                label={`Output Price (1M) - ${pricingType === 'cris' ? 'CRIS' : 'In Region'}`}
                values={modelData.map(d => ({ value: formatPrice(d.outputPrice) }))}
                isLight={isLight}
                priceRegions={modelData.map(d => d.priceRegions)}
                pricingType={pricingType}
                crisSupported={modelData.map(d => d.crisSupported)}
              />
              <BooleanRow
                label="Active"
                values={modelData.map(d => d.isActive)}
                isLight={isLight}
              />
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
