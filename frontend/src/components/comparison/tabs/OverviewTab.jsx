import { useMemo, useState } from 'react'
import { Check, X, MessageSquare, Image, FileText, Video, Mic, Trophy, DollarSign, Globe, ChevronDown, ChevronRight, Cpu, Copy, Info, Radar as RadarIcon } from 'lucide-react'
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
  const { maxContext, maxRegions, maxCost, minCost } = benchmarks
  
  return modelData.map(d => {
    // Cost Efficiency: lower price = higher score (0-10)
    // Models with no pricing data get 0
    // Models with pricing: cheapest gets 10, most expensive gets 1 (not 0)
    const totalCost = (d.inputPrice || 0) + (d.outputPrice || 0)
    let costScore = 0
    if (totalCost > 0 && maxCost > 0) {
      if (maxCost === minCost) {
        // All models with pricing have the same cost - give them all 10
        costScore = 10
      } else {
        // Scale from 1 (most expensive) to 10 (cheapest)
        // Formula: 10 - 9 * (cost - minCost) / (maxCost - minCost)
        costScore = 10 - 9 * (totalCost - minCost) / (maxCost - minCost)
      }
    }
    
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

function MetricRow({ label, values, isLight, bestIndices = null, tooltip = null }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10 cursor-default',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        <span title={tooltip}>
          {label}
        </span>
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

function ContextRow({ label, values, isLight, bestIndices = null, tooltip = null }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10 cursor-default',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        <span title={tooltip}>
          {label}
        </span>
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

function BooleanRow({ label, values, isLight, tooltip = null }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10 cursor-default',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        <span title={tooltip}>
          {label}
        </span>
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

function CustomizationRow({ label, values, isLight, tooltip = null }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10 cursor-default',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        <span title={tooltip}>
          {label}
        </span>
      </td>
      {values.map((options, idx) => (
        <td key={idx} className="px-3 py-2.5">
          {options.length > 0 ? (
            <div className="flex justify-center gap-1 flex-wrap">
              {options.map(opt => (
                <span
                  key={opt}
                  className={cn(
                    'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium',
                    isLight ? 'bg-violet-50 text-violet-600' : 'bg-violet-500/10 text-violet-400'
                  )}
                >
                  {prettifyLabel(opt)}
                </span>
              ))}
            </div>
          ) : (
            <div className="flex justify-center">
              <X className="h-4 w-4 text-red-400/40" />
            </div>
          )}
        </td>
      ))}
    </tr>
  )
}

// Section header row for the comparison table
function SectionHeader({ label, colSpan, isLight }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-200 bg-stone-50/80' : 'border-white/[0.08] bg-white/[0.02]'
    )}>
      <td
        colSpan={colSpan}
        className={cn(
          'px-4 py-2 text-[10px] font-semibold uppercase tracking-wide',
          isLight ? 'text-stone-500' : 'text-slate-500'
        )}
      >
        {label}
      </td>
    </tr>
  )
}


function ModalitiesRow({ label, values, isLight, isOutput = false, tooltip = null }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10 cursor-default',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        <span title={tooltip}>
          {label}
        </span>
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
function WinnerRow({ icon, label, winners, value, isLight, modelData }) {
  const hasWinners = winners && winners.length > 0
  
  return (
    <div className={cn(
      'py-2 px-3 rounded-lg border',
      isLight 
        ? 'bg-white/60 border-stone-200/40' 
        : 'bg-white/[0.02] border-white/[0.04]'
    )}>
      {/* Header row with label and value */}
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span className={cn(
            isLight ? 'text-stone-400' : 'text-slate-500'
          )}>{icon}</span>
          <span className={cn(
            'text-xs font-medium',
            isLight ? 'text-stone-700' : 'text-slate-300'
          )}>{label}</span>
        </div>
        <span className={cn(
          'text-xs font-semibold',
          hasWinners
            ? isLight ? 'text-stone-900' : 'text-white'
            : isLight ? 'text-stone-300' : 'text-slate-600'
        )}>
          {hasWinners ? value : '—'}
        </span>
      </div>
      
      {/* Winner models */}
      {hasWinners ? (
        <div className="flex flex-wrap gap-1">
          {winners.slice(0, 3).map((w, i) => {
            const modelIndex = modelData.findIndex(m => m === w)
            return (
              <span
                key={i}
                className={cn(
                  'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium',
                  isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/[0.06] text-slate-400'
                )}
              >
                <span
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ backgroundColor: radarColors[modelIndex % radarColors.length] }}
                />
                {w.model.model_name?.split(' ').slice(-2).join(' ') || w.model.model_id.split('.').pop()}
              </span>
            )
          })}
          {winners.length > 3 && (
            <span className={cn(
              'text-[10px] px-1.5 py-0.5',
              isLight ? 'text-stone-400' : 'text-slate-500'
            )}>
              +{winners.length - 3} more
            </span>
          )}
        </div>
      ) : (
        <p className={cn(
          'text-[10px]',
          isLight ? 'text-stone-400' : 'text-slate-600'
        )}>
          No data available
        </p>
      )}
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

// Convert snake_case to Title Case
function prettifyLabel(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

// Helper to get all regions for a model (on-demand + CRIS + Mantle + Batch)
function getAllModelRegions(model) {
  const onDemand = model.availability?.on_demand?.regions ?? model.in_region ?? model.regions_available ?? []
  const cris = model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions ?? []
  const mantle = model.availability?.mantle?.regions ?? []
  const batch = model.availability?.batch?.regions ?? model.batch_inference_supported?.supported_regions ?? []
  return [...new Set([...onDemand, ...cris, ...mantle, ...batch])]
}

// Detect which pricing types are available for a model
function getAvailablePricingTypes(pricing) {
  const fullPricing = pricing?.fullPricing
  if (!fullPricing?.regions) return {
    inRegion: false, inRegionBatch: false,
    crisGlobal: false, crisGlobalBatch: false,
    crisGeo: false, crisGeoBatch: false,
    mantle: false,
    reserved1mGlobal: false, reserved1mGeo: false,
    reserved3mGlobal: false, reserved3mGeo: false,
    provisioned: false,
    customModel: false,
  }
  
  const result = {
    inRegion: false, inRegionBatch: false,
    crisGlobal: false, crisGlobalBatch: false,
    crisGeo: false, crisGeoBatch: false,
    mantle: false,
    reserved1mGlobal: false, reserved1mGeo: false,
    reserved3mGlobal: false, reserved3mGeo: false,
    provisioned: false,
    customModel: false,
  }
  
  for (const regionData of Object.values(fullPricing.regions)) {
    const groups = regionData?.pricing_groups || {}
    
    // In-Region
    if (groups['On-Demand'] || groups['On-Demand Long Context']) result.inRegion = true
    if (groups['Batch'] || groups['Batch Long Context']) result.inRegionBatch = true
    
    // CRIS Global
    if (groups['On-Demand Global'] || groups['On-Demand Long Context Global']) result.crisGlobal = true
    if (groups['Batch Global'] || groups['Batch Long Context Global']) result.crisGlobalBatch = true
    
    // CRIS Geo
    if (groups['On-Demand Geo'] || groups['On-Demand Long Context Geo']) result.crisGeo = true
    if (groups['Batch Geo'] || groups['Batch Long Context Geo']) result.crisGeoBatch = true
    
    // Mantle
    if (groups['Mantle']) result.mantle = true
    
    // Reserved
    if (groups['Reserved 1 Month Global']) result.reserved1mGlobal = true
    if (groups['Reserved 1 Month Geo']) result.reserved1mGeo = true
    if (groups['Reserved 3 Month Global']) result.reserved3mGlobal = true
    if (groups['Reserved 3 Month Geo']) result.reserved3mGeo = true
    
    // Provisioned
    if (groups['Provisioned Throughput']) result.provisioned = true
    
    // Custom Model
    if (groups['Custom Model']) result.customModel = true
  }
  
  return result
}

// Helper to get pricing for a specific consumption type with dimension filtering
// Includes fallback logic: In-Region and CRIS Geo pricing are equivalent for many models
function getPricesByType(pricing, region, pricingType, options = {}, model = null) {
  const { crisType = 'global', reservedTerm = '1m', reservedScope = 'global', batchMode = false } = options
  
  const fullPricing = pricing?.fullPricing
  if (!fullPricing?.regions) return { inputPrice: null, outputPrice: null, availableRegions: [], hasData: false, usedFallback: false }
  
  // Skip In-Region pricing when hide_in_region is true
  if (pricingType === 'in_region' && model?.availability?.hide_in_region) {
    return { inputPrice: null, outputPrice: null, availableRegions: [], hasData: false, usedFallback: false }
  }
  
  // Try the specified region first, then us-east-1, then any available region
  const regionData = fullPricing.regions[region] || 
                     fullPricing.regions['us-east-1'] || 
                     Object.values(fullPricing.regions)[0]
  
  if (!regionData?.pricing_groups) return { inputPrice: null, outputPrice: null, availableRegions: [], hasData: false, usedFallback: false }
  
  const pricingGroups = regionData.pricing_groups
  
  // Helper to extract prices from a set of group names
  const extractPrices = (groupNames) => {
    let inputPrice = null
    let outputPrice = null
    
    for (const groupName of groupNames) {
      let items = pricingGroups[groupName]
      if (!items || items.length === 0) continue
      
      // Filter out cache pricing and mantle source (unless we're looking at mantle)
      items = items.filter(item => {
        const dim = (item.dimension || '').toLowerCase()
        const desc = (item.description || '').toLowerCase()
        const isCache = dim.includes('cache') || desc.includes('cache')
        if (isCache) return false
        
        // For non-mantle pricing, exclude mantle source
        if (pricingType !== 'mantle') {
          const dims = item.dimensions || {}
          if (dims.source === 'mantle') return false
        }
        return true
      })
      
      for (const item of items) {
        const price = item.price_per_thousand != null 
          ? item.price_per_thousand * 1000 
          : item.price_per_unit
        
        if (price == null) continue
        
        const dim = (item.dimension || '').toLowerCase()
        const desc = (item.description || '').toLowerCase()
        
        const isInput = item.is_input || dim.includes('input') || desc.includes('input')
        const isOutput = item.is_output || dim.includes('output') || desc.includes('output')
        
        if (isInput && inputPrice === null) inputPrice = price
        if (isOutput && outputPrice === null) outputPrice = price
        
        if (inputPrice !== null && outputPrice !== null) break
      }
      
      if (inputPrice !== null || outputPrice !== null) break
    }
    
    return { inputPrice, outputPrice }
  }
  
  // Determine which groups to check based on pricing type
  let groupNames = []
  let fallbackGroupNames = []
  
  switch (pricingType) {
    case 'in_region':
      groupNames = batchMode 
        ? ['Batch', 'Batch Long Context']
        : ['On-Demand', 'On-Demand Long Context']
      // Fallback: CRIS Geo pricing is equivalent to In-Region pricing
      fallbackGroupNames = batchMode
        ? ['Batch Geo', 'Batch Long Context Geo']
        : ['On-Demand Geo', 'On-Demand Long Context Geo']
      break
    case 'cris':
      if (crisType === 'geo') {
        groupNames = batchMode
          ? ['Batch Geo', 'Batch Long Context Geo']
          : ['On-Demand Geo', 'On-Demand Long Context Geo']
        // Fallback: In-Region pricing is equivalent to CRIS Geo pricing
        fallbackGroupNames = batchMode
          ? ['Batch', 'Batch Long Context']
          : ['On-Demand', 'On-Demand Long Context']
      } else {
        groupNames = batchMode
          ? ['Batch Global', 'Batch Long Context Global']
          : ['On-Demand Global', 'On-Demand Long Context Global']
        // No fallback for CRIS Global - it has different pricing
        fallbackGroupNames = []
      }
      break
    case 'mantle':
      groupNames = ['Mantle']
      break
    case 'reserved':
      const term = reservedTerm === '3m' ? '3 Month' : '1 Month'
      const scope = reservedScope === 'geo' ? 'Geo' : 'Global'
      groupNames = [`Reserved ${term} ${scope}`]
      break
    case 'provisioned':
      groupNames = ['Provisioned Throughput']
      break
    case 'custom_model':
      groupNames = ['Custom Model']
      break
    default:
      groupNames = ['On-Demand', 'On-Demand Long Context']
  }
  
  // Try primary groups first
  let { inputPrice, outputPrice } = extractPrices(groupNames)
  let usedFallback = false
  
  // If no pricing found and we have fallback groups, try them
  // Only use fallback if the model has availability for the target type
  if (inputPrice === null && outputPrice === null && fallbackGroupNames.length > 0) {
    const consumptionOptions = model?.consumption_options || []
    const hasInRegionAvailability = consumptionOptions.includes('on_demand') || 
                                    (model?.in_region && model.in_region.length > 0) ||
                                    (model?.availability?.on_demand?.regions?.length > 0)
    const hasCrisAvailability = consumptionOptions.includes('cross_region_inference') ||
                                model?.availability?.cross_region?.supported
    
    // For In-Region: only fallback to CRIS Geo if model has In-Region availability
    // For CRIS Geo: only fallback to In-Region if model has CRIS availability
    const canUseFallback = (pricingType === 'in_region' && hasInRegionAvailability) ||
                           (pricingType === 'cris' && crisType === 'geo' && hasCrisAvailability)
    
    if (canUseFallback) {
      const fallbackPrices = extractPrices(fallbackGroupNames)
      if (fallbackPrices.inputPrice !== null || fallbackPrices.outputPrice !== null) {
        inputPrice = fallbackPrices.inputPrice
        outputPrice = fallbackPrices.outputPrice
        usedFallback = true
      }
    }
  }
  
  // Get regions that have this pricing type (including fallback groups)
  const allGroupNames = [...groupNames, ...fallbackGroupNames]
  const availableRegions = Object.keys(fullPricing.regions).filter(r => {
    const rData = fullPricing.regions[r]
    if (!rData?.pricing_groups) return false
    return allGroupNames.some(g => rData.pricing_groups[g]?.length > 0)
  })
  
  return { inputPrice, outputPrice, availableRegions, hasData: inputPrice !== null || outputPrice !== null, usedFallback }
}

// Helper function to get pricing label// Helper function to get pricing label
function getPricingLabel(pricingType, options = {}) {
  const { crisType, reservedTerm, reservedScope, batchMode } = options
  
  switch (pricingType) {
    case 'in_region':
      return batchMode ? 'In-Region Batch' : 'In-Region'
    case 'cris':
      const crisLabel = crisType === 'geo' ? 'CRIS Geo' : 'CRIS Global'
      return batchMode ? `${crisLabel} Batch` : crisLabel
    case 'mantle':
      return 'Mantle'
    case 'reserved':
      const term = reservedTerm === '3m' ? '3M' : '1M'
      const scope = reservedScope === 'geo' ? 'Geo' : 'Global'
      return `Reserved ${term} ${scope}`
    case 'provisioned':
      return 'Provisioned'
    case 'custom_model':
      return 'Custom Model'
    default:
      return 'On-Demand'
  }
}

function PriceRow({ label, values, isLight, bestIndices = null, tooltip = null }) {
  return (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-100' : 'border-white/[0.04]'
    )}>
      <td className={cn(
        'px-4 py-2.5 font-medium text-xs whitespace-nowrap sticky left-0 z-10 cursor-default',
        isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
      )}>
        <span title={tooltip}>
          {label}
        </span>
      </td>
      {values.map((data, idx) => {
        const isNA = data.value === 'N/A'
        const isBest = bestIndices?.has(idx)
        return (
          <td
            key={idx}
            className={cn(
              'px-3 py-2.5 text-center text-sm',
              isNA 
                ? isLight ? 'text-stone-300' : 'text-slate-700'
                : isBest
                  ? 'text-emerald-600 font-medium'
                  : isLight ? 'text-stone-900 font-medium' : 'text-white font-medium'
            )}
          >
            <div className="flex items-center justify-center gap-1">
              {data.value}
              {isBest && !isNA && <Trophy className="h-3 w-3 text-emerald-500 flex-shrink-0" />}
            </div>
          </td>
        )
      })}
    </tr>
  )
}

export function OverviewTab({ selectedModels, getPricingForModel, allModels, isLight }) {
  const [radarCollapsed, setRadarCollapsed] = useState(false)
  const [pricingType, setPricingType] = useState('in_region') // 'in_region', 'cris', 'mantle', 'reserved', 'provisioned', 'custom_model'
  const [crisType, setCrisType] = useState('global') // 'global' or 'geo'
  const [reservedTerm, setReservedTerm] = useState('1m') // '1m' or '3m'
  const [reservedScope, setReservedScope] = useState('global') // 'global' or 'geo'
  const [batchMode, setBatchMode] = useState(false) // toggle for batch pricing
  const [copiedModelId, setCopiedModelId] = useState(null) // track which model ID was copied
  const [expandedDimension, setExpandedDimension] = useState(null) // 'context' | 'cost' | 'regions' | null

  // Handle copy to clipboard with visual feedback
  const handleCopyModelId = async (modelId) => {
    try {
      await navigator.clipboard.writeText(modelId)
      setCopiedModelId(modelId)
      setTimeout(() => setCopiedModelId(null), 1500)
    } catch (err) {
      console.error('Failed to copy model ID:', err)
    }
  }

  const canViewProvisioned = true

  const modelData = useMemo(() => selectedModels.map(({ model, region }) => {
    const pricing = getPricingForModel?.(model, region)
    const priceData = getPricesByType(pricing, region, pricingType, {
      crisType,
      reservedTerm,
      reservedScope,
      batchMode,
    }, model)
    const availablePricingTypes = getAvailablePricingTypes(pricing)
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
    
    // Technical details
    const modelId = model.model_id
    const customizationOptions = model.customization?.customization_supported || []

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
      modelId,
      customizationOptions,
      inputPrice: priceData.inputPrice,
      outputPrice: priceData.outputPrice,
      priceRegions: priceData.availableRegions,
      availablePricingTypes,
    }
  }), [selectedModels, getPricingForModel, pricingType, crisType, reservedTerm, reservedScope, batchMode])

  // Relative benchmarks from only the compared models
  const relativeBenchmarks = useMemo(() => {
    const allCosts = modelData.map(d => ((d.inputPrice || 0) + (d.outputPrice || 0))).filter(c => c > 0)
    const maxCost = allCosts.length > 0 ? Math.max(...allCosts) : 1
    const minCost = allCosts.length > 0 ? Math.min(...allCosts) : 0
    const maxContext = Math.max(...modelData.map(d => d.effectiveContextWindow || d.contextWindow), 1)
    const maxRegions = Math.max(...modelData.map(d => d.regions.length), 1)
    return { maxContext, maxRegions, maxCost, minCost }
  }, [modelData])

  // Radar chart data
  const radarScores = useMemo(() => computeRadarScores(modelData, relativeBenchmarks), [modelData, relativeBenchmarks])

  const radarChartData = useMemo(() => {
    const axes = ['Context Window', 'Lowest Cost', 'Most Regions']
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
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <DollarSign className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
              Lowest Cost Input ({getPricingLabel(pricingType, { crisType, reservedTerm, reservedScope, batchMode })})
            </span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
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
      {modelData.length >= 1 && (
        modelData.length >= 2 ? (
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
          </div>

          {!radarCollapsed && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-0">
              {/* Radar Chart - takes 2 columns on large screens */}
              <div className={cn(
                'lg:col-span-2 lg:border-r relative',
                isLight ? 'border-stone-200/60' : 'border-white/[0.06]'
              )} style={{ height: 280 }}>
                {/* Pricing Selector - positioned in top-right corner */}
                <div className="absolute top-2 right-3 z-10 flex items-center gap-1.5">
                  <span className={cn(
                    'text-[9px] font-medium uppercase tracking-wider',
                    isLight ? 'text-stone-400' : 'text-slate-500'
                  )}>
                    Pricing:
                  </span>
                  
                  {/* Main pricing type buttons */}
                  <div className={cn(
                    'inline-flex rounded-md border overflow-hidden h-5',
                    isLight ? 'border-stone-200 bg-white/80 backdrop-blur-sm' : 'border-[#373a40] bg-[#1a1b1e]/80 backdrop-blur-sm'
                  )}>
                    <button
                      onClick={() => setPricingType('in_region')}
                      className={cn(
                        'px-2 py-0.5 text-[9px] font-medium transition-colors',
                        pricingType === 'in_region'
                          ? isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
                          : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-transparent text-[#9a9b9f] hover:bg-[#2c2d32]'
                      )}
                    >
                      In-Region
                    </button>
                    <button
                      onClick={() => setPricingType('cris')}
                      className={cn(
                        'px-2 py-0.5 text-[9px] font-medium transition-colors border-l',
                        isLight ? 'border-stone-200' : 'border-[#373a40]',
                        pricingType === 'cris'
                          ? isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
                          : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-transparent text-[#9a9b9f] hover:bg-[#2c2d32]'
                      )}
                    >
                      CRIS
                    </button>
                  </div>
                  
                  {/* CRIS sub-selector */}
                  {pricingType === 'cris' && (
                    <div className={cn(
                      'inline-flex rounded-md border overflow-hidden h-5',
                      isLight ? 'border-stone-200 bg-white/80 backdrop-blur-sm' : 'border-[#373a40] bg-[#1a1b1e]/80 backdrop-blur-sm'
                    )}>
                      <button
                        onClick={() => setCrisType('global')}
                        className={cn(
                          'px-2 py-0.5 text-[9px] font-medium transition-colors',
                          crisType === 'global'
                            ? isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
                            : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-transparent text-[#9a9b9f] hover:bg-[#2c2d32]'
                        )}
                      >
                        Global
                      </button>
                      <button
                        onClick={() => setCrisType('geo')}
                        className={cn(
                          'px-2 py-0.5 text-[9px] font-medium transition-colors border-l',
                          isLight ? 'border-stone-200' : 'border-[#373a40]',
                          crisType === 'geo'
                            ? isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
                            : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-transparent text-[#9a9b9f] hover:bg-[#2c2d32]'
                        )}
                      >
                        Geo
                      </button>
                    </div>
                  )}
                </div>
                
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
                isLight ? 'bg-stone-50/50' : 'bg-white/[0.01]'
              )}>
                <h4 className={cn(
                  'text-xs font-semibold mb-3',
                  isLight ? 'text-stone-700' : 'text-slate-300'
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
                
                {/* Cost Efficiency Winner */}
                <WinnerRow
                  icon={<DollarSign className="h-3 w-3" />}
                  label={`Lowest Cost (${getPricingLabel(pricingType, { crisType, reservedTerm, reservedScope, batchMode })})`}
                  winners={[...inputPriceBestSet].map(i => modelData[i])}
                  value={minInputPrice !== null ? `$${minInputPrice < 0.01 ? minInputPrice.toFixed(4) : minInputPrice.toFixed(2)}` : '—'}
                  isLight={isLight}
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
              </div>
            </div>
          )}

          {/* Explanatory text - BELOW the radar chart */}
          {!radarCollapsed && (
            <div className={cn(
              'px-4 py-3 border-t',
              isLight ? 'bg-stone-50/50 border-stone-200/60' : 'bg-white/[0.02] border-white/[0.06]'
            )}>
              {/* Compact info buttons row */}
              <div className="flex items-center justify-center gap-4">
                {/* Context Window */}
                <button
                  onClick={() => setExpandedDimension(expandedDimension === 'context' ? null : 'context')}
                  className={cn(
                    'flex items-center gap-1.5 px-2 py-1 rounded-md transition-all duration-200',
                    expandedDimension === 'context'
                      ? isLight
                        ? 'bg-amber-100/80 text-amber-700'
                        : 'bg-[#1A9E7A]/20 text-[#1A9E7A]'
                      : isLight
                        ? 'hover:bg-stone-100 text-stone-600'
                        : 'hover:bg-white/[0.05] text-slate-400'
                  )}
                >
                  <Cpu className={cn(
                    'h-3.5 w-3.5 transition-colors',
                    expandedDimension === 'context'
                      ? isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
                      : isLight ? 'text-stone-500' : 'text-slate-500'
                  )} />
                  <span className="text-[10px] font-medium">Context Window</span>
                  <Info className={cn(
                    'h-3 w-3 transition-opacity',
                    expandedDimension === 'context' ? 'opacity-70' : 'opacity-40 group-hover:opacity-60'
                  )} />
                </button>

                {/* Lowest Cost */}
                <button
                  onClick={() => setExpandedDimension(expandedDimension === 'cost' ? null : 'cost')}
                  className={cn(
                    'flex items-center gap-1.5 px-2 py-1 rounded-md transition-all duration-200',
                    expandedDimension === 'cost'
                      ? isLight
                        ? 'bg-amber-100/80 text-amber-700'
                        : 'bg-[#1A9E7A]/20 text-[#1A9E7A]'
                      : isLight
                        ? 'hover:bg-stone-100 text-stone-600'
                        : 'hover:bg-white/[0.05] text-slate-400'
                  )}
                >
                  <DollarSign className={cn(
                    'h-3.5 w-3.5 transition-colors',
                    expandedDimension === 'cost'
                      ? isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
                      : isLight ? 'text-stone-500' : 'text-slate-500'
                  )} />
                  <span className="text-[10px] font-medium">Lowest Cost</span>
                  <Info className={cn(
                    'h-3 w-3 transition-opacity',
                    expandedDimension === 'cost' ? 'opacity-70' : 'opacity-40 group-hover:opacity-60'
                  )} />
                </button>

                {/* Most Regions */}
                <button
                  onClick={() => setExpandedDimension(expandedDimension === 'regions' ? null : 'regions')}
                  className={cn(
                    'flex items-center gap-1.5 px-2 py-1 rounded-md transition-all duration-200',
                    expandedDimension === 'regions'
                      ? isLight
                        ? 'bg-amber-100/80 text-amber-700'
                        : 'bg-[#1A9E7A]/20 text-[#1A9E7A]'
                      : isLight
                        ? 'hover:bg-stone-100 text-stone-600'
                        : 'hover:bg-white/[0.05] text-slate-400'
                  )}
                >
                  <Globe className={cn(
                    'h-3.5 w-3.5 transition-colors',
                    expandedDimension === 'regions'
                      ? isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
                      : isLight ? 'text-stone-500' : 'text-slate-500'
                  )} />
                  <span className="text-[10px] font-medium">Most Regions</span>
                  <Info className={cn(
                    'h-3 w-3 transition-opacity',
                    expandedDimension === 'regions' ? 'opacity-70' : 'opacity-40 group-hover:opacity-60'
                  )} />
                </button>
              </div>

              {/* Expanded content panel */}
              <div className={cn(
                'grid transition-all duration-300 ease-in-out',
                expandedDimension ? 'grid-rows-[1fr] mt-3' : 'grid-rows-[0fr]'
              )}>
                <div className="overflow-hidden">
                  {expandedDimension && (
                    <div className={cn(
                      'p-3 rounded-lg border',
                      isLight ? 'bg-white/80 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
                    )}>
                      {expandedDimension === 'context' && (
                        <div className="space-y-2">
                          <p className={cn('text-[10px]', isLight ? 'text-stone-600' : 'text-slate-400')}>
                            The maximum input size a model can handle. For models with multiple context options (e.g., standard and extended), the highest available context window is used.
                          </p>
                          <div className={cn(
                            'text-[9px] font-mono px-2 py-1.5 rounded inline-block',
                            isLight ? 'bg-stone-100 text-stone-700' : 'bg-black/20 text-slate-300'
                          )}>
                            Score = 10 × (model_context / max_context)
                          </div>
                          <p className={cn('text-[9px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
                            Model with largest context window gets 10/10
                          </p>
                        </div>
                      )}
                      {expandedDimension === 'cost' && (
                        <div className="space-y-2">
                          <p className={cn('text-[10px]', isLight ? 'text-stone-600' : 'text-slate-400')}>
                            Compares total cost (input + output price per 1M tokens).
                          </p>
                          <div className={cn(
                            'text-[9px] font-mono px-2 py-1.5 rounded inline-block',
                            isLight ? 'bg-stone-100 text-stone-700' : 'bg-black/20 text-slate-300'
                          )}>
                            Score = 10 × (1 - model_cost / max_cost)
                          </div>
                          <p className={cn('text-[9px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
                            Lower cost = higher score (10 = best value, 0 = highest cost)
                          </p>
                        </div>
                      )}
                      {expandedDimension === 'regions' && (
                        <div className="space-y-2">
                          <p className={cn('text-[10px]', isLight ? 'text-stone-600' : 'text-slate-400')}>
                            Counts total AWS regions where the model is available.
                          </p>
                          <div className={cn(
                            'text-[9px] font-mono px-2 py-1.5 rounded inline-block',
                            isLight ? 'bg-stone-100 text-stone-700' : 'bg-black/20 text-slate-300'
                          )}>
                            Score = 10 × (model_regions / max_regions)
                          </div>
                          <p className={cn('text-[9px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
                            Model available in most regions gets 10/10
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              
              {/* Note about relative scoring */}
              <p className={cn(
                'text-[9px] mt-3 text-center',
                isLight ? 'text-stone-400' : 'text-slate-600'
              )}>
                All scores are relative to the selected models only — not the entire catalog.
              </p>
            </div>
          )}
        </div>
        ) : (
          // Placeholder for single model
          <div className={cn(
            'rounded-lg border overflow-hidden',
            isLight
              ? 'bg-white/70 border-stone-200/60 backdrop-blur-xl'
              : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
          )}>
            <div className={cn(
              'px-4 py-2.5 border-b flex items-center gap-2',
              isLight ? 'bg-stone-50/60 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
            )}>
              <h3 className={cn('font-semibold text-xs', isLight ? 'text-stone-900' : 'text-white')}>
                Model Comparison Radar
              </h3>
            </div>
            <div className="flex flex-col items-center justify-center py-12 px-4">
              <div className={cn(
                'w-16 h-16 rounded-full flex items-center justify-center mb-4',
                isLight ? 'bg-stone-100' : 'bg-white/[0.04]'
              )}>
                <RadarIcon className={cn('h-8 w-8', isLight ? 'text-stone-300' : 'text-slate-600')} />
              </div>
              <p className={cn(
                'text-sm font-medium mb-1',
                isLight ? 'text-stone-600' : 'text-slate-400'
              )}>
                Add another model to compare
              </p>
              <p className={cn(
                'text-xs text-center max-w-xs',
                isLight ? 'text-stone-400' : 'text-slate-500'
              )}>
                The radar chart visualizes how models compare across context window, pricing, and availability dimensions.
              </p>
            </div>
          </div>
        )
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
                    {/* Model ID with copy button */}
                    <div className="flex items-center justify-center gap-1 mt-1">
                      <span
                        className={cn(
                          'text-[9px] font-mono max-w-[100px] truncate inline-block',
                          isLight ? 'text-stone-500' : 'text-slate-500'
                        )}
                        title={model.model_id}
                      >
                        {model.model_id}
                      </span>
                      <button
                        onClick={() => handleCopyModelId(model.model_id)}
                        className={cn(
                          'p-0.5 rounded transition-colors flex-shrink-0',
                          isLight
                            ? 'text-stone-400 hover:text-amber-700 hover:bg-stone-100'
                            : 'text-slate-600 hover:text-[#1A9E7A] hover:bg-white/[0.05]'
                        )}
                        title="Copy model ID"
                      >
                        {copiedModelId === model.model_id ? (
                          <Check className="h-3 w-3 text-emerald-500" />
                        ) : (
                          <Copy className="h-3 w-3" />
                        )}
                      </button>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* SPECIFICATIONS */}
              <SectionHeader label="Specifications" colSpan={modelData.length + 1} isLight={isLight} />
              <ContextRow
                label="Context Window"
                tooltip="Maximum input size the model can process in a single request"
                values={modelData.map(d => ({
                  formatted: formatNumber(d.effectiveContextWindow),
                  baseFormatted: d.effectiveContextWindow > d.contextWindow ? formatNumber(d.contextWindow) : null,
                  hasLongContext: d.hasLongContext,
                }))}
                isLight={isLight}
                bestIndices={contextBestSet}
              />
              <MetricRow
                label="Max Output Tokens"
                tooltip="Maximum number of tokens the model can generate in a response"
                values={modelData.map(d => formatNumber(d.maxOutput))}
                isLight={isLight}
                bestIndices={outputBestSet}
              />
              
              {/* MODALITIES */}
              <SectionHeader label="Modalities" colSpan={modelData.length + 1} isLight={isLight} />
              <ModalitiesRow
                label="Input"
                tooltip="Types of content the model can accept as input"
                values={modelData.map(d => d.inputModalities)}
                isLight={isLight}
              />
              <ModalitiesRow
                label="Output"
                tooltip="Types of content the model can generate"
                values={modelData.map(d => d.outputModalities)}
                isLight={isLight}
                isOutput={true}
              />
              
              {/* AVAILABILITY */}
              <SectionHeader label="Availability" colSpan={modelData.length + 1} isLight={isLight} />
              <MetricRow
                label="AWS Regions"
                tooltip="Total number of AWS regions where this model is available (combining all consumption types)"
                values={modelData.map(d => `${d.regions.length}`)}
                isLight={isLight}
                bestIndices={regionsBestSet}
              />
              <BooleanRow
                label="Active Status"
                tooltip="Whether the model is actively supported or in legacy/deprecated status"
                values={modelData.map(d => d.isActive)}
                isLight={isLight}
              />
              
              {/* PRICING */}
              <SectionHeader label={`Pricing (${getPricingLabel(pricingType, { crisType, reservedTerm, reservedScope, batchMode })})`} colSpan={modelData.length + 1} isLight={isLight} />
              <PriceRow
                label="Input (per 1M tokens)"
                tooltip="Cost per 1 million input tokens for the selected pricing type"
                values={modelData.map(d => ({ value: formatPrice(d.inputPrice) }))}
                isLight={isLight}
                bestIndices={inputPriceBestSet}
              />
              <PriceRow
                label="Output (per 1M tokens)"
                tooltip="Cost per 1 million output tokens for the selected pricing type"
                values={modelData.map(d => ({ value: formatPrice(d.outputPrice) }))}
                isLight={isLight}
                bestIndices={outputPriceBestSet}
              />
              
              {/* FEATURES */}
              <SectionHeader label="Features" colSpan={modelData.length + 1} isLight={isLight} />
              <BooleanRow
                label="Streaming Support"
                tooltip="Whether the model supports streaming responses in real-time"
                values={modelData.map(d => d.streamingSupported)}
                isLight={isLight}
              />
              <BooleanRow
                label="Cross-Region Inference"
                tooltip="Whether the model supports Cross-Region Inference (CRIS) for global availability"
                values={modelData.map(d => d.crisSupported)}
                isLight={isLight}
              />
              <BooleanRow
                label="Batch Processing"
                tooltip="Whether the model supports batch inference for processing multiple requests"
                values={modelData.map(d => d.batchSupported)}
                isLight={isLight}
              />
              <BooleanRow
                label="Mantle Support"
                tooltip="Whether the model is available through Amazon Mantle (OpenAI-compatible endpoint)"
                values={modelData.map(d => d.mantleSupported)}
                isLight={isLight}
              />
              
              {/* ADDITIONAL INFO */}
              <SectionHeader label="Additional Info" colSpan={modelData.length + 1} isLight={isLight} />
              <MetricRow
                label="Capabilities"
                tooltip="Number of documented capabilities this model has"
                values={modelData.map(d => `${d.capabilitiesCount}`)}
                isLight={isLight}
              />
              <MetricRow
                label="Use Cases"
                tooltip="Number of documented use cases for this model"
                values={modelData.map(d => `${d.useCasesCount}`)}
                isLight={isLight}
              />
              <CustomizationRow
                label="Customization"
                tooltip="Available customization options like fine-tuning or continued pre-training"
                values={modelData.map(d => d.customizationOptions)}
                isLight={isLight}
              />
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
