import { useState } from 'react'
import { DollarSign, Trophy, TrendingDown, Info, ChevronDown, ChevronRight, Zap, Globe, Package, Server } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { providerColorClasses } from '@/config/constants'

const providerColors = providerColorClasses

// Dimension badge component for displaying pricing dimensions
function DimensionBadge({ dimension, value }) {
  if (!value) return null
  
  const colors = {
    mantle: 'bg-purple-100 text-purple-800 dark:bg-purple-500/20 dark:text-purple-400',
    standard: 'bg-gray-100 text-gray-800 dark:bg-white/10 dark:text-slate-400',
    global: 'bg-blue-100 text-blue-800 dark:bg-blue-500/20 dark:text-blue-400',
    regional: 'bg-green-100 text-green-800 dark:bg-green-500/20 dark:text-green-400',
    flex: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-500/20 dark:text-yellow-400',
    priority: 'bg-red-100 text-red-800 dark:bg-red-500/20 dark:text-red-400',
    long: 'bg-orange-100 text-orange-800 dark:bg-orange-500/20 dark:text-orange-400',
  }
  
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[value] || colors.standard}`}>
      {value}
    </span>
  )
}

function formatPrice(price) {
  if (price === null || price === undefined) return null
  if (price === 0) return '$0'
  if (price < 0.01) return `$${price.toFixed(4)}`
  return `$${price.toFixed(2)}`
}

// Simplify pricing group names for display
const groupLabels = {
  'On-Demand': 'In Region',
  'On-Demand Global': 'CRIS Global',
  'On-Demand Geo': 'CRIS Geo',
  'On-Demand Long Context': 'In Region Long Context',
  'On-Demand Long Context Global': 'Long Context CRIS Global',
  'On-Demand Long Context Geo': 'Long Context CRIS Geo',
  'Batch': 'Batch',
  'Batch Global': 'Batch CRIS Global',
  'Batch Geo': 'Batch CRIS Geo',
  'Batch Long Context': 'Batch Long Context',
  'Batch Long Context Global': 'Batch Long Context CRIS Global',
  'Batch Long Context Geo': 'Batch Long Context CRIS Geo',
  'Provisioned Throughput': 'Provisioned Throughput',
  'Custom Model': 'Custom Model',
}

// Simplify item descriptions for comparison rows
function simplifyDescription(item) {
  const dim = item.dimension || ''
  const desc = item.description || ''

  // For token-based items, try to determine input/output
  // Prefer is_input/is_output flags from new structure
  const dimLower = dim.toLowerCase()
  const descLower = desc.toLowerCase()

  // Use is_input/is_output flags first (new structure)
  if (item.is_input) return 'Input'
  if (item.is_output) return 'Output'
  
  // Fallback to dimension/description parsing
  if (dimLower.includes('input') || descLower.includes('input')) return 'Input'
  if (dimLower.includes('output') || descLower.includes('output')) return 'Output'
  if (dimLower.includes('cache-read') || descLower.includes('cache read')) return 'Cache Read'
  if (dimLower.includes('cache-write') || descLower.includes('cache write')) return 'Cache Write'

  // Provisioned throughput
  if (item.pricing_type === 'model_unit' || dimLower.includes('model-unit') || dimLower.includes('modelunit')) {
    // Extract commitment info
    if (dimLower.includes('nocommitment') || dimLower.includes('no-commitment')) return 'No Commitment'
    if (dimLower.includes('1-month') || dimLower.includes('1month')) return '1-Month Commitment'
    if (dimLower.includes('6-month') || dimLower.includes('6month')) return '6-Month Commitment'
    return 'Model Unit'
  }

  // Image pricing
  if (dimLower.includes('t2i')) {
    const resMatch = dimLower.match(/(\d{3,4})/)
    const tier = dimLower.includes('premium') ? 'Premium' : 'Standard'
    return `Text-to-Image ${resMatch ? resMatch[1] + 'px' : ''} ${tier}`
  }
  if (dimLower.includes('i2i')) {
    const resMatch = dimLower.match(/(\d{3,4})/)
    const tier = dimLower.includes('premium') ? 'Premium' : 'Standard'
    return `Image-to-Image ${resMatch ? resMatch[1] + 'px' : ''} ${tier}`
  }

  // Video pricing
  if (dimLower.includes('t2v') || dimLower.includes('i2v')) {
    const type = dimLower.includes('t2v') ? 'T2V' : 'I2V'
    const fpsMatch = dimLower.match(/(low|med|high)fps/i)
    const resMatch = dimLower.match(/(sd|hd|fhd)res/i)
    const parts = [type]
    if (fpsMatch) parts.push(fpsMatch[1].charAt(0).toUpperCase() + fpsMatch[1].slice(1) + ' FPS')
    if (resMatch) parts.push(resMatch[1].toUpperCase())
    return parts.join(' · ')
  }

  // Flex pricing
  if (dimLower.includes('-flex')) {
    if (dimLower.includes('input')) return 'Input (Flex)'
    if (dimLower.includes('output')) return 'Output (Flex)'
    return 'Flex'
  }

  // Priority pricing
  if (dimLower.includes('-priority')) {
    if (dimLower.includes('input')) return 'Input (Priority)'
    if (dimLower.includes('output')) return 'Output (Priority)'
    return 'Priority'
  }

  // Latency optimized
  if (dimLower.includes('latency')) {
    if (dimLower.includes('input')) return 'Input (Low Latency)'
    if (dimLower.includes('output')) return 'Output (Low Latency)'
    return 'Low Latency'
  }

  // Fallback: use description or dimension, truncated
  if (desc && desc.length < 60) return desc
  if (dim && dim.length < 60) return dim
  return desc.slice(0, 50) || dim.slice(0, 50) || 'Other'
}

function getItemPrice(item) {
  if (item.price_per_thousand != null) return { value: item.price_per_thousand * 1000, unit: item.unit_label || 'per 1M tokens' }
  if (item.price_per_unit != null) return { value: item.price_per_unit, unit: item.unit_label || `per ${item.unit || 'unit'}` }
  return { value: null, unit: '' }
}

function formatPriceValue(priceInfo) {
  if (!priceInfo || priceInfo.value === null || priceInfo.value === undefined) return 'N/A'
  if (priceInfo.value === 0) return '$0'
  const v = priceInfo.value
  if (v < 0.01) return `$${v.toFixed(4)}`
  return `$${v.toFixed(2)}`
}

// Build a unified row key for comparison across models
function getRowKey(groupName, item) {
  const desc = simplifyDescription(item)
  return `${groupName}::${desc}`
}

// Consumption type configuration
const consumptionTypes = {
  on_demand: {
    label: 'In Region',
    icon: Zap,
    groups: ['On-Demand', 'On-Demand Long Context'],
    description: 'Standard on-demand pricing'
  },
  cross_region: {
    label: 'Cross-Region (CRIS)',
    icon: Globe,
    groups: ['On-Demand Global', 'On-Demand Long Context Global'],
    description: 'Cross-region inference'
  },
  batch: {
    label: 'Batch',
    icon: Package,
    groups: ['Batch', 'Batch Global', 'Batch Long Context', 'Batch Long Context Global'],
    description: 'Batch processing'
  },
  provisioned: {
    label: 'Provisioned',
    icon: Server,
    groups: ['Provisioned Throughput'],
    description: 'Reserved capacity'
  }
}

function PricingGroupSection({ groupName, rows, models, pricingByModel, isLight }) {
  const [expanded, setExpanded] = useState(true)
  const label = groupLabels[groupName] || groupName

  // Check if any model has data for this group
  const hasData = rows.some(row =>
    models.some((_, idx) => {
      const entry = pricingByModel[idx]?.[row.key]
      return entry && entry.value !== null
    })
  )

  if (!hasData) return null

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight
        ? 'bg-white/80 border-stone-200/80'
        : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
    )}>
      {/* Group header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          'w-full flex items-center justify-between px-4 py-2.5 text-left transition-colors',
          isLight
            ? 'hover:bg-stone-50'
            : 'hover:bg-white/5'
        )}
      >
        <div className="flex items-center gap-2">
          {expanded
            ? <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
            : <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
          }
          <span className={cn(
            'font-semibold text-sm',
            isLight ? 'text-stone-900' : 'text-white'
          )}>
            {label}
          </span>
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            {rows.length} {rows.length === 1 ? 'tier' : 'tiers'}
          </Badge>
        </div>
      </button>

      {expanded && (
        <div className="overflow-auto max-h-[400px]">
          <table className="w-full">
            <thead className="sticky top-0 z-20">
              <tr className={cn(
                'border-t border-b',
                isLight ? 'border-stone-200 bg-stone-50' : 'border-white/[0.06] bg-[#1a1b1e]'
              )}>
                <th className={cn(
                  'px-5 py-2.5 text-left text-sm font-semibold w-44 min-w-[140px] sticky left-0 z-30',
                  isLight ? 'text-stone-700 bg-stone-50' : 'text-slate-300 bg-[#1a1b1e]'
                )}>
                  Dimension
                </th>
                {models.map((m) => (
                  <th key={m.model.model_id} className="px-3 py-2.5 text-center min-w-[130px]">
                    <Badge className={cn(
                      'text-[9px] mb-0.5',
                      isLight ? 'text-[#faf9f5]' : 'text-white',
                      providerColors[m.model.model_provider] || providerColors.default
                    )}>
                      {m.model.model_provider}
                    </Badge>
                    <p className={cn(
                      'text-[10px] font-semibold line-clamp-1 max-w-[120px] mx-auto',
                      isLight ? 'text-stone-700' : 'text-slate-300'
                    )}>
                      {m.model.model_name || m.model.model_id}
                    </p>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const prices = models.map((_, idx) => pricingByModel[idx]?.[row.key]?.value ?? null)
                const validPrices = prices.filter(p => p !== null && p > 0)
                const minPrice = validPrices.length > 1 ? Math.min(...validPrices) : null
                const maxPrice = validPrices.length > 0 ? Math.max(...validPrices) : null

                return (
                  <tr
                    key={row.key}
                    className={cn(
                      'border-b last:border-b-0',
                      isLight ? 'border-stone-100' : 'border-white/[0.04]'
                    )}
                  >
                    <td className={cn(
                      'px-5 py-3 text-sm font-medium sticky left-0 z-10',
                      isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]'
                    )}>
                      <div>
                        {row.label}
                        {row.unit && (
                          <span className={cn(
                            'block text-[10px] font-normal mt-0.5',
                            isLight ? 'text-stone-400' : 'text-slate-500'
                          )}>
                            {row.unit}
                          </span>
                        )}
                        {/* Dimension badges */}
                        {row.dimensions && (
                          <div className="flex gap-1 mt-1 flex-wrap">
                            {row.dimensions.source && row.dimensions.source !== 'standard' && (
                              <DimensionBadge dimension="source" value={row.dimensions.source} />
                            )}
                            {row.dimensions.geo && (
                              <DimensionBadge dimension="geo" value={row.dimensions.geo} />
                            )}
                            {row.dimensions.tier && (
                              <DimensionBadge dimension="tier" value={row.dimensions.tier} />
                            )}
                            {row.dimensions.context && row.dimensions.context !== 'standard' && (
                              <DimensionBadge dimension="context" value={row.dimensions.context} />
                            )}
                          </div>
                        )}
                      </div>
                    </td>
                    {models.map((m, idx) => {
                      const entry = pricingByModel[idx]?.[row.key]
                      const price = entry?.value ?? null
                      const isCheapest = price !== null && price > 0 && price === minPrice && validPrices.length > 1
                      const formattedPrice = formatPriceValue(entry)
                      // Bar width: relative to max price in this row
                      const barWidth = price !== null && price > 0 && maxPrice > 0
                        ? Math.max((price / maxPrice) * 100, 8)
                        : 0

                      return (
                        <td
                          key={m.model.model_id}
                          className="px-3 py-3"
                        >
                          {formattedPrice === 'N/A' ? (
                            <div className="text-center">
                              <span className={cn('text-xs', isLight ? 'text-stone-300' : 'text-slate-600')}>—</span>
                            </div>
                          ) : (
                            <div className="flex flex-col items-center gap-1">
                              <div className="flex items-center gap-1">
                                <span className={cn(
                                  'text-sm font-semibold tabular-nums',
                                  isCheapest
                                    ? 'text-emerald-600'
                                    : isLight ? 'text-stone-900' : 'text-white'
                                )}>
                                  {formattedPrice}
                                </span>
                                {isCheapest && (
                                  <Trophy className="h-3 w-3 text-emerald-500 flex-shrink-0" />
                                )}
                              </div>
                              <div className={cn(
                                'w-full h-1.5 rounded-full overflow-hidden',
                                isLight ? 'bg-stone-100' : 'bg-white/[0.04]'
                              )}>
                                <div
                                  className={cn(
                                    'h-full rounded-full transition-all',
                                    isCheapest
                                      ? 'bg-emerald-500'
                                      : isLight ? 'bg-amber-400/60' : 'bg-[#1A9E7A]/40'
                                  )}
                                  style={{ width: `${barWidth}%` }}
                                />
                              </div>
                            </div>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// Quick comparison table for selected consumption type
function QuickComparisonTable({ selectedModels, pricingByModel, consumptionType, isLight }) {
  const config = consumptionTypes[consumptionType]
  if (!config) return null

  // Get input/output prices for each model from the selected consumption type groups
  const comparisonData = selectedModels.map((m, idx) => {
    const modelPricing = pricingByModel[idx] || {}
    let inputPrice = null
    let outputPrice = null

    // Look through all groups for this consumption type
    for (const groupName of config.groups) {
      const inputKey = `${groupName}::Input`
      const outputKey = `${groupName}::Output`
      
      if (modelPricing[inputKey]?.value != null && inputPrice === null) {
        inputPrice = modelPricing[inputKey].value
      }
      if (modelPricing[outputKey]?.value != null && outputPrice === null) {
        outputPrice = modelPricing[outputKey].value
      }
      
      if (inputPrice !== null && outputPrice !== null) break
    }

    return {
      model: m.model,
      inputPrice,
      outputPrice,
      hasPricing: inputPrice !== null || outputPrice !== null
    }
  })

  // Find cheapest
  const validInputPrices = comparisonData.filter(d => d.inputPrice != null).map(d => d.inputPrice)
  const validOutputPrices = comparisonData.filter(d => d.outputPrice != null).map(d => d.outputPrice)
  const minInput = validInputPrices.length > 0 ? Math.min(...validInputPrices) : null
  const minOutput = validOutputPrices.length > 0 ? Math.min(...validOutputPrices) : null

  const modelsWithPricing = comparisonData.filter(d => d.hasPricing).length

  if (modelsWithPricing === 0) {
    return (
      <div className={cn(
        'text-center py-6 rounded-lg border',
        isLight
          ? 'bg-white/80 border-stone-200/80 text-stone-500'
          : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl text-slate-500'
      )}>
        <DollarSign className="h-6 w-6 mx-auto mb-2 opacity-30" />
        <p className="text-sm">No {config.label} pricing available for selected models.</p>
      </div>
    )
  }

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight
        ? 'bg-white/80 border-stone-200/80'
        : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
    )}>
      <div className="overflow-auto">
        <table className="w-full">
          <thead>
            <tr className={cn(
              'border-b',
              isLight ? 'border-stone-200 bg-stone-50' : 'border-white/[0.06] bg-[#1a1b1e]'
            )}>
              <th className={cn(
                'px-4 py-2.5 text-left text-xs font-semibold w-32 sticky left-0 z-10',
                isLight ? 'text-stone-700 bg-stone-50' : 'text-slate-300 bg-[#1a1b1e]'
              )}>
                Model
              </th>
              <th className={cn(
                'px-3 py-2.5 text-center text-xs font-semibold min-w-[100px]',
                isLight ? 'text-stone-700' : 'text-slate-300'
              )}>
                Input (1M)
              </th>
              <th className={cn(
                'px-3 py-2.5 text-center text-xs font-semibold min-w-[100px]',
                isLight ? 'text-stone-700' : 'text-slate-300'
              )}>
                Output (1M)
              </th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.map((d, idx) => {
              const isInputCheapest = d.inputPrice !== null && d.inputPrice === minInput && validInputPrices.length > 1
              const isOutputCheapest = d.outputPrice !== null && d.outputPrice === minOutput && validOutputPrices.length > 1

              return (
                <tr
                  key={d.model.model_id}
                  className={cn(
                    'border-b last:border-b-0',
                    isLight ? 'border-stone-100' : 'border-white/[0.04]'
                  )}
                >
                  <td className={cn(
                    'px-4 py-3 sticky left-0 z-10',
                    isLight ? 'bg-white' : 'bg-[#1a1b1e]'
                  )}>
                    <Badge className={cn(
                      'text-[9px] mb-0.5',
                      isLight ? 'text-[#faf9f5]' : 'text-white',
                      providerColors[d.model.model_provider] || providerColors.default
                    )}>
                      {d.model.model_provider}
                    </Badge>
                    <p className={cn(
                      'text-xs font-medium line-clamp-1',
                      isLight ? 'text-stone-700' : 'text-slate-300'
                    )}>
                      {d.model.model_name || d.model.model_id}
                    </p>
                  </td>
                  <td className="px-3 py-3 text-center">
                    {d.inputPrice !== null ? (
                      <div className="flex items-center justify-center gap-1">
                        <span className={cn(
                          'text-sm font-semibold tabular-nums',
                          isInputCheapest ? 'text-emerald-600' : isLight ? 'text-stone-900' : 'text-white'
                        )}>
                          {formatPrice(d.inputPrice)}
                        </span>
                        {isInputCheapest && <Trophy className="h-3 w-3 text-emerald-500" />}
                      </div>
                    ) : (
                      <span className={cn('text-xs', isLight ? 'text-stone-300' : 'text-slate-600')}>—</span>
                    )}
                  </td>
                  <td className="px-3 py-3 text-center">
                    {d.outputPrice !== null ? (
                      <div className="flex items-center justify-center gap-1">
                        <span className={cn(
                          'text-sm font-semibold tabular-nums',
                          isOutputCheapest ? 'text-emerald-600' : isLight ? 'text-stone-900' : 'text-white'
                        )}>
                          {formatPrice(d.outputPrice)}
                        </span>
                        {isOutputCheapest && <Trophy className="h-3 w-3 text-emerald-500" />}
                      </div>
                    ) : (
                      <span className={cn('text-xs', isLight ? 'text-stone-300' : 'text-slate-600')}>—</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function PricingTab({ selectedModels, getPricingForModel, isLight }) {
  const [consumptionType, setConsumptionType] = useState('on_demand')

  // Calculate pricing for each model
  const pricingData = selectedModels.map(({ model, region }) => {
    const pricing = getPricingForModel?.(model, region)
    return {
      model,
      region,
      pricing,
      inputPrice: pricing?.summary?.inputPrice,
      outputPrice: pricing?.summary?.outputPrice,
      imagePrice: pricing?.summary?.imagePrice,
      pricingType: pricing?.summary?.pricingType,
      unitLabel: pricing?.summary?.unitLabel,
    }
  })

  // Build per-model pricing maps keyed by row key
  const pricingByModel = pricingData.map(({ model, region, pricing }) => {
    const map = {}
    const fullPricing = pricing?.fullPricing
    // Try selected region first, then us-east-1, then any available region
    const regionData = fullPricing?.regions?.[region] || 
                       fullPricing?.regions?.['us-east-1'] || 
                       Object.values(fullPricing?.regions || {})[0]
    const pricingGroups = regionData?.pricing_groups || {}

    Object.entries(pricingGroups).forEach(([groupName, items]) => {
      if (!items || items.length === 0) return
      items.forEach(item => {
        const key = getRowKey(groupName, item)
        const priceInfo = getItemPrice(item)
        map[key] = priceInfo
      })
    })

    return map
  })

  // Collect all unique pricing groups and rows across all models
  const allGroups = new Map() // groupName -> Set of row keys
  const rowMeta = {} // key -> { label, unit, dimensions }

  pricingData.forEach(({ region, pricing }) => {
    const fullPricing = pricing?.fullPricing
    const regionData = fullPricing?.regions?.[region] || 
                       fullPricing?.regions?.['us-east-1'] ||
                       Object.values(fullPricing?.regions || {})[0]
    const pricingGroups = regionData?.pricing_groups || {}

    Object.entries(pricingGroups).forEach(([groupName, items]) => {
      if (!items || items.length === 0) return
      if (!allGroups.has(groupName)) allGroups.set(groupName, new Set())
      const group = allGroups.get(groupName)

      items.forEach(item => {
        const key = getRowKey(groupName, item)
        group.add(key)
        if (!rowMeta[key]) {
          const priceInfo = getItemPrice(item)
          rowMeta[key] = {
            key,
            label: simplifyDescription(item),
            unit: priceInfo.unit,
            dimensions: item.dimensions || null,
          }
        }
      })
    })
  })

  // Filter groups by selected consumption type
  const selectedConfig = consumptionTypes[consumptionType]
  const filteredGroups = [...allGroups.entries()]
    .filter(([groupName]) => selectedConfig?.groups.includes(groupName))
    .sort((a, b) => {
      const ai = selectedConfig.groups.indexOf(a[0])
      const bi = selectedConfig.groups.indexOf(b[0])
      return ai - bi
    })

  // Find best on-demand prices for summary
  const validInputPrices = pricingData.filter(d => d.inputPrice !== null && d.inputPrice !== undefined)
  const validOutputPrices = pricingData.filter(d => d.outputPrice !== null && d.outputPrice !== undefined)

  const minInputPrice = validInputPrices.length > 0 ? Math.min(...validInputPrices.map(d => d.inputPrice)) : null
  const minOutputPrice = validOutputPrices.length > 0 ? Math.min(...validOutputPrices.map(d => d.outputPrice)) : null

  // Find ALL models that are cheapest (handles ties)
  const cheapestInputModels = minInputPrice !== null ? pricingData.filter(d => d.inputPrice === minInputPrice) : []
  const cheapestOutputModels = minOutputPrice !== null ? pricingData.filter(d => d.outputPrice === minOutputPrice) : []

  // Check which consumption types have data
  const availableTypes = Object.entries(consumptionTypes).filter(([key, config]) => {
    return config.groups.some(groupName => allGroups.has(groupName))
  }).map(([key]) => key)

  return (
    <div className="mt-4 space-y-3">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-emerald-50/50 border-emerald-200' : 'bg-emerald-500/10 border-emerald-500/30'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <TrendingDown className="h-3.5 w-3.5 text-emerald-500" />
            <span className={cn('text-[10px]', isLight ? 'text-emerald-700' : 'text-emerald-400')}>Best Input</span>
          </div>
          <p className="text-lg font-bold text-emerald-600">
            {minInputPrice !== null ? formatPrice(minInputPrice) : '—'}
          </p>
          {cheapestInputModels.length > 0 && (
            <p className={cn('text-[10px] truncate', isLight ? 'text-stone-400' : 'text-slate-500')}>
              {cheapestInputModels.map(d => d.model.model_name || d.model.model_id).join(', ')}
            </p>
          )}
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-emerald-50/50 border-emerald-200' : 'bg-emerald-500/10 border-emerald-500/30'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <TrendingDown className="h-3.5 w-3.5 text-emerald-500" />
            <span className={cn('text-[10px]', isLight ? 'text-emerald-700' : 'text-emerald-400')}>Best Output</span>
          </div>
          <p className="text-lg font-bold text-emerald-600">
            {minOutputPrice !== null ? formatPrice(minOutputPrice) : '—'}
          </p>
          {cheapestOutputModels.length > 0 && (
            <p className={cn('text-[10px] truncate', isLight ? 'text-stone-400' : 'text-slate-500')}>
              {cheapestOutputModels.map(d => d.model.model_name || d.model.model_id).join(', ')}
            </p>
          )}
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <DollarSign className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>Pricing Groups</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {allGroups.size}
          </p>
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <Info className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>With Pricing</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {pricingData.filter(d => d.inputPrice != null || d.outputPrice != null || d.imagePrice != null).length}/{pricingData.length}
          </p>
        </div>
      </div>

      {/* Consumption Type Filter */}
      <div className={cn(
        'flex flex-wrap items-center gap-2 px-3 py-2.5 rounded-lg border',
        isLight
          ? 'bg-stone-50/80 border-stone-200/60'
          : 'bg-white/[0.02] border-white/[0.06]'
      )}>
        <span className={cn('text-xs font-medium mr-1', isLight ? 'text-stone-600' : 'text-slate-400')}>
          Consumption:
        </span>
        {Object.entries(consumptionTypes).map(([key, config]) => {
          const Icon = config.icon
          const isAvailable = availableTypes.includes(key)
          const isSelected = consumptionType === key
          
          return (
            <Button
              key={key}
              variant="ghost"
              size="sm"
              disabled={!isAvailable}
              className={cn(
                'h-7 px-2.5 text-xs rounded-md gap-1.5',
                isSelected
                  ? isLight
                    ? 'bg-amber-100 text-amber-800 hover:bg-amber-100'
                    : 'bg-[#1A9E7A]/20 text-[#1A9E7A] hover:bg-[#1A9E7A]/20'
                  : isAvailable
                    ? isLight
                      ? 'text-stone-600 hover:text-stone-800 hover:bg-stone-100'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.06]'
                    : isLight
                      ? 'text-stone-300 cursor-not-allowed'
                      : 'text-slate-600 cursor-not-allowed'
              )}
              onClick={() => isAvailable && setConsumptionType(key)}
            >
              <Icon className="h-3 w-3" />
              {config.label}
            </Button>
          )
        })}
      </div>

      {/* Quick Comparison Table */}
      <QuickComparisonTable
        selectedModels={selectedModels}
        pricingByModel={pricingByModel}
        consumptionType={consumptionType}
        isLight={isLight}
      />

      {/* Region info */}
      <div className={cn(
        'flex items-center gap-2 px-3 py-2 rounded-lg text-xs',
        isLight
          ? 'bg-stone-50 text-stone-500 border border-stone-200/60'
          : 'bg-white/[0.02] text-slate-500 border border-white/[0.06]'
      )}>
        <Info className="h-3.5 w-3.5 flex-shrink-0" />
        Prices shown for us-east-1. Actual pricing may vary by region.
      </div>

      {/* Detailed Pricing groups for selected consumption type */}
      {filteredGroups.map(([groupName, rowKeys]) => {
        const rows = [...rowKeys].map(key => rowMeta[key]).filter(Boolean)
        // Sort rows: Input before Output before others
        rows.sort((a, b) => {
          const order = ['Input', 'Output', 'Cache Read', 'Cache Write']
          const ai = order.indexOf(a.label)
          const bi = order.indexOf(b.label)
          if (ai !== -1 && bi !== -1) return ai - bi
          if (ai !== -1) return -1
          if (bi !== -1) return 1
          return a.label.localeCompare(b.label)
        })

        return (
          <PricingGroupSection
            key={groupName}
            groupName={groupName}
            rows={rows}
            models={selectedModels}
            pricingByModel={pricingByModel}
            isLight={isLight}
          />
        )
      })}

      {/* No pricing data fallback */}
      {allGroups.size === 0 && (
        <div className={cn(
          'text-center py-12 rounded-lg border',
          isLight
            ? 'bg-white/80 border-stone-200/80 text-stone-500'
            : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl text-slate-500'
        )}>
          <DollarSign className="h-8 w-8 mx-auto mb-2 opacity-30" />
          <p className="text-sm">No pricing data available for the selected models and regions.</p>
        </div>
      )}
    </div>
  )
}
