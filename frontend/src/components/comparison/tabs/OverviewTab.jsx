import { useMemo, useState } from 'react'
import { Check, X, MessageSquare, Image, FileText, Video, Mic, Trophy, Info, DollarSign, Globe } from 'lucide-react'
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
  return `$${price < 0.0001 ? price.toFixed(6) : price.toFixed(4)}`
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
  const quotas = model.model_service_quotas || {}
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

// Radar scoring (0-10 scale, benchmarked against global or relative set)
function computeRadarScores(modelData, benchmarks) {
  const { maxContext, maxRegions, maxFeatures, maxCost } = benchmarks
  
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
    
    // Features: relative to the max features count
    let featureCount = 0
    if (d.streamingSupported) featureCount++
    if (d.batchSupported) featureCount++
    if (d.crisSupported) featureCount++
    if (d.mantleSupported) featureCount++
    const featureScore = maxFeatures > 0
      ? Math.min(10 * (featureCount / maxFeatures), 10)
      : 0
    
    return {
      name: d.model.model_name || d.model.model_id,
      costScore: Math.round(costScore * 10) / 10,
      contextScore: Math.round(contextScore * 10) / 10,
      regionScore: Math.round(regionScore * 10) / 10,
      featureScore: Math.round(featureScore * 10) / 10,
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

// Helper to get all regions for a model (on-demand + CRIS + Mantle)
function getAllModelRegions(model) {
  const onDemand = model.in_region || []
  const cris = model.cross_region_inference?.source_regions || []
  const mantle = model.mantle_inference?.mantle_regions || []
  return [...new Set([...onDemand, ...cris, ...mantle])]
}

export function OverviewTab({ selectedModels, getPricingForModel, allModels, isLight }) {
  const [scoringMode, setScoringMode] = useState('global') // 'global' or 'relative'
  const [showScoringInfo, setShowScoringInfo] = useState(false)

  const modelData = selectedModels.map(({ model, region }) => {
    const pricing = getPricingForModel?.(model, region)
    const contextWindow = model.converse_data?.context_window || 0
    const maxOutput = model.converse_data?.max_output_tokens || 0
    const inputModalities = model.model_modalities?.input_modalities || []
    const outputModalities = model.model_modalities?.output_modalities || []
    const regions = getAllModelRegions(model)
    const isActive = model.model_lifecycle?.status === 'ACTIVE' || model.model_status === 'ACTIVE'
    const streamingSupported = model.streaming_supported || false
    const crisSupported = model.cross_region_inference?.supported || false
    const batchSupported = (model.consumption_options || []).includes('batch')
    const mantleSupported = model.mantle_inference?.supported || model.is_mantle || false
    const hasLongContext = detectLongContext(pricing, region)
    const extendedContext = getExtendedContextWindow(model)
    const effectiveContextWindow = Math.max(contextWindow, extendedContext || 0)
    const useCasesCount = (model.model_use_cases || []).length
    const capabilitiesCount = (model.model_capabilities || []).length

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
      inputPrice: pricing?.summary?.inputPrice,
      outputPrice: pricing?.summary?.outputPrice,
    }
  })

  // Global benchmarks from ALL models in the catalog
  const globalBenchmarks = useMemo(() => {
    if (!allModels || allModels.length === 0) return null
    
    let maxContext = 0
    let maxRegions = 0
    let maxFeatures = 0
    
    allModels.forEach(m => {
      // Context window
      const ctx = m.converse_data?.context_window || 0
      const extCtx = getExtendedContextWindow(m)
      const effectiveCtx = Math.max(ctx, extCtx || 0)
      if (effectiveCtx > maxContext) maxContext = effectiveCtx
      
      // Regions (total: on-demand + CRIS + Mantle)
      const regionCount = getAllModelRegions(m).length
      if (regionCount > maxRegions) maxRegions = regionCount
      
      // Features count
      let features = 0
      if (m.streaming_supported) features++
      if ((m.consumption_options || []).includes('batch')) features++
      if (m.cross_region_inference?.supported) features++
      if (m.mantle_inference?.supported || m.is_mantle) features++
      if (features > maxFeatures) maxFeatures = features
    })
    
    // Max cost from the comparison set (we don't have pricing for all models easily)
    const allCosts = modelData
      .map(d => ((d.inputPrice || 0) + (d.outputPrice || 0)))
      .filter(c => c > 0)
    const maxCost = allCosts.length > 0 ? Math.max(...allCosts) : 1
    
    return { maxContext, maxRegions, maxFeatures: Math.max(maxFeatures, 1), maxCost }
  }, [allModels, modelData])

  // Relative benchmarks from only the compared models
  const relativeBenchmarks = useMemo(() => {
    const allCosts = modelData.map(d => ((d.inputPrice || 0) + (d.outputPrice || 0))).filter(c => c > 0)
    const maxCost = allCosts.length > 0 ? Math.max(...allCosts) : 1
    const maxContext = Math.max(...modelData.map(d => d.effectiveContextWindow || d.contextWindow), 1)
    const maxRegions = Math.max(...modelData.map(d => d.regions.length), 1)
    let maxFeatures = 0
    modelData.forEach(d => {
      let f = 0
      if (d.streamingSupported) f++
      if (d.batchSupported) f++
      if (d.crisSupported) f++
      if (d.mantleSupported) f++
      if (f > maxFeatures) maxFeatures = f
    })
    return { maxContext, maxRegions, maxFeatures: Math.max(maxFeatures, 1), maxCost }
  }, [modelData])

  const activeBenchmarks = scoringMode === 'global' && globalBenchmarks ? globalBenchmarks : relativeBenchmarks

  // Radar chart data
  const radarScores = useMemo(() => computeRadarScores(modelData, activeBenchmarks), [modelData, activeBenchmarks])

  const radarChartData = useMemo(() => {
    const axes = ['Context Window', 'Cost Efficiency', 'Availability', 'Features']
    const scoreKeys = ['contextScore', 'costScore', 'regionScore', 'featureScore']
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
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <Trophy className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>Best Score</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {radarScores.length > 0 ? Math.max(...radarScores.map(s => s.costScore + s.contextScore + s.regionScore + s.featureScore)).toFixed(1) : '—'}
          </p>
          {radarScores.length > 0 && (() => {
            const best = radarScores.reduce((a, b) => {
              const aTotal = a.costScore + a.contextScore + a.regionScore + a.featureScore
              const bTotal = b.costScore + b.contextScore + b.regionScore + b.featureScore
              return bTotal > aTotal ? b : a
            })
            return <p className={cn('text-[10px] truncate', isLight ? 'text-stone-400' : 'text-slate-500')}>{best.name}</p>
          })()}
        </div>

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
            <span className={cn('text-[10px]', isLight ? 'text-emerald-700' : 'text-emerald-400')}>Cheapest Input</span>
          </div>
          <p className="text-lg font-bold text-emerald-600">
            {minInputPrice !== null ? `$${minInputPrice < 0.01 ? minInputPrice.toFixed(4) : minInputPrice.toFixed(3)}` : '—'}
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

      {/* Radar Chart */}
      {modelData.length >= 2 && (
        <div className={cn(
          'rounded-lg border overflow-hidden',
          isLight
            ? 'bg-white/70 border-stone-200/60 backdrop-blur-xl'
            : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
        )}>
          <div className={cn(
            'px-4 py-2.5 border-b flex items-center justify-between',
            isLight ? 'bg-stone-50/60 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
          )}>
            <h3 className={cn('font-semibold text-xs', isLight ? 'text-stone-900' : 'text-white')}>
              Model Comparison Radar
            </h3>
            <div className={cn(
              'inline-flex rounded-md border overflow-hidden h-6',
              isLight ? 'border-stone-300' : 'border-[#373a40]'
            )}>
              <button
                onClick={() => setScoringMode('global')}
                className={cn(
                  'px-2 py-0.5 text-[10px] font-medium transition-colors',
                  scoringMode === 'global'
                    ? isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
                    : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-[#1a1b1e] text-[#9a9b9f] hover:bg-[#2c2d32]'
                )}
              >
                Global
              </button>
              <button
                onClick={() => setScoringMode('relative')}
                className={cn(
                  'px-2 py-0.5 text-[10px] font-medium transition-colors border-l',
                  isLight ? 'border-stone-300' : 'border-[#373a40]',
                  scoringMode === 'relative'
                    ? isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
                    : isLight ? 'bg-transparent text-stone-500 hover:bg-stone-50' : 'bg-[#1a1b1e] text-[#9a9b9f] hover:bg-[#2c2d32]'
                )}
              >
                Relative
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3">
            <div className="lg:col-span-2" style={{ height: 380 }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarChartData} cx="50%" cy="50%" outerRadius="70%">
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

            {/* Score summary per model */}
            <div className="px-4 py-3 lg:pr-6 overflow-y-auto" style={{ maxHeight: 380 }}>
              <div className="space-y-1.5">
                {radarScores.map((scores, idx) => {
                  const total = scores.costScore + scores.contextScore + scores.regionScore + scores.featureScore
                  return (
                    <div key={scores.name} className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{ backgroundColor: radarColors[idx % radarColors.length] }}
                        />
                        <span className={cn(
                          'text-[11px] truncate max-w-[120px]',
                          isLight ? 'text-stone-700' : 'text-slate-300'
                        )}>
                          {scores.name}
                        </span>
                      </div>
                      <span className={cn(
                        'text-[11px] font-bold tabular-nums',
                        isLight ? 'text-stone-900' : 'text-white'
                      )}>
                        {total.toFixed(1)}/40
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Scoring Info */}
      <div className="flex items-center">
        <button
          onClick={() => setShowScoringInfo(!showScoringInfo)}
          className={cn(
            'flex items-center gap-1 text-xs rounded-md px-2 py-1 transition-colors',
            isLight ? 'text-stone-400 hover:text-stone-600 hover:bg-stone-100' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
          )}
        >
          <Info className="h-3.5 w-3.5" />
          <span>Scoring (0–10 scale)</span>
        </button>
      </div>

      {/* Scoring Info Panel (collapsible) */}
      {showScoringInfo && (
        <div className={cn(
          'rounded-lg border p-4 text-xs space-y-2',
          isLight ? 'bg-stone-50 border-stone-200 text-stone-600' : 'bg-white/[0.02] border-white/[0.06] text-slate-400'
        )}>
          <p className={cn('font-semibold text-sm mb-2', isLight ? 'text-stone-700' : 'text-slate-300')}>
            How Scoring Works
          </p>
          <p>Each model is scored on 4 dimensions (0–10 scale each, max total 40):</p>
          <div className="space-y-1.5 mt-2">
            <div><span className="font-medium">Cost Efficiency</span> — Lower price per token = higher score. Cheapest model scores 10, most expensive scores 0. Models without pricing data score 0.</div>
            <div><span className="font-medium">Context Window</span> — Larger context = higher score. The model with the largest context window scores 10, others proportionally.</div>
            <div><span className="font-medium">Availability</span> — More regions = higher score. The model available in the most regions scores 10, others proportionally.</div>
            <div><span className="font-medium">Features</span> — More features = higher score. Counts: Streaming, Batch, Cross-Region Inference (CRIS), and Mantle support.</div>
          </div>
          <div className={cn('mt-3 pt-2 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
            <p><span className="font-medium">Global mode</span> — Scores are relative to ALL {allModels?.length || 0} models in the catalog. Consistent across comparisons.</p>
            <p><span className="font-medium">Relative mode</span> — Scores are relative to only the models being compared. Best in the group always scores 10.</p>
          </div>
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
              <MetricRow
                label="Input Price (1K)"
                values={modelData.map(d => formatPrice(d.inputPrice))}
                isLight={isLight}
                bestIndices={inputPriceBestSet}
              />
              <MetricRow
                label="Output Price (1K)"
                values={modelData.map(d => formatPrice(d.outputPrice))}
                isLight={isLight}
                bestIndices={outputPriceBestSet}
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
