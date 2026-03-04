import { useState, useMemo } from 'react'
import { Star, GitCompare, ExternalLink, Globe, MessageSquare, Image, FileText, Video, Mic, Check, X, MapPin, Radio, ArrowRight, CheckCircle2, Copy, Search, Clock, Cpu } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useComparisonStore } from '@/stores/comparisonStore'
import { providerColors, consumptionLabels, getContextSizeCategory } from '@/config/constants'
import { trackEvent } from '@/services/analytics'
import { getCrisGeoScopes } from '@/utils/filters'

function InfoTooltip({ children, content, side = "bottom", sideOffset = 4 }) {
  return (
    <Tooltip delayDuration={100}>
      <TooltipTrigger asChild>
        {children}
      </TooltipTrigger>
      <TooltipContent side={side} sideOffset={sideOffset} className="max-w-[220px]">
        <p>{content}</p>
      </TooltipContent>
    </Tooltip>
  )
}

// Modality icons and labels
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

function formatNumber(num) {
  if (typeof num !== 'number' || isNaN(num)) return 'N/A'
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(0)}K`
  return num.toString()
}

function getProviderColor(provider) {
  return providerColors[provider] || providerColors.default
}

// Returns '#ffffff' or '#000000' based on background luminance for readable contrast
function getContrastColor(hexColor) {
  if (!hexColor) return '#ffffff'
  const hex = hexColor.replace('#', '')
  const r = parseInt(hex.substring(0, 2), 16)
  const g = parseInt(hex.substring(2, 4), 16)
  const b = parseInt(hex.substring(4, 6), 16)
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  return luminance > 0.75 ? '#000000' : '#ffffff'
}

function extractPricing(model, preferredRegion = 'us-east-1') {
  const pricing = model.pricing || model.model_pricing || model.comprehensive_pricing || {}

  let inputPrice = null
  let outputPrice = null

  if (pricing.by_region) {
    const region = pricing.by_region[preferredRegion] || pricing.by_region['us-east-1'] || pricing.by_region['us-west-2'] || Object.values(pricing.by_region)[0]
    if (region) {
      if (region.text) {
        inputPrice = region.text.input_per_1k_tokens
        outputPrice = region.text.output_per_1k_tokens
      } else if (region.input_per_1k_tokens !== undefined) {
        inputPrice = region.input_per_1k_tokens
        outputPrice = region.output_per_1k_tokens
      } else if (region.on_demand) {
        const inputTokens = region.on_demand.input_tokens?.[0]
        const outputTokens = region.on_demand.output_tokens?.[0]
        if (inputTokens?.price) inputPrice = parseFloat(inputTokens.price)
        if (outputTokens?.price) outputPrice = parseFloat(outputTokens.price)
      }
    }
  }

  return { inputPrice, outputPrice }
}

// Visual progress bar for specs
function SpecBar({ label, value, maxValue, isLight }) {
  const percentage = value && maxValue ? Math.min((value / maxValue) * 100, 100) : 0
  const displayValue = formatNumber(value)

  return (
    <div className="mb-2 last:mb-0">
      <div className="flex justify-between items-center text-xs mb-1">
        <span className={cn(isLight ? 'text-stone-500' : 'text-slate-400')}>{label}</span>
        <span className={cn('font-semibold', isLight ? 'text-stone-700' : 'text-slate-200')}>
          {displayValue}
        </span>
      </div>
      <div className={cn(
        'h-1.5 rounded-full overflow-hidden',
        isLight ? 'bg-stone-200' : 'bg-white/[0.06]'
      )}>
        <div
          className={cn(
            'h-full rounded-full transition-all duration-500 ease-out',
            isLight ? 'bg-gradient-to-r from-amber-400 to-amber-500' : 'bg-gradient-to-r from-[#158567] to-[#1A9E7A]'
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

// Helper to get display model ID (strips version suffix like :0)
function getDisplayModelId(modelId) {
  if (!modelId) return ''
  const colonIndex = modelId.lastIndexOf(':')
  // Only strip if what's after the colon looks like a version number
  if (colonIndex > 0) {
    const suffix = modelId.slice(colonIndex + 1)
    if (/^\d+$/.test(suffix)) {
      return modelId.slice(0, colonIndex)
    }
  }
  return modelId
}

// Copyable model ID
function CopyableModelId({ modelId, isLight }) {
  const [copied, setCopied] = useState(false)
  const displayId = getDisplayModelId(modelId)

  const handleCopy = async (e) => {
    e.stopPropagation()
    // Copy the display ID (without version suffix)
    await navigator.clipboard.writeText(displayId)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
      <button
        onClick={handleCopy}
        title={copied ? "Copied!" : "Click to copy model ID"}
        className={cn(
          'flex items-center gap-1 text-[11px] font-mono truncate max-w-full transition-colors group/copy',
          isLight
            ? 'text-stone-400 hover:text-stone-600'
            : 'text-slate-400 hover:text-[#c0c1c5]'
        )}
      >
        <span className="truncate">{displayId}</span>
        {copied ? (
          <Check className="h-3 w-3 flex-shrink-0 text-emerald-500" />
        ) : (
          <Copy className="h-3 w-3 flex-shrink-0 opacity-0 group-hover/copy:opacity-100 transition-opacity" />
        )}
      </button>
  )
}

// Helper to get status styles for a given status value
function getStatusStylesForValue(statusValue, isLight) {
  const normalizedStatus = (statusValue || 'ACTIVE').toUpperCase()
  switch (normalizedStatus) {
    case 'ACTIVE':
      return isLight
        ? 'bg-emerald-100 text-emerald-700 border border-emerald-200'
        : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
    case 'LEGACY':
      return isLight
        ? 'bg-amber-100 text-amber-700 border border-amber-200'
        : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
    case 'EOL':
      return isLight
        ? 'bg-red-100 text-red-700 border border-red-200'
        : 'bg-red-500/20 text-red-400 border border-red-500/30'
    case 'MIXED':
      return isLight
        ? 'bg-purple-100 text-purple-700 border border-purple-200'
        : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
    default:
      return isLight
        ? 'bg-stone-100 text-stone-700 border border-stone-200'
        : 'bg-white/10 text-slate-400 border border-white/20'
  }
}

// Helper to get status label
function getStatusLabelForValue(statusValue) {
  const normalizedStatus = (statusValue || 'ACTIVE').toUpperCase()
  switch (normalizedStatus) {
    case 'ACTIVE': return 'Active'
    case 'LEGACY': return 'Legacy'
    case 'EOL': return 'EOL'
    case 'MIXED': return 'Mixed'
    default: return normalizedStatus
  }
}

// Status pill component - handles both single status and MIXED status with multiple badges
function StatusPill({ status, globalStatus, statusSummary, isLight }) {
  // If global_status is MIXED and we have status_summary, render multiple badges
  if (globalStatus === 'MIXED' && statusSummary) {
    // Get statuses that have regions, in order: LEGACY, ACTIVE, EOL
    const statusOrder = ['LEGACY', 'ACTIVE', 'EOL']
    const activeStatuses = statusOrder.filter(s => statusSummary[s]?.length > 0)
    
    if (activeStatuses.length > 0) {
      return (
        <div className="flex items-center gap-1">
          {activeStatuses.map(s => (
            <div
              key={s}
              className={cn(
                'px-1.5 py-0.5 rounded-full text-[9px] font-semibold uppercase tracking-wide',
                getStatusStylesForValue(s, isLight)
              )}
            >
              {getStatusLabelForValue(s)}
            </div>
          ))}
        </div>
      )
    }
  }
  
  // Single status display (backward compatible)
  const normalizedStatus = (status || 'ACTIVE').toUpperCase()
  
  return (
    <div className={cn(
      'px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide',
      getStatusStylesForValue(normalizedStatus, isLight)
    )}>
      {getStatusLabelForValue(normalizedStatus)}
    </div>
  )
}

// Feature indicator
function FeatureIndicator({ supported, icon: Icon, label, isLight }) {
  return (
    <InfoTooltip content={label}>
      <div className={cn(
        'flex items-center gap-0.5 cursor-default',
        supported
          ? isLight ? 'text-emerald-600' : 'text-emerald-400'
          : isLight ? 'text-stone-300' : 'text-slate-600'
      )}>
        <Icon className="h-3.5 w-3.5" />
        {supported ? (
          <Check className="h-2.5 w-2.5" />
        ) : (
          <X className="h-2.5 w-2.5" />
        )}
      </div>
    </InfoTooltip>
  )
}

// Modality descriptions for tooltips
const modalityDescriptions = {
  TEXT: 'Text content',
  IMAGE: 'Image content',
  DOCUMENT: 'Document files',
  VIDEO: 'Video content',
  AUDIO: 'Audio content',
  SPEECH: 'Speech/voice',
}

// Consumption option descriptions
const consumptionDescriptions = {
  on_demand: 'In Region',
  provisioned: 'Reserved capacity',
  batch: 'Batch processing',
  cross_region_inference: 'Cross-region routing',
}

export function ModelCard({ model, onViewDetails, onCompare, onToggleFavorite, isFavorite = false, preferredRegion = 'us-east-1', getPricingForModel }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  // Comparison store
  const { toggleModel, isModelSelected } = useComparisonStore()
  const isSelectedForComparison = isModelSelected(model.model_id)

  const contextWindow = model.specs?.context_window ?? model.converse_data?.context_window
  const extendedContext = model.specs?.extended_context ?? model.converse_data?.extended_context
  const hasExtendedContext = model.specs?.extended_context != null || model.converse_data?.has_extended_context
  const maxOutput = model.specs?.max_output ?? model.specs?.max_output_tokens ?? model.converse_data?.max_output_tokens
  const inputModalities = model.modalities?.input_modalities || model.model_modalities?.input_modalities || []
  const outputModalities = model.modalities?.output_modalities || model.model_modalities?.output_modalities || []
  const capabilities = model.capabilities || model.model_capabilities || []
  const regions = model.availability?.on_demand?.regions || model.in_region || []
  const lifecycleStatus = model.lifecycle?.status || model.model_lifecycle?.status || model.model_status || 'ACTIVE'

  // Get pricing from new pricing data source, fallback to old method
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
  const pricingSummary = pricingResult?.summary || extractPricing(model, preferredRegion)
  const { inputPrice, outputPrice, pricingType, unitLabel, imagePrice, imagePrices, videoPrice, videoPrices, hasMantlePricing, availableDimensions } = pricingSummary

  const crisSupported = model.availability?.cross_region?.supported ?? model.cross_region_inference?.supported ?? false
  const crisGeoScopes = getCrisGeoScopes(model)
  const mantleSupported = model.availability?.mantle?.supported ?? false
  const streamingSupported = model.streaming ?? model.streaming_supported ?? false
  const consumptionOptions = useMemo(() => {
    const opts = [...(model.consumption_options || [])]
    // Ensure provisioned_throughput is shown if model supports it
    const provisionedSupported = model.availability?.provisioned?.supported ?? model.provisioned_throughput?.supported
    if (provisionedSupported && !opts.includes('provisioned_throughput') && !opts.includes('provisioned')) {
      opts.push('provisioned_throughput')
    }
    // Ensure mantle is shown if model supports it
    const mantleInferenceSupported = model.availability?.mantle?.supported
    if (mantleInferenceSupported && !opts.includes('mantle')) {
      opts.push('mantle')
    }
    // Ensure cross_region_inference is shown if model supports it
    const crossRegionSupported = model.availability?.cross_region?.supported ?? model.cross_region_inference?.supported
    if (crossRegionSupported && !opts.includes('cross_region_inference')) {
      opts.push('cross_region_inference')
    }
    return opts
  }, [model])
  const providerColor = getProviderColor(model.model_provider)

  return (
    <TooltipProvider>
      <Card className={cn(
        'group relative flex flex-col h-full overflow-hidden',
        isSelectedForComparison
          ? isLight
            ? 'ring-2 ring-amber-500 border-amber-400'
            : 'ring-2 ring-[#1A9E7A] border-[#1A9E7A]/50'
          : isLight
            ? 'hover:border-stone-300 hover:shadow-lg'
            : 'hover:border-white/[0.12] hover:shadow-xl hover:shadow-black/20'
      )}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 pb-2">
          <div className="flex items-center gap-1.5">
            <Badge
              className="text-[10px] font-semibold"
              style={{ backgroundColor: providerColor, color: getContrastColor(providerColor) }}
            >
              {model.model_provider}
            </Badge>
            {model.availability?.mantle?.only && (
              <span className={cn(
                'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold',
                isLight
                  ? 'bg-violet-100 text-violet-700 border border-violet-200'
                  : 'bg-violet-500/15 text-violet-400 border border-violet-500/30'
              )}>
                Mantle Only
              </span>
            )}
            {!model.availability?.mantle?.only && (model.availability?.mantle?.supported) && (
              <span className={cn(
                'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold',
                isLight
                  ? 'bg-violet-100 text-violet-700 border border-violet-200'
                  : 'bg-violet-500/15 text-violet-400 border border-violet-500/30'
              )}>
                Mantle
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <StatusPill 
              status={lifecycleStatus} 
              globalStatus={model.lifecycle?.global_status || model.model_lifecycle?.global_status}
              statusSummary={model.lifecycle?.status_summary || model.model_lifecycle?.status_summary}
              isLight={isLight} 
            />
            <button
              className={cn(
                'p-1 rounded transition-colors',
                isLight ? 'hover:bg-stone-100' : 'hover:bg-white/[0.06]'
              )}
              onClick={() => {
                trackEvent('favorite_toggle', { modelId: model.model_id, provider: model.model_provider, modelName: model.model_name, section: 'explorer' })
                onToggleFavorite?.(model.model_id)
              }}
            >
              <Star
                className={cn(
                  'h-4 w-4 transition-colors',
                  isFavorite
                    ? 'fill-amber-400 text-amber-400'
                    : isLight ? 'text-stone-300 hover:text-stone-400' : 'text-slate-500 hover:text-slate-400'
                )}
              />
            </button>
          </div>
        </div>

        {/* Title */}
        <div className="px-4 pb-3">
          <h3 className={cn(
            'font-semibold text-[15px] leading-tight line-clamp-2 mb-1',
            isLight ? 'text-stone-900' : 'text-slate-200'
          )}>
            {model.model_name || model.model_id}
          </h3>
          <CopyableModelId modelId={model.model_id} isLight={isLight} />
        </div>

        <CardContent className="flex-1 flex flex-col gap-3 pt-0">
          {/* Context/Output boxed display */}
            <div className={cn(
              'rounded-lg p-2.5',
              isLight
                ? 'bg-gradient-to-r from-amber-50/80 to-orange-50/60 border border-amber-100/50'
                : 'bg-gradient-to-r from-white/5 to-white/[0.02] border border-white/10'
            )}>
              <div className="flex items-center justify-between">
                <div className="flex-1 text-center border-r border-current/10">
                  <p className={cn('text-[10px] uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>Context</p>
                  <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
                    {hasExtendedContext ? (
                      <Tooltip delayDuration={200}>
                        <TooltipTrigger asChild>
                          <span className="cursor-default">
                            {formatNumber(contextWindow)}
                            <span className={cn('ml-1', isLight ? 'text-amber-500' : 'text-emerald-400')}>
                              | {formatNumber(extendedContext)}
                            </span>
                          </span>
                        </TooltipTrigger>
                        <TooltipContent side="bottom" className="max-w-[220px] text-xs">
                          <p><strong>{formatNumber(contextWindow)}</strong> base context window</p>
                          <p><strong>{formatNumber(extendedContext)}</strong> extended context (beta)</p>
                        </TooltipContent>
                      </Tooltip>
                    ) : (
                      formatNumber(contextWindow)
                    )}
                  </p>
                </div>
                <div className="flex-1 text-center">
                  <p className={cn('text-[10px] uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>Output</p>
                  <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
                    {formatNumber(maxOutput)}
                  </p>
                </div>
              </div>
            </div>

          {/* Modalities & Features Row */}
          <div className="flex items-center justify-between">
            {/* Modalities */}
            <div className="flex items-center gap-1">
              {inputModalities.slice(0, 3).map(mod => {
                const Icon = modalityIcons[mod] || MessageSquare
                return (
                  <InfoTooltip key={`in-${mod}`} content={`Input: ${modalityDescriptions[mod]}`}>
                    <div className={cn(
                      'p-1.5 rounded cursor-default',
                      isLight ? 'bg-stone-100' : 'bg-white/[0.06]'
                    )}>
                      <Icon className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-300')} />
                    </div>
                  </InfoTooltip>
                )
              })}
              {inputModalities.length > 0 && outputModalities.length > 0 && (
                <ArrowRight className={cn('h-3 w-3 mx-0.5', isLight ? 'text-stone-300' : 'text-slate-600')} />
              )}
              {outputModalities.slice(0, 2).map(mod => {
                const Icon = modalityIcons[mod] || MessageSquare
                return (
                  <InfoTooltip key={`out-${mod}`} content={`Output: ${modalityDescriptions[mod]}`}>
                    <div className={cn(
                      'p-1.5 rounded cursor-default',
                      isLight ? 'bg-emerald-50' : 'bg-emerald-500/10'
                    )}>
                      <Icon className={cn('h-3.5 w-3.5', isLight ? 'text-emerald-600' : 'text-emerald-400')} />
                    </div>
                  </InfoTooltip>
                )
              })}
            </div>

            {/* Features */}
            <div className="flex items-center gap-2">
              <FeatureIndicator
                supported={streamingSupported}
                icon={Radio}
                label={streamingSupported ? "Streaming supported" : "No streaming"}
                isLight={isLight}
              />
              <InfoTooltip
                content={crisSupported && crisGeoScopes.length > 0 ? (
                  <div className="space-y-1.5">
                    <p className="font-medium">Cross-region inference</p>
                    <div className="flex flex-wrap gap-1">
                      {crisGeoScopes.map(scope => (
                        <span key={scope} className="text-[10px] font-semibold px-1.5 py-0.5 bg-white/20 rounded">
                          {scope}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : crisSupported ? "Cross-region inference" : "No cross-region"}
              >
                <div className={cn(
                  'flex items-center gap-0.5 cursor-default',
                  crisSupported
                    ? isLight ? 'text-emerald-600' : 'text-emerald-400'
                    : isLight ? 'text-stone-300' : 'text-slate-600'
                )}>
                  <Globe className="h-3.5 w-3.5" />
                  {crisSupported ? (
                    <Check className="h-2.5 w-2.5" />
                  ) : (
                    <X className="h-2.5 w-2.5" />
                  )}
                </div>
              </InfoTooltip>
              <FeatureIndicator
                supported={mantleSupported}
                icon={Cpu}
                label={mantleSupported ? "Mantle inference" : "No Mantle"}
                isLight={isLight}
              />
              {(() => {
                const onDemandRegions = model.availability?.on_demand?.regions || model.in_region || []
                const crisRegions = model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions ?? []
                const mantleRegions = model.availability?.mantle?.regions || []
                const allRegions = new Set([...onDemandRegions, ...crisRegions, ...mantleRegions])
                const totalRegionCount = allRegions.size
                return (
                  <InfoTooltip content={`Available in ${totalRegionCount} regions`}>
                    <div className={cn(
                      'flex items-center gap-1 text-xs cursor-default',
                      isLight ? 'text-stone-500' : 'text-slate-300'
                    )}>
                      <MapPin className="h-3 w-3" />
                      <span className="font-medium">{totalRegionCount}</span>
                    </div>
                  </InfoTooltip>
                )
              })()}
            </div>
          </div>

          {/* Pricing - boxed style */}
            <div>
              {pricingType === 'video_generation' || pricingType === 'video_second' ? (
                <div className={cn(
                  'text-xs rounded-md p-2',
                  isLight ? 'bg-stone-100/60' : 'bg-white/5'
                )}>
                  <div className="text-center">
                    <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      <Video className="h-3 w-3 inline mr-1" />
                      Per {pricingType === 'video_second' ? 'Second' : 'Video'}
                    </p>
                    <p className={cn('font-semibold text-lg', isLight ? 'text-stone-800' : 'text-[#f0f1f3]')}>
                      ${videoPrice !== null ? (videoPrice < 0.01 ? videoPrice.toFixed(4) : videoPrice.toFixed(2)) : 'N/A'}
                    </p>
                  </div>
                </div>
              ) : pricingType === 'image_generation' ? (
                <div className={cn(
                  'text-xs rounded-md p-2',
                  isLight ? 'bg-stone-100/60' : 'bg-white/5'
                )}>
                  <div className="text-center">
                    <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      <Image className="h-3 w-3 inline mr-1" />
                      Per Image
                    </p>
                    <p className={cn('font-semibold text-lg', isLight ? 'text-stone-800' : 'text-[#f0f1f3]')}>
                      ${imagePrice !== null ? (imagePrice < 0.01 ? imagePrice.toFixed(4) : imagePrice.toFixed(2)) : 'N/A'}
                    </p>
                  </div>
                </div>
              ) : pricingType === 'search_unit' ? (
                <div className={cn(
                  'text-xs rounded-md p-2',
                  isLight ? 'bg-stone-100/60' : 'bg-white/5'
                )}>
                  <div className="text-center">
                    <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      <Search className="h-3 w-3 inline mr-1" />
                      Per 1K Units
                    </p>
                    <p className={cn('font-semibold text-lg', isLight ? 'text-stone-800' : 'text-[#f0f1f3]')}>
                      ${inputPrice !== null ? (inputPrice < 0.01 ? inputPrice.toFixed(4) : inputPrice.toFixed(2)) : 'N/A'}
                    </p>
                  </div>
                </div>
              ) : inputPrice !== null ? (
                <div className={cn(
                  'grid grid-cols-2 gap-2 text-xs rounded-md p-2',
                  isLight ? 'bg-stone-100/60' : 'bg-white/5'
                )}>
                  <div className="text-center">
                    <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>Input</p>
                    <p className={cn('font-semibold', isLight ? 'text-stone-800' : 'text-[#f0f1f3]')}>
                      ${inputPrice < 0.01 ? inputPrice.toFixed(4) : inputPrice.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>Output</p>
                    <p className={cn('font-semibold', isLight ? 'text-stone-800' : 'text-[#f0f1f3]')}>
                      ${outputPrice !== null ? (outputPrice < 0.01 ? outputPrice.toFixed(4) : outputPrice.toFixed(2)) : 'N/A'}
                    </p>
                  </div>
                  <div className="col-span-2 text-center -mt-1 flex items-center justify-center gap-1">
                    <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-400')}>
                      {unitLabel || 'per 1M tokens'}
                    </span>
                    {hasMantlePricing && (
                      <span 
                        className={cn(
                          'text-[9px] px-1.5 py-0.5 rounded',
                          isLight ? 'bg-purple-100 text-purple-700' : 'bg-purple-500/15 text-purple-400'
                        )}
                        title="Mantle (OpenAI-compatible) pricing available"
                      >
                        + Mantle
                      </span>
                    )}
                  </div>
                </div>
              ) : (
                <div className={cn(
                  'text-center text-xs py-2 rounded-md',
                  isLight ? 'bg-stone-100/60 text-stone-500' : 'bg-white/5 text-[#b0b1b5]'
                )}>
                  Pricing unavailable
                </div>
              )}
            </div>

          {/* Capabilities */}
          {capabilities.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {capabilities.slice(0, 3).map(cap => (
                <span
                  key={cap}
                  className={cn(
                    'text-[10px] px-1.5 py-0.5 rounded',
                    isLight ? 'bg-stone-100 text-stone-500' : 'bg-white/[0.06] text-slate-400'
                  )}
                >
                  {cap}
                </span>
              ))}
              {capabilities.length > 3 && (
                <span className={cn(
                  'text-[10px] px-1.5 py-0.5 rounded',
                  isLight ? 'bg-stone-100 text-stone-500' : 'bg-white/[0.06] text-slate-400'
                )}>
                  +{capabilities.length - 3}
                </span>
              )}
            </div>
          )}

          {/* Spacer */}
          <div className="flex-1 min-h-1" />

          {/* Action Buttons */}
          <div className={cn(
            'flex gap-2 pt-3 border-t',
            isLight ? 'border-stone-200' : 'border-white/[0.06]'
          )}>
            <Button
              variant="outline"
              size="sm"
              className="flex-1 text-xs"
              onClick={() => {
                trackEvent('model_detail_open', { modelId: model.model_id, provider: model.model_provider, modelName: model.model_name, section: 'explorer' })
                onViewDetails?.(model)
              }}
            >
              <ExternalLink className="h-3.5 w-3.5 mr-1.5" />
              Details
            </Button>
            <Button
              variant={isSelectedForComparison ? "default" : "outline"}
              size="sm"
              className={cn(
                "flex-1 text-xs",
                isSelectedForComparison && (isLight
                  ? "bg-amber-600 hover:bg-amber-700"
                  : "bg-[#1A9E7A] hover:bg-[#22b38d]")
              )}
              style={isSelectedForComparison ? { color: '#ffffff' } : undefined}
              onClick={() => {
                const wasSelected = isSelectedForComparison
                toggleModel(model, preferredRegion)
                trackEvent(wasSelected ? 'comparison_remove' : 'comparison_add', { modelId: model.model_id, provider: model.model_provider, modelName: model.model_name, section: 'explorer' })
              }}
              disabled={false}
            >
              {isSelectedForComparison ? (
                <>
                  <CheckCircle2 className="h-3.5 w-3.5 mr-1.5" />
                  Selected
                </>
              ) : (
                <>
                  <GitCompare className="h-3.5 w-3.5 mr-1.5" />
                  Compare
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </TooltipProvider>
  )
}
