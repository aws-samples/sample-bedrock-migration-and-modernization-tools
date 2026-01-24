import { Check, X, MessageSquare, Image, FileText, Video, Mic, Trophy } from 'lucide-react'
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

// Use providerColorClasses from constants
const providerColors = providerColorClasses

function MetricRow({ label, values, isLight, bestIndex = null, lowerIsBetter = false }) {
  return (
    <div className={cn(
      'grid items-center py-3 border-b',
      isLight ? 'border-stone-200/60' : 'border-slate-700/50'
    )} style={{ gridTemplateColumns: `minmax(120px, 150px) repeat(${values.length}, minmax(80px, 1fr))` }}>
      <div className={cn(
        'font-medium text-xs sm:text-sm pr-2',
        isLight ? 'text-stone-700' : 'text-slate-300'
      )}>
        {label}
      </div>
      {values.map((value, idx) => {
        const isBest = bestIndex === idx
        return (
          <div
            key={idx}
            className={cn(
              'text-center text-xs sm:text-sm font-medium',
              isBest
                ? isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
                : isLight ? 'text-stone-600' : 'text-slate-400'
            )}
          >
            <div className="flex items-center justify-center gap-1 sm:gap-1.5">
              {value}
              {isBest && (
                <Trophy className={cn(
                  'h-3 w-3 sm:h-3.5 sm:w-3.5',
                  isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
                )} />
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function BooleanRow({ label, values, isLight }) {
  return (
    <div className={cn(
      'grid items-center py-3 border-b',
      isLight ? 'border-stone-200/60' : 'border-slate-700/50'
    )} style={{ gridTemplateColumns: `minmax(120px, 150px) repeat(${values.length}, minmax(80px, 1fr))` }}>
      <div className={cn(
        'font-medium text-xs sm:text-sm pr-2',
        isLight ? 'text-stone-700' : 'text-slate-300'
      )}>
        {label}
      </div>
      {values.map((value, idx) => (
        <div key={idx} className="flex justify-center">
          {value ? (
            <div className="flex items-center gap-1">
              <Check className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-emerald-500" />
              <span className="text-xs text-emerald-500 hidden sm:inline">Yes</span>
            </div>
          ) : (
            <div className="flex items-center gap-1">
              <X className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-red-400/60" />
              <span className={cn('text-xs hidden sm:inline', isLight ? 'text-stone-400' : 'text-slate-500')}>No</span>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function ModalitiesRow({ label, values, isLight }) {
  return (
    <div className={cn(
      'grid items-center py-3 border-b',
      isLight ? 'border-stone-200/60' : 'border-slate-700/50'
    )} style={{ gridTemplateColumns: `minmax(120px, 150px) repeat(${values.length}, minmax(80px, 1fr))` }}>
      <div className={cn(
        'font-medium text-xs sm:text-sm pr-2',
        isLight ? 'text-stone-700' : 'text-slate-300'
      )}>
        {label}
      </div>
      {values.map((modalities, idx) => (
        <div key={idx} className="flex justify-center gap-0.5 sm:gap-1 flex-wrap">
          {modalities.map(mod => {
            const Icon = modalityIcons[mod] || MessageSquare
            return (
              <div
                key={mod}
                className={cn(
                  'p-0.5 sm:p-1 rounded',
                  isLight ? 'bg-stone-100' : 'bg-white/5'
                )}
                title={mod}
              >
                <Icon className={cn('h-3 w-3 sm:h-3.5 sm:w-3.5', isLight ? 'text-stone-600' : 'text-slate-400')} />
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

export function OverviewTab({ selectedModels, getPricingForModel, isLight }) {
  // Extract data for comparison
  const modelData = selectedModels.map(({ model, region }) => {
    const pricing = getPricingForModel?.(model, region)
    const contextWindow = model.converse_data?.context_window || 0
    const maxOutput = model.converse_data?.max_output_tokens || 0
    const inputModalities = model.model_modalities?.input_modalities || []
    const outputModalities = model.model_modalities?.output_modalities || []
    const regions = model.regions_available || []
    const isActive = model.model_lifecycle?.status === 'ACTIVE' || model.model_status === 'ACTIVE'
    const streamingSupported = model.streaming_supported || false
    const crisSupported = model.cross_region_inference?.supported || false
    const batchSupported = (model.consumption_options || []).includes('batch')

    return {
      model,
      region,
      contextWindow,
      maxOutput,
      inputModalities,
      outputModalities,
      regions,
      isActive,
      streamingSupported,
      crisSupported,
      batchSupported,
      inputPrice: pricing?.summary?.inputPrice,
      outputPrice: pricing?.summary?.outputPrice,
    }
  })

  // Find best values
  const maxContext = Math.max(...modelData.map(d => d.contextWindow))
  const maxOutputTokens = Math.max(...modelData.map(d => d.maxOutput))
  const maxRegions = Math.max(...modelData.map(d => d.regions.length))

  const validInputPrices = modelData.filter(d => d.inputPrice !== null && d.inputPrice !== undefined)
  const minInputPrice = validInputPrices.length > 0
    ? Math.min(...validInputPrices.map(d => d.inputPrice))
    : null

  const validOutputPrices = modelData.filter(d => d.outputPrice !== null && d.outputPrice !== undefined)
  const minOutputPrice = validOutputPrices.length > 0
    ? Math.min(...validOutputPrices.map(d => d.outputPrice))
    : null

  const contextBestIdx = modelData.findIndex(d => d.contextWindow === maxContext)
  const outputBestIdx = modelData.findIndex(d => d.maxOutput === maxOutputTokens)
  const regionsBestIdx = modelData.findIndex(d => d.regions.length === maxRegions)
  const inputPriceBestIdx = minInputPrice !== null
    ? modelData.findIndex(d => d.inputPrice === minInputPrice)
    : -1
  const outputPriceBestIdx = minOutputPrice !== null
    ? modelData.findIndex(d => d.outputPrice === minOutputPrice)
    : -1

  return (
    <div className="overflow-x-auto -mx-3 sm:mx-0">
      <div className={cn(
        'rounded-lg border p-3 sm:p-4 mt-4 min-w-[400px]',
        isLight
          ? 'bg-white/80 border-stone-200/80 backdrop-blur-xl'
          : 'bg-[#161d26]/80 border-slate-700/50 backdrop-blur-xl'
      )}>
        {/* Model Headers */}
        <div className={cn(
          'grid items-center py-3 border-b-2',
          isLight ? 'border-stone-300' : 'border-slate-600'
        )} style={{ gridTemplateColumns: `minmax(120px, 150px) repeat(${selectedModels.length}, minmax(80px, 1fr))` }}>
          <div className={cn(
            'font-semibold text-xs sm:text-sm',
            isLight ? 'text-stone-900' : 'text-white'
          )}>
            Model
          </div>
          {modelData.map(({ model }, idx) => (
            <div key={idx} className="text-center px-1 sm:px-2">
              <Badge className={cn(
                'text-[9px] sm:text-[10px] mb-1',
                isLight ? 'text-[#faf9f5]' : 'text-white',
                providerColors[model.model_provider] || providerColors.default
              )}>
                {model.model_provider}
              </Badge>
              <p className={cn(
                'text-xs sm:text-sm font-semibold line-clamp-2',
                isLight ? 'text-stone-900' : 'text-white'
              )}>
                {model.model_name || model.model_id}
              </p>
            </div>
          ))}
        </div>

      {/* Metrics */}
      <MetricRow
        label="Context Window"
        values={modelData.map(d => formatNumber(d.contextWindow))}
        isLight={isLight}
        bestIndex={contextBestIdx}
      />

      <MetricRow
        label="Max Output Tokens"
        values={modelData.map(d => formatNumber(d.maxOutput))}
        isLight={isLight}
        bestIndex={outputBestIdx}
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
      />

      <BooleanRow
        label="Streaming"
        values={modelData.map(d => d.streamingSupported)}
        isLight={isLight}
      />

      <BooleanRow
        label="Cross-Region Inference"
        values={modelData.map(d => d.crisSupported)}
        isLight={isLight}
      />

      <BooleanRow
        label="Batch Inference"
        values={modelData.map(d => d.batchSupported)}
        isLight={isLight}
      />

      <MetricRow
        label="Available Regions"
        values={modelData.map(d => `${d.regions.length} regions`)}
        isLight={isLight}
        bestIndex={regionsBestIdx}
      />

      <MetricRow
        label="Input Price (per 1K)"
        values={modelData.map(d => formatPrice(d.inputPrice))}
        isLight={isLight}
        bestIndex={inputPriceBestIdx}
        lowerIsBetter
      />

      <MetricRow
        label="Output Price (per 1K)"
        values={modelData.map(d => formatPrice(d.outputPrice))}
        isLight={isLight}
        bestIndex={outputPriceBestIdx}
        lowerIsBetter
      />

      <BooleanRow
        label="Active Status"
        values={modelData.map(d => d.isActive)}
        isLight={isLight}
      />
      </div>
    </div>
  )
}
