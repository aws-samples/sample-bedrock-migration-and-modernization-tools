import { Star, GitCompare, ExternalLink, Zap, Globe, MessageSquare, Image, FileText, Video, Mic, Check, X, MapPin, Radio, ArrowRight, DollarSign, Gauge, CheckCircle2, Search, Clock } from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useComparisonStore } from '@/stores/comparisonStore'
import { providerColorClasses, consumptionLabels, getContextSizeCategory } from '@/config/constants'

// Tooltip wrapper for consistent styling
function InfoTooltip({ children, content, side = "bottom", sideOffset = 4 }) {
  return (
    <Tooltip delayDuration={200}>
      <TooltipTrigger asChild>
        {children}
      </TooltipTrigger>
      <TooltipContent side={side} sideOffset={sideOffset} className="max-w-[220px]">
        <p>{content}</p>
      </TooltipContent>
    </Tooltip>
  )
}

// Modality icons
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

function getProviderColor(provider) {
  return providerColorClasses[provider] || providerColorClasses.default
}

function extractPricing(model, preferredRegion = 'us-east-1') {
  const pricing = model.model_pricing || model.comprehensive_pricing || {}

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

// Use getContextSizeCategory from constants as getModelSize
const getModelSize = getContextSizeCategory

// Section divider component
function SectionDivider({ isLight }) {
  return (
    <div className={cn(
      'h-px w-full',
      isLight ? 'bg-stone-200/60' : 'bg-white/5'
    )} />
  )
}

export function ModelCard({ model, onViewDetails, onCompare, onToggleFavorite, isFavorite = false, preferredRegion = 'us-east-1', getPricingForModel }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  // Comparison store
  const { toggleModel, isModelSelected, canAddMore } = useComparisonStore()
  const isSelectedForComparison = isModelSelected(model.model_id)

  const contextWindow = model.converse_data?.context_window
  const maxOutput = model.converse_data?.max_output_tokens
  const inputModalities = model.model_modalities?.input_modalities || []
  const outputModalities = model.model_modalities?.output_modalities || []
  const capabilities = model.model_capabilities || []
  const regions = model.regions_available || []
  const isActive = model.model_lifecycle?.status === 'ACTIVE' || model.model_status === 'ACTIVE'

  // Get pricing from new pricing data source, fallback to old method
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
  const pricingSummary = pricingResult?.summary || extractPricing(model, preferredRegion)
  const { inputPrice, outputPrice, pricingType, unitLabel, imagePrice, imagePrices, videoPrice, videoPrices } = pricingSummary

  const crisSupported = model.cross_region_inference?.supported || false
  const streamingSupported = model.streaming_supported || false
  const consumptionOptions = model.consumption_options || []
  const modelSize = getModelSize(contextWindow)

  // Get unique modalities for compact display
  const allModalities = [...new Set([...inputModalities, ...outputModalities])]

  // Modality descriptions for tooltips
  const modalityDescriptions = {
    TEXT: 'Text content - written language input/output',
    IMAGE: 'Image content - visual media processing',
    DOCUMENT: 'Document files - PDFs, Word docs, etc.',
    VIDEO: 'Video content - moving visual media',
    AUDIO: 'Audio content - sound files and music',
    SPEECH: 'Speech content - voice and spoken language',
  }

  // Consumption option descriptions
  const consumptionDescriptions = {
    on_demand: 'Pay per use with no commitments',
    provisioned: 'Reserved capacity for consistent performance',
    batch: 'Process large volumes at lower cost',
    cross_region_inference: 'Route requests across regions',
  }

  return (
    <TooltipProvider>
      <Card className={cn(
        'group relative flex flex-col h-full transition-all duration-300',
        isSelectedForComparison
          ? isLight
            ? 'ring-2 ring-amber-500 border-amber-400 shadow-lg shadow-amber-500/20'
            : 'ring-2 ring-[#1A9E7A] border-[#1A9E7A]/50 shadow-lg shadow-[#1A9E7A]/20'
          : isLight
            ? 'hover:border-amber-300/80 hover:shadow-xl hover:shadow-amber-900/10 hover:ring-1 hover:ring-amber-200/50'
            : 'hover:border-[#1A9E7A]/50 hover:shadow-[0_0_30px_-5px_rgba(26,158,122,0.3)] hover:ring-1 hover:ring-[#1A9E7A]/20'
      )}>
        <CardHeader className="pb-2">
          {/* Top row: Provider badge, Size badge, Status, Favorite */}
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <InfoTooltip content="The company that created and maintains this model" side="right" sideOffset={8}>
                <Badge className={cn('text-xs font-medium cursor-help', isLight ? 'text-[#faf9f5]' : 'text-white', getProviderColor(model.model_provider))}>
                  {model.model_provider}
                </Badge>
              </InfoTooltip>
              <InfoTooltip content={`Model size based on context window: Small (<32K), Medium (<128K), Large (<500K), XL (500K+)`} side="right" sideOffset={8}>
                <Badge className={cn('text-xs font-medium cursor-help', isLight ? 'text-[#faf9f5]' : 'text-white', modelSize.color)}>
                  {modelSize.label}
                </Badge>
              </InfoTooltip>
            </div>
            <div className="flex items-center gap-1">
              <InfoTooltip content={isActive ? "Model is actively supported and recommended" : "Legacy model - consider newer alternatives"} side="left" sideOffset={8}>
                <Badge variant={isActive ? 'success' : 'warning'} className="text-xs px-2 py-0.5 cursor-help">
                  {isActive ? 'Active' : 'Legacy'}
                </Badge>
              </InfoTooltip>
              <InfoTooltip content="Add to favorites for quick access" side="left" sideOffset={8}>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => onToggleFavorite?.(model.model_id)}
                >
                  <Star
                    className={cn(
                      'h-4 w-4',
                      isFavorite ? 'fill-yellow-500 text-yellow-500' : 'text-slate-400'
                    )}
                  />
                </Button>
              </InfoTooltip>
            </div>
          </div>

          {/* Model name and ID */}
          <div className="mt-2">
            <h3 className={cn(
              'font-semibold text-base leading-tight line-clamp-2',
              isLight ? 'text-stone-900' : 'text-white'
            )}>
              {model.model_name || model.model_id}
            </h3>
            <InfoTooltip content="Unique identifier used in API calls">
              <p className={cn(
                'text-xs mt-1 truncate font-mono cursor-help',
                isLight ? 'text-stone-500' : 'text-slate-500'
              )}>
                {model.model_id}
              </p>
            </InfoTooltip>
          </div>
        </CardHeader>

        <CardContent className="flex-1 flex flex-col gap-2.5 pt-0">

          {/* ═══ CAPACITY SECTION ═══ */}
          <InfoTooltip content="Token capacity: Context is max input size, Output is max response length">
            <div className={cn(
              'rounded-lg p-2.5 cursor-help',
              isLight
                ? 'bg-gradient-to-r from-amber-50/80 to-orange-50/60 border border-amber-100/50'
                : 'bg-gradient-to-r from-white/5 to-white/[0.02] border border-white/10'
            )}>
              <div className="flex items-center justify-between">
                <div className="flex-1 text-center border-r border-current/10">
                  <p className={cn('text-[10px] uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-500')}>Context</p>
                  <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
                    {formatNumber(contextWindow)}
                  </p>
                </div>
                <div className="flex-1 text-center">
                  <p className={cn('text-[10px] uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-500')}>Output</p>
                  <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
                    {formatNumber(maxOutput)}
                  </p>
                </div>
              </div>
            </div>
          </InfoTooltip>

          {/* ═══ MODALITIES ═══ */}
          <div className="flex items-center gap-1.5">
            <div className="flex items-center gap-1">
              {inputModalities.map(mod => {
                const Icon = modalityIcons[mod] || MessageSquare
                return (
                  <InfoTooltip key={`in-${mod}`} content={`Input: ${modalityDescriptions[mod] || mod}`}>
                    <div className={cn(
                      'p-1 rounded cursor-help',
                      isLight ? 'bg-stone-100' : 'bg-white/5'
                    )}>
                      <Icon className={cn('h-3.5 w-3.5', isLight ? 'text-stone-600' : 'text-slate-400')} />
                    </div>
                  </InfoTooltip>
                )
              })}
            </div>
            <InfoTooltip content="Input types → Output types">
              <ArrowRight className={cn('h-3 w-3 flex-shrink-0 cursor-help', isLight ? 'text-stone-400' : 'text-slate-600')} />
            </InfoTooltip>
            <div className="flex items-center gap-1">
              {outputModalities.map(mod => {
                const Icon = modalityIcons[mod] || MessageSquare
                return (
                  <InfoTooltip key={`out-${mod}`} content={`Output: ${modalityDescriptions[mod] || mod}`}>
                    <div className={cn(
                      'p-1 rounded cursor-help',
                      isLight ? 'bg-emerald-100' : 'bg-emerald-500/10'
                    )}>
                      <Icon className={cn('h-3.5 w-3.5', isLight ? 'text-emerald-600' : 'text-emerald-400')} />
                    </div>
                  </InfoTooltip>
                )
              })}
            </div>
          </div>

          <SectionDivider isLight={isLight} />

          {/* ═══ FEATURES ROW ═══ */}
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-3">
              {/* Streaming */}
              <InfoTooltip content={streamingSupported
                ? "Streaming supported - receive responses in real-time as they're generated"
                : "Streaming not supported - full response delivered at once"}>
                <div className="flex items-center gap-1 cursor-help">
                  <Radio className={cn('h-3.5 w-3.5', streamingSupported ? 'text-emerald-500' : 'text-slate-400')} />
                  {streamingSupported ? (
                    <Check className="h-3 w-3 text-emerald-500" />
                  ) : (
                    <X className="h-3 w-3 text-red-400/60" />
                  )}
                </div>
              </InfoTooltip>
              {/* CRIS */}
              <InfoTooltip content={crisSupported
                ? "Cross-Region Inference - route requests to other regions for better availability"
                : "Cross-Region Inference not available for this model"}>
                <div className="flex items-center gap-1 cursor-help">
                  <Globe className={cn('h-3.5 w-3.5', crisSupported ? 'text-blue-500' : 'text-slate-400')} />
                  {crisSupported ? (
                    <Check className="h-3 w-3 text-emerald-500" />
                  ) : (
                    <X className="h-3 w-3 text-red-400/60" />
                  )}
                </div>
              </InfoTooltip>
            </div>
            {/* Regions count */}
            <InfoTooltip content={`Available in ${regions.length} AWS regions worldwide. Click Details for full list.`}>
              <div className="flex items-center gap-1 cursor-help">
                <MapPin className={cn('h-3.5 w-3.5', isLight ? 'text-blue-600' : 'text-blue-400')} />
                <span className={cn('font-medium', isLight ? 'text-stone-700' : 'text-slate-300')}>
                  {regions.length} regions
                </span>
              </div>
            </InfoTooltip>
          </div>

          <SectionDivider isLight={isLight} />

          {/* ═══ PRICING & DEPLOYMENT ═══ */}
          <div className="space-y-2">
            {/* Pricing - handles different pricing types */}
            {pricingType === 'video_generation' ? (
              /* Video Generation Pricing */
              <InfoTooltip content="Cost per generated video. Varies by resolution, duration, and fps." side="top">
                <div className="cursor-help">
                  {videoPrice !== null ? (
                    <div className={cn(
                      'text-xs rounded-md p-2',
                      isLight ? 'bg-stone-100/60' : 'bg-white/5'
                    )}>
                      <div className="text-center">
                        <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-500')}>
                          <Video className="h-3 w-3 inline mr-1" />
                          Per Video
                        </p>
                        <p className={cn('font-semibold text-lg', isLight ? 'text-stone-800' : 'text-white')}>
                          ${videoPrice < 0.01 ? videoPrice.toFixed(4) : videoPrice.toFixed(2)}
                        </p>
                      </div>
                      {videoPrices && Object.keys(videoPrices).length > 1 && (
                        <p className={cn('text-center text-[10px] mt-1', isLight ? 'text-stone-400' : 'text-slate-600')}>
                          {Object.keys(videoPrices).length} pricing tiers
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className={cn(
                      'text-center text-xs py-2 rounded-md',
                      isLight ? 'bg-stone-100/60 text-stone-500' : 'bg-white/5 text-slate-500'
                    )}>
                      Pricing unavailable
                    </div>
                  )}
                </div>
              </InfoTooltip>
            ) : pricingType === 'image_generation' ? (
              /* Image Generation Pricing */
              <InfoTooltip content="Cost per generated image. Varies by resolution and quality tier." side="top">
                <div className="cursor-help">
                  {imagePrice !== null ? (
                    <div className={cn(
                      'text-xs rounded-md p-2',
                      isLight ? 'bg-stone-100/60' : 'bg-white/5'
                    )}>
                      <div className="text-center">
                        <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-500')}>
                          <Image className="h-3 w-3 inline mr-1" />
                          Per Image
                        </p>
                        <p className={cn('font-semibold text-lg', isLight ? 'text-stone-800' : 'text-white')}>
                          ${imagePrice < 0.01 ? imagePrice.toFixed(4) : imagePrice.toFixed(2)}
                        </p>
                      </div>
                      {imagePrices && Object.keys(imagePrices).length > 1 && (
                        <p className={cn('text-center text-[10px] mt-1', isLight ? 'text-stone-400' : 'text-slate-600')}>
                          {Object.keys(imagePrices).length} pricing tiers
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className={cn(
                      'text-center text-xs py-2 rounded-md',
                      isLight ? 'bg-stone-100/60 text-stone-500' : 'bg-white/5 text-slate-500'
                    )}>
                      Pricing unavailable
                    </div>
                  )}
                </div>
              </InfoTooltip>
            ) : pricingType === 'search_unit' ? (
              /* Search Unit Pricing (Rerank models) */
              <InfoTooltip content="Cost per 1,000 search units. Used for reranking search results." side="top">
                <div className="cursor-help">
                  {inputPrice !== null ? (
                    <div className={cn(
                      'text-xs rounded-md p-2',
                      isLight ? 'bg-stone-100/60' : 'bg-white/5'
                    )}>
                      <div className="text-center">
                        <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-500')}>
                          <Search className="h-3 w-3 inline mr-1" />
                          Per 1K Search Units
                        </p>
                        <p className={cn('font-semibold text-lg', isLight ? 'text-stone-800' : 'text-white')}>
                          ${inputPrice < 0.01 ? inputPrice.toFixed(4) : inputPrice.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className={cn(
                      'text-center text-xs py-2 rounded-md',
                      isLight ? 'bg-stone-100/60 text-stone-500' : 'bg-white/5 text-slate-500'
                    )}>
                      Pricing unavailable
                    </div>
                  )}
                </div>
              </InfoTooltip>
            ) : pricingType === 'video_second' ? (
              /* Video Per-Second Pricing (Luma AI) */
              <InfoTooltip content="Cost per second of generated video." side="top">
                <div className="cursor-help">
                  {videoPrice !== null ? (
                    <div className={cn(
                      'text-xs rounded-md p-2',
                      isLight ? 'bg-stone-100/60' : 'bg-white/5'
                    )}>
                      <div className="text-center">
                        <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-500')}>
                          <Clock className="h-3 w-3 inline mr-1" />
                          Per Second
                        </p>
                        <p className={cn('font-semibold text-lg', isLight ? 'text-stone-800' : 'text-white')}>
                          ${videoPrice < 0.01 ? videoPrice.toFixed(4) : videoPrice.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className={cn(
                      'text-center text-xs py-2 rounded-md',
                      isLight ? 'bg-stone-100/60 text-stone-500' : 'bg-white/5 text-slate-500'
                    )}>
                      Pricing unavailable
                    </div>
                  )}
                </div>
              </InfoTooltip>
            ) : (
              /* Token-based Pricing (default) */
              <InfoTooltip content="Cost per 1,000 tokens. Input = what you send, Output = what model generates" side="top">
                <div className="cursor-help">
                  {inputPrice !== null ? (
                    <div className={cn(
                      'grid grid-cols-2 gap-2 text-xs rounded-md p-2',
                      isLight ? 'bg-stone-100/60' : 'bg-white/5'
                    )}>
                      <div className="text-center">
                        <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-500')}>Input</p>
                        <p className={cn('font-semibold', isLight ? 'text-stone-800' : 'text-white')}>
                          ${inputPrice < 0.00001 ? inputPrice.toFixed(7) : inputPrice < 0.001 ? inputPrice.toFixed(6) : inputPrice.toFixed(4)}
                        </p>
                      </div>
                      <div className="text-center">
                        <p className={cn('text-[10px] uppercase tracking-wide mb-0.5', isLight ? 'text-stone-500' : 'text-slate-500')}>Output</p>
                        <p className={cn('font-semibold', isLight ? 'text-stone-800' : 'text-white')}>
                          ${outputPrice !== null ? (outputPrice < 0.00001 ? outputPrice.toFixed(7) : outputPrice < 0.001 ? outputPrice.toFixed(6) : outputPrice.toFixed(4)) : 'N/A'}
                        </p>
                      </div>
                      <p className={cn('col-span-2 text-center text-[10px] -mt-1', isLight ? 'text-stone-400' : 'text-slate-600')}>
                        {unitLabel || 'per 1K tokens'}
                      </p>
                    </div>
                  ) : (
                    <div className={cn(
                      'text-center text-xs py-2 rounded-md',
                      isLight ? 'bg-stone-100/60 text-stone-500' : 'bg-white/5 text-slate-500'
                    )}>
                      Pricing unavailable
                    </div>
                  )}
                </div>
              </InfoTooltip>
            )}

            {/* Deployment Options - show all consumption options except cross_region_inference */}
            {consumptionOptions.filter(opt => opt !== 'cross_region_inference').length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap">
                {consumptionOptions.filter(opt => opt !== 'cross_region_inference').map(opt => (
                  <InfoTooltip key={opt} content={consumptionDescriptions[opt] || opt} side="top">
                    <Badge variant="info" className="text-[10px] py-0 px-1.5 font-medium cursor-help">
                      {consumptionLabels[opt] || opt}
                    </Badge>
                  </InfoTooltip>
                ))}
              </div>
            )}
          </div>

          {/* ═══ CAPABILITIES ═══ */}
          {capabilities.length > 0 && (
            <>
              <SectionDivider isLight={isLight} />
              <InfoTooltip content="Tasks and features this model excels at" side="top">
                <div className="flex flex-wrap gap-1 cursor-help">
                  {capabilities.slice(0, 4).map(cap => (
                    <Badge key={cap} variant="secondary" className="text-[10px] py-0 px-1.5">
                      {cap}
                    </Badge>
                  ))}
                  {capabilities.length > 4 && (
                    <Badge variant="secondary" className="text-[10px] py-0 px-1.5">
                      +{capabilities.length - 4}
                    </Badge>
                  )}
                </div>
              </InfoTooltip>
            </>
          )}

          {/* Spacer to push buttons to bottom */}
          <div className="flex-1 min-h-2" />

          {/* ═══ ACTION BUTTONS ═══ */}
          <div className={cn(
            'flex gap-2 pt-2 border-t',
            isLight ? 'border-stone-200/60' : 'border-white/5'
          )}>
            <Button
              variant="outline"
              size="sm"
              className="flex-1 text-xs"
              onClick={() => onViewDetails?.(model)}
            >
              <ExternalLink className="h-3.5 w-3.5 mr-1" />
              Details
            </Button>
            <Button
              variant={isSelectedForComparison ? "default" : "outline"}
              size="sm"
              className={cn(
                "text-xs",
                isSelectedForComparison && (isLight
                  ? "bg-amber-600 hover:bg-amber-700 text-white"
                  : "bg-[#1A9E7A] hover:bg-[#158567] text-white")
              )}
              onClick={() => toggleModel(model, preferredRegion)}
              disabled={!isSelectedForComparison && !canAddMore()}
            >
              {isSelectedForComparison ? (
                <>
                  <CheckCircle2 className="h-3.5 w-3.5 mr-1" />
                  Selected
                </>
              ) : (
                <>
                  <GitCompare className="h-3.5 w-3.5 mr-1" />
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
