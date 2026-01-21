import { DollarSign, Trophy, TrendingDown, Info, Image } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

// Provider colors
const providerColors = {
  Amazon: 'bg-[#FF9900]',
  Anthropic: 'bg-[#D4A27F]',
  Meta: 'bg-[#0082FB]',
  Mistral: 'bg-[#F54E42]',
  Cohere: 'bg-[#39594D]',
  'AI21 Labs': 'bg-[#6C5CE7]',
  AI21: 'bg-[#6C5CE7]',
  'Stability AI': 'bg-[#7C5CFF]',
  Stability: 'bg-[#7C5CFF]',
  Luma: 'bg-[#6366F1]',
  default: 'bg-slate-500',
}

function formatPrice(price) {
  if (price === null || price === undefined) return 'N/A'
  return `$${price < 0.0001 ? price.toFixed(6) : price.toFixed(4)}`
}

function formatImagePrice(price) {
  if (price === null || price === undefined) return 'N/A'
  return `$${price < 0.01 ? price.toFixed(4) : price.toFixed(2)}`
}

function PricingCard({ model, region, pricing, isLight, isCheapestInput, isCheapestOutput, isCheapestImage }) {
  const inputPrice = pricing?.summary?.inputPrice
  const outputPrice = pricing?.summary?.outputPrice
  const pricingType = pricing?.summary?.pricingType
  const imagePrice = pricing?.summary?.imagePrice
  const imagePrices = pricing?.summary?.imagePrices
  const unitLabel = pricing?.summary?.unitLabel

  // Get full pricing details if available
  const fullPricing = pricing?.fullPricing
  const regionData = fullPricing?.regions?.[region] || fullPricing?.regions?.['us-east-1']
  const pricingGroups = regionData?.pricing_groups || {}

  return (
    <div className={cn(
      'rounded-lg border p-4',
      isLight
        ? 'bg-white/80 border-stone-200/80'
        : 'bg-[#161d26]/80 border-slate-700/50'
    )}>
      {/* Model Header */}
      <div className="mb-4">
        <Badge className={cn(
          'text-[10px] mb-1',
          isLight ? 'text-[#faf9f5]' : 'text-white',
          providerColors[model.model_provider] || providerColors.default
        )}>
          {model.model_provider}
        </Badge>
        <h4 className={cn(
          'font-semibold text-sm line-clamp-2',
          isLight ? 'text-stone-900' : 'text-white'
        )}>
          {model.model_name || model.model_id}
        </h4>
        <p className={cn(
          'text-xs',
          isLight ? 'text-stone-500' : 'text-slate-500'
        )}>
          Region: {region}
        </p>
      </div>

      {/* Main Pricing */}
      <div className="space-y-3">
        {pricingType === 'image_generation' ? (
          /* Image Generation Pricing */
          <>
            <div className={cn(
              'rounded-lg p-3',
              isCheapestImage
                ? isLight ? 'bg-emerald-50 border border-emerald-200' : 'bg-emerald-500/10 border border-emerald-500/30'
                : isLight ? 'bg-stone-50' : 'bg-white/5'
            )}>
              <div className="flex items-center justify-between mb-1">
                <span className={cn('text-xs flex items-center gap-1', isLight ? 'text-stone-500' : 'text-slate-500')}>
                  <Image className="h-3 w-3" />
                  Per Image
                </span>
                {isCheapestImage && (
                  <Badge className={cn(
                    'text-[9px] px-1.5 py-0',
                    isLight ? 'bg-emerald-100 text-emerald-700' : 'bg-emerald-500/20 text-emerald-400'
                  )}>
                    <Trophy className="h-2.5 w-2.5 mr-0.5" />
                    Lowest
                  </Badge>
                )}
              </div>
              <p className={cn(
                'text-lg font-bold',
                isCheapestImage
                  ? 'text-emerald-600'
                  : isLight ? 'text-stone-900' : 'text-white'
              )}>
                {formatImagePrice(imagePrice)}
              </p>
            </div>
            {/* Image pricing tiers */}
            {imagePrices && Object.keys(imagePrices).length > 1 && (
              <div className={cn(
                'rounded-lg p-3',
                isLight ? 'bg-stone-50' : 'bg-white/5'
              )}>
                <p className={cn(
                  'text-xs font-medium mb-2',
                  isLight ? 'text-stone-700' : 'text-slate-300'
                )}>
                  Pricing Tiers
                </p>
                <div className="space-y-1.5">
                  {Object.entries(imagePrices).slice(0, 6).map(([key, data]) => (
                    <div key={key} className="flex justify-between items-center text-xs">
                      <span className={cn(
                        'truncate mr-2',
                        isLight ? 'text-stone-600' : 'text-slate-400'
                      )}>
                        {data.type === 'text_to_image' ? 'T2I' : data.type === 'image_to_image' ? 'I2I' : ''} {data.resolution}px {data.tier}
                      </span>
                      <span className={cn(
                        'font-medium whitespace-nowrap',
                        isLight ? 'text-stone-900' : 'text-white'
                      )}>
                        {formatImagePrice(data.price)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          /* Token-based Pricing (default) */
          <>
            {/* Input Price */}
            <div className={cn(
              'rounded-lg p-3',
              isCheapestInput
                ? isLight ? 'bg-emerald-50 border border-emerald-200' : 'bg-emerald-500/10 border border-emerald-500/30'
                : isLight ? 'bg-stone-50' : 'bg-white/5'
            )}>
              <div className="flex items-center justify-between mb-1">
                <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-500')}>
                  Input ({unitLabel || 'per 1K tokens'})
                </span>
                {isCheapestInput && (
                  <Badge className={cn(
                    'text-[9px] px-1.5 py-0',
                    isLight ? 'bg-emerald-100 text-emerald-700' : 'bg-emerald-500/20 text-emerald-400'
                  )}>
                    <Trophy className="h-2.5 w-2.5 mr-0.5" />
                    Lowest
                  </Badge>
                )}
              </div>
              <p className={cn(
                'text-lg font-bold',
                isCheapestInput
                  ? 'text-emerald-600'
                  : isLight ? 'text-stone-900' : 'text-white'
              )}>
                {formatPrice(inputPrice)}
              </p>
            </div>

            {/* Output Price */}
            <div className={cn(
              'rounded-lg p-3',
              isCheapestOutput
                ? isLight ? 'bg-emerald-50 border border-emerald-200' : 'bg-emerald-500/10 border border-emerald-500/30'
                : isLight ? 'bg-stone-50' : 'bg-white/5'
            )}>
              <div className="flex items-center justify-between mb-1">
                <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-500')}>
                  Output ({unitLabel || 'per 1K tokens'})
                </span>
                {isCheapestOutput && (
                  <Badge className={cn(
                    'text-[9px] px-1.5 py-0',
                    isLight ? 'bg-emerald-100 text-emerald-700' : 'bg-emerald-500/20 text-emerald-400'
                  )}>
                    <Trophy className="h-2.5 w-2.5 mr-0.5" />
                    Lowest
                  </Badge>
                )}
              </div>
              <p className={cn(
                'text-lg font-bold',
                isCheapestOutput
                  ? 'text-emerald-600'
                  : isLight ? 'text-stone-900' : 'text-white'
              )}>
                {formatPrice(outputPrice)}
              </p>
            </div>
          </>
        )}

        {/* Additional Pricing Groups */}
        {Object.entries(pricingGroups).map(([groupName, items]) => {
          if (groupName === 'On-Demand') return null // Already shown above
          if (!items || items.length === 0) return null

          return (
            <div
              key={groupName}
              className={cn(
                'rounded-lg p-3',
                isLight ? 'bg-stone-50' : 'bg-white/5'
              )}
            >
              <p className={cn(
                'text-xs font-medium mb-2',
                isLight ? 'text-stone-700' : 'text-slate-300'
              )}>
                {groupName}
              </p>
              <div className="space-y-1.5">
                {items.slice(0, 4).map((item, idx) => (
                  <div key={idx} className="flex justify-between items-center text-xs">
                    <span className={cn(
                      'truncate mr-2',
                      isLight ? 'text-stone-600' : 'text-slate-400'
                    )}>
                      {item.description || item.dimension}
                    </span>
                    <span className={cn(
                      'font-medium whitespace-nowrap',
                      isLight ? 'text-stone-900' : 'text-white'
                    )}>
                      ${item.price_per_thousand?.toFixed(4) || item.price || 'N/A'}
                    </span>
                  </div>
                ))}
                {items.length > 4 && (
                  <p className={cn(
                    'text-[10px]',
                    isLight ? 'text-stone-500' : 'text-slate-500'
                  )}>
                    +{items.length - 4} more items
                  </p>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function PricingTab({ selectedModels, getPricingForModel, isLight }) {
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
    }
  })

  // Find cheapest prices for token-based models
  const validInputPrices = pricingData.filter(d => d.inputPrice !== null && d.inputPrice !== undefined)
  const validOutputPrices = pricingData.filter(d => d.outputPrice !== null && d.outputPrice !== undefined)
  const validImagePrices = pricingData.filter(d => d.imagePrice !== null && d.imagePrice !== undefined)

  const minInputPrice = validInputPrices.length > 0
    ? Math.min(...validInputPrices.map(d => d.inputPrice))
    : null

  const minOutputPrice = validOutputPrices.length > 0
    ? Math.min(...validOutputPrices.map(d => d.outputPrice))
    : null

  const minImagePrice = validImagePrices.length > 0
    ? Math.min(...validImagePrices.map(d => d.imagePrice))
    : null

  return (
    <div className="mt-4 space-y-4">
      {/* Info banner */}
      <div className={cn(
        'flex items-center gap-2 p-3 rounded-lg text-sm',
        isLight
          ? 'bg-amber-50 text-amber-800 border border-amber-200'
          : 'bg-[#1A9E7A]/10 text-[#1A9E7A] border border-[#1A9E7A]/30'
      )}>
        <Info className="h-4 w-4 flex-shrink-0" />
        <p>
          Prices shown are based on each model's selected region. Change the region in the model card above to see different pricing.
        </p>
      </div>

      {/* Cost comparison summary */}
      {(validInputPrices.length > 1 || validImagePrices.length > 1) && (
        <div className={cn(
          'p-4 rounded-lg border',
          isLight
            ? 'bg-emerald-50/50 border-emerald-200'
            : 'bg-emerald-500/5 border-emerald-500/20'
        )}>
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className={cn(
              'h-4 w-4',
              isLight ? 'text-emerald-600' : 'text-emerald-400'
            )} />
            <span className={cn(
              'font-medium text-sm',
              isLight ? 'text-emerald-800' : 'text-emerald-300'
            )}>
              Cost Optimization Tip
            </span>
          </div>
          <p className={cn(
            'text-sm',
            isLight ? 'text-emerald-700' : 'text-emerald-400/80'
          )}>
            {minInputPrice !== null && (
              <>
                Lowest input cost: <strong>{formatPrice(minInputPrice)}</strong>/1K tokens
                {(minOutputPrice !== null || minImagePrice !== null) && ' | '}
              </>
            )}
            {minOutputPrice !== null && (
              <>
                Lowest output cost: <strong>{formatPrice(minOutputPrice)}</strong>/1K tokens
                {minImagePrice !== null && ' | '}
              </>
            )}
            {minImagePrice !== null && (
              <>
                Lowest image cost: <strong>{formatImagePrice(minImagePrice)}</strong>/image
              </>
            )}
          </p>
        </div>
      )}

      {/* Pricing Cards Grid */}
      <div className={cn(
        'grid gap-4',
        selectedModels.length === 1 && 'grid-cols-1 max-w-md',
        selectedModels.length === 2 && 'grid-cols-2',
        selectedModels.length === 3 && 'grid-cols-3',
        selectedModels.length === 4 && 'grid-cols-2 lg:grid-cols-4',
        selectedModels.length === 5 && 'grid-cols-2 lg:grid-cols-5'
      )}>
        {pricingData.map(({ model, region, pricing, inputPrice, outputPrice, imagePrice }, idx) => (
          <PricingCard
            key={model.model_id}
            model={model}
            region={region}
            pricing={pricing}
            isLight={isLight}
            isCheapestInput={inputPrice === minInputPrice && minInputPrice !== null}
            isCheapestOutput={outputPrice === minOutputPrice && minOutputPrice !== null}
            isCheapestImage={imagePrice === minImagePrice && minImagePrice !== null}
          />
        ))}
      </div>
    </div>
  )
}
