import { Check, X, MessageSquare, Image, FileText, Video, Mic } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { providerColorClasses, consumptionLabels } from '@/config/constants'

const providerColors = providerColorClasses

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

// Detect long-context support from pricing data
function detectLongContext(pricing, region) {
  const fullPricing = pricing?.fullPricing
  const regionData = fullPricing?.regions?.[region] || fullPricing?.regions?.['us-east-1']
  const groups = regionData?.pricing_groups || {}
  return !!(groups['On-Demand Long Context'] && groups['On-Demand Long Context'].length > 0)
}

// Extract extended context window from quota names (e.g., "1M Context Length")
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

// Convert snake_case to Title Case
function prettifyLabel(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

const inferenceTypeLabels = {
  'ON_DEMAND': 'In Region',
  'INFERENCE_PROFILE': 'Inference Profile',
  'MANTLE_ONLY': 'Mantle Only',
}

export function TechSpecsTab({ selectedModels, getPricingForModel, isLight }) {
  const specsData = selectedModels.map(({ model, region }) => {
    const pricing = getPricingForModel?.(model, region)
    const baseContext = model.specs?.context_window ?? model.converse_data?.context_window
    const extendedContext = getExtendedContextWindow(model)
    const effectiveContext = Math.max(baseContext || 0, extendedContext || 0)
    const hasLongCtx = detectLongContext(pricing, region) || (extendedContext != null && extendedContext > (baseContext || 0))
    const isMantleOnly = model.availability?.mantle?.only
    const mantleRegions = model.availability?.mantle?.regions ?? []
    
    // Get CRIS data with proper fallbacks
    const crisData = model.availability?.cross_region ?? model.cross_region_inference ?? {}
    const crisSupported = isMantleOnly ? false : (
      crisData.supported ?? 
      (crisData.source_regions?.length > 0) ?? 
      (crisData.profiles?.length > 0) ??
      false
    )
    
    // Get batch data with proper fallbacks
    const batchData = model.availability?.batch ?? model.batch_inference_supported ?? {}
    const batchSupported = isMantleOnly ? false : (
      batchData.supported ?? 
      (batchData.supported_regions?.length > 0) ??
      (model.consumption_options || []).includes('batch')
    )
    
    return {
      model,
      region,
      contextWindow: baseContext,
      effectiveContext,
      hasExtendedContext: extendedContext != null && extendedContext > (baseContext || 0),
      maxOutput: model.specs?.max_output ?? model.specs?.max_output_tokens ?? model.converse_data?.max_output_tokens,
      inputModalities: model.modalities?.input_modalities ?? model.model_modalities?.input_modalities ?? [],
      outputModalities: model.modalities?.output_modalities ?? model.model_modalities?.output_modalities ?? [],
      streamingSupported: model.streaming ?? model.streaming_supported ?? false,
      crisSupported,
      crisProfilesCount: crisData.profiles?.length ?? crisData.profiles_count ?? 0,
      crisSourceRegions: (crisData.regions ?? crisData.source_regions ?? []).length,
      mantleSupported: model.availability?.mantle?.supported ?? false,
      mantleRegions: mantleRegions.length,
      consumptionOptions: model.consumption_options || [],
      languages: model.languages ?? model.languages_supported ?? [],
      customizations: model.customization?.customization_supported || [],
      isActive: (model.lifecycle?.status ?? model.model_lifecycle?.status) === 'ACTIVE' || model.model_status === 'ACTIVE',
      hasLongContext: hasLongCtx,
      batchSupported,
      batchRegions: (batchData.regions ?? batchData.supported_regions ?? []).length,
      batchCoverage: null, // Removed - coverage_percentage field no longer exists
      // Total regions: on-demand + CRIS + Mantle + Batch
      totalRegions: new Set([
        ...(model.availability?.on_demand?.regions ?? model.in_region ?? model.regions_available ?? []),
        ...(crisData.regions ?? crisData.source_regions ?? []),
        ...mantleRegions,
        ...(batchData.regions ?? batchData.supported_regions ?? [])
      ]).size,
      // For Mantle-only models, show "Mantle Only" instead of empty inference types
      inferenceTypes: isMantleOnly ? ['MANTLE_ONLY'] : (model.inference_types_supported || []),
      isMantleOnly,
    }
  })

  const SpecRow = ({ label, children }) => (
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
      {children}
    </tr>
  )

  const TextCell = ({ value, subtitle = null }) => (
    <td className={cn(
      'px-3 py-2.5 text-center',
      isLight ? 'text-stone-900' : 'text-white'
    )}>
      <span className="text-sm font-medium">{value}</span>
      {subtitle && (
        <span className={cn(
          'block text-[10px] mt-0.5',
          isLight ? 'text-stone-400' : 'text-slate-500'
        )}>
          {subtitle}
        </span>
      )}
    </td>
  )

  const BoolCell = ({ value }) => (
    <td className="px-3 py-2.5 text-center">
      {value ? (
        <Check className="h-4 w-4 text-emerald-500 mx-auto" />
      ) : (
        <X className="h-4 w-4 text-red-400/40 mx-auto" />
      )}
    </td>
  )

  const ModalitiesCell = ({ modalities, isOutput = false }) => (
    <td className="px-3 py-2.5">
      <div className="flex justify-center gap-1 flex-wrap">
        {modalities.length > 0 ? modalities.map(mod => {
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
        }) : (
          <span className={cn('text-xs', isLight ? 'text-stone-300' : 'text-slate-600')}>—</span>
        )}
      </div>
    </td>
  )

  const BadgesCell = ({ items, maxShow = 3, labelMap = null }) => (
    <td className="px-3 py-2.5">
      <div className="flex justify-center gap-0.5 flex-wrap">
        {items.length > 0 ? (
          <>
            {items.slice(0, maxShow).map(item => (
              <Badge
                key={item}
                variant="secondary"
                className="text-[9px] py-0 px-1"
              >
                {(labelMap || consumptionLabels)[item] || prettifyLabel(item)}
              </Badge>
            ))}
            {items.length > maxShow && (
              <Badge variant="secondary" className="text-[9px] py-0 px-1">
                +{items.length - maxShow}
              </Badge>
            )}
          </>
        ) : (
          <span className={cn('text-xs', isLight ? 'text-stone-300' : 'text-slate-600')}>—</span>
        )}
      </div>
    </td>
  )

  return (
    <div className="mt-4 space-y-3">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>Max Context</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {(() => { const max = Math.max(...specsData.map(d => d.effectiveContext || 0)); return max >= 1000000 ? `${(max/1000000).toFixed(0)}M` : max >= 1000 ? `${(max/1000).toFixed(0)}K` : max || '—' })()}
          </p>
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>Streaming</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {specsData.filter(d => d.streamingSupported).length}/{specsData.length}
          </p>
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>CRIS Support</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {specsData.filter(d => d.crisSupported).length}/{specsData.length}
          </p>
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>Batch Inference</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {specsData.filter(d => d.batchSupported).length}/{specsData.length}
          </p>
        </div>

        <div className={cn(
          'px-3 py-2.5 rounded-lg border',
          isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>Mantle</span>
          </div>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {specsData.filter(d => d.mantleSupported).length}/{specsData.length}
          </p>
        </div>
      </div>

      <div className={cn(
        'rounded-lg border overflow-hidden',
        isLight
          ? 'bg-white/80 border-stone-200/80 backdrop-blur-xl'
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
                'px-4 py-3 text-left text-xs font-semibold w-44 min-w-[140px] sticky left-0 z-30',
                isLight ? 'text-stone-900 bg-stone-50' : 'text-white bg-[#1a1b1e]'
              )}>
                Specification
              </th>
              {specsData.map(({ model }) => (
                <th key={model.model_id} className="px-3 py-3 text-center min-w-[110px]">
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
            {/* Context Window with long-context indicator */}
            <SpecRow label="Context Window">
              {specsData.map(d => (
                <td key={d.model.model_id} className={cn(
                  'px-3 py-2.5 text-center',
                  isLight ? 'text-stone-900' : 'text-white'
                )}>
                  <span className="text-sm font-medium">{formatNumber(d.effectiveContext)}</span>
                  {d.hasExtendedContext && (
                    <span className={cn(
                      'block text-[10px] mt-0.5',
                      isLight ? 'text-stone-400' : 'text-slate-500'
                    )}>
                      base: {formatNumber(d.contextWindow)}
                    </span>
                  )}
                  {d.hasLongContext && (
                    <span className={cn(
                      'block text-[10px] mt-0.5 px-1.5 py-0.5 rounded mx-auto w-fit',
                      isLight ? 'bg-purple-100 text-purple-700' : 'bg-purple-500/15 text-purple-400'
                    )}>
                      Long context
                    </span>
                  )}
                </td>
              ))}
            </SpecRow>

            <SpecRow label="Max Output Tokens">
              {specsData.map(d => (
                <TextCell key={d.model.model_id} value={formatNumber(d.maxOutput)} />
              ))}
            </SpecRow>

            <SpecRow label="Input Modalities">
              {specsData.map(d => (
                <ModalitiesCell key={d.model.model_id} modalities={d.inputModalities} />
              ))}
            </SpecRow>

            <SpecRow label="Output Modalities">
              {specsData.map(d => (
                <ModalitiesCell key={d.model.model_id} modalities={d.outputModalities} isOutput={true} />
              ))}
            </SpecRow>

            <SpecRow label="Streaming">
              {specsData.map(d => (
                <BoolCell key={d.model.model_id} value={d.streamingSupported} />
              ))}
            </SpecRow>

            <SpecRow label="Cross-Region (CRIS)">
              {specsData.map(d => (
                <td key={d.model.model_id} className="px-3 py-2.5 text-center">
                  {d.crisSupported ? (
                    <div className="flex flex-col items-center gap-0.5">
                      <Check className="h-4 w-4 text-emerald-500" />
                      <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
                        {d.crisSourceRegions} regions
                      </span>
                    </div>
                  ) : (
                    <X className="h-4 w-4 text-red-400/40 mx-auto" />
                  )}
                </td>
              ))}
            </SpecRow>

            <SpecRow label="Mantle">
              {specsData.map(d => (
                <td key={d.model.model_id} className="px-3 py-2.5 text-center">
                  {d.mantleSupported ? (
                    <div className="flex flex-col items-center gap-0.5">
                      <Check className="h-4 w-4 text-emerald-500" />
                      <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
                        {d.mantleRegions} regions
                      </span>
                    </div>
                  ) : (
                    <X className="h-4 w-4 text-red-400/40 mx-auto" />
                  )}
                </td>
              ))}
            </SpecRow>

            <SpecRow label="Batch Inference">
              {specsData.map(d => (
                <td key={d.model.model_id} className="px-3 py-2.5 text-center">
                  {d.batchSupported ? (
                    <div className="flex flex-col items-center gap-0.5">
                      <Check className="h-4 w-4 text-emerald-500" />
                      <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
                        {d.batchRegions} regions{d.batchCoverage != null ? ` (${Math.round(d.batchCoverage)}%)` : ''}
                      </span>
                    </div>
                  ) : (
                    <X className="h-4 w-4 text-red-400/40 mx-auto" />
                  )}
                </td>
              ))}
            </SpecRow>

            <SpecRow label="Consumption">
              {specsData.map(d => (
                <BadgesCell key={d.model.model_id} items={d.consumptionOptions} maxShow={4} />
              ))}
            </SpecRow>

            <SpecRow label="Customization">
              {specsData.map(d => (
                <BadgesCell key={d.model.model_id} items={d.customizations} />
              ))}
            </SpecRow>

            <SpecRow label="Languages">
              {specsData.map(d => (
                <td key={d.model.model_id} className={cn(
                  'px-3 py-2.5 text-center text-xs font-medium',
                  isLight ? 'text-stone-900' : 'text-white'
                )}>
                  {d.languages.length > 0 ? `${d.languages.length}` : '—'}
                </td>
              ))}
            </SpecRow>

            <SpecRow label="Inference Types">
              {specsData.map(d => (
                <BadgesCell key={d.model.model_id} items={d.inferenceTypes} maxShow={2} labelMap={inferenceTypeLabels} />
              ))}
            </SpecRow>

            <SpecRow label="Status">
              {specsData.map(d => (
                <td key={d.model.model_id} className="px-3 py-2.5 text-center">
                  <Badge variant={d.isActive ? 'success' : 'warning'} className="text-[10px]">
                    {d.isActive ? 'Active' : 'Legacy'}
                  </Badge>
                </td>
              ))}
            </SpecRow>

            <SpecRow label="Model ID">
              {specsData.map(d => (
                <td key={d.model.model_id} className={cn(
                  'px-3 py-2.5 text-center',
                  isLight ? 'text-stone-400' : 'text-slate-500'
                )}>
                  <code className="text-[9px] font-mono break-all leading-tight">
                    {d.model.model_id || '—'}
                  </code>
                </td>
              ))}
            </SpecRow>
          </tbody>
        </table>
      </div>
      </div>
    </div>
  )
}
