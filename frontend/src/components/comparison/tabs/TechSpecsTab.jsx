import { Check, X, MessageSquare, Image, FileText, Video, Mic } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

const modalityIcons = {
  TEXT: MessageSquare,
  IMAGE: Image,
  DOCUMENT: FileText,
  VIDEO: Video,
  AUDIO: Mic,
  SPEECH: Mic,
}

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

function formatNumber(num) {
  if (typeof num !== 'number' || isNaN(num)) return 'N/A'
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(0)}K`
  return num.toString()
}

// Consumption option labels
const consumptionLabels = {
  on_demand: 'On-Demand',
  provisioned: 'Provisioned',
  batch: 'Batch',
  cross_region_inference: 'CRIS'
}

export function TechSpecsTab({ selectedModels, isLight }) {
  // Extract specs for each model
  const specsData = selectedModels.map(({ model }) => ({
    model,
    contextWindow: model.converse_data?.context_window,
    maxOutput: model.converse_data?.max_output_tokens,
    inputModalities: model.model_modalities?.input_modalities || [],
    outputModalities: model.model_modalities?.output_modalities || [],
    streamingSupported: model.streaming_supported || false,
    crisSupported: model.cross_region_inference?.supported || false,
    consumptionOptions: model.consumption_options || [],
    languages: model.languages_supported || [],
    capabilities: model.model_capabilities || [],
    customizations: model.customization?.customization_supported || [],
    isActive: model.model_lifecycle?.status === 'ACTIVE' || model.model_status === 'ACTIVE',
    arn: model.model_arn,
  }))

  const SpecRow = ({ label, children }) => (
    <tr className={cn(
      'border-b',
      isLight ? 'border-stone-200/60' : 'border-slate-700/50'
    )}>
      <td className={cn(
        'px-4 py-3 font-medium text-sm',
        isLight ? 'text-stone-700' : 'text-slate-300'
      )}>
        {label}
      </td>
      {children}
    </tr>
  )

  const TextCell = ({ value }) => (
    <td className={cn(
      'px-4 py-3 text-center text-sm',
      isLight ? 'text-stone-600' : 'text-slate-400'
    )}>
      {value}
    </td>
  )

  const BoolCell = ({ value }) => (
    <td className="px-4 py-3 text-center">
      {value ? (
        <div className="flex items-center justify-center gap-1">
          <Check className="h-4 w-4 text-emerald-500" />
          <span className="text-xs text-emerald-500">Yes</span>
        </div>
      ) : (
        <div className="flex items-center justify-center gap-1">
          <X className="h-4 w-4 text-red-400/60" />
          <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-slate-500')}>No</span>
        </div>
      )}
    </td>
  )

  const ModalitiesCell = ({ modalities }) => (
    <td className="px-4 py-3">
      <div className="flex justify-center gap-1 flex-wrap">
        {modalities.length > 0 ? modalities.map(mod => {
          const Icon = modalityIcons[mod] || MessageSquare
          return (
            <div
              key={mod}
              className={cn(
                'p-1 rounded',
                isLight ? 'bg-stone-100' : 'bg-white/5'
              )}
              title={mod}
            >
              <Icon className={cn('h-3.5 w-3.5', isLight ? 'text-stone-600' : 'text-slate-400')} />
            </div>
          )
        }) : (
          <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-slate-500')}>N/A</span>
        )}
      </div>
    </td>
  )

  const BadgesCell = ({ items, maxShow = 3 }) => (
    <td className="px-4 py-3">
      <div className="flex justify-center gap-1 flex-wrap">
        {items.length > 0 ? (
          <>
            {items.slice(0, maxShow).map(item => (
              <Badge
                key={item}
                variant="secondary"
                className="text-[10px] py-0 px-1.5"
              >
                {consumptionLabels[item] || item}
              </Badge>
            ))}
            {items.length > maxShow && (
              <Badge variant="secondary" className="text-[10px] py-0 px-1.5">
                +{items.length - maxShow}
              </Badge>
            )}
          </>
        ) : (
          <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-slate-500')}>N/A</span>
        )}
      </div>
    </td>
  )

  return (
    <div className={cn(
      'mt-4 rounded-lg border overflow-hidden',
      isLight
        ? 'bg-white/80 border-stone-200/80 backdrop-blur-xl'
        : 'bg-[#161d26]/80 border-slate-700/50 backdrop-blur-xl'
    )}>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className={cn(
              'border-b-2',
              isLight ? 'border-stone-300 bg-stone-50/80' : 'border-slate-600 bg-slate-800/50'
            )}>
              <th className={cn(
                'px-4 py-3 text-left text-sm font-semibold w-48',
                isLight ? 'text-stone-900' : 'text-white'
              )}>
                Specification
              </th>
              {specsData.map(({ model }) => (
                <th key={model.model_id} className="px-4 py-3 text-center min-w-[150px]">
                  <Badge className={cn(
                    'text-[10px] mb-1',
                    isLight ? 'text-[#faf9f5]' : 'text-white',
                    providerColors[model.model_provider] || providerColors.default
                  )}>
                    {model.model_provider}
                  </Badge>
                  <p className={cn(
                    'text-sm font-semibold line-clamp-2',
                    isLight ? 'text-stone-900' : 'text-white'
                  )}>
                    {model.model_name || model.model_id}
                  </p>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {/* Context Window */}
            <SpecRow label="Context Window">
              {specsData.map(d => (
                <TextCell key={d.model.model_id} value={formatNumber(d.contextWindow)} />
              ))}
            </SpecRow>

            {/* Max Output */}
            <SpecRow label="Max Output Tokens">
              {specsData.map(d => (
                <TextCell key={d.model.model_id} value={formatNumber(d.maxOutput)} />
              ))}
            </SpecRow>

            {/* Input Modalities */}
            <SpecRow label="Input Modalities">
              {specsData.map(d => (
                <ModalitiesCell key={d.model.model_id} modalities={d.inputModalities} />
              ))}
            </SpecRow>

            {/* Output Modalities */}
            <SpecRow label="Output Modalities">
              {specsData.map(d => (
                <ModalitiesCell key={d.model.model_id} modalities={d.outputModalities} />
              ))}
            </SpecRow>

            {/* Streaming */}
            <SpecRow label="Streaming Support">
              {specsData.map(d => (
                <BoolCell key={d.model.model_id} value={d.streamingSupported} />
              ))}
            </SpecRow>

            {/* CRIS */}
            <SpecRow label="Cross-Region Inference">
              {specsData.map(d => (
                <BoolCell key={d.model.model_id} value={d.crisSupported} />
              ))}
            </SpecRow>

            {/* Consumption Options */}
            <SpecRow label="Consumption Options">
              {specsData.map(d => (
                <BadgesCell key={d.model.model_id} items={d.consumptionOptions} maxShow={4} />
              ))}
            </SpecRow>

            {/* Customizations */}
            <SpecRow label="Customization Options">
              {specsData.map(d => (
                <BadgesCell key={d.model.model_id} items={d.customizations} />
              ))}
            </SpecRow>

            {/* Capabilities */}
            <SpecRow label="Capabilities">
              {specsData.map(d => (
                <BadgesCell key={d.model.model_id} items={d.capabilities} maxShow={3} />
              ))}
            </SpecRow>

            {/* Languages */}
            <SpecRow label="Languages">
              {specsData.map(d => (
                <td key={d.model.model_id} className={cn(
                  'px-4 py-3 text-center text-sm',
                  isLight ? 'text-stone-600' : 'text-slate-400'
                )}>
                  {d.languages.length > 0 ? `${d.languages.length} languages` : 'N/A'}
                </td>
              ))}
            </SpecRow>

            {/* Status */}
            <SpecRow label="Status">
              {specsData.map(d => (
                <td key={d.model.model_id} className="px-4 py-3 text-center">
                  <Badge variant={d.isActive ? 'success' : 'warning'} className="text-xs">
                    {d.isActive ? 'Active' : 'Legacy'}
                  </Badge>
                </td>
              ))}
            </SpecRow>

            {/* Model ARN */}
            <SpecRow label="Model ARN">
              {specsData.map(d => (
                <td key={d.model.model_id} className={cn(
                  'px-4 py-3 text-center',
                  isLight ? 'text-stone-500' : 'text-slate-500'
                )}>
                  <code className="text-[10px] font-mono break-all">
                    {d.arn || 'N/A'}
                  </code>
                </td>
              ))}
            </SpecRow>
          </tbody>
        </table>
      </div>
    </div>
  )
}
