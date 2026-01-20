import { X, Globe, MessageSquare, Image, FileText, Video, Mic, Radio, Check } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { RegionSelector } from '@/components/models/RegionSelector'
import { useTheme } from '@/components/layout/ThemeProvider'
import { cn } from '@/lib/utils'

// Provider color mapping
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
  return providerColors[provider] || providerColors.default
}

export function ComparisonCard({ model, region, onRemove, onRegionChange }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const contextWindow = model.converse_data?.context_window
  const maxOutput = model.converse_data?.max_output_tokens
  const inputModalities = model.model_modalities?.input_modalities || []
  const outputModalities = model.model_modalities?.output_modalities || []
  const isActive = model.model_lifecycle?.status === 'ACTIVE' || model.model_status === 'ACTIVE'
  const streamingSupported = model.streaming_supported || false
  const crisSupported = model.cross_region_inference?.supported || false

  const allModalities = [...new Set([...inputModalities, ...outputModalities])]

  return (
    <Card className={cn(
      'relative flex flex-col',
      isLight
        ? 'bg-white/80 border-stone-200/80 backdrop-blur-xl'
        : 'bg-[#161d26]/80 border-slate-700/50 backdrop-blur-xl'
    )}>
      {/* Remove button */}
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          'absolute top-2 right-2 h-6 w-6 z-10',
          isLight
            ? 'hover:bg-stone-200/80 text-stone-500'
            : 'hover:bg-slate-700/80 text-slate-400'
        )}
        onClick={() => onRemove(model.model_id)}
      >
        <X className="h-4 w-4" />
      </Button>

      <CardContent className="p-3 flex flex-col gap-2">
        {/* Provider and Status */}
        <div className="flex items-center gap-1.5 pr-6">
          <Badge className={cn('text-[10px] font-medium', isLight ? 'text-[#faf9f5]' : 'text-white', getProviderColor(model.model_provider))}>
            {model.model_provider}
          </Badge>
          <Badge variant={isActive ? 'success' : 'warning'} className="text-[10px] px-1.5 py-0">
            {isActive ? 'Active' : 'Legacy'}
          </Badge>
        </div>

        {/* Model name */}
        <div>
          <h4 className={cn(
            'font-semibold text-sm leading-tight line-clamp-2',
            isLight ? 'text-stone-900' : 'text-white'
          )}>
            {model.model_name || model.model_id}
          </h4>
          <p className={cn(
            'text-[10px] mt-0.5 truncate font-mono',
            isLight ? 'text-stone-500' : 'text-slate-500'
          )}>
            {model.model_id}
          </p>
        </div>

        {/* Capacity */}
        <div className={cn(
          'grid grid-cols-2 gap-1 text-center text-xs rounded p-1.5',
          isLight ? 'bg-stone-100/60' : 'bg-white/5'
        )}>
          <div>
            <p className={cn('text-[9px] uppercase', isLight ? 'text-stone-500' : 'text-slate-500')}>Context</p>
            <p className={cn('font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
              {formatNumber(contextWindow)}
            </p>
          </div>
          <div>
            <p className={cn('text-[9px] uppercase', isLight ? 'text-stone-500' : 'text-slate-500')}>Output</p>
            <p className={cn('font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
              {formatNumber(maxOutput)}
            </p>
          </div>
        </div>

        {/* Modalities */}
        <div className="flex items-center gap-1">
          {allModalities.slice(0, 4).map(mod => {
            const Icon = modalityIcons[mod] || MessageSquare
            return (
              <div
                key={mod}
                className={cn(
                  'p-1 rounded',
                  isLight ? 'bg-stone-100' : 'bg-white/5'
                )}
              >
                <Icon className={cn('h-3 w-3', isLight ? 'text-stone-600' : 'text-slate-400')} />
              </div>
            )
          })}
        </div>

        {/* Features */}
        <div className="flex items-center gap-2 text-[10px]">
          <div className="flex items-center gap-0.5">
            <Radio className={cn('h-3 w-3', streamingSupported ? 'text-emerald-500' : 'text-slate-400')} />
            {streamingSupported && <Check className="h-2.5 w-2.5 text-emerald-500" />}
          </div>
          <div className="flex items-center gap-0.5">
            <Globe className={cn('h-3 w-3', crisSupported ? 'text-blue-500' : 'text-slate-400')} />
            {crisSupported && <Check className="h-2.5 w-2.5 text-emerald-500" />}
          </div>
        </div>

        {/* Region selector */}
        <div className="mt-auto pt-2">
          <p className={cn('text-[10px] mb-1', isLight ? 'text-stone-500' : 'text-slate-500')}>
            Price Region
          </p>
          <RegionSelector
            value={region}
            onChange={(newRegion) => onRegionChange(model.model_id, newRegion)}
            className="h-8 text-xs"
          />
        </div>
      </CardContent>
    </Card>
  )
}
