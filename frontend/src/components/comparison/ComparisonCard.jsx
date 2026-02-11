import { X, Globe, MessageSquare, Image, FileText, Video, Mic, Radio } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/components/layout/ThemeProvider'
import { cn } from '@/lib/utils'
import { providerColorClasses } from '@/config/constants'

const providerColors = providerColorClasses

const modalityConfig = {
  TEXT: { icon: MessageSquare, label: 'Text' },
  IMAGE: { icon: Image, label: 'Image' },
  DOCUMENT: { icon: FileText, label: 'Doc' },
  VIDEO: { icon: Video, label: 'Video' },
  AUDIO: { icon: Mic, label: 'Audio' },
  SPEECH: { icon: Mic, label: 'Speech' },
}

function formatNumber(num) {
  if (typeof num !== 'number' || isNaN(num)) return 'N/A'
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(0)}K`
  return num.toString()
}

export function ComparisonCard({ model, onRemove }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const contextWindow = model.converse_data?.context_window
  const maxOutput = model.converse_data?.max_output_tokens
  const inputModalities = model.model_modalities?.input_modalities || []
  const outputModalities = model.model_modalities?.output_modalities || []
  const isActive = model.model_lifecycle?.status === 'ACTIVE' || model.model_status === 'ACTIVE'
  const streamingSupported = model.streaming_supported || false
  const crisSupported = model.cross_region_inference?.supported || false

  return (
    <Card className={cn(
      'relative flex flex-col group',
      isLight
        ? 'bg-white/70 border-stone-200/60 backdrop-blur-xl shadow-[0_2px_15px_-3px_rgba(120,113,108,0.08)]'
        : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl shadow-[0_2px_15px_-3px_rgba(0,0,0,0.3)]'
    )}>
      {/* Remove button */}
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          'absolute top-1.5 right-1.5 h-5 w-5 z-10 opacity-0 group-hover:opacity-100 transition-opacity',
          isLight
            ? 'hover:bg-stone-200/80 text-stone-400'
            : 'hover:bg-white/[0.08] text-slate-500'
        )}
        onClick={() => onRemove(model.model_id)}
      >
        <X className="h-3 w-3" />
      </Button>

      <CardContent className="p-2.5 flex flex-col gap-1.5">
        {/* Provider + Status row */}
        <div className="flex items-center gap-1 pr-5">
          <Badge className={cn(
            'text-[9px] font-medium px-1.5 py-0',
            isLight ? 'text-[#faf9f5]' : 'text-white',
            providerColors[model.model_provider] || providerColors.default
          )}>
            {model.model_provider}
          </Badge>
          {isActive ? (
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" title="Active" />
          ) : (
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" title="Legacy" />
          )}
        </div>

        {/* Model name */}
        <h4 className={cn(
          'font-semibold text-xs leading-tight line-clamp-2',
          isLight ? 'text-stone-900' : 'text-white'
        )}>
          {model.model_name || model.model_id}
        </h4>

        {/* Compact stats row */}
        <div className="flex items-center gap-2 text-[10px]">
          <span className={cn(
            'font-medium',
            isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
          )}>
            {formatNumber(contextWindow)}
          </span>
          <span className={cn('text-[8px]', isLight ? 'text-stone-300' : 'text-slate-600')}>|</span>
          <span className={cn(
            'font-medium',
            isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
          )}>
            {formatNumber(maxOutput)}
          </span>
        </div>

        {/* Modalities */}
        <div className="flex flex-wrap items-center gap-1 text-[9px]">
          {inputModalities.length > 0 && inputModalities.map(mod => {
            const cfg = modalityConfig[mod] || { icon: MessageSquare, label: mod }
            const Icon = cfg.icon
            return (
              <span key={`in-${mod}`} className={cn(
                'flex items-center gap-0.5 px-1 py-0.5 rounded',
                isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/5 text-slate-400'
              )}>
                <Icon className="h-2.5 w-2.5" />
                {cfg.label}
              </span>
            )
          })}
          {outputModalities.filter(m => !inputModalities.includes(m)).map(mod => {
            const cfg = modalityConfig[mod] || { icon: MessageSquare, label: mod }
            const Icon = cfg.icon
            return (
              <span key={`out-${mod}`} className={cn(
                'flex items-center gap-0.5 px-1 py-0.5 rounded',
                isLight ? 'bg-blue-50 text-blue-600' : 'bg-blue-500/10 text-blue-400'
              )}>
                <Icon className="h-2.5 w-2.5" />
                {cfg.label}
              </span>
            )
          })}
        </div>

        {/* Features row */}
        <div className="flex items-center gap-1 text-[9px]">
          {streamingSupported && (
            <span className={cn(
              'flex items-center gap-0.5 px-1 py-0.5 rounded',
              isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/5 text-slate-400'
            )}>
              <Radio className="h-2.5 w-2.5" />
              Stream
            </span>
          )}
          {crisSupported && (
            <span className={cn(
              'flex items-center gap-0.5 px-1 py-0.5 rounded',
              isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/5 text-slate-400'
            )}>
              <Globe className="h-2.5 w-2.5" />
              CRIS
            </span>
          )}
        </div>

      </CardContent>
    </Card>
  )
}
