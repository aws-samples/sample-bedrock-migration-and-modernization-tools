import { X } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/components/layout/ThemeProvider'
import { cn } from '@/lib/utils'
import { providerColorClasses } from '@/config/constants'

const providerColors = providerColorClasses

export function ComparisonCard({ model, onRemove }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const isActive = (model.lifecycle?.status ?? model.model_lifecycle?.status) === 'ACTIVE' || model.model_status === 'ACTIVE'

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
        <div className="flex items-center gap-1 pr-5 flex-wrap">
          <Badge className={cn(
            'text-[9px] font-medium px-1.5 py-0',
            isLight ? 'text-[#faf9f5]' : 'text-white',
            providerColors[model.model_provider] || providerColors.default
          )}>
            {model.model_provider}
          </Badge>
          {model.availability?.mantle?.supported && (
            <span className={cn(
              'inline-flex items-center px-1.5 py-0 rounded-full text-[9px] font-semibold',
              isLight
                ? 'bg-violet-100 text-violet-700 border border-violet-200'
                : 'bg-violet-500/15 text-violet-400 border border-violet-500/30'
            )}>
              Mantle
            </span>
          )}
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
      </CardContent>
    </Card>
  )
}
