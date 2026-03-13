import { Menu, Clock } from 'lucide-react'
import { useMemo } from 'react'
import { cn } from '@/lib/utils'
import { useTheme } from './ThemeProvider'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { BedrockIcon } from '@/components/icons/BedrockIcon'
import { useModels } from '@/hooks/useModels'

export function MainContent({ children, className, onMenuToggle }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const { metadata } = useModels()

  const { lastUpdatedLabel, lastUpdatedFull } = useMemo(() => {
    const ts = metadata?.collection_timestamp
    if (!ts) return { lastUpdatedLabel: null, lastUpdatedFull: null }

    try {
      const d = new Date(ts)
      const now = new Date()
      const diffMs = now - d
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

      let label = null
      if (diffHours < 1) label = 'Updated just now'
      else if (diffHours < 24) label = `Updated ${diffHours}h ago`
      else if (diffDays === 1) label = 'Updated yesterday'
      else label = `Updated ${diffDays}d ago`

      const fullDate = d.toLocaleString('en-US', {
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short'
      })

      return { lastUpdatedLabel: label, lastUpdatedFull: fullDate }
    } catch {
      return { lastUpdatedLabel: null, lastUpdatedFull: null }
    }
  }, [metadata])

  return (
    <main
      className={cn(
        'flex-1 overflow-auto relative',
        isLight ? 'bg-[#faf9f5]' : 'bg-slate-950',
        className
      )}
    >
      {/* Theme gradient mesh background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {isLight ? (
          <>
            {/* Light theme: warm amber/cream gradients */}
            <div className="absolute -top-[20%] -right-[10%] w-[50%] h-[50%] bg-amber-200/30 rounded-full blur-[100px]" />
            <div className="absolute -bottom-[10%] -left-[10%] w-[40%] h-[40%] bg-orange-200/20 rounded-full blur-[80px]" />
            <div className="absolute top-[30%] left-[40%] w-[30%] h-[30%] bg-stone-200/30 rounded-full blur-[60px]" />
          </>
        ) : (
          <>
            {/* Dark theme: Bedrock green gradients */}
            <div className="absolute -top-[30%] -left-[20%] w-[60%] h-[60%] bg-[#1A9E7A]/10 rounded-full blur-[120px]" />
            <div className="absolute -bottom-[20%] -right-[10%] w-[50%] h-[50%] bg-[#1A9E7A]/5 rounded-full blur-[100px]" />
            <div className="absolute top-[40%] left-[30%] w-[40%] h-[40%] bg-slate-700/20 rounded-full blur-[80px]" />
          </>
        )}
      </div>

      {/* Mobile Header */}
      <div className={cn(
        'lg:hidden sticky top-0 z-40 flex items-center gap-3 px-3 py-2 border-b',
        isLight
          ? 'bg-white/90 border-stone-200/80 backdrop-blur-xl'
          : 'bg-[#141517]/95 border-[#2c2d32]/60 backdrop-blur-xl'
      )}>
        <Button
          variant="ghost"
          size="icon"
          onClick={onMenuToggle}
          className="h-9 w-9"
        >
          <Menu className={cn(
            'h-5 w-5',
            isLight ? 'text-stone-600' : 'text-slate-300'
          )} />
        </Button>
        <div className="flex items-center gap-2">
          <BedrockIcon className={cn(
            'h-8 w-8',
            isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
          )} />
          <span className={cn(
            'font-semibold text-sm',
            isLight ? 'text-stone-900' : 'text-white'
          )}>
            Bedrock Profiler
          </span>
        </div>
        {lastUpdatedLabel && (
          <TooltipProvider delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className={cn(
                  'ml-auto flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors cursor-default',
                  isLight
                    ? 'bg-stone-100/80 hover:bg-stone-100'
                    : 'bg-[#2c2d32]/60 hover:bg-[#2c2d32]'
                )}>
                  <Clock className={cn(
                    'h-3 w-3',
                    isLight ? 'text-stone-500' : 'text-slate-400'
                  )} />
                  <span className={cn(
                    'text-xs font-medium whitespace-nowrap',
                    isLight ? 'text-stone-600' : 'text-slate-300'
                  )}>
                    {lastUpdatedLabel}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={8}>
                <div className="text-xs">
                  <div className="font-medium mb-0.5">Data last refreshed</div>
                  <div className="text-muted-foreground">{lastUpdatedFull}</div>
                  <div className="text-muted-foreground mt-1 opacity-70">Refreshed automatically every 12 hours</div>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      {/* Desktop Header - timestamp only */}
      <div className={cn(
        'hidden lg:flex sticky top-0 z-40 items-center justify-end px-6 py-3 border-b',
        isLight
          ? 'bg-white/70 border-stone-200/60 backdrop-blur-xl'
          : 'bg-[#141517]/90 border-[#2c2d32]/50 backdrop-blur-xl'
      )}>
        {lastUpdatedLabel && (
          <TooltipProvider delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className={cn(
                  'flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors cursor-default',
                  isLight
                    ? 'bg-stone-100/80 hover:bg-stone-100'
                    : 'bg-[#2c2d32]/60 hover:bg-[#2c2d32]'
                )}>
                  <Clock className={cn(
                    'h-3.5 w-3.5',
                    isLight ? 'text-stone-500' : 'text-slate-400'
                  )} />
                  <span className={cn(
                    'text-sm font-medium whitespace-nowrap',
                    isLight ? 'text-stone-600' : 'text-slate-300'
                  )}>
                    {lastUpdatedLabel}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={8}>
                <div className="text-xs">
                  <div className="font-medium mb-0.5">Data last refreshed</div>
                  <div className="text-muted-foreground">{lastUpdatedFull}</div>
                  <div className="text-muted-foreground mt-1 opacity-70">Refreshed automatically every 12 hours</div>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      <div className="relative p-4 sm:p-6">
        {children}
      </div>
    </main>
  )
}
