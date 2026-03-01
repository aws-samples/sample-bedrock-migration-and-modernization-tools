import { Menu } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTheme } from './ThemeProvider'
import { Button } from '@/components/ui/button'
import { BedrockIcon } from '@/components/icons/BedrockIcon'

export function MainContent({ children, className, onMenuToggle }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

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
          : 'bg-slate-900/90 border-slate-800/50 backdrop-blur-xl'
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
      </div>

      <div className="relative p-4 sm:p-6">
        {children}
      </div>
    </main>
  )
}
