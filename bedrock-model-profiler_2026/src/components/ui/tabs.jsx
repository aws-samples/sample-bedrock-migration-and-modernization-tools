import * as React from 'react'
import * as TabsPrimitive from '@radix-ui/react-tabs'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'

const Tabs = TabsPrimitive.Root

const TabsList = React.forwardRef(({ className, ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <TabsPrimitive.List
      ref={ref}
      className={cn(
        'inline-flex h-10 items-center justify-center rounded-lg p-1',
        isLight
          ? 'bg-stone-100/80 text-stone-500 backdrop-blur-sm border border-stone-200/50'
          : 'bg-white/5 text-slate-400 backdrop-blur-sm border border-white/10',
        className
      )}
      {...props}
    />
  )
})
TabsList.displayName = TabsPrimitive.List.displayName

const TabsTrigger = React.forwardRef(({ className, ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <TabsPrimitive.Trigger
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
        isLight
          ? 'ring-offset-white focus-visible:ring-amber-600 data-[state=active]:bg-white/90 data-[state=active]:text-stone-900 data-[state=active]:shadow data-[state=active]:backdrop-blur-sm'
          : 'ring-offset-transparent focus-visible:ring-[#1A9E7A] data-[state=active]:bg-white/10 data-[state=active]:text-white data-[state=active]:shadow data-[state=active]:backdrop-blur-sm',
        className
      )}
      {...props}
    />
  )
})
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName

const TabsContent = React.forwardRef(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      'mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-950 focus-visible:ring-offset-2 dark:ring-offset-slate-950 dark:focus-visible:ring-slate-300',
      className
    )}
    {...props}
  />
))
TabsContent.displayName = TabsPrimitive.Content.displayName

export { Tabs, TabsList, TabsTrigger, TabsContent }
