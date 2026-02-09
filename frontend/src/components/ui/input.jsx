import * as React from 'react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'

const Input = React.forwardRef(({ className, type, ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <input
      type={type}
      className={cn(
        'flex h-9 w-full rounded-md border px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium focus-visible:outline-none focus-visible:ring-1 disabled:cursor-not-allowed disabled:opacity-50',
        isLight
          ? 'border-stone-300 bg-white text-stone-900 placeholder:text-stone-500 focus-visible:ring-amber-600'
          : 'border-[#373a40] bg-[#1a1b1e] text-[#e4e5e7] placeholder:text-[#6d6e72] focus-visible:ring-[#1A9E7A]',
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
Input.displayName = 'Input'

export { Input }
