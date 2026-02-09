import * as React from 'react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'

const lightVariants = {
  default: 'border-transparent bg-amber-100 text-amber-900 shadow',
  secondary: 'border-transparent bg-stone-200 text-stone-700',
  destructive: 'border-transparent bg-red-600 text-[#faf9f5] shadow',
  outline: 'border-stone-400 bg-transparent text-stone-700',
  success: 'border-transparent bg-emerald-600 text-[#faf9f5] shadow',
  warning: 'border-transparent bg-amber-700 text-[#faf9f5] shadow',
  info: 'border-transparent bg-amber-700 text-[#faf9f5] shadow',
}

const darkVariants = {
  default: 'border-transparent bg-[#373a40] text-[#e4e5e7] shadow',
  secondary: 'border-transparent bg-[#2c2d32] text-[#a0a1a5]',
  destructive: 'border-transparent bg-red-600 text-white shadow',
  outline: 'border-[#4a4d54] bg-transparent text-[#a0a1a5]',
  success: 'border-transparent bg-emerald-600/90 text-white shadow',
  warning: 'border-transparent bg-amber-600/90 text-white shadow',
  info: 'border-transparent bg-[#1A9E7A]/90 text-white shadow',
}

const Badge = React.forwardRef(({ className, variant = 'default', ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const variants = isLight ? lightVariants : darkVariants

  return (
    <div
      ref={ref}
      className={cn(
        'inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2',
        variants[variant] || variants.default,
        className
      )}
      {...props}
    />
  )
})
Badge.displayName = 'Badge'

export { Badge }
