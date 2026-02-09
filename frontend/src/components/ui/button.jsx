import * as React from 'react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'

const lightVariants = {
  default: 'bg-amber-700 text-[#faf9f5] shadow hover:bg-amber-800',
  destructive: 'bg-red-600 text-[#faf9f5] shadow-sm hover:bg-red-700',
  outline: 'border border-stone-400 bg-transparent text-stone-700 shadow-sm hover:bg-stone-100',
  secondary: 'bg-stone-200 text-stone-700 shadow-sm hover:bg-stone-300',
  ghost: 'text-stone-600 hover:bg-stone-100 hover:text-stone-900',
  link: 'text-amber-700 underline-offset-4 hover:underline',
}

const darkVariants = {
  default: 'bg-[#1A9E7A] text-white shadow hover:bg-[#22b38d]',
  destructive: 'bg-red-600 text-white shadow-sm hover:bg-red-700',
  outline: 'border border-[#4a4d54] bg-transparent text-[#e4e5e7] shadow-sm hover:bg-[#373a40] hover:border-[#5a5d64]',
  secondary: 'bg-[#2c2d32] text-[#a0a1a5] shadow-sm hover:bg-[#373a40] hover:text-[#e4e5e7]',
  ghost: 'text-[#a0a1a5] hover:bg-[#2c2d32] hover:text-[#e4e5e7]',
  link: 'text-[#1A9E7A] underline-offset-4 hover:underline hover:text-[#22b38d]',
}

const sizes = {
  default: 'h-9 px-4 py-2',
  sm: 'h-8 rounded-md px-3 text-xs',
  lg: 'h-10 rounded-md px-8',
  icon: 'h-9 w-9',
}

const Button = React.forwardRef(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    const { theme } = useTheme()
    const isLight = theme === 'light'
    const variants = isLight ? lightVariants : darkVariants

    return (
      <button
        className={cn(
          'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-slate-400 disabled:pointer-events-none disabled:opacity-50',
          variants[variant] || variants.default,
          sizes[size] || sizes.default,
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

export { Button }
