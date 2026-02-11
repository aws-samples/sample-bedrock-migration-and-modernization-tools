import * as React from 'react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'

const Card = React.forwardRef(({ className, ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <div
      ref={ref}
      className={cn(
        'rounded-xl border card-hover-lift backdrop-blur-xl',
        isLight
          ? 'border-stone-200/60 bg-white/70 text-stone-900 shadow-[0_2px_15px_-3px_rgba(120,113,108,0.08)] ring-1 ring-stone-100/50'
          : 'border-white/[0.06] bg-white/[0.03] text-[#e4e5e7] shadow-[0_2px_15px_-3px_rgba(0,0,0,0.3)] ring-1 ring-white/[0.03]',
        className
      )}
      {...props}
    />
  )
})
Card.displayName = 'Card'

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex flex-col space-y-1.5 p-6', className)}
    {...props}
  />
))
CardHeader.displayName = 'CardHeader'

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn('font-semibold leading-none tracking-tight', className)}
    {...props}
  />
))
CardTitle.displayName = 'CardTitle'

const CardDescription = React.forwardRef(({ className, ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <p
      ref={ref}
      className={cn('text-sm', isLight ? 'text-stone-600' : 'text-[#a0a1a5]', className)}
      {...props}
    />
  )
})
CardDescription.displayName = 'CardDescription'

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn('p-6 pt-0', className)} {...props} />
))
CardContent.displayName = 'CardContent'

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex items-center p-6 pt-0', className)}
    {...props}
  />
))
CardFooter.displayName = 'CardFooter'

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }
