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
        'rounded-xl border card-hover-lift',
        isLight
          ? 'border-stone-200/80 bg-white/90 text-stone-900 backdrop-blur-xl shadow-md shadow-stone-900/5 ring-1 ring-stone-100/50'
          : 'border-[#373a40] bg-[#25262b] text-[#e4e5e7] shadow-lg shadow-black/15 ring-1 ring-white/[0.03]',
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
