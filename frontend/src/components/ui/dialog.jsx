import * as React from 'react'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'

const Dialog = DialogPrimitive.Root
const DialogTrigger = DialogPrimitive.Trigger
const DialogPortal = DialogPrimitive.Portal
const DialogClose = DialogPrimitive.Close

const DialogOverlay = React.forwardRef(({ className, ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <DialogPrimitive.Overlay
      ref={ref}
      className={cn(
        'fixed inset-0 z-50 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
        isLight
          ? 'bg-stone-900/40 backdrop-blur-sm'
          : 'bg-black/60 backdrop-blur-sm',
        className
      )}
      {...props}
    />
  )
})
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName

const DialogContent = React.forwardRef(
  ({ className, children, ...props }, ref) => {
    const { theme } = useTheme()
    const isLight = theme === 'light'

    return (
      <DialogPortal>
        <DialogOverlay />
        <DialogPrimitive.Content
          ref={ref}
          className={cn(
            'fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border p-6 duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-xl',
            isLight
              ? 'border-stone-300/80 bg-white/95 backdrop-blur-xl shadow-xl shadow-stone-900/10 ring-1 ring-stone-200/50'
              : 'border-[#373a40] bg-[#25262b]/98 backdrop-blur-xl shadow-2xl shadow-black/40 ring-1 ring-white/5',
            className
          )}
          {...props}
        >
          {children}
          <DialogPrimitive.Close className={cn(
            'absolute right-4 top-4 rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:pointer-events-none',
            isLight
              ? 'ring-offset-white focus:ring-amber-600 data-[state=open]:bg-stone-100 data-[state=open]:text-stone-500'
              : 'ring-offset-transparent focus:ring-[#1A9E7A] data-[state=open]:bg-[#373a40] data-[state=open]:text-[#c0c1c5]'
          )}>
            <X className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-[#c0c1c5]')} />
            <span className="sr-only">Close</span>
          </DialogPrimitive.Close>
        </DialogPrimitive.Content>
      </DialogPortal>
    )
  }
)
DialogContent.displayName = DialogPrimitive.Content.displayName

const DialogHeader = ({ className, ...props }) => (
  <div
    className={cn(
      'flex flex-col space-y-1.5 text-center sm:text-left',
      className
    )}
    {...props}
  />
)
DialogHeader.displayName = 'DialogHeader'

const DialogFooter = ({ className, ...props }) => (
  <div
    className={cn(
      'flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2',
      className
    )}
    {...props}
  />
)
DialogFooter.displayName = 'DialogFooter'

const DialogTitle = React.forwardRef(({ className, ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <DialogPrimitive.Title
      ref={ref}
      className={cn(
        'text-lg font-semibold leading-none tracking-tight',
        isLight ? 'text-stone-900' : 'text-white',
        className
      )}
      {...props}
    />
  )
})
DialogTitle.displayName = DialogPrimitive.Title.displayName

const DialogDescription = React.forwardRef(({ className, ...props }, ref) => {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <DialogPrimitive.Description
      ref={ref}
      className={cn(
        'text-sm',
        isLight ? 'text-stone-500' : 'text-[#c0c1c5]',
        className
      )}
      {...props}
    />
  )
})
DialogDescription.displayName = DialogPrimitive.Description.displayName

export {
  Dialog,
  DialogPortal,
  DialogOverlay,
  DialogTrigger,
  DialogClose,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
}
