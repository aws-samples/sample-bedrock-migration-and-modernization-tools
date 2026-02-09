import { useState } from 'react'
import {
  LayoutGrid,
  Star,
  GitCompare,
  ChevronLeft,
  ChevronRight,
  X,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { ThemeToggle } from './ThemeToggle'
import { useTheme } from './ThemeProvider'
import { BedrockIcon } from '@/components/icons/BedrockIcon'
import { UserProfile } from './UserProfile'

const navigationItems = [
  {
    id: 'explorer',
    label: 'Model Explorer',
    icon: LayoutGrid,
  },
  {
    id: 'favorites',
    label: 'Favorites',
    icon: Star,
  },
  {
    id: 'comparison',
    label: 'Model Comparison',
    icon: GitCompare,
  },
]

export function Sidebar({ activeSection, onSectionChange, mobileMenuOpen, setMobileMenuOpen }) {
  const [collapsed, setCollapsed] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const handleNavigation = (sectionId) => {
    onSectionChange(sectionId)
    setMobileMenuOpen?.(false)
  }

  const sidebarContent = (
    <>
      {/* Header */}
      <div className={cn(
        'flex items-center p-4 border-b',
        collapsed && !mobileMenuOpen ? 'flex-col gap-2' : 'justify-between',
        isLight ? 'border-stone-200' : 'border-[#373a40]'
      )}>
        {(!collapsed || mobileMenuOpen) ? (
          <div className="flex items-center gap-2">
            <BedrockIcon className={cn(
              'h-14 w-14 flex-shrink-0',
              isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
            )} />
            <span className={cn(
              'font-semibold text-base leading-tight',
              isLight ? 'text-stone-900' : 'text-white'
            )}>
              Bedrock Model Profiler
            </span>
          </div>
        ) : (
          <BedrockIcon className={cn(
            'h-13 w-13',
            isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
          )} />
        )}
        {/* Desktop collapse button */}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(!collapsed)}
          className="h-8 w-8 hidden lg:flex"
        >
          {collapsed ? (
            <ChevronRight className={cn(
              'h-4 w-4',
              isLight ? 'text-stone-500' : 'text-[#9a9b9f]'
            )} />
          ) : (
            <ChevronLeft className={cn(
              'h-4 w-4',
              isLight ? 'text-stone-500' : 'text-[#9a9b9f]'
            )} />
          )}
        </Button>
        {/* Mobile close button */}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setMobileMenuOpen?.(false)}
          className="h-8 w-8 lg:hidden"
        >
          <X className={cn(
            'h-5 w-5',
            isLight ? 'text-stone-500' : 'text-[#9a9b9f]'
          )} />
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-1">
        {navigationItems.map((item) => {
          const Icon = item.icon
          const isActive = activeSection === item.id

          return (
            <button
              key={item.id}
              onClick={() => handleNavigation(item.id)}
              className={cn(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-left',
                isActive
                  ? isLight
                    ? 'bg-amber-700 text-[#faf9f5]'
                    : 'bg-[#1A9E7A] text-white'
                  : isLight
                    ? 'text-stone-600 hover:bg-stone-100 hover:text-stone-900'
                    : 'text-[#c0c1c5] hover:bg-[#2c2d32] hover:text-white'
              )}
            >
              <Icon className={cn('h-5 w-5 flex-shrink-0', isActive && (isLight ? 'text-[#faf9f5]' : 'text-white'))} />
              {(!collapsed || mobileMenuOpen) && (
                <span className="text-sm font-medium">{item.label}</span>
              )}
            </button>
          )
        })}
      </nav>

      <Separator className={isLight ? 'bg-stone-200' : 'bg-[#373a40]'} />

      {/* User Profile */}
      <UserProfile collapsed={collapsed} mobileMenuOpen={mobileMenuOpen} />

      {/* Footer */}
      <div className={cn(
        'p-4 flex items-center',
        collapsed && !mobileMenuOpen ? 'justify-center' : 'justify-between'
      )}>
        {(!collapsed || mobileMenuOpen) && (
          <span className={cn(
            'text-xs',
            isLight ? 'text-stone-500' : 'text-[#6d6e72]'
          )}>
            v1.0.0
          </span>
        )}
        <ThemeToggle />
      </div>
    </>
  )

  return (
    <>
      {/* Desktop Sidebar - hidden on mobile */}
      <aside
        className={cn(
          'hidden lg:flex flex-col h-screen border-r transition-all duration-300',
          collapsed ? 'w-16' : 'w-64',
          isLight
            ? 'bg-white/80 border-stone-200/80 backdrop-blur-xl'
            : 'bg-[#1a1b1e]/95 border-[#373a40]/50 backdrop-blur-xl'
        )}
      >
        {sidebarContent}
      </aside>

      {/* Mobile Sidebar Overlay */}
      {mobileMenuOpen && (
        <div className="lg:hidden fixed inset-0 z-50">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen?.(false)}
          />
          {/* Mobile sidebar */}
          <aside
            className={cn(
              'absolute left-0 top-0 bottom-0 w-72 flex flex-col border-r transition-transform duration-300',
              isLight
                ? 'bg-white border-stone-200'
                : 'bg-[#1a1b1e] border-[#373a40]'
            )}
          >
            {sidebarContent}
          </aside>
        </div>
      )}
    </>
  )
}
