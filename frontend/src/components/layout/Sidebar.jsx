import { useState, useMemo } from 'react'
import {
  LayoutGrid,
  Star,
  GitCompare,
  BarChart3,
  Map,
  PanelLeftClose,
  PanelLeftOpen,
  X,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { ThemeToggle } from './ThemeToggle'
import { useTheme } from './ThemeProvider'
import { BedrockIcon } from '@/components/icons/BedrockIcon'
import { useAuthStore } from '@/stores/authStore'
import { canViewRoadmap, canViewAnalytics, getSectionBadge } from '@/config/admin'

const baseNavigationItems = [
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

function NavButton({ item, isActive, isLight, collapsed, mobileMenuOpen, onClick }) {
  const Icon = item.icon
  const showLabel = !collapsed || mobileMenuOpen

  const button = (
    <button
      onClick={onClick}
      className={cn(
        'w-full flex items-center rounded-lg transition-all duration-200 text-left group',
        collapsed && !mobileMenuOpen ? 'justify-center px-0 py-2.5' : 'gap-3 px-3 py-2.5',
        isActive
          ? isLight
            ? 'bg-amber-700 text-[#faf9f5] shadow-sm'
            : 'bg-[#1A9E7A]/90 text-white shadow-sm shadow-[#1A9E7A]/20'
          : isLight
            ? 'text-stone-600 hover:bg-stone-100 hover:text-stone-900'
            : 'text-[#9a9b9f] hover:bg-[#2c2d32] hover:text-white'
      )}
    >
      <Icon className={cn(
        'h-[18px] w-[18px] flex-shrink-0 transition-colors',
        isActive
          ? 'text-current'
          : isLight
            ? 'text-stone-400 group-hover:text-stone-600'
            : 'text-[#6d6e72] group-hover:text-[#c0c1c5]'
      )} />
      {showLabel && (
        <>
          <span className="text-[13px] font-medium">{item.label}</span>
          {item.badge && (
            <span className={cn(
              'ml-auto text-[9px] font-bold leading-none px-1.5 py-0.5 rounded-full border',
              isLight ? item.badge.light : item.badge.dark
            )}>
              {item.badge.text}
            </span>
          )}
        </>
      )}
    </button>
  )

  if (collapsed && !mobileMenuOpen) {
    return (
      <TooltipProvider delayDuration={0}>
        <Tooltip>
          <TooltipTrigger asChild>{button}</TooltipTrigger>
          <TooltipContent side="right" sideOffset={8} className="z-[100]">
            {item.label}
            {item.badge && (
              <span className={cn(
                'ml-1.5 text-[9px] font-bold leading-none px-1.5 py-0.5 rounded-full border inline-block',
                isLight ? item.badge.light : item.badge.dark
              )}>
                {item.badge.text}
              </span>
            )}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  return button
}

export function Sidebar({ activeSection, onSectionChange, mobileMenuOpen, setMobileMenuOpen }) {
  const [collapsed, setCollapsed] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const showExpanded = !collapsed || mobileMenuOpen
  const user = useAuthStore((s) => s.user)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)

  const navigationItems = useMemo(() => {
    const items = [...baseNavigationItems]
    if (isAuthenticated && canViewRoadmap(user)) {
      items.push({ id: 'roadmap', label: 'Region Roadmap', icon: Map, badge: getSectionBadge(user, 'roadmap') })
    }
    if (isAuthenticated && canViewAnalytics(user)) {
      items.push({ id: 'admin', label: 'Analytics', icon: BarChart3, badge: getSectionBadge(user, 'admin') })
    }
    return items
  }, [isAuthenticated, user])

  const handleNavigation = (sectionId) => {
    onSectionChange(sectionId)
    setMobileMenuOpen?.(false)
  }

  const sidebarContent = (
    <>
      {/* Header */}
      <div className={cn(
        'flex items-center p-4',
        collapsed && !mobileMenuOpen ? 'justify-center' : 'gap-3',
      )}>
        {showExpanded ? (
          <>
            <BedrockIcon className={cn(
              'h-9 w-9 flex-shrink-0',
              isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
            )} />
            <span className={cn(
              'font-semibold text-[15px] leading-tight tracking-tight',
              isLight ? 'text-stone-900' : 'text-white'
            )}>
              Bedrock Model<br/>Profiler
            </span>
          </>
        ) : (
          <BedrockIcon className={cn(
            'h-8 w-8',
            isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
          )} />
        )}

        {/* Mobile close button */}
        {mobileMenuOpen && (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setMobileMenuOpen?.(false)}
            className="h-8 w-8 lg:hidden ml-auto"
          >
            <X className={cn(
              'h-5 w-5',
              isLight ? 'text-stone-500' : 'text-[#9a9b9f]'
            )} />
          </Button>
        )}
      </div>

      {/* Divider */}
      <div className={cn(
        'mx-3 h-px',
        isLight ? 'bg-stone-200/80' : 'bg-[#2c2d32]'
      )} />

      {/* Navigation */}
      <nav className={cn(
        'flex-1 space-y-0.5 overflow-y-auto',
        collapsed && !mobileMenuOpen ? 'px-2 py-3' : 'px-3 py-3'
      )}>
        {navigationItems.map((item) => (
          <NavButton
            key={item.id}
            item={item}
            isActive={activeSection === item.id}
            isLight={isLight}
            collapsed={collapsed}
            mobileMenuOpen={mobileMenuOpen}
            onClick={() => handleNavigation(item.id)}
          />
        ))}
      </nav>

      {/* Footer — collapse toggle + separator + version + theme */}
      <div className="px-3 py-3">
        {/* Collapse/Expand button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className={cn(
            'w-full flex items-center rounded-lg transition-all duration-200 hidden lg:flex',
            collapsed && !mobileMenuOpen ? 'justify-center py-2' : 'gap-3 px-3 py-2',
            isLight
              ? 'text-stone-500 hover:bg-stone-100 hover:text-stone-700'
              : 'text-[#6d6e72] hover:bg-[#2c2d32] hover:text-[#c0c1c5]'
          )}
        >
          {collapsed ? (
            <PanelLeftOpen className="h-4 w-4 flex-shrink-0" />
          ) : (
            <PanelLeftClose className="h-4 w-4 flex-shrink-0" />
          )}
          {showExpanded && (
            <span className="text-[12px] font-medium">Collapse</span>
          )}
        </button>

        {/* Separator */}
        <div className={cn(
          'my-2 border-t',
          isLight ? 'border-stone-200/80' : 'border-[#2c2d32]'
        )} />

        {/* Theme & Version row */}
        <div className={cn(
          'flex items-center',
          collapsed && !mobileMenuOpen ? 'justify-center' : 'justify-between'
        )}>
          {showExpanded && (
            <span className={cn(
              'text-[11px] font-medium tracking-wide uppercase',
              isLight ? 'text-stone-400' : 'text-[#4a4d54]'
            )}>
              v1.0.0
            </span>
          )}
          <ThemeToggle />
        </div>
      </div>
    </>
  )

  return (
    <>
      {/* Desktop Sidebar */}
      <aside
        className={cn(
          'hidden lg:flex flex-col h-screen border-r transition-all duration-300 ease-in-out',
          collapsed ? 'w-[60px]' : 'w-60',
          isLight
            ? 'bg-white/80 border-stone-200/80 backdrop-blur-xl'
            : 'bg-[#141517]/95 border-[#2c2d32]/60 backdrop-blur-xl'
        )}
      >
        {sidebarContent}
      </aside>

      {/* Mobile Sidebar Overlay */}
      {mobileMenuOpen && (
        <div className="lg:hidden fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen?.(false)}
          />
          <aside
            className={cn(
              'absolute left-0 top-0 bottom-0 w-72 flex flex-col border-r transition-transform duration-300',
              isLight
                ? 'bg-white border-stone-200'
                : 'bg-[#141517] border-[#2c2d32]'
            )}
          >
            {sidebarContent}
          </aside>
        </div>
      )}
    </>
  )
}
