import { useState } from 'react'
import {
  LayoutGrid,
  Star,
  GitCompare,
  Globe2,
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
  {
    id: 'availability',
    label: 'Regional Availability',
    icon: Globe2,
  },
]

function NavButton({ item, isActive, isLight, collapsed, mobileMenuOpen, onClick }) {
  const Icon = item.icon
  const isCollapsedDesktop = collapsed && !mobileMenuOpen

  const button = (
    <button
      onClick={onClick}
      className={cn(
        'w-full flex items-center rounded-lg transition-all duration-200 text-left group px-3 py-2.5 gap-3',
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
      <div className={cn(
        'flex items-center min-w-0 overflow-hidden transition-all duration-150 ease-in-out',
        isCollapsedDesktop ? 'w-0 opacity-0 flex-none' : 'w-auto opacity-100 flex-1'
      )}>
        <span className="text-[13px] font-medium whitespace-nowrap">{item.label}</span>
      </div>
    </button>
  )

  if (isCollapsedDesktop) {
    return (
      <TooltipProvider delayDuration={0}>
        <Tooltip>
          <TooltipTrigger asChild>{button}</TooltipTrigger>
          <TooltipContent side="right" sideOffset={8} className="z-[100]">
            {item.label}
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

  const handleNavigation = (sectionId) => {
    onSectionChange(sectionId)
    setMobileMenuOpen?.(false)
  }

  const sidebarContent = (
    <>
      {/* Header */}
      <div className={cn(
        'flex items-center p-4 min-h-[68px] transition-all duration-150 gap-3',
      )}>
        <BedrockIcon className={cn(
          'flex-shrink-0 transition-all duration-150',
          collapsed && !mobileMenuOpen ? 'h-8 w-8' : 'h-9 w-9',
          isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
        )} />
        <div className={cn(
          'overflow-hidden transition-all duration-150 ease-in-out',
          collapsed && !mobileMenuOpen ? 'w-0 opacity-0 flex-none' : 'w-auto opacity-100'
        )}>
          <span className={cn(
            'font-semibold text-[15px] leading-tight tracking-tight whitespace-nowrap',
            isLight ? 'text-stone-900' : 'text-white'
          )}>
            Bedrock Model<br/>Profiler
          </span>
        </div>

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
        'flex-1 space-y-0.5 overflow-y-auto px-3 py-3',
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
            'w-full flex items-center rounded-lg transition-all duration-200 hidden lg:flex py-2',
            collapsed && !mobileMenuOpen ? 'justify-center px-0 gap-0' : 'px-3 gap-3',
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
          <div className={cn(
            'overflow-hidden transition-all duration-150 ease-in-out',
            collapsed && !mobileMenuOpen ? 'w-0 opacity-0 flex-none' : 'w-auto opacity-100'
          )}>
            <span className="text-[12px] font-medium whitespace-nowrap">Collapse</span>
          </div>
        </button>

        {/* Separator */}
        <div className={cn(
          'my-2 border-t',
          isLight ? 'border-stone-200/80' : 'border-[#2c2d32]'
        )} />

        {/* Theme & Version row */}
        <div className={cn(
          'flex items-center transition-all duration-150',
          collapsed && !mobileMenuOpen ? 'justify-center' : 'justify-between gap-2',
        )}>
          <div className={cn(
            'overflow-hidden transition-all duration-150 ease-in-out',
            collapsed && !mobileMenuOpen ? 'w-0 opacity-0 flex-none' : 'w-auto opacity-100'
          )}>
            <span className={cn(
              'text-[11px] font-medium tracking-wide uppercase whitespace-nowrap',
              isLight ? 'text-stone-400' : 'text-[#4a4d54]'
            )}>
              v1.0.0
            </span>
          </div>
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
          'hidden lg:flex flex-col h-full border-r overflow-hidden transition-all duration-150 ease-in-out',
          collapsed ? 'w-[68px]' : 'w-60',
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
              'absolute left-0 top-0 bottom-0 w-72 flex flex-col border-r transition-transform duration-150',
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
