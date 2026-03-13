import { useState } from 'react'
import { Sidebar } from './Sidebar'
import { MainContent } from './MainContent'
import { ThemeProvider, useTheme } from './ThemeProvider'
import { cn } from '@/lib/utils'

function ConfidentialBanner() {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  
  return (
<div
  className={cn(
    'flex-shrink-0 text-center py-1 text-xs font-medium tracking-wide flex items-center justify-center gap-3',
    isLight 
      ? 'bg-stone-200 text-stone-600' 
      : 'bg-[#1a1b1e] text-slate-400'
  )}
>
  <span>Amazon Confidential — Internal Only</span>
  <span className="opacity-40">•</span>
  <span>All data is fetched from publicly available sources</span>
</div>
  )
}

export function Layout({ children }) {
  const [activeSection, setActiveSection] = useState('explorer')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <ThemeProvider defaultTheme="light">
      <div className="flex flex-col h-screen bg-slate-950">
        <ConfidentialBanner />
        <div className="flex flex-1 min-h-0">
        <Sidebar
          activeSection={activeSection}
          onSectionChange={setActiveSection}
          mobileMenuOpen={mobileMenuOpen}
          setMobileMenuOpen={setMobileMenuOpen}
        />
        <MainContent onMenuToggle={() => setMobileMenuOpen(true)}>
          {typeof children === 'function'
            ? children({ activeSection, setActiveSection })
            : children
          }
        </MainContent>
        </div>
      </div>
    </ThemeProvider>
  )
}
