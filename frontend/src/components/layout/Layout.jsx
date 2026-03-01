import { useState } from 'react'
import { Sidebar } from './Sidebar'
import { MainContent } from './MainContent'
import { ThemeProvider, useTheme } from './ThemeProvider'
import { cn } from '@/lib/utils'

function ConfidentialBanner() {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  
  return (
    <div className={cn(
      'flex-shrink-0 text-center py-1 text-xs font-medium tracking-wide flex items-center justify-center gap-3',
      isLight 
        ? 'bg-stone-200 text-stone-600' 
        : 'bg-slate-800 text-slate-400'
    )}>
      <span>Amazon Confidential — Internal Only - BETA</span>
      <span className="opacity-40">•</span>
      <a
        href="https://quip-amazon.com/uyKFALmfnmaV/Bedrock-Model-Profiler-Report-Issues-and-Feedback"
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          'underline hover:no-underline',
          isLight ? 'hover:text-stone-800' : 'hover:text-slate-200'
        )}
      >
        Report Issues & Feedback
      </a>
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
