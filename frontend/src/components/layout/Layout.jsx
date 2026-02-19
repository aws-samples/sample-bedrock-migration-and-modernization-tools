import { useState } from 'react'
import { Sidebar } from './Sidebar'
import { MainContent } from './MainContent'
import { ThemeProvider } from './ThemeProvider'

export function Layout({ children }) {
  const [activeSection, setActiveSection] = useState('explorer')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <ThemeProvider defaultTheme="light">
      <div className="flex h-screen bg-slate-950">
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
    </ThemeProvider>
  )
}
