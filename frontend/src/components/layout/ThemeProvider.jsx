import { createContext, useContext, useState, useEffect } from 'react'

const ThemeContext = createContext({
  theme: 'dark',
  setTheme: () => null,
  toggleTheme: () => null,
})

export function ThemeProvider({ children, defaultTheme = 'dark', storageKey = 'bedrock-profiler-theme' }) {
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(storageKey) || defaultTheme
    }
    return defaultTheme
  })

  useEffect(() => {
    const root = window.document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(theme)
  }, [theme])

  const value = {
    theme,
    setTheme: (newTheme) => {
      localStorage.setItem(storageKey, newTheme)
      setTheme(newTheme)
    },
    toggleTheme: () => {
      const newTheme = theme === 'dark' ? 'light' : 'dark'
      const apply = () => {
        localStorage.setItem(storageKey, newTheme)
        setTheme(newTheme)
      }

      // Use View Transitions API for a smooth cross-fade if available
      if (document.startViewTransition) {
        document.startViewTransition(apply)
      } else {
        // Fallback: add temporary transition class
        document.documentElement.classList.add('theme-transitioning')
        apply()
        setTimeout(() => {
          document.documentElement.classList.remove('theme-transitioning')
        }, 500)
      }
    },
  }

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}
