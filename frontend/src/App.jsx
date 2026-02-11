import { useEffect, useRef } from 'react'
import { Layout } from '@/components/layout/Layout'
import { ModelExplorer } from '@/components/models/ModelExplorer'
import { ModelComparison } from '@/components/comparison/ModelComparison'
import { AdminDashboard } from '@/components/admin/AdminDashboard'
import { RegionRoadmap } from '@/components/admin/RegionRoadmap'
import { AuthGate } from '@/auth/AuthGate'
import { initAnalytics, trackEvent, shutdownAnalytics, setUserGeo } from '@/services/analytics'
import { useAuthStore } from '@/stores/authStore'
import { canViewRoadmap, canViewAnalytics } from '@/config/admin'

function App() {
  const prevSection = useRef(null)
  const user = useAuthStore((s) => s.user)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)

  useEffect(() => {
    initAnalytics()
    trackEvent('page_view', { section: 'explorer' })
    return () => shutdownAnalytics()
  }, [])

  // When user authenticates, update analytics with Midway geo data (country/region only, no PII)
  useEffect(() => {
    if (user?.country || user?.region) {
      setUserGeo({ country: user.country, region: user.region })
    }
  }, [user])

  return (
    <AuthGate>
      <Layout>
        {({ activeSection, setActiveSection }) => {
          // Track section changes
          if (prevSection.current !== null && prevSection.current !== activeSection) {
            trackEvent('section_change', { section: activeSection, from: prevSection.current })
          }
          prevSection.current = activeSection

          switch (activeSection) {
            case 'explorer':
              return <ModelExplorer />
            case 'favorites':
              return (
                <div className="text-center py-16">
                  <h2 className="text-xl font-semibold text-white">Favorites</h2>
                  <p className="text-slate-400 mt-2">Coming soon...</p>
                </div>
              )
            case 'comparison':
              return (
                <ModelComparison
                  onNavigateToExplorer={() => setActiveSection('explorer')}
                />
              )
            case 'calculator':
              return (
                <div className="text-center py-16">
                  <h2 className="text-xl font-semibold text-white">Pricing Calculator</h2>
                  <p className="text-slate-400 mt-2">Coming soon...</p>
                </div>
              )
            case 'admin':
              return isAuthenticated && canViewAnalytics(user) ? <AdminDashboard /> : <ModelExplorer />
            case 'roadmap':
              return isAuthenticated && canViewRoadmap(user) ? <RegionRoadmap /> : <ModelExplorer />
            default:
              return <ModelExplorer />
          }
        }}
      </Layout>
    </AuthGate>
  )
}

export default App
