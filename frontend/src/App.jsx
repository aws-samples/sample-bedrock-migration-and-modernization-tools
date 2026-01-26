import { Layout } from '@/components/layout/Layout'
import { ModelExplorer } from '@/components/models/ModelExplorer'
import { ModelComparison } from '@/components/comparison/ModelComparison'
import { AuthGate } from '@/auth/AuthGate'

function App() {
  return (
    <AuthGate>
      <Layout>
        {({ activeSection, setActiveSection }) => {
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
            default:
              return <ModelExplorer />
          }
        }}
      </Layout>
    </AuthGate>
  )
}

export default App
