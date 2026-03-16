import { Layout } from '@/components/layout/Layout'
import { ModelExplorer } from '@/components/models/ModelExplorer'
import { Favorites } from '@/components/models/Favorites'
import { ModelComparison } from '@/components/comparison/ModelComparison'
import { RegionalAvailability } from '@/components/models/RegionalAvailability'

function App() {
  return (
    <Layout>
      {({ activeSection, setActiveSection }) => {
        switch (activeSection) {
          case 'explorer':
            return <ModelExplorer />
          case 'favorites':
            return <Favorites onNavigateToExplorer={() => setActiveSection('explorer')} />
          case 'comparison':
            return (
              <ModelComparison
                onNavigateToExplorer={() => setActiveSection('explorer')}
              />
            )
          case 'availability':
            return <RegionalAvailability />
          default:
            return <ModelExplorer />
        }
      }}
    </Layout>
  )
}

export default App
