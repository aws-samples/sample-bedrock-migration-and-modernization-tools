import { Database, Users, Globe, Cpu, Image, Zap } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { useTheme } from '@/components/layout/ThemeProvider'
import { cn } from '@/lib/utils'

const statCards = [
  { key: 'totalModels', label: 'Total Models', icon: Database, color: 'text-blue-500' },
  { key: 'totalProviders', label: 'Providers', icon: Users, color: 'text-purple-500' },
  { key: 'totalRegions', label: 'Regions', icon: Globe, color: 'text-emerald-500' },
  { key: 'activeModels', label: 'Active', icon: Zap, color: 'text-green-500' },
  { key: 'multimodalModels', label: 'Multimodal', icon: Image, color: 'text-orange-500' },
]

export function ModelOverview({ stats }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  if (!stats) return null

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
      {statCards.map(({ key, label, icon: Icon, color }) => (
        <Card key={key}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={cn(
                  'text-xs uppercase tracking-wider',
                  isLight ? 'text-slate-500' : 'text-slate-400'
                )}>{label}</p>
                <p className={cn(
                  'text-2xl font-bold mt-1',
                  isLight ? 'text-slate-900' : 'text-white'
                )}>
                  {stats[key]?.toLocaleString() || 0}
                </p>
              </div>
              <Icon className={`h-8 w-8 ${color} opacity-80`} />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
