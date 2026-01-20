import { useState } from 'react'
import { GitCompare, Trash2, ArrowLeft, BarChart3, Globe, DollarSign, Cpu } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useComparisonStore } from '@/stores/comparisonStore'
import { useModels } from '@/hooks/useModels'
import { ComparisonCard } from './ComparisonCard'
import { OverviewTab } from './tabs/OverviewTab'
import { PricingTab } from './tabs/PricingTab'
import { AvailabilityTab } from './tabs/AvailabilityTab'
import { TechSpecsTab } from './tabs/TechSpecsTab'
import { cn } from '@/lib/utils'

function EmptyState({ isLight, onNavigateToExplorer }) {
  return (
    <div className={cn(
      'flex flex-col items-center justify-center py-20 px-4 rounded-xl border',
      isLight
        ? 'bg-white/60 border-stone-200/80 backdrop-blur-xl'
        : 'bg-[#161d26]/60 border-slate-700/50 backdrop-blur-xl'
    )}>
      <div className={cn(
        'w-16 h-16 rounded-full flex items-center justify-center mb-4',
        isLight ? 'bg-amber-100' : 'bg-[#1A9E7A]/20'
      )}>
        <GitCompare className={cn(
          'h-8 w-8',
          isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
        )} />
      </div>
      <h2 className={cn(
        'text-xl font-semibold mb-2',
        isLight ? 'text-stone-900' : 'text-white'
      )}>
        No Models Selected
      </h2>
      <p className={cn(
        'text-center max-w-md mb-6',
        isLight ? 'text-stone-600' : 'text-slate-400'
      )}>
        Select up to 5 models from the Model Explorer to compare their features,
        pricing, and availability side by side.
      </p>
      <Button
        onClick={onNavigateToExplorer}
        className={cn(
          isLight
            ? 'bg-amber-600 hover:bg-amber-700 text-white'
            : 'bg-[#1A9E7A] hover:bg-[#158567] text-white'
        )}
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Go to Model Explorer
      </Button>
    </div>
  )
}

export function ModelComparison({ onNavigateToExplorer }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const [activeTab, setActiveTab] = useState('overview')

  const { selectedModels, removeModel, updateRegion, clearAll, maxModels } = useComparisonStore()
  const { getPricingForModel } = useModels()

  if (selectedModels.length === 0) {
    return <EmptyState isLight={isLight} onNavigateToExplorer={onNavigateToExplorer} />
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className={cn(
            'p-2 rounded-lg',
            isLight ? 'bg-amber-100' : 'bg-[#1A9E7A]/20'
          )}>
            <GitCompare className={cn(
              'h-5 w-5',
              isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
            )} />
          </div>
          <div>
            <h1 className={cn(
              'text-lg sm:text-xl font-bold',
              isLight ? 'text-stone-900' : 'text-white'
            )}>
              Model Comparison
            </h1>
            <p className={cn(
              'text-sm',
              isLight ? 'text-stone-600' : 'text-slate-400'
            )}>
              Comparing {selectedModels.length} of {maxModels} models
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onNavigateToExplorer}
          >
            <ArrowLeft className="h-4 w-4 sm:mr-2" />
            <span className="hidden sm:inline">Add More</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearAll}
            className="text-red-500 hover:text-red-600 hover:bg-red-500/10"
          >
            <Trash2 className="h-4 w-4 sm:mr-2" />
            <span className="hidden sm:inline">Clear All</span>
          </Button>
        </div>
      </div>

      {/* Selected Models Grid */}
      <div className={cn(
        'grid gap-3',
        'grid-cols-1 sm:grid-cols-2',
        selectedModels.length >= 3 && 'lg:grid-cols-3',
        selectedModels.length >= 4 && 'xl:grid-cols-4',
        selectedModels.length >= 5 && '2xl:grid-cols-5'
      )}>
        {selectedModels.map(({ model, region }) => (
          <ComparisonCard
            key={model.model_id}
            model={model}
            region={region}
            onRemove={removeModel}
            onRegionChange={updateRegion}
          />
        ))}
      </div>

      {/* Comparison Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full justify-start overflow-x-auto">
          <TabsTrigger value="overview" className="gap-1 sm:gap-2">
            <BarChart3 className="h-4 w-4" />
            <span className="hidden sm:inline">Overview</span>
          </TabsTrigger>
          <TabsTrigger value="pricing" className="gap-1 sm:gap-2">
            <DollarSign className="h-4 w-4" />
            <span className="hidden sm:inline">Pricing</span>
          </TabsTrigger>
          <TabsTrigger value="availability" className="gap-1 sm:gap-2">
            <Globe className="h-4 w-4" />
            <span className="hidden sm:inline">Availability</span>
          </TabsTrigger>
          <TabsTrigger value="specs" className="gap-1 sm:gap-2">
            <Cpu className="h-4 w-4" />
            <span className="hidden sm:inline">Tech Specs</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <OverviewTab
            selectedModels={selectedModels}
            getPricingForModel={getPricingForModel}
            isLight={isLight}
          />
        </TabsContent>

        <TabsContent value="pricing">
          <PricingTab
            selectedModels={selectedModels}
            getPricingForModel={getPricingForModel}
            isLight={isLight}
          />
        </TabsContent>

        <TabsContent value="availability">
          <AvailabilityTab
            selectedModels={selectedModels}
            isLight={isLight}
          />
        </TabsContent>

        <TabsContent value="specs">
          <TechSpecsTab
            selectedModels={selectedModels}
            isLight={isLight}
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}
