import { useState, useMemo, useRef, useEffect } from 'react'
import { GitCompare, Trash2, ArrowLeft, BarChart3, Globe, DollarSign, Cpu, ChevronDown, ChevronUp, Plus, Search, X } from 'lucide-react'
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
import { providerColorClasses } from '@/config/constants'

function AddModelSearch({ isLight, models, addModel, isModelSelected }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [providerFilter, setProviderFilter] = useState(null)
  const inputRef = useRef(null)
  const dropdownRef = useRef(null)

  useEffect(() => {
    if (open && inputRef.current) inputRef.current.focus()
  }, [open])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setOpen(false)
        setQuery('')
        setProviderFilter(null)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  // Get sorted providers list
  const providers = useMemo(() => {
    const provs = new Map()
    models.forEach(m => {
      const p = m.model_provider || 'Unknown'
      provs.set(p, (provs.get(p) || 0) + 1)
    })
    return [...provs.entries()].sort((a, b) => b[1] - a[1])
  }, [models])

  // Sorted models: by provider then name
  const sortedModels = useMemo(() =>
    [...models].sort((a, b) => {
      const p = (a.model_provider || '').localeCompare(b.model_provider || '')
      if (p !== 0) return p
      return (a.model_name || a.model_id).localeCompare(b.model_name || b.model_id)
    }),
    [models]
  )

  // Filter models
  const filtered = useMemo(() => {
    let list = sortedModels
    if (providerFilter) {
      list = list.filter(m => m.model_provider === providerFilter)
    }
    if (query.trim()) {
      const q = query.toLowerCase()
      list = list.filter(m =>
        (m.model_name || '').toLowerCase().includes(q) ||
        m.model_id.toLowerCase().includes(q)
      )
    }
    return list
  }, [sortedModels, query, providerFilter])

  // Group filtered models by provider
  const grouped = useMemo(() => {
    const map = new Map()
    filtered.forEach(m => {
      const p = m.model_provider || 'Unknown'
      if (!map.has(p)) map.set(p, [])
      map.get(p).push(m)
    })
    return map
  }, [filtered])

  if (!open) {
    return (
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(true)}
      >
        <Plus className="h-4 w-4 sm:mr-2" />
        <span className="hidden sm:inline">Add Models</span>
      </Button>
    )
  }

  return (
    <div ref={dropdownRef} className="relative">
      <div className={cn(
        'flex items-center gap-1 rounded-md border px-2 py-1',
        isLight ? 'bg-white border-stone-300' : 'bg-[#131a24] border-slate-600'
      )}>
        <Search className={cn('h-3.5 w-3.5 flex-shrink-0', isLight ? 'text-stone-400' : 'text-slate-500')} />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search models..."
          className={cn(
            'bg-transparent outline-none text-sm w-48',
            isLight ? 'text-stone-900 placeholder:text-stone-400' : 'text-white placeholder:text-slate-500'
          )}
        />
        <button onClick={() => { setOpen(false); setQuery(''); setProviderFilter(null) }} className="p-0.5">
          <X className={cn('h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-slate-500')} />
        </button>
      </div>

      {/* Dropdown */}
      <div className={cn(
        'absolute top-full right-0 mt-1 w-96 rounded-lg border shadow-xl z-50 flex flex-col',
        isLight ? 'bg-white border-stone-200' : 'bg-[#131a24] border-slate-700'
      )} style={{ maxHeight: '420px' }}>
        {/* Provider filter pills */}
        <div className={cn(
          'flex items-center gap-1 px-3 py-2 overflow-x-auto flex-shrink-0 border-b',
          isLight ? 'border-stone-100' : 'border-slate-800/50'
        )}>
          <button
            onClick={() => setProviderFilter(null)}
            className={cn(
              'px-2 py-0.5 rounded-full text-[10px] font-medium whitespace-nowrap transition-colors',
              !providerFilter
                ? isLight ? 'bg-amber-100 text-amber-800' : 'bg-[#1A9E7A]/20 text-[#1A9E7A]'
                : isLight ? 'bg-stone-100 text-stone-500 hover:bg-stone-200' : 'bg-white/5 text-slate-400 hover:bg-white/10'
            )}
          >
            All ({models.length})
          </button>
          {providers.map(([prov, count]) => (
            <button
              key={prov}
              onClick={() => setProviderFilter(providerFilter === prov ? null : prov)}
              className={cn(
                'px-2 py-0.5 rounded-full text-[10px] font-medium whitespace-nowrap transition-colors',
                providerFilter === prov
                  ? isLight ? 'bg-amber-100 text-amber-800' : 'bg-[#1A9E7A]/20 text-[#1A9E7A]'
                  : isLight ? 'bg-stone-100 text-stone-500 hover:bg-stone-200' : 'bg-white/5 text-slate-400 hover:bg-white/10'
              )}
            >
              {prov} ({count})
            </button>
          ))}
        </div>

        {/* Results count */}
        <div className={cn(
          'px-3 py-1.5 text-[10px] flex-shrink-0 border-b',
          isLight ? 'text-stone-400 border-stone-100' : 'text-slate-500 border-slate-800/50'
        )}>
          {filtered.length} model{filtered.length !== 1 ? 's' : ''}
          {query && ` matching "${query}"`}
        </div>

        {/* Model list grouped by provider */}
        <div className="overflow-y-auto flex-1 min-h-0">
          {filtered.length === 0 ? (
            <p className={cn('px-3 py-6 text-sm text-center', isLight ? 'text-stone-500' : 'text-slate-500')}>
              No models found
            </p>
          ) : (
            [...grouped.entries()].map(([provider, provModels]) => (
              <div key={provider}>
                {/* Provider group header (only when showing All) */}
                {!providerFilter && (
                  <div className={cn(
                    'px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider sticky top-0',
                    isLight ? 'bg-stone-50 text-stone-500 border-b border-stone-100' : 'bg-[#0e1319] text-slate-500 border-b border-slate-800/50'
                  )}>
                    {provider} ({provModels.length})
                  </div>
                )}
                {provModels.map(model => {
                  const selected = isModelSelected(model.model_id)
                  return (
                    <button
                      key={model.model_id}
                      onClick={() => { if (!selected) addModel(model) }}
                      disabled={selected}
                      className={cn(
                        'w-full flex items-center gap-2 px-3 py-1.5 text-left transition-colors',
                        selected
                          ? isLight ? 'bg-stone-50/50' : 'bg-slate-800/20'
                          : isLight ? 'hover:bg-amber-50/50' : 'hover:bg-white/5',
                      )}
                    >
                      <div className="min-w-0 flex-1">
                        <p className={cn(
                          'text-xs truncate',
                          selected
                            ? isLight ? 'text-stone-400' : 'text-slate-600'
                            : isLight ? 'text-stone-900' : 'text-white'
                        )}>
                          {model.model_name || model.model_id}
                        </p>
                      </div>
                      {selected ? (
                        <span className={cn(
                          'text-[10px] flex-shrink-0 px-1.5 py-0.5 rounded',
                          isLight ? 'bg-emerald-50 text-emerald-600' : 'bg-emerald-500/10 text-emerald-400'
                        )}>
                          Added
                        </span>
                      ) : (
                        <Plus className={cn(
                          'h-3.5 w-3.5 flex-shrink-0 opacity-0 group-hover:opacity-100',
                          isLight ? 'text-stone-400' : 'text-slate-500'
                        )} />
                      )}
                    </button>
                  )
                })}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

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
        Select models from the Model Explorer to compare their features,
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
  const [cardsCollapsed, setCardsCollapsed] = useState(false)

  const { selectedModels, removeModel, clearAll, addModel, isModelSelected } = useComparisonStore()
  const { models, getPricingForModel } = useModels()

  // Sort models by provider then by name for consistent display
  const sortedModels = useMemo(() =>
    [...selectedModels].sort((a, b) => {
      const provCmp = (a.model.model_provider || '').localeCompare(b.model.model_provider || '')
      if (provCmp !== 0) return provCmp
      return (a.model.model_name || a.model.model_id).localeCompare(b.model.model_name || b.model.model_id)
    }),
    [selectedModels]
  )

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
              Comparing {selectedModels.length} models
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <AddModelSearch
            isLight={isLight}
            models={models}
            addModel={addModel}
            isModelSelected={isModelSelected}
          />
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

      {/* Selected Models — collapsible */}
      <div className={cn(
        'rounded-lg border overflow-hidden',
        isLight
          ? 'bg-white/50 border-stone-200/80'
          : 'bg-[#131a24]/50 border-slate-700/40'
      )}>
        <button
          onClick={() => setCardsCollapsed(prev => !prev)}
          className={cn(
            'w-full px-4 py-2 flex items-center justify-between cursor-pointer',
            'transition-colors',
            isLight
              ? 'hover:bg-stone-50'
              : 'hover:bg-slate-800/30'
          )}
        >
          <div className="flex items-center gap-2">
            <span className={cn(
              'text-xs font-semibold',
              isLight ? 'text-stone-700' : 'text-slate-300'
            )}>
              Selected Models
            </span>
            <span className={cn(
              'text-xs',
              isLight ? 'text-stone-400' : 'text-slate-500'
            )}>
              ({selectedModels.length})
            </span>
            {cardsCollapsed && (
              <div className="flex items-center gap-1 ml-1">
                {sortedModels.slice(0, 6).map(({ model }) => (
                  <Badge
                    key={model.model_id}
                    variant="secondary"
                    className={cn(
                      'text-[9px] py-0 px-1.5',
                      isLight ? 'bg-stone-100 text-stone-600' : 'bg-slate-700/50 text-slate-400'
                    )}
                  >
                    {model.model_name || model.model_id}
                  </Badge>
                ))}
                {sortedModels.length > 6 && (
                  <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                    +{sortedModels.length - 6}
                  </span>
                )}
              </div>
            )}
          </div>
          {cardsCollapsed
            ? <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
            : <ChevronUp className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
          }
        </button>

        {!cardsCollapsed && (
          <div className={cn(
            'px-3 pb-3 pt-1 grid gap-3',
            'grid-cols-2 sm:grid-cols-3',
            sortedModels.length >= 4 && 'lg:grid-cols-4',
            sortedModels.length >= 5 && 'xl:grid-cols-5',
            sortedModels.length >= 6 && '2xl:grid-cols-6'
          )}>
            {sortedModels.map(({ model }) => (
              <ComparisonCard
                key={model.model_id}
                model={model}
                onRemove={removeModel}
              />
            ))}
          </div>
        )}
      </div>

      {/* Comparison Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList>
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
            selectedModels={sortedModels}
            getPricingForModel={getPricingForModel}
            isLight={isLight}
          />
        </TabsContent>

        <TabsContent value="pricing">
          <PricingTab
            selectedModels={sortedModels}
            getPricingForModel={getPricingForModel}
            isLight={isLight}
          />
        </TabsContent>

        <TabsContent value="availability">
          <AvailabilityTab
            selectedModels={sortedModels}
            isLight={isLight}
          />
        </TabsContent>

        <TabsContent value="specs">
          <TechSpecsTab
            selectedModels={sortedModels}
            getPricingForModel={getPricingForModel}
            isLight={isLight}
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}
