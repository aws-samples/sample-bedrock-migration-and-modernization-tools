import { useState, useMemo } from 'react'
import { Loader2, LayoutGrid, ArrowUpDown } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ModelOverview } from './ModelOverview'
import { ModelGrid } from './ModelGrid'
import { ModelCardExpanded } from './ModelCardExpanded'
import { ModelFilters } from './ModelFilters'
import { Pagination } from './Pagination'
import { useModels } from '@/hooks/useModels'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useComparisonStore } from '@/stores/comparisonStore'
import { useFavoritesStore } from '@/stores/favoritesStore'
import { applyFilters, initialFilterState, sortOptions, sortModels } from '@/utils/filters'
import { cn } from '@/lib/utils'

const gridColumnOptions = [
  { value: '3', label: '3 per row' },
  { value: '4', label: '4 per row' },
  { value: '5', label: '5 per row' },
  { value: '6', label: '6 per row' },
]

export function ModelExplorer() {
  const { models, providers, capabilities, useCases, customizations, languages, stats, loading, error, getPricingForModel } = useModels()
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const { toggleModel, isModelSelected } = useComparisonStore()
  const { favoriteIds, toggleFavorite } = useFavoritesStore()

  // Local state
  const [filters, setFilters] = useState(initialFilterState)
  const [sortBy, setSortBy] = useState('newest')
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [columnsPerRow, setColumnsPerRow] = useState(4)
  const [selectedModel, setSelectedModel] = useState(null)

  // Filter and sort models
  const filteredModels = useMemo(() => {
    const filtered = applyFilters(models, filters, getPricingForModel)
    return sortModels(filtered, sortBy, getPricingForModel, filters.primaryRegion)
  }, [models, filters, sortBy, getPricingForModel])

  // Paginate
  const totalPages = Math.ceil(filteredModels.length / pageSize)
  const paginatedModels = useMemo(() => {
    const start = (currentPage - 1) * pageSize
    return filteredModels.slice(start, start + pageSize)
  }, [filteredModels, currentPage, pageSize])

  // Handlers
  const handleFiltersChange = (newFilters) => {
    setFilters(newFilters)
    setCurrentPage(1)
  }

  const handleViewDetails = (model) => {
    setSelectedModel(model)
  }

  const handleCompare = (model) => {
    console.log('Compare:', model.model_id)
  }

  const handlePageChange = (page) => {
    setCurrentPage(page)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handlePageSizeChange = (size) => {
    setPageSize(size)
    setCurrentPage(1)
  }

  // Loading state
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64">
        <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
        <p className={cn('mt-4', isLight ? 'text-slate-500' : 'text-slate-400')}>Loading models...</p>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-red-500">
        <p className="text-lg font-medium">Failed to load models</p>
        <p className="text-sm mt-1">{error.message}</p>
      </div>
    )
  }

  return (
    <div>
      {/* Page header */}
      <div className="mb-4 sm:mb-6">
        <h1 className={cn('text-xl sm:text-2xl font-bold', isLight ? 'text-slate-900' : 'text-white')}>Model Explorer</h1>
        <p className={cn('mt-1 text-sm sm:text-base', isLight ? 'text-slate-500' : 'text-slate-400')}>
          Browse and explore Amazon Bedrock foundation models
        </p>
      </div>

      {/* Overview stats */}
      <ModelOverview stats={stats} />

      {/* Filters */}
      <div className="mb-6">
        <ModelFilters
          filters={filters}
          onFiltersChange={handleFiltersChange}
          availableProviders={providers}
          availableCapabilities={capabilities}
          availableUseCases={useCases}
          availableCustomizations={customizations}
          availableLanguages={languages}
          models={models}
        />
      </div>

      {/* Results bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-4">
        <div className={cn('text-sm', isLight ? 'text-slate-500' : 'text-slate-400')}>
          {filteredModels.length === models.length ? (
            <span>Showing all {models.length} models</span>
          ) : (
            <span>
              Found {filteredModels.length} model{filteredModels.length !== 1 ? 's' : ''}
              {filters.searchQuery && <span className="hidden sm:inline"> matching "{filters.searchQuery}"</span>}
            </span>
          )}
        </div>

        {/* Sort and grid controls */}
        <div className="flex items-center gap-3">
          {/* Sort dropdown */}
          <div className="flex items-center gap-2">
            <ArrowUpDown className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-400')} />
            <Select value={sortBy} onValueChange={(v) => { setSortBy(v); setCurrentPage(1) }}>
              <SelectTrigger className="w-[180px] h-9 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {sortOptions.map(opt => (
                  <SelectItem key={opt.value} value={opt.value} className="text-xs">
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Grid columns selector - hidden on mobile */}
          <div className="hidden md:flex items-center gap-2">
            <LayoutGrid className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-400')} />
            <Select
              value={columnsPerRow.toString()}
              onValueChange={(v) => setColumnsPerRow(parseInt(v))}
            >
              <SelectTrigger className="w-[120px] h-9 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {gridColumnOptions.map(opt => (
                  <SelectItem key={opt.value} value={opt.value} className="text-xs">
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Model grid */}
      <ModelGrid
        models={paginatedModels}
        onViewDetails={handleViewDetails}
        onCompare={handleCompare}
        onToggleFavorite={toggleFavorite}
        favorites={favoriteIds}
        columnsPerRow={columnsPerRow}
        preferredRegion={filters.primaryRegion}
        getPricingForModel={getPricingForModel}
      />

      {/* Pagination */}
      {filteredModels.length > 0 && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          pageSize={pageSize}
          totalItems={filteredModels.length}
          onPageChange={handlePageChange}
          onPageSizeChange={handlePageSizeChange}
        />
      )}

      {/* Model details modal */}
      <ModelCardExpanded
        model={selectedModel}
        open={!!selectedModel}
        onOpenChange={(open) => !open && setSelectedModel(null)}
        onToggleFavorite={toggleFavorite}
        isFavorite={selectedModel ? favoriteIds.includes(selectedModel.model_id) : false}
        onToggleCompare={toggleModel}
        isInComparison={selectedModel ? isModelSelected(selectedModel.model_id) : false}
        getPricingForModel={getPricingForModel}
        preferredRegion={filters.primaryRegion}
      />
    </div>
  )
}
