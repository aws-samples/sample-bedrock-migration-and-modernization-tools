import { useState, useMemo, useEffect } from 'react'
import { Loader2, LayoutGrid, Star, ArrowRight } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ModelGrid } from './ModelGrid'
import { ModelCardExpanded } from './ModelCardExpanded'
import { ModelFilters } from './ModelFilters'
import { Pagination } from './Pagination'
import { useModels } from '@/hooks/useModels'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useComparisonStore } from '@/stores/comparisonStore'
import { useFavoritesStore } from '@/stores/favoritesStore'
import { applyFilters, initialFilterState } from '@/utils/filters'
import { cn } from '@/lib/utils'

const gridColumnOptions = [
  { value: '3', label: '3 per row' },
  { value: '4', label: '4 per row' },
  { value: '5', label: '5 per row' },
  { value: '6', label: '6 per row' },
]

export function Favorites({ onNavigateToExplorer }) {
  const { models, providers, capabilities, useCases, customizations, languages, loading, error, getPricingForModel } = useModels()
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const { toggleModel, isModelSelected } = useComparisonStore()
  const { favoriteIds, toggleFavorite } = useFavoritesStore()

  const [filters, setFilters] = useState(initialFilterState)
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [columnsPerRow, setColumnsPerRow] = useState(4)
  const [selectedModel, setSelectedModel] = useState(null)

  const handleViewDetails = (model) => {
    setSelectedModel(model)
  }

  const handleToggleFavorite = (modelId) => {
    toggleFavorite(modelId)
  }

  // Filter to favorites first, then apply user filters
  const favoriteModels = useMemo(
    () => models.filter(m => favoriteIds.includes(m.model_id)),
    [models, favoriteIds]
  )

  const filteredModels = useMemo(
    () => applyFilters(favoriteModels, filters, getPricingForModel),
    [favoriteModels, filters, getPricingForModel]
  )

  // Paginate
  const totalPages = Math.ceil(filteredModels.length / pageSize)
  const paginatedModels = useMemo(() => {
    const start = (currentPage - 1) * pageSize
    return filteredModels.slice(start, start + pageSize)
  }, [filteredModels, currentPage, pageSize])

  const handleFiltersChange = (newFilters) => {
    setFilters(newFilters)
    setCurrentPage(1)
  }

  const handlePageChange = (page) => {
    setCurrentPage(page)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handlePageSizeChange = (size) => {
    setPageSize(size)
    setCurrentPage(1)
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64">
        <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
        <p className={cn('mt-4', isLight ? 'text-slate-500' : 'text-slate-400')}>Loading models...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-red-500">
        <p className="text-lg font-medium">Failed to load models</p>
        <p className="text-sm mt-1">{error.message}</p>
      </div>
    )
  }

  // Empty state — no favorites yet
  if (favoriteIds.length === 0) {
    return (
      <div>
        <div className="mb-4 sm:mb-6">
          <h1 className={cn('text-xl sm:text-2xl font-bold', isLight ? 'text-slate-900' : 'text-white')}>Favorites</h1>
          <p className={cn('mt-1 text-sm sm:text-base', isLight ? 'text-slate-500' : 'text-slate-400')}>
            Your favorited Bedrock models
          </p>
        </div>
        <div className={cn(
          'flex flex-col items-center justify-center py-24 rounded-2xl border backdrop-blur-xl',
          isLight
            ? 'bg-white/70 border-stone-200/60'
            : 'bg-white/[0.03] border-white/[0.06]',
        )}>
          <Star className={cn('h-12 w-12 mb-4', isLight ? 'text-stone-300' : 'text-[#4a4d54]')} />
          <p className={cn('text-lg font-medium mb-1', isLight ? 'text-stone-700' : 'text-[#e4e5e7]')}>
            No favorites yet
          </p>
          <p className={cn('text-sm mb-6', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
            Star models in the Model Explorer to add them here
          </p>
          <button
            onClick={onNavigateToExplorer}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              isLight
                ? 'bg-amber-700 text-white hover:bg-amber-800'
                : 'bg-emerald-500 text-white hover:bg-emerald-600',
            )}
          >
            Browse Models
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <div className="flex items-center gap-3">
          <h1 className={cn('text-xl sm:text-2xl font-bold', isLight ? 'text-slate-900' : 'text-white')}>Favorites</h1>
          <span className={cn(
            'text-xs font-medium px-2 py-0.5 rounded-full',
            isLight ? 'bg-amber-100 text-amber-700' : 'bg-emerald-500/15 text-emerald-400',
          )}>
            {favoriteIds.length}
          </span>
        </div>
        <p className={cn('mt-1 text-sm sm:text-base', isLight ? 'text-slate-500' : 'text-slate-400')}>
          Your favorited Bedrock models
        </p>
      </div>

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
          models={favoriteModels}
        />
      </div>

      {/* Results bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-4">
        <div className={cn('text-sm', isLight ? 'text-slate-500' : 'text-slate-400')}>
          {filteredModels.length === favoriteModels.length ? (
            <span>Showing all {favoriteModels.length} favorite{favoriteModels.length !== 1 ? 's' : ''}</span>
          ) : (
            <span>
              Found {filteredModels.length} of {favoriteModels.length} favorite{favoriteModels.length !== 1 ? 's' : ''}
              {filters.searchQuery && <span className="hidden sm:inline"> matching "{filters.searchQuery}"</span>}
            </span>
          )}
        </div>

        <div className="hidden md:flex items-center gap-2">
          <LayoutGrid className="h-4 w-4 text-slate-400" />
          <Select
            value={columnsPerRow.toString()}
            onValueChange={(v) => setColumnsPerRow(parseInt(v))}
          >
            <SelectTrigger className="w-[120px] h-8 text-xs">
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

      {/* Model grid */}
      <ModelGrid
        models={paginatedModels}
        onViewDetails={handleViewDetails}
        onCompare={() => {}}
        onToggleFavorite={handleToggleFavorite}
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
        onToggleFavorite={handleToggleFavorite}
        isFavorite={selectedModel ? favoriteIds.includes(selectedModel.model_id) : false}
        onToggleCompare={toggleModel}
        isInComparison={selectedModel ? isModelSelected(selectedModel.model_id) : false}
        getPricingForModel={getPricingForModel}
        preferredRegion={filters.primaryRegion}
      />
    </div>
  )
}
