import { ModelCard } from './ModelCard'
import { cn } from '@/lib/utils'

// Grid column class mappings for different column counts
// On mobile (<640px), always show 1 column for better readability
const gridClasses = {
  3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
  4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
  5: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5',
  6: 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6',
}

export function ModelGrid({
  models,
  onViewDetails,
  onCompare,
  onToggleFavorite,
  favorites = [],
  columnsPerRow = 4,
  preferredRegion = 'us-east-1',
  getPricingForModel,
}) {
  if (!models || models.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-slate-400">
        <p className="text-lg">No models found</p>
        <p className="text-sm mt-1">Try adjusting your filters</p>
      </div>
    )
  }

  const gridClass = gridClasses[columnsPerRow] || gridClasses[4]

  return (
    <div className={cn('grid gap-4', gridClass)}>
      {models.map(model => (
        <ModelCard
          key={model.model_id}
          model={model}
          onViewDetails={onViewDetails}
          onCompare={onCompare}
          onToggleFavorite={onToggleFavorite}
          isFavorite={favorites.includes(model.model_id)}
          preferredRegion={preferredRegion}
          getPricingForModel={getPricingForModel}
        />
      ))}
    </div>
  )
}
