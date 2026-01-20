import { Check, X, Globe, MapPin, Trophy, Map } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { awsRegions } from '@/utils/filters'
import { RegionMap } from '../RegionMap'

// Provider colors
const providerColors = {
  Amazon: 'bg-[#FF9900]',
  Anthropic: 'bg-[#D4A27F]',
  Meta: 'bg-[#0082FB]',
  Mistral: 'bg-[#F54E42]',
  Cohere: 'bg-[#39594D]',
  'AI21 Labs': 'bg-[#6C5CE7]',
  AI21: 'bg-[#6C5CE7]',
  'Stability AI': 'bg-[#7C5CFF]',
  Stability: 'bg-[#7C5CFF]',
  Luma: 'bg-[#6366F1]',
  default: 'bg-slate-500',
}

// Group regions by geography
const geoGroups = {
  US: { label: 'United States', regions: [] },
  EU: { label: 'Europe', regions: [] },
  AP: { label: 'Asia Pacific', regions: [] },
  CA: { label: 'Canada', regions: [] },
  SA: { label: 'South America', regions: [] },
  ME: { label: 'Middle East', regions: [] },
  AF: { label: 'Africa', regions: [] },
}

// Populate geo groups
awsRegions.forEach(r => {
  if (geoGroups[r.geo]) {
    geoGroups[r.geo].regions.push(r)
  }
})

// Create region lookup map
const regionLookup = {}
awsRegions.forEach(r => {
  regionLookup[r.value] = r
})

export function AvailabilityTab({ selectedModels, isLight }) {
  // Get all unique regions across selected models
  const allRegions = new Set()
  selectedModels.forEach(({ model }) => {
    (model.regions_available || []).forEach(r => allRegions.add(r))
  })

  // Calculate coverage for each model
  const modelCoverage = selectedModels.map(({ model }) => {
    const regions = model.regions_available || []
    return {
      model,
      regions,
      count: regions.length,
      coverage: allRegions.size > 0 ? (regions.length / allRegions.size * 100).toFixed(0) : 0,
    }
  })

  // Find max coverage
  const maxCount = Math.max(...modelCoverage.map(m => m.count))

  // Find common regions (available in all selected models)
  const commonRegions = [...allRegions].filter(region =>
    selectedModels.every(({ model }) => (model.regions_available || []).includes(region))
  )

  return (
    <div className="mt-4 space-y-4">
      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className={cn(
          'p-4 rounded-lg border',
          isLight
            ? 'bg-white/80 border-stone-200/80'
            : 'bg-[#161d26]/80 border-slate-700/50'
        )}>
          <div className="flex items-center gap-2 mb-2">
            <Globe className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
            <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-500')}>
              Total Unique Regions
            </span>
          </div>
          <p className={cn('text-2xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {allRegions.size}
          </p>
        </div>

        <div className={cn(
          'p-4 rounded-lg border',
          isLight
            ? 'bg-emerald-50/50 border-emerald-200'
            : 'bg-emerald-500/10 border-emerald-500/30'
        )}>
          <div className="flex items-center gap-2 mb-2">
            <MapPin className="h-4 w-4 text-emerald-500" />
            <span className={cn('text-xs', isLight ? 'text-emerald-700' : 'text-emerald-400')}>
              Common Regions
            </span>
          </div>
          <p className={cn('text-2xl font-bold', 'text-emerald-600')}>
            {commonRegions.length}
          </p>
        </div>

        <div className={cn(
          'p-4 rounded-lg border',
          isLight
            ? 'bg-white/80 border-stone-200/80'
            : 'bg-[#161d26]/80 border-slate-700/50'
        )}>
          <div className="flex items-center gap-2 mb-2">
            <Trophy className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
            <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-500')}>
              Best Coverage
            </span>
          </div>
          <p className={cn('text-2xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            {maxCount} regions
          </p>
        </div>
      </div>

      {/* Interactive Map */}
      <div className={cn(
        'rounded-lg border overflow-hidden',
        isLight
          ? 'bg-white/80 border-stone-200/80'
          : 'bg-[#161d26]/80 border-slate-700/50'
      )}>
        <div className={cn(
          'px-4 py-3 border-b flex items-center gap-2',
          isLight ? 'bg-stone-50 border-stone-200' : 'bg-slate-800/50 border-slate-700'
        )}>
          <Map className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <h3 className={cn(
            'font-semibold text-sm',
            isLight ? 'text-stone-900' : 'text-white'
          )}>
            Global Availability Map
          </h3>
        </div>
        <RegionMap selectedModels={selectedModels} isLight={isLight} />
      </div>

      {/* Model Coverage Bars */}
      <div className={cn(
        'p-4 rounded-lg border',
        isLight
          ? 'bg-white/80 border-stone-200/80'
          : 'bg-[#161d26]/80 border-slate-700/50'
      )}>
        <h3 className={cn(
          'font-semibold mb-4',
          isLight ? 'text-stone-900' : 'text-white'
        )}>
          Regional Coverage
        </h3>
        <div className="space-y-3">
          {modelCoverage.map(({ model, count, coverage }, idx) => (
            <div key={model.model_id}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <Badge className={cn(
                    'text-[10px]',
                    isLight ? 'text-[#faf9f5]' : 'text-white',
                    providerColors[model.model_provider] || providerColors.default
                  )}>
                    {model.model_provider}
                  </Badge>
                  <span className={cn(
                    'text-sm font-medium truncate max-w-[200px]',
                    isLight ? 'text-stone-700' : 'text-slate-300'
                  )}>
                    {model.model_name || model.model_id}
                  </span>
                  {count === maxCount && (
                    <Trophy className={cn(
                      'h-3.5 w-3.5 flex-shrink-0',
                      isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
                    )} />
                  )}
                </div>
                <span className={cn(
                  'text-sm font-medium',
                  isLight ? 'text-stone-600' : 'text-slate-400'
                )}>
                  {count} regions ({coverage}%)
                </span>
              </div>
              <div className={cn(
                'h-2 rounded-full overflow-hidden',
                isLight ? 'bg-stone-200' : 'bg-slate-700'
              )}>
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-500',
                    count === maxCount
                      ? isLight ? 'bg-amber-500' : 'bg-[#1A9E7A]'
                      : isLight ? 'bg-stone-400' : 'bg-slate-500'
                  )}
                  style={{ width: `${coverage}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Region Table by Geography */}
      {Object.entries(geoGroups).map(([geo, { label, regions }]) => {
        const relevantRegions = regions.filter(r => allRegions.has(r.value))
        if (relevantRegions.length === 0) return null

        return (
          <div
            key={geo}
            className={cn(
              'rounded-lg border overflow-hidden',
              isLight
                ? 'bg-white/80 border-stone-200/80'
                : 'bg-[#161d26]/80 border-slate-700/50'
            )}
          >
            <div className={cn(
              'px-4 py-2 border-b',
              isLight ? 'bg-stone-100/80 border-stone-200' : 'bg-slate-800/50 border-slate-700'
            )}>
              <h3 className={cn(
                'font-semibold text-sm',
                isLight ? 'text-stone-900' : 'text-white'
              )}>
                {label}
              </h3>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className={cn(
                    'border-b',
                    isLight ? 'border-stone-200' : 'border-slate-700'
                  )}>
                    <th className={cn(
                      'px-4 py-2 text-left text-xs font-medium',
                      isLight ? 'text-stone-600' : 'text-slate-400'
                    )}>
                      Region
                    </th>
                    {selectedModels.map(({ model }) => (
                      <th
                        key={model.model_id}
                        className={cn(
                          'px-2 py-2 text-center text-xs font-medium',
                          isLight ? 'text-stone-600' : 'text-slate-400'
                        )}
                      >
                        <span className="truncate block max-w-[100px]" title={model.model_name || model.model_id}>
                          {(model.model_name || model.model_id).split(/[-_]/).slice(-2).join(' ')}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {relevantRegions.map(region => {
                    const isCommon = commonRegions.includes(region.value)
                    return (
                      <tr
                        key={region.value}
                        className={cn(
                          'border-b last:border-b-0',
                          isCommon
                            ? isLight ? 'bg-emerald-50/50' : 'bg-emerald-500/5'
                            : '',
                          isLight ? 'border-stone-100' : 'border-slate-800'
                        )}
                      >
                        <td className="px-4 py-2">
                          <div>
                            <p className={cn(
                              'text-sm font-medium',
                              isLight ? 'text-stone-900' : 'text-white'
                            )}>
                              {region.label}
                            </p>
                            <p className={cn(
                              'text-xs font-mono',
                              isLight ? 'text-stone-500' : 'text-slate-500'
                            )}>
                              {region.value}
                            </p>
                          </div>
                        </td>
                        {selectedModels.map(({ model }) => {
                          const available = (model.regions_available || []).includes(region.value)
                          return (
                            <td key={model.model_id} className="px-2 py-2 text-center">
                              {available ? (
                                <Check className="h-4 w-4 text-emerald-500 mx-auto" />
                              ) : (
                                <X className="h-4 w-4 text-red-400/40 mx-auto" />
                              )}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )
      })}
    </div>
  )
}
