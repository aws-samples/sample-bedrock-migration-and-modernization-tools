import { useState } from 'react'
import { Check, Globe, MapPin, Trophy, Map, ChevronDown, ChevronRight, Filter, Maximize2, Minimize2, Cpu } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { awsRegions } from '@/utils/filters'
import { RegionMap } from '../RegionMap'
import { providerColorClasses } from '@/config/constants'

const providerColors = providerColorClasses

// Group regions by geography
const geoGroups = [
  { key: 'US', label: 'United States' },
  { key: 'EU', label: 'Europe' },
  { key: 'AP', label: 'Asia Pacific' },
  { key: 'CA', label: 'Canada' },
  { key: 'SA', label: 'South America' },
  { key: 'ME', label: 'Middle East' },
  { key: 'AF', label: 'Africa' },
]

const regionsByGeo = {}
geoGroups.forEach(g => { regionsByGeo[g.key] = [] })
awsRegions.forEach(r => {
  if (regionsByGeo[r.geo]) {
    regionsByGeo[r.geo].push(r)
  }
})

// Extract friendly name from label like "N. Virginia (us-east-1)" -> "N. Virginia"
function friendlyName(label) {
  return label.replace(/\s*\(.*\)$/, '')
}

// Helper to get all regions for a model (on-demand + CRIS + Mantle)
function getAllModelRegions(model) {
  const onDemand = model.on_demand_regions || []
  const cris = model.cross_region_inference?.source_regions || []
  const mantle = model.mantle_inference?.mantle_regions || []
  return [...new Set([...onDemand, ...cris, ...mantle])]
}

export function AvailabilityTab({ selectedModels, isLight }) {
  const [collapsedGeos, setCollapsedGeos] = useState(new Set())
  const [filterMode, setFilterMode] = useState('all')
  const [mapFullscreen, setMapFullscreen] = useState(false)

  // Check if any model is Mantle-only
  const mantleOnlyModels = selectedModels.filter(({ model }) => model.mantle_only)
  const hasMantleOnlyModels = mantleOnlyModels.length > 0

  const allRegions = new Set()
  selectedModels.forEach(({ model }) => {
    getAllModelRegions(model).forEach(r => allRegions.add(r))
  })

  const modelCoverage = selectedModels.map(({ model }) => {
    const isMantleOnly = model.mantle_only
    const regions = getAllModelRegions(model)
    const mantleRegions = model.mantle_inference?.mantle_regions || []
    return {
      model,
      regions,
      count: regions.length,
      coverage: allRegions.size > 0 ? (regions.length / allRegions.size * 100).toFixed(0) : 0,
      isMantleOnly,
      mantleRegions,
    }
  })

  const maxCount = Math.max(...modelCoverage.map(m => m.count), 0)
  const bestModels = modelCoverage.filter(m => m.count === maxCount)

  const commonRegions = [...allRegions].filter(region =>
    selectedModels.every(({ model }) => getAllModelRegions(model).includes(region))
  )

  const diffRegions = [...allRegions].filter(region => {
    const available = selectedModels.filter(({ model }) => getAllModelRegions(model).includes(region))
    return available.length > 0 && available.length < selectedModels.length
  })

  const exclusiveCount = [...allRegions].filter(region => {
    const available = selectedModels.filter(({ model }) => getAllModelRegions(model).includes(region))
    return available.length === 1
  }).length

  const toggleGeo = (key) => {
    setCollapsedGeos(prev => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }

  // Build region groups for the table
  const tableGroups = []
  geoGroups.forEach(({ key, label }) => {
    const regions = regionsByGeo[key].filter(r => allRegions.has(r.value))
    if (regions.length === 0) return

    const filteredRegions = regions.filter(r => {
      if (filterMode === 'common') return commonRegions.includes(r.value)
      if (filterMode === 'diff') return diffRegions.includes(r.value)
      return true
    })

    if (filteredRegions.length === 0 && filterMode !== 'all') return

    tableGroups.push({
      key,
      label,
      totalCount: regions.length,
      filteredCount: filteredRegions.length,
      regions: filteredRegions,
    })
  })

  return (
    <div className="mt-4 space-y-3">
      {/* Mantle-only models notice */}
      {hasMantleOnlyModels && (
        <div className={cn(
          'rounded-lg p-3 border',
          isLight
            ? 'bg-violet-50 border-violet-200'
            : 'bg-violet-500/10 border border-violet-500/20'
        )}>
          <div className="flex items-center gap-2 mb-1">
            <Cpu className={cn('h-4 w-4', isLight ? 'text-violet-600' : 'text-violet-400')} />
            <span className={cn('text-sm font-medium', isLight ? 'text-violet-700' : 'text-violet-400')}>
              Mantle-Only Models in Comparison
            </span>
          </div>
          <p className={cn('text-xs', isLight ? 'text-violet-600' : 'text-violet-300')}>
            {mantleOnlyModels.map(({ model }) => model.model_name || model.model_id).join(', ')} {mantleOnlyModels.length === 1 ? 'is' : 'are'} available 
            exclusively via Mantle Inference and {mantleOnlyModels.length === 1 ? 'does' : 'do'} not have standard AWS region availability. 
            See the Mantle Inference section below for {mantleOnlyModels.length === 1 ? 'its' : 'their'} regional coverage.
          </p>
        </div>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <SummaryCard
          icon={<Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />}
          label="Total Regions"
          value={allRegions.size}
          isLight={isLight}
        />
        <SummaryCard
          icon={<MapPin className="h-3.5 w-3.5 text-emerald-500" />}
          label="Common"
          value={commonRegions.length}
          isLight={isLight}
          variant="emerald"
        />
        <SummaryCard
          icon={<Filter className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-amber-400')} />}
          label="Exclusive"
          value={exclusiveCount}
          isLight={isLight}
          variant="amber"
        />
        <SummaryCard
          icon={<Trophy className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />}
          label="Best Coverage"
          value={maxCount}
          subtitle={bestModels.map(m => m.model.model_name || m.model.model_id).join(', ')}
          isLight={isLight}
        />
      </div>

      {/* Map (left) + Coverage bars (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Map - takes 2/3 on left */}
        <div className={cn(
          'lg:col-span-2 rounded-lg border overflow-hidden relative backdrop-blur-xl',
          isLight
            ? 'bg-white/70 border-stone-200/60'
            : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <div className={cn(
            'px-3 py-1.5 border-b flex items-center justify-between',
            isLight ? 'bg-stone-50/60 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
          )}>
            <div className="flex items-center gap-2">
              <Map className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <h3 className={cn('font-semibold text-xs', isLight ? 'text-stone-900' : 'text-white')}>
                Global Availability
              </h3>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className={cn(
                'h-6 w-6 p-0',
                isLight ? 'text-stone-400 hover:text-stone-600' : 'text-slate-500 hover:text-slate-300'
              )}
              onClick={() => setMapFullscreen(true)}
              title="Expand map"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </Button>
          </div>
          <RegionMap selectedModels={selectedModels} isLight={isLight} height="350px" />
        </div>

        {/* Coverage bars - 1/3 on right */}
        <div className={cn(
          'px-4 py-3 rounded-lg border backdrop-blur-xl',
          isLight
            ? 'bg-white/70 border-stone-200/60'
            : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <h3 className={cn(
            'font-semibold text-xs mb-3',
            isLight ? 'text-stone-900' : 'text-white'
          )}>
            Regional Coverage
          </h3>
          <div className="space-y-3">
            {[...modelCoverage]
              .sort((a, b) => {
                // Sort Mantle-only models to the end
                if (a.isMantleOnly && !b.isMantleOnly) return 1
                if (!a.isMantleOnly && b.isMantleOnly) return -1
                return b.count - a.count
              })
              .map(({ model, count, coverage, isMantleOnly, mantleRegions }) => (
              <div key={model.model_id}>
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <Badge className={cn(
                      'text-[9px] px-1.5 py-0 flex-shrink-0',
                      isLight ? 'text-[#faf9f5]' : 'text-white',
                      providerColors[model.model_provider] || providerColors.default
                    )}>
                      {model.model_provider}
                    </Badge>
                    <span className={cn(
                      'text-xs font-medium truncate',
                      isLight ? 'text-stone-700' : 'text-slate-300'
                    )}>
                      {model.model_name || model.model_id}
                    </span>
                    {!isMantleOnly && count === maxCount && maxCount > 0 && (
                      <Trophy className={cn(
                        'h-3 w-3 flex-shrink-0',
                        isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
                      )} />
                    )}
                    {isMantleOnly && (
                      <span className={cn(
                        'inline-flex items-center gap-0.5 px-1 py-0 rounded text-[9px] font-medium flex-shrink-0',
                        isLight
                          ? 'bg-violet-100 text-violet-600'
                          : 'bg-violet-500/15 text-violet-400'
                      )}>
                        <Cpu className="h-2.5 w-2.5" />
                        Mantle
                      </span>
                    )}
                  </div>
                  <span className={cn(
                    'text-xs font-semibold tabular-nums flex-shrink-0 ml-2',
                    isMantleOnly
                      ? isLight ? 'text-violet-600' : 'text-violet-400'
                      : isLight ? 'text-stone-600' : 'text-slate-400'
                  )}>
                    {count}
                    {!isMantleOnly && (
                      <span className={cn('font-normal ml-0.5', isLight ? 'text-stone-400' : 'text-slate-500')}>
                        ({coverage}%)
                      </span>
                    )}
                  </span>
                </div>
                <div className={cn(
                  'h-2 rounded-full overflow-hidden',
                  isLight ? 'bg-stone-200' : 'bg-white/[0.06]'
                )}>
                  {isMantleOnly ? (
                    // For Mantle-only models, show a violet bar based on their Mantle regions
                    <div
                      className={cn(
                        'h-full rounded-full transition-all duration-500',
                        isLight ? 'bg-violet-400' : 'bg-violet-500'
                      )}
                      style={{ width: mantleRegions.length > 0 ? '100%' : '0%' }}
                    />
                  ) : (
                    <div
                      className={cn(
                        'h-full rounded-full transition-all duration-500',
                        count === maxCount && maxCount > 0
                          ? isLight ? 'bg-amber-500' : 'bg-[#1A9E7A]'
                          : isLight ? 'bg-stone-400' : 'bg-slate-500'
                      )}
                      style={{ width: `${coverage}%` }}
                    />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Fullscreen map overlay */}
      {mapFullscreen && (
        <div className="fixed inset-0 z-50 flex flex-col" style={{ backgroundColor: isLight ? '#fafaf9' : '#0d1117' }}>
          <div className={cn(
            'px-4 py-2.5 border-b flex items-center justify-between flex-shrink-0',
            isLight ? 'bg-white/90 border-stone-200' : 'bg-slate-950/90 border-white/[0.06] backdrop-blur-xl'
          )}>
            <div className="flex items-center gap-2">
              <Map className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <h3 className={cn('font-semibold text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                Global Availability
              </h3>
              <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                {selectedModels.length} models
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-7 gap-1.5"
              onClick={() => setMapFullscreen(false)}
            >
              <Minimize2 className="h-3.5 w-3.5" />
              Close
            </Button>
          </div>
          <div className="flex-1 overflow-hidden">
            <RegionMap selectedModels={selectedModels} isLight={isLight} height="calc(100vh - 45px)" />
          </div>
        </div>
      )}

      {/* CRIS Coverage — collapsible, column format */}
      {selectedModels.some(({ model }) => model.cross_region_inference?.supported) && (
        <CrisSection selectedModels={selectedModels} isLight={isLight} />
      )}

      {/* Mantle Inference — collapsible, column format */}
      {selectedModels.some(({ model }) => model.mantle_inference?.supported || model.is_mantle) && (
        <MantleSection selectedModels={selectedModels} isLight={isLight} />
      )}

      {/* Region comparison table */}
      <div className={cn(
        'rounded-lg border overflow-hidden',
        isLight
          ? 'bg-white/80 border-stone-200/80'
          : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
      )}>
        {/* Table filter bar */}
        <div className={cn(
          'px-4 py-2.5 border-b flex items-center gap-2',
          isLight ? 'bg-stone-50/60 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
        )}>
          <span className={cn('text-sm font-medium mr-1', isLight ? 'text-stone-600' : 'text-slate-400')}>Show:</span>
          {[
            { key: 'all', label: 'All Regions' },
            { key: 'common', label: `Common (${commonRegions.length})` },
            { key: 'diff', label: `Differences (${diffRegions.length})` },
          ].map(({ key, label }) => (
            <Button
              key={key}
              variant="ghost"
              size="sm"
              className={cn(
                'h-7 px-3 text-xs rounded-md',
                filterMode === key
                  ? isLight
                    ? 'bg-amber-100 text-amber-800 hover:bg-amber-100'
                    : 'bg-[#1A9E7A]/20 text-[#1A9E7A] hover:bg-[#1A9E7A]/20'
                  : isLight
                    ? 'text-stone-500 hover:text-stone-700 hover:bg-stone-100'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.06]'
              )}
              onClick={() => setFilterMode(key)}
            >
              {label}
            </Button>
          ))}
        </div>

        <div className="overflow-auto max-h-[500px]">
          <table className="w-full">
            <thead className="sticky top-0 z-20">
              <tr className={cn(
                'border-b-2',
                isLight ? 'border-stone-200 bg-stone-50' : 'border-white/[0.06] bg-[#1a1b1e]'
              )}>
                <th className={cn(
                  'px-5 py-3 text-left text-sm font-semibold min-w-[200px] sticky left-0 z-30',
                  isLight ? 'text-stone-900 bg-stone-50' : 'text-white bg-[#1a1b1e]'
                )}>
                  Region
                </th>
                {selectedModels.map(({ model }) => (
                  <th
                    key={model.model_id}
                    className={cn(
                      'px-3 py-3 text-center font-semibold min-w-[120px]',
                      isLight ? 'text-stone-900' : 'text-white'
                    )}
                  >
                    <Badge className={cn(
                      'text-[9px] mb-1',
                      isLight ? 'text-[#faf9f5]' : 'text-white',
                      providerColors[model.model_provider] || providerColors.default
                    )}>
                      {model.model_provider}
                    </Badge>
                    <p className={cn(
                      'text-xs font-semibold line-clamp-2 max-w-[140px] mx-auto leading-tight',
                      isLight ? 'text-stone-700' : 'text-slate-300'
                    )}>
                      {model.model_name || model.model_id}
                    </p>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableGroups.map(group => (
                <GeoSection
                  key={group.key}
                  group={group}
                  collapsed={collapsedGeos.has(group.key)}
                  onToggle={() => toggleGeo(group.key)}
                  selectedModels={selectedModels}
                  commonRegions={commonRegions}
                  isLight={isLight}
                />
              ))}
              {tableGroups.length === 0 && (
                <tr>
                  <td
                    colSpan={1 + selectedModels.length}
                    className={cn(
                      'px-5 py-10 text-center text-sm',
                      isLight ? 'text-stone-400' : 'text-slate-500'
                    )}
                  >
                    No regions match this filter
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function SummaryCard({ icon, label, value, subtitle, isLight, variant }) {
  const bgClass = variant === 'emerald'
    ? isLight ? 'bg-emerald-50/50 border-emerald-200' : 'bg-emerald-500/10 border-emerald-500/30'
    : variant === 'amber'
    ? isLight ? 'bg-amber-50/50 border-amber-200' : 'bg-amber-500/10 border-amber-500/30'
    : isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'

  const valueClass = variant === 'emerald'
    ? 'text-emerald-600'
    : variant === 'amber'
    ? isLight ? 'text-amber-700' : 'text-amber-500'
    : isLight ? 'text-stone-900' : 'text-white'

  const labelClass = variant === 'emerald'
    ? isLight ? 'text-emerald-700' : 'text-emerald-400'
    : variant === 'amber'
    ? isLight ? 'text-amber-700' : 'text-amber-400'
    : isLight ? 'text-stone-500' : 'text-slate-500'

  return (
    <div className={cn('px-3 py-2.5 rounded-lg border', bgClass)}>
      <div className="flex items-center gap-1.5 mb-0.5">
        {icon}
        <span className={cn('text-[10px]', labelClass)}>{label}</span>
      </div>
      <p className={cn('text-lg font-bold', valueClass)}>{value}</p>
      {subtitle && (
        <p className={cn('text-[10px] truncate', isLight ? 'text-stone-400' : 'text-slate-500')}>
          {subtitle}
        </p>
      )}
    </div>
  )
}

function GeoSection({ group, collapsed, onToggle, selectedModels, commonRegions, isLight }) {
  return (
    <>
      {/* Collapsible geo header */}
      <tr
        className={cn(
          'border-b cursor-pointer select-none',
          isLight
            ? 'border-stone-200 bg-stone-100/80 hover:bg-stone-100'
            : 'border-white/[0.06] bg-white/[0.03] hover:bg-white/[0.05]'
        )}
        onClick={onToggle}
      >
        <td
          colSpan={1 + selectedModels.length}
          className={cn(
            'px-5 py-2.5 text-sm font-bold',
            isLight ? 'text-stone-800' : 'text-white'
          )}
        >
          <div className="flex items-center gap-2">
            {collapsed
              ? <ChevronRight className="h-3.5 w-3.5 flex-shrink-0" />
              : <ChevronDown className="h-3.5 w-3.5 flex-shrink-0" />
            }
            <span>{group.label}</span>
            <span className={cn(
              'text-xs font-normal',
              isLight ? 'text-stone-400' : 'text-slate-500'
            )}>
              {group.filteredCount} region{group.filteredCount !== 1 ? 's' : ''}
            </span>
          </div>
        </td>
      </tr>

      {/* Region rows */}
      {!collapsed && group.regions.map(region => {
        // Only consider non-Mantle-only models for common regions
        const nonMantleOnlyModels = selectedModels.filter(({ model }) => !model.mantle_only)
        const isCommon = nonMantleOnlyModels.length > 0 && commonRegions.includes(region.value)
        return (
          <tr
            key={region.value}
            className={cn(
              'border-b last:border-b-0',
              isCommon
                ? isLight ? 'bg-emerald-50/30' : 'bg-emerald-500/5'
                : '',
              isLight ? 'border-stone-100' : 'border-white/[0.04]'
            )}
          >
            <td className={cn(
              'px-5 py-2.5 sticky left-0 z-10',
              isCommon
                ? isLight ? 'bg-emerald-50/30' : 'bg-emerald-500/5'
                : isLight ? 'bg-white' : 'bg-[#1a1b1e]'
            )}>
              <div className="flex items-center gap-2">
                {isCommon && (
                  <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
                )}
                <div>
                  <span className={cn(
                    'text-sm font-medium',
                    isLight ? 'text-stone-900' : 'text-white'
                  )}>
                    {friendlyName(region.label)}
                  </span>
                  <span className={cn(
                    'text-xs font-mono ml-2',
                    isLight ? 'text-stone-400' : 'text-slate-500'
                  )}>
                    {region.value}
                  </span>
                </div>
              </div>
            </td>
            {selectedModels.map(({ model }) => {
              const available = getAllModelRegions(model).includes(region.value)
              return (
                <td key={model.model_id} className="px-3 py-2.5 text-center">
                  {available ? (
                    <Check className="h-4.5 w-4.5 text-emerald-500 mx-auto" />
                  ) : (
                    <span className={cn(
                      'text-sm',
                      isLight ? 'text-stone-300' : 'text-slate-600'
                    )}>--</span>
                  )}
                </td>
              )
            })}
          </tr>
        )
      })}
    </>
  )
}

function CrisSection({ selectedModels, isLight }) {
  const [expanded, setExpanded] = useState(true)

  const crisCount = selectedModels.filter(({ model }) => model.cross_region_inference?.supported).length

  // Build region lookup: code → { name, geo }
  const regionLookup = {}
  awsRegions.forEach(r => {
    regionLookup[r.value] = { name: friendlyName(r.label), geo: r.geo }
  })

  // Geo labels
  const geoLabels = { US: 'United States', EU: 'Europe', AP: 'Asia Pacific', CA: 'Canada', SA: 'South America', ME: 'Middle East', AF: 'Africa' }

  // Collect all CRIS source regions and group by geo
  const allSourceRegions = new Set()
  selectedModels.forEach(({ model }) => {
    (model.cross_region_inference?.source_regions || []).forEach(r => allSourceRegions.add(r))
  })

  const geoGrouped = {}
  allSourceRegions.forEach(region => {
    const info = regionLookup[region]
    const geo = info?.geo || 'Other'
    if (!geoGrouped[geo]) geoGrouped[geo] = []
    geoGrouped[geo].push(region)
  })
  // Sort regions within each geo
  Object.values(geoGrouped).forEach(arr => arr.sort())

  // Order geos consistently
  const geoOrder = ['US', 'EU', 'AP', 'CA', 'SA', 'ME', 'AF', 'Other']
  const sortedGeos = geoOrder.filter(g => geoGrouped[g])

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight
        ? 'bg-white/80 border-stone-200/80'
        : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
    )}>
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          'w-full px-4 py-2.5 flex items-center justify-between cursor-pointer transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/5'
        )}
      >
        <div className="flex items-center gap-2">
          {expanded
            ? <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-slate-500')} />
            : <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-slate-500')} />
          }
          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-semibold text-xs', isLight ? 'text-stone-900' : 'text-white')}>
            Cross-Region Inference (CRIS)
          </span>
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            {crisCount}/{selectedModels.length} supported
          </Badge>
        </div>
      </button>

      {expanded && (
        <div className="overflow-auto max-h-[400px]">
          <table className="w-full">
            <thead className="sticky top-0 z-20">
              <tr className={cn(
                'border-t border-b',
                isLight ? 'border-stone-200 bg-stone-50' : 'border-white/[0.06] bg-[#1a1b1e]'
              )}>
                <th className={cn(
                  'px-5 py-2.5 text-left text-xs font-semibold min-w-[140px] sticky left-0 z-30',
                  isLight ? 'text-stone-700 bg-stone-50' : 'text-slate-300 bg-[#1a1b1e]'
                )}>
                  Attribute
                </th>
                {selectedModels.map(({ model }) => (
                  <th key={model.model_id} className={cn(
                    'px-3 py-2.5 text-center min-w-[100px]',
                    isLight ? 'bg-stone-50' : 'bg-[#1a1b1e]'
                  )}>
                    <Badge className={cn(
                      'text-[9px] mb-0.5',
                      isLight ? 'text-[#faf9f5]' : 'text-white',
                      providerColors[model.model_provider] || providerColors.default
                    )}>
                      {model.model_provider}
                    </Badge>
                    <p className={cn(
                      'text-[10px] font-semibold line-clamp-1 max-w-[100px] mx-auto',
                      isLight ? 'text-stone-700' : 'text-slate-300'
                    )}>
                      {model.model_name || model.model_id}
                    </p>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* Supported row */}
              <tr className={cn('border-b', isLight ? 'border-stone-100' : 'border-white/[0.04]')}>
                <td className={cn('px-5 py-2.5 text-sm font-medium sticky left-0 z-10', isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]')}>
                  Supported
                </td>
                {selectedModels.map(({ model }) => {
                  const supported = model.cross_region_inference?.supported
                  return (
                    <td key={model.model_id} className="px-3 py-2.5 text-center">
                      {supported ? (
                        <Check className="h-4 w-4 text-emerald-500 mx-auto" />
                      ) : (
                        <span className={cn('text-sm', isLight ? 'text-stone-300' : 'text-slate-600')}>--</span>
                      )}
                    </td>
                  )
                })}
              </tr>

              {/* Profiles row */}
              <tr className={cn('border-b', isLight ? 'border-stone-100' : 'border-white/[0.04]')}>
                <td className={cn('px-5 py-2.5 text-sm font-medium sticky left-0 z-10', isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]')}>
                  Profiles
                </td>
                {selectedModels.map(({ model }) => {
                  // Count unique profile IDs (same logic as ModelCardExpanded)
                  const profiles = model.cross_region_inference?.profiles || []
                  const uniqueEndpoints = new Set(
                    profiles
                      .map(p => p.profile_id || p.inference_profile_id)
                      .filter(Boolean)
                  ).size
                  return (
                    <td key={model.model_id} className={cn(
                      'px-3 py-2.5 text-center text-sm font-medium',
                      isLight ? 'text-stone-900' : 'text-white'
                    )}>
                      {uniqueEndpoints || '--'}
                    </td>
                  )
                })}
              </tr>

              {/* Source regions count row */}
              <tr className={cn('border-b', isLight ? 'border-stone-100' : 'border-white/[0.04]')}>
                <td className={cn('px-5 py-2.5 text-sm font-medium sticky left-0 z-10', isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]')}>
                  Source Regions
                </td>
                {selectedModels.map(({ model }) => {
                  const count = (model.cross_region_inference?.source_regions || []).length
                  return (
                    <td key={model.model_id} className={cn(
                      'px-3 py-2.5 text-center text-sm font-medium',
                      isLight ? 'text-stone-900' : 'text-white'
                    )}>
                      {count || '--'}
                    </td>
                  )
                })}
              </tr>

              {/* Geo group rows — each cell shows region badges */}
              {sortedGeos.map(geo => {
                const regions = geoGrouped[geo]
                return (
                  <tr key={geo} className={cn(
                    'border-b last:border-b-0',
                    isLight ? 'border-stone-100' : 'border-white/[0.04]'
                  )}>
                    <td className={cn('px-5 py-3 align-top sticky left-0 z-10', isLight ? 'text-stone-800 bg-white' : 'text-slate-200 bg-[#1a1b1e]')}>
                      <div className="text-sm font-semibold">{geoLabels[geo] || geo}</div>
                      <div className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                        {regions.length} region{regions.length !== 1 ? 's' : ''}
                      </div>
                    </td>
                    {selectedModels.map(({ model }) => {
                      const modelRegions = model.cross_region_inference?.source_regions || []
                      const supported = model.cross_region_inference?.supported
                      const matchingRegions = regions.filter(r => modelRegions.includes(r))

                      return (
                        <td key={model.model_id} className="px-2 py-3 align-top">
                          {!supported ? (
                            <div className="flex justify-center">
                              <span className={cn('text-sm', isLight ? 'text-stone-300' : 'text-slate-600')}>--</span>
                            </div>
                          ) : matchingRegions.length === 0 ? (
                            <div className="flex justify-center">
                              <span className={cn('text-sm', isLight ? 'text-stone-300' : 'text-slate-600')}>--</span>
                            </div>
                          ) : (
                            <div className="flex flex-wrap gap-1 justify-center">
                              {matchingRegions.map(r => {
                                const info = regionLookup[r]
                                return (
                                  <span
                                    key={r}
                                    title={r}
                                    className={cn(
                                      'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium',
                                      isLight
                                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                        : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                    )}
                                  >
                                    {info?.name || r}
                                  </span>
                                 )
                              })}
                            </div>
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
      )}
    </div>
  )
}

function MantleSection({ selectedModels, isLight }) {
  const [expanded, setExpanded] = useState(true)

  const mantleCount = selectedModels.filter(({ model }) => model.mantle_inference?.supported || model.is_mantle).length

  // Build region lookup: code → { name, geo }
  const regionLookup = {}
  awsRegions.forEach(r => {
    regionLookup[r.value] = { name: friendlyName(r.label), geo: r.geo }
  })

  // Geo labels
  const geoLabels = { US: 'United States', EU: 'Europe', AP: 'Asia Pacific', CA: 'Canada', SA: 'South America', ME: 'Middle East', AF: 'Africa' }

  // Collect all Mantle regions and group by geo
  const allMantleRegions = new Set()
  selectedModels.forEach(({ model }) => {
    (model.mantle_inference?.mantle_regions || []).forEach(r => allMantleRegions.add(r))
  })

  const geoGrouped = {}
  allMantleRegions.forEach(region => {
    const info = regionLookup[region]
    const geo = info?.geo || 'Other'
    if (!geoGrouped[geo]) geoGrouped[geo] = []
    geoGrouped[geo].push(region)
  })
  // Sort regions within each geo
  Object.values(geoGrouped).forEach(arr => arr.sort())

  // Order geos consistently
  const geoOrder = ['US', 'EU', 'AP', 'CA', 'SA', 'ME', 'AF', 'Other']
  const sortedGeos = geoOrder.filter(g => geoGrouped[g])

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight
        ? 'bg-white/80 border-stone-200/80'
        : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl'
    )}>
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          'w-full px-4 py-2.5 flex items-center justify-between cursor-pointer transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/5'
        )}
      >
        <div className="flex items-center gap-2">
          {expanded
            ? <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-slate-500')} />
            : <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-slate-500')} />
          }
          <Cpu className={cn('h-3.5 w-3.5', isLight ? 'text-purple-600' : 'text-purple-400')} />
          <span className={cn('font-semibold text-xs', isLight ? 'text-stone-900' : 'text-white')}>
            Mantle Inference
          </span>
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            {mantleCount}/{selectedModels.length} supported
          </Badge>
        </div>
      </button>

      {expanded && (
        <div className="overflow-auto max-h-[400px]">
          <table className="w-full">
            <thead className="sticky top-0 z-20">
              <tr className={cn(
                'border-t border-b',
                isLight ? 'border-stone-200 bg-stone-50' : 'border-white/[0.06] bg-[#1a1b1e]'
              )}>
                <th className={cn(
                  'px-5 py-2.5 text-left text-xs font-semibold min-w-[140px] sticky left-0 z-30',
                  isLight ? 'text-stone-700 bg-stone-50' : 'text-slate-300 bg-[#1a1b1e]'
                )}>
                  Attribute
                </th>
                {selectedModels.map(({ model }) => (
                  <th key={model.model_id} className={cn(
                    'px-3 py-2.5 text-center min-w-[100px]',
                    isLight ? 'bg-stone-50' : 'bg-[#1a1b1e]'
                  )}>
                    <Badge className={cn(
                      'text-[9px] mb-0.5',
                      isLight ? 'text-[#faf9f5]' : 'text-white',
                      providerColors[model.model_provider] || providerColors.default
                    )}>
                      {model.model_provider}
                    </Badge>
                    <p className={cn(
                      'text-[10px] font-semibold line-clamp-1 max-w-[100px] mx-auto',
                      isLight ? 'text-stone-700' : 'text-slate-300'
                    )}>
                      {model.model_name || model.model_id}
                    </p>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* Supported row */}
              <tr className={cn('border-b', isLight ? 'border-stone-100' : 'border-white/[0.04]')}>
                <td className={cn('px-5 py-2.5 text-sm font-medium sticky left-0 z-10', isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]')}>
                  Supported
                </td>
                {selectedModels.map(({ model }) => {
                  const supported = model.mantle_inference?.supported || model.is_mantle || false
                  return (
                    <td key={model.model_id} className="px-3 py-2.5 text-center">
                      {supported ? (
                        <Check className="h-4 w-4 text-emerald-500 mx-auto" />
                      ) : (
                        <span className={cn('text-sm', isLight ? 'text-stone-300' : 'text-slate-600')}>--</span>
                      )}
                    </td>
                  )
                })}
              </tr>

              {/* Regions count row */}
              <tr className={cn('border-b', isLight ? 'border-stone-100' : 'border-white/[0.04]')}>
                <td className={cn('px-5 py-2.5 text-sm font-medium sticky left-0 z-10', isLight ? 'text-stone-700 bg-white' : 'text-slate-300 bg-[#1a1b1e]')}>
                  Regions
                </td>
                {selectedModels.map(({ model }) => {
                  const count = (model.mantle_inference?.mantle_regions || []).length
                  return (
                    <td key={model.model_id} className={cn(
                      'px-3 py-2.5 text-center text-sm font-medium',
                      isLight ? 'text-stone-900' : 'text-white'
                    )}>
                      {count || '--'}
                    </td>
                  )
                })}
              </tr>

              {/* Geo group rows — each cell shows region badges */}
              {sortedGeos.map(geo => {
                const regions = geoGrouped[geo]
                return (
                  <tr key={geo} className={cn(
                    'border-b last:border-b-0',
                    isLight ? 'border-stone-100' : 'border-white/[0.04]'
                  )}>
                    <td className={cn('px-5 py-3 align-top sticky left-0 z-10', isLight ? 'text-stone-800 bg-white' : 'text-slate-200 bg-[#1a1b1e]')}>
                      <div className="text-sm font-semibold">{geoLabels[geo] || geo}</div>
                      <div className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                        {regions.length} region{regions.length !== 1 ? 's' : ''}
                      </div>
                    </td>
                    {selectedModels.map(({ model }) => {
                      const modelRegions = model.mantle_inference?.mantle_regions || []
                      const supported = model.mantle_inference?.supported || model.is_mantle || false
                      const matchingRegions = regions.filter(r => modelRegions.includes(r))

                      return (
                        <td key={model.model_id} className="px-2 py-3 align-top">
                          {!supported ? (
                            <div className="flex justify-center">
                              <span className={cn('text-sm', isLight ? 'text-stone-300' : 'text-slate-600')}>--</span>
                            </div>
                          ) : matchingRegions.length === 0 ? (
                            <div className="flex justify-center">
                              <span className={cn('text-sm', isLight ? 'text-stone-300' : 'text-slate-600')}>--</span>
                            </div>
                          ) : (
                            <div className="flex flex-wrap gap-1 justify-center">
                              {matchingRegions.map(r => {
                                const info = regionLookup[r]
                                return (
                                  <span
                                    key={r}
                                    title={r}
                                    className={cn(
                                      'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium',
                                      isLight
                                        ? 'bg-purple-50 text-purple-700 border border-purple-200'
                                        : 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                                    )}
                                  >
                                    {info?.name || r}
                                  </span>
                                )
                              })}
                            </div>
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
      )}
    </div>
  )
}
