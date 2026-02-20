import { useState, useMemo } from 'react'
import {
  Globe, Map, Calendar, ChevronDown, ChevronRight, Search,
  Check, X, Clock, Grid3X3, LayoutGrid, CalendarDays,
  MapPin, ArrowRight, Pencil,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useAuthStore } from '@/stores/authStore'
import { canEditRoadmap } from '@/config/admin'
import roadmapData from '@/data/region-roadmap-data.json'

const GEO_GROUPS = [
  { key: 'americas', label: 'Americas', icon: '🌎' },
  { key: 'europe', label: 'Europe', icon: '🌍' },
  { key: 'apac', label: 'Asia Pacific', icon: '🌏' },
  { key: 'mea', label: 'Middle East & Africa', icon: '🌍' },
]

// Color constants for bars, dots, stat text — theme-aware
const COLORS = {
  available: { bar: ['bg-emerald-500', 'bg-emerald-400'], dot: ['bg-emerald-500', 'bg-emerald-400'], text: ['text-emerald-700', 'text-emerald-300'] },
  cris:      { bar: ['bg-amber-600', 'bg-sky-400'],      dot: ['bg-amber-600', 'bg-sky-400'],      text: ['text-amber-700', 'text-sky-300'] },
  planned:   { bar: ['bg-violet-400', 'bg-violet-400'],     dot: ['bg-violet-400', 'bg-violet-400'],   text: ['text-violet-700', 'text-violet-300'] },
  np:        { bar: ['bg-stone-300', 'bg-[#4a4d54]'],     dot: ['bg-stone-400', 'bg-[#4a4d54]'],   text: ['text-stone-400', 'text-[#4a4d54]'] },
}
function c(status, prop, isLight) { return COLORS[status]?.[prop]?.[isLight ? 0 : 1] || '' }

const STATUS_CONFIG = {
  available: {
    dark: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/20',
    light: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    icon: Check, label: 'In Region',
  },
  cris: {
    dark: 'bg-sky-400/20 text-sky-300 border-sky-400/30',
    light: 'bg-amber-100 text-amber-700 border-amber-200',
    icon: Globe, label: 'CRIS',
  },
  in_region_cris: {
    dark: 'bg-sky-400/20 text-sky-200 border-sky-400/30',
    light: 'bg-amber-100 text-amber-800 border-amber-300',
    icon: Globe, label: 'In Region + CRIS',
  },
  cris_profile: {
    dark: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
    light: 'bg-amber-50 text-amber-600 border-amber-200',
    icon: Globe, label: 'CRIS Profile',
  },
  planned: {
    dark: 'bg-violet-500/20 text-violet-300 border-violet-500/30',
    light: 'bg-violet-100 text-violet-700 border-violet-200',
    icon: Calendar, label: 'Planned',
  },
  np: {
    dark: 'bg-white/[0.02] text-[#3a3d44] border-white/[0.03]',
    light: 'bg-stone-100 text-stone-400 border-stone-200',
    icon: X, label: 'Not Planned',
  },
  other: {
    dark: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    light: 'bg-purple-100 text-purple-700 border-purple-200',
    icon: Clock, label: 'Other',
  },
}

// --- Sub-components ---

function StatusCell({ status, isLight }) {
  if (!status) return <div className="w-8 h-8" />
  const cfg = STATUS_CONFIG[status.status] || STATUS_CONFIG.other
  const Icon = cfg.icon
  const colorCls = isLight ? cfg.light : cfg.dark

  return (
    <div className="relative group">
      <div className={cn(
        'w-8 h-8 rounded-md border flex items-center justify-center transition-transform hover:scale-110 cursor-default',
        colorCls,
      )}>
        <Icon className="w-3.5 h-3.5" />
      </div>
      {/* Tooltip */}
      <div className={cn(
        'absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 rounded-lg text-xs',
        'whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50',
        'shadow-lg border',
        isLight
          ? 'bg-white border-stone-200 text-stone-700'
          : 'bg-[#1a1b1e] border-white/10 text-[#e4e5e7]',
      )}>
        <div className="font-medium">{status.label}</div>
        {status.date && <div className="text-[10px] mt-0.5 opacity-70">{status.date}</div>}
      </div>
    </div>
  )
}

function shortProviderName(name) {
  return name
    .replace('Amazon ', '')
    .replace('Anthropic ', '')
    .replace('Meta ', '')
    .replace('Google ', '')
    .replace(' AI', '')
    .replace(' Labs', '')
    .replace('Bedrock ', '')
}

function OverviewCell({ region, provider, isLight, onClick }) {
  const items = provider.items
  let avail = 0, cris = 0, planned = 0, np = 0
  for (const item of items) {
    const s = region.availability[item.id]
    if (!s) continue
    if (s.status === 'available') avail++
    else if (['cris', 'in_region_cris', 'cris_profile'].includes(s.status)) cris++
    else if (s.status === 'planned') planned++
    else if (s.status === 'np') np++
  }

  const accessible = avail + cris
  const total = items.length
  const pct = total > 0 ? accessible / total : 0
  const hasData = avail + cris + planned + np > 0

  if (!hasData) return <div className="w-14 h-8" />

  let bgCls, textCls
  if (pct === 0) {
    bgCls = isLight ? 'bg-stone-100/60' : 'bg-white/[0.03] border-white/[0.04]'
    textCls = isLight ? 'text-stone-400' : 'text-[#6d6e72]'
  } else if (pct < 0.25) {
    bgCls = isLight ? 'bg-emerald-50 border-emerald-200/60' : 'bg-white/[0.05] border-white/[0.07]'
    textCls = isLight ? 'text-emerald-600' : 'text-[#a0a1a5]'
  } else if (pct < 0.5) {
    bgCls = isLight ? 'bg-emerald-100 border-emerald-200' : 'bg-white/[0.07] border-white/[0.10]'
    textCls = isLight ? 'text-emerald-700' : 'text-[#d0d1d5]'
  } else if (pct < 0.75) {
    bgCls = isLight ? 'bg-emerald-200 border-emerald-300' : 'bg-emerald-500/10 border-emerald-500/15'
    textCls = isLight ? 'text-emerald-800' : 'text-emerald-300'
  } else {
    bgCls = isLight ? 'bg-emerald-300/70 border-emerald-400' : 'bg-emerald-500/15 border-emerald-500/20'
    textCls = isLight ? 'text-emerald-900' : 'text-emerald-200'
  }

  return (
    <div className="relative group">
      <div
        className={cn(
          'w-14 h-8 rounded-md border flex items-center justify-center cursor-pointer transition-all hover:scale-105 hover:shadow-sm',
          bgCls,
        )}
        onClick={onClick}
      >
        <span className={cn('text-[10px] font-semibold tabular-nums', textCls)}>
          {accessible}/{total}
        </span>
        {planned > 0 && (
          <div className={cn('absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full ring-1',
            isLight ? 'bg-violet-400 ring-violet-500/30' : 'bg-violet-400 ring-violet-500/30'
          )} />
        )}
      </div>
      {/* Tooltip */}
      <div className={cn(
        'absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 rounded-lg text-xs',
        'whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50',
        'shadow-lg border',
        isLight
          ? 'bg-white border-stone-200 text-stone-700'
          : 'bg-[#1a1b1e] border-white/10 text-[#e4e5e7]',
      )}>
        <div className="font-semibold mb-1">{provider.name}</div>
        <div className="space-y-0.5">
          {avail > 0 && <div className="flex items-center gap-1.5"><span className={cn('w-2 h-2 rounded-sm', c('available','dot',isLight))} />{avail} In Region</div>}
          {cris > 0 && <div className="flex items-center gap-1.5"><span className={cn('w-2 h-2 rounded-sm', c('cris','dot',isLight))} />{cris} CRIS</div>}
          {planned > 0 && <div className="flex items-center gap-1.5"><span className={cn('w-2 h-2 rounded-sm', c('planned','dot',isLight))} />{planned} Planned</div>}
          {np > 0 && <div className="flex items-center gap-1.5"><span className={cn('w-2 h-2 rounded-sm', c('np','dot',isLight))} />{np} Not Planned</div>}
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon: Icon, label, value, accent, isLight, cardCls }) {
  return (
    <div className={cn(cardCls, 'flex items-center gap-3')}>
      <div className={cn(
        'w-10 h-10 rounded-xl flex items-center justify-center',
        isLight ? 'bg-stone-100' : 'bg-white/[0.05]',
      )}>
        <Icon className={cn('w-5 h-5', accent)} />
      </div>
      <div>
        <div className={cn('text-2xl font-bold tabular-nums', isLight ? 'text-stone-900' : 'text-white')}>
          {value}
        </div>
        <div className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>{label}</div>
      </div>
    </div>
  )
}

// --- Heatmap View ---
function HeatmapView({ providers, regions, selectedProvider, setSelectedProvider, search, isLight, cardCls }) {
  const [collapsedGeos, setCollapsedGeos] = useState(new Set())

  const modelProviders = useMemo(() => providers.filter(p => p.category === 'models'), [providers])
  const isOverview = selectedProvider === 'all'

  const items = useMemo(() => {
    if (isOverview) return null
    if (selectedProvider === 'features') return providers.filter(p => p.category === 'features').flatMap(p => p.items)
    const prov = providers.find(p => p.name === selectedProvider)
    return prov ? prov.items : []
  }, [providers, selectedProvider, isOverview])

  const filteredRegions = useMemo(() => {
    if (!search) return regions
    const q = search.toLowerCase()
    return regions.filter(r =>
      r.name.toLowerCase().includes(q) ||
      r.code.toLowerCase().includes(q) ||
      r.shortCode.toLowerCase().includes(q)
    )
  }, [regions, search])

  const toggleGeo = (key) => {
    setCollapsedGeos(prev => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }

  const grouped = useMemo(() => {
    return GEO_GROUPS.map(g => ({
      ...g,
      regions: filteredRegions.filter(r => r.geo === g.key),
    })).filter(g => g.regions.length > 0)
  }, [filteredRegions])

  return (
    <div className={cn(cardCls, 'overflow-hidden p-0')}>
      <div className="overflow-x-auto">
        <div className="min-w-max">
          {/* Header row */}
          <div className={cn(
            'flex items-end border-b sticky top-0 z-10',
            isLight ? 'bg-white/90 border-stone-200/80 backdrop-blur-xl' : 'bg-white/[0.03] border-white/[0.06] backdrop-blur-xl',
          )}>
            <div className={cn(
              'w-52 flex-shrink-0 px-4 py-3 sticky left-0 z-20',
              isLight ? 'bg-white/90 backdrop-blur-xl' : 'bg-white/[0.03] backdrop-blur-xl',
            )}>
              <span className={cn('text-xs font-semibold uppercase tracking-wider',
                isLight ? 'text-stone-400' : 'text-[#8a8b8f]'
              )}>Region</span>
            </div>

            {isOverview ? (
              /* Overview mode: provider columns with horizontal labels */
              <div className="flex gap-1 px-2 py-2">
                {modelProviders.map(prov => (
                  <div
                    key={prov.name}
                    className={cn(
                      'w-14 flex-shrink-0 flex flex-col items-center justify-end cursor-pointer rounded-lg px-0.5 py-1 transition-colors',
                      isLight ? 'hover:bg-amber-50' : 'hover:bg-white/[0.06]',
                    )}
                    onClick={() => setSelectedProvider(prov.name)}
                    title={`${prov.name} (${prov.items.length} models) — Click for details`}
                  >
                    <span className={cn('text-[10px] font-semibold leading-tight text-center',
                      isLight ? 'text-stone-600' : 'text-[#c0c1c5]',
                    )}>{shortProviderName(prov.name)}</span>
                    <span className={cn('text-[9px]',
                      isLight ? 'text-stone-400' : 'text-[#6d6e72]',
                    )}>({prov.items.length})</span>
                  </div>
                ))}
              </div>
            ) : (
              /* Detail mode: individual model columns */
              <div className="flex gap-1 px-2 py-2">
                {items.map(item => (
                  <div key={item.id} className="w-8 flex-shrink-0 flex items-end justify-center relative group/col">
                    <span className={cn(
                      'text-[9px] font-medium whitespace-nowrap',
                      isLight ? 'text-stone-500' : 'text-[#9a9b9f]',
                    )} style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)', maxHeight: 80 }}>
                      {item.name.length > 14 ? item.name.slice(0, 12) + '...' : item.name}
                    </span>
                    <div className={cn(
                      'absolute top-full left-1/2 -translate-x-1/2 mt-1 px-2.5 py-1 rounded-lg text-[10px] font-medium',
                      'whitespace-nowrap opacity-0 group-hover/col:opacity-100 transition-opacity pointer-events-none z-50',
                      'shadow-lg border',
                      isLight
                        ? 'bg-white border-stone-200 text-stone-700'
                        : 'bg-[#1a1b1e] border-white/10 text-[#e4e5e7]',
                    )}>
                      {item.name}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Region rows grouped by geo */}
          {grouped.map(group => (
            <div key={group.key}>
              {/* Geo header */}
              <button
                onClick={() => toggleGeo(group.key)}
                className={cn(
                  'w-full flex items-center gap-2 px-4 py-2 text-left transition-colors',
                  isLight ? 'bg-stone-50/80 hover:bg-stone-100/80' : 'bg-white/[0.04] hover:bg-white/[0.06]',
                )}
              >
                {collapsedGeos.has(group.key)
                  ? <ChevronRight className={cn('w-3.5 h-3.5', isLight ? 'text-stone-400' : 'text-[#9a9b9f]')} />
                  : <ChevronDown className={cn('w-3.5 h-3.5', isLight ? 'text-stone-400' : 'text-[#9a9b9f]')} />
                }
                <span className="text-sm">{group.icon}</span>
                <span className={cn('text-xs font-semibold uppercase tracking-wider',
                  isLight ? 'text-stone-500' : 'text-[#c0c1c5]'
                )}>{group.label}</span>
                <span className={cn('text-[10px] ml-1',
                  isLight ? 'text-stone-400' : 'text-[#8a8b8f]'
                )}>({group.regions.length})</span>
              </button>

              {!collapsedGeos.has(group.key) && group.regions.map(region => (
                <div key={region.shortCode || region.code} className={cn(
                  'flex items-center border-b',
                  isLight ? 'border-stone-100 hover:bg-stone-50/50' : 'border-white/[0.04] hover:bg-white/[0.03]',
                )}>
                  <div className={cn(
                    'w-52 flex-shrink-0 px-4 py-1.5 sticky left-0 z-10',
                    isLight ? 'bg-white/95 backdrop-blur-sm' : 'bg-white/[0.03] backdrop-blur-xl',
                  )}>
                    <div className="flex items-center gap-2">
                      <span className={cn('text-xs font-bold', isLight ? 'text-stone-700' : 'text-white')}>
                        {region.shortCode}
                      </span>
                      <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-[#8a8b8f]')}>
                        {region.code}
                      </span>
                    </div>
                  </div>

                  {isOverview ? (
                    <div className="flex gap-1 px-2 py-1">
                      {modelProviders.map(prov => (
                        <OverviewCell
                          key={prov.name}
                          region={region}
                          provider={prov}
                          isLight={isLight}
                          onClick={() => setSelectedProvider(prov.name)}
                        />
                      ))}
                    </div>
                  ) : (
                    <div className="flex gap-1 px-2 py-1">
                      {items.map(item => (
                        <StatusCell
                          key={item.id}
                          status={region.availability[item.id]}
                          isLight={isLight}
                        />
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// --- Region Detail Modal ---
function RegionModal({ region, providers, isLight, onClose }) {
  if (!region) return null

  const allProviders = providers.filter(p => p.items.length > 0)

  // Build per-provider breakdown for this region
  const providerData = []
  let totalAvail = 0, totalCris = 0, totalPlanned = 0, totalNp = 0, totalItems = 0

  for (const prov of allProviders) {
    const available = []
    const cris = []
    const planned = []
    const np = []
    for (const item of prov.items) {
      const s = region.availability[item.id]
      if (!s) continue
      if (s.status === 'available') available.push(item)
      else if (['cris', 'in_region_cris', 'cris_profile'].includes(s.status)) cris.push(item)
      else if (s.status === 'planned') planned.push({ ...item, date: s.date, dateLabel: s.label })
      else if (s.status === 'np') np.push(item)
    }
    const hasData = available.length + cris.length + planned.length + np.length > 0
    if (hasData) {
      providerData.push({ name: prov.name, short: shortProviderName(prov.name), category: prov.category, available, cris, planned, np, total: prov.items.length })
      totalAvail += available.length
      totalCris += cris.length
      totalPlanned += planned.length
      totalNp += np.length
    }
    totalItems += prov.items.length
  }

  const accessible = totalAvail + totalCris
  const coverage = totalItems > 0 ? Math.round(accessible / totalItems * 100) : 0

  const statPills = [
    { label: 'In Region', count: totalAvail, dot: c('available','dot',isLight), cls: c('available','text',isLight) },
    { label: 'CRIS', count: totalCris, dot: c('cris','dot',isLight), cls: c('cris','text',isLight) },
    { label: 'Planned', count: totalPlanned, dot: c('planned','dot',isLight), cls: c('planned','text',isLight) },
    { label: 'Not Planned', count: totalNp, dot: c('np','dot',isLight), cls: c('np','text',isLight) },
  ]

  const statusPill = (status, name, dateLabel) => {
    const styles = {
      available: isLight ? 'bg-emerald-50 text-emerald-700 border-emerald-200/60' : 'bg-emerald-400/10 text-emerald-300 border-emerald-400/20',
      cris: isLight ? 'bg-amber-50 text-amber-700 border-amber-200/60' : 'bg-sky-400/10 text-sky-300 border-sky-400/20',
      planned: isLight ? 'bg-violet-50 text-violet-700 border-violet-200/60' : 'bg-violet-500/10 text-violet-300 border-violet-500/20',
      np: isLight ? 'bg-stone-50 text-stone-400 border-stone-200/60' : 'bg-white/[0.02] text-[#4a4d54] border-white/[0.04]',
    }
    return (
      <span className={cn('inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md border font-medium', styles[status])}>
        {name}
        {dateLabel && <span className="opacity-60 text-[10px]">({dateLabel})</span>}
      </span>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 lg:p-6" onClick={onClose}>
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

      {/* Modal */}
      <div
        className={cn(
          'relative w-full max-w-6xl max-h-[90vh] flex flex-col rounded-2xl border shadow-2xl overflow-hidden',
          isLight
            ? 'bg-[#faf9f5] border-stone-200/80'
            : 'bg-[#18191c] border-white/[0.08]',
        )}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className={cn(
          'flex-shrink-0 px-6 py-5 border-b',
          isLight ? 'border-stone-200/80' : 'border-white/[0.06]',
        )}>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-3">
                <MapPin className={cn('w-5 h-5', isLight ? 'text-amber-600' : 'text-emerald-400')} />
                <h2 className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>
                  {region.shortCode}
                </h2>
                <span className={cn('text-sm', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>
                  {region.name}
                </span>
                <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-400' : 'text-[#4a4d54]')}>
                  {region.code}
                </span>
                <span className={cn(
                  'text-xs px-2 py-0.5 rounded-full font-medium ml-1',
                  region.launchStatus === 'Available'
                    ? (isLight ? 'bg-emerald-100 text-emerald-700' : 'bg-emerald-500/20 text-emerald-400')
                    : (isLight ? 'bg-amber-100 text-amber-700' : 'bg-amber-500/20 text-amber-400'),
                )}>{region.launchStatus || 'TBD'}</span>
              </div>

              {/* Coverage bar + summary stats */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-3 flex-1 max-w-md">
                  <div className={cn('flex-1 h-2.5 rounded-full overflow-hidden',
                    isLight ? 'bg-stone-200' : 'bg-white/[0.06]'
                  )}>
                    <div className="h-full flex">
                      {totalAvail > 0 && <div className={cn(c('available','bar',isLight), 'h-full')} style={{ width: `${totalAvail / totalItems * 100}%` }} />}
                      {totalCris > 0 && <div className={cn(c('cris','bar',isLight), 'h-full')} style={{ width: `${totalCris / totalItems * 100}%` }} />}
                      {totalPlanned > 0 && <div className={cn(c('planned','bar',isLight), 'h-full')} style={{ width: `${totalPlanned / totalItems * 100}%` }} />}
                    </div>
                  </div>
                  <span className={cn('text-lg font-bold tabular-nums', isLight ? 'text-stone-800' : 'text-white')}>
                    {coverage}%
                  </span>
                </div>
                <div className="flex gap-3">
                  {statPills.map(s => s.count > 0 && (
                    <span key={s.label} className="flex items-center gap-1.5 text-[11px]">
                      <span className={cn('w-2 h-2 rounded-full', s.dot)} />
                      <span className={cn('font-semibold tabular-nums', s.cls)}>{s.count}</span>
                      <span className={isLight ? 'text-stone-400' : 'text-[#4a4d54]'}>{s.label}</span>
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <button
              onClick={onClose}
              className={cn(
                'w-8 h-8 rounded-lg flex items-center justify-center transition-colors ml-4 flex-shrink-0',
                isLight ? 'hover:bg-stone-100 text-stone-400' : 'hover:bg-white/[0.06] text-[#6d6e72]',
              )}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Body — provider cards in a grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {providerData.map(prov => {
              const provAccessible = prov.available.length + prov.cris.length
              const provCoverage = prov.total > 0 ? Math.round(provAccessible / prov.total * 100) : 0

              return (
                <div
                  key={prov.name}
                  className={cn(
                    'rounded-xl border p-4',
                    isLight
                      ? 'bg-white/60 border-stone-200/60'
                      : 'bg-white/[0.02] border-white/[0.05]',
                  )}
                >
                  {/* Provider header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      {prov.category === 'features' && <Grid3X3 className={cn('w-3.5 h-3.5', isLight ? 'text-stone-400' : 'text-[#6d6e72]')} />}
                      <span className={cn('text-sm font-semibold', isLight ? 'text-stone-800' : 'text-white')}>
                        {prov.short}
                      </span>
                      <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-[#4a4d54]')}>
                        {provAccessible}/{prov.total}
                      </span>
                    </div>
                    <div className={cn(
                      'text-xs font-bold tabular-nums px-2 py-0.5 rounded-full',
                      provCoverage >= 75
                        ? (isLight ? 'bg-emerald-100 text-emerald-700' : 'bg-emerald-500/15 text-emerald-400')
                        : provCoverage >= 25
                          ? (isLight ? 'bg-amber-100 text-amber-700' : 'bg-amber-500/15 text-amber-400')
                          : (isLight ? 'bg-stone-100 text-stone-500' : 'bg-white/[0.05] text-[#6d6e72]'),
                    )}>
                      {provCoverage}%
                    </div>
                  </div>

                  {/* Mini coverage bar */}
                  <div className={cn('h-1.5 rounded-full overflow-hidden mb-3',
                    isLight ? 'bg-stone-100' : 'bg-white/[0.04]'
                  )}>
                    <div className="h-full flex">
                      {prov.available.length > 0 && <div className={cn(c('available','bar',isLight), 'h-full')} style={{ width: `${prov.available.length / prov.total * 100}%` }} />}
                      {prov.cris.length > 0 && <div className={cn(c('cris','bar',isLight), 'h-full')} style={{ width: `${prov.cris.length / prov.total * 100}%` }} />}
                      {prov.planned.length > 0 && <div className={cn(c('planned','bar',isLight), 'h-full')} style={{ width: `${prov.planned.length / prov.total * 100}%` }} />}
                    </div>
                  </div>

                  {/* Model pills grouped by status */}
                  <div className="space-y-2">
                    {prov.available.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {prov.available.map(m => <span key={m.id}>{statusPill('available', m.name)}</span>)}
                      </div>
                    )}
                    {prov.cris.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {prov.cris.map(m => <span key={m.id}>{statusPill('cris', m.name)}</span>)}
                      </div>
                    )}
                    {prov.planned.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {prov.planned.map(m => <span key={m.id}>{statusPill('planned', m.name, m.dateLabel)}</span>)}
                      </div>
                    )}
                    {prov.np.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {prov.np.map(m => <span key={m.id}>{statusPill('np', m.name)}</span>)}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

// --- Region Cards View ---
function RegionCardsView({ providers, regions, search, isLight, cardCls }) {
  const [selectedRegion, setSelectedRegion] = useState(null)

  const allModelItems = useMemo(() =>
    providers.filter(p => p.category === 'models').flatMap(p => p.items),
    [providers]
  )

  const filteredRegions = useMemo(() => {
    if (!search) return regions
    const q = search.toLowerCase()
    return regions.filter(r =>
      r.name.toLowerCase().includes(q) || r.code.toLowerCase().includes(q) || r.shortCode.toLowerCase().includes(q)
    )
  }, [regions, search])

  const grouped = useMemo(() =>
    GEO_GROUPS.map(g => ({
      ...g,
      regions: filteredRegions.filter(r => r.geo === g.key),
    })).filter(g => g.regions.length > 0),
    [filteredRegions]
  )

  function getBreakdown(region) {
    let avail = 0, cris = 0, planned = 0, np = 0
    for (const item of allModelItems) {
      const s = region.availability[item.id]
      if (!s) continue
      if (s.status === 'available') avail++
      else if (['cris', 'in_region_cris', 'cris_profile'].includes(s.status)) cris++
      else if (s.status === 'planned') planned++
      else if (s.status === 'np') np++
    }
    return { avail, cris, planned, np, total: allModelItems.length }
  }

  const drawerRegion = selectedRegion ? regions.find(r => r.shortCode === selectedRegion) : null

  return (
    <>
      <div className="space-y-6">
        {grouped.map(group => (
          <div key={group.key}>
            <h3 className={cn('text-xs font-semibold uppercase tracking-wider mb-3 flex items-center gap-2',
              isLight ? 'text-stone-500' : 'text-[#9a9b9f]'
            )}>
              <span>{group.icon}</span> {group.label}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {group.regions.map(region => {
                const bd = getBreakdown(region)
                const coverage = bd.total > 0 ? Math.round((bd.avail + bd.cris) / bd.total * 100) : 0
                const isSelected = selectedRegion === region.shortCode

                return (
                  <div
                    key={region.shortCode || region.code}
                    className={cn(cardCls, 'p-4 cursor-pointer transition-all hover:scale-[1.01]',
                      isSelected && (isLight ? 'ring-2 ring-amber-300' : 'ring-2 ring-emerald-400/50')
                    )}
                    onClick={() => setSelectedRegion(isSelected ? null : region.shortCode)}
                  >
                    {/* Card header */}
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className={cn('text-sm font-bold', isLight ? 'text-stone-800' : 'text-white')}>
                          {region.shortCode}
                        </span>
                        <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>
                          {region.name}
                        </span>
                      </div>
                      <span className={cn(
                        'text-xs px-2 py-0.5 rounded-full font-medium',
                        region.launchStatus === 'Available'
                          ? (isLight ? 'bg-emerald-100 text-emerald-700' : 'bg-emerald-500/20 text-emerald-400')
                          : (isLight ? 'bg-amber-100 text-amber-700' : 'bg-amber-500/20 text-amber-400'),
                      )}>{region.launchStatus || 'TBD'}</span>
                    </div>

                    {/* Coverage bar */}
                    <div className="flex items-center gap-3 mb-2">
                      <div className={cn('flex-1 h-2 rounded-full overflow-hidden',
                        isLight ? 'bg-stone-200' : 'bg-white/[0.06]'
                      )}>
                        <div className="h-full flex">
                          {bd.avail > 0 && <div className={cn(c('available','bar',isLight), 'h-full')} style={{ width: `${bd.avail / bd.total * 100}%` }} />}
                          {bd.cris > 0 && <div className={cn(c('cris','bar',isLight), 'h-full')} style={{ width: `${bd.cris / bd.total * 100}%` }} />}
                          {bd.planned > 0 && <div className={cn(c('planned','bar',isLight), 'h-full')} style={{ width: `${bd.planned / bd.total * 100}%` }} />}
                        </div>
                      </div>
                      <span className={cn('text-sm font-bold tabular-nums', isLight ? 'text-stone-700' : 'text-white')}>
                        {coverage}%
                      </span>
                    </div>

                    {/* Mini stats */}
                    <div className="flex gap-3 text-[10px]">
                      {bd.avail > 0 && <span className="flex items-center gap-1"><span className={cn('w-1.5 h-1.5 rounded-sm', c('available','dot',isLight))} /><span className={c('available','text',isLight)}>{bd.avail} in-region</span></span>}
                      {bd.cris > 0 && <span className="flex items-center gap-1"><span className={cn('w-1.5 h-1.5 rounded-sm', c('cris','dot',isLight))} /><span className={c('cris','text',isLight)}>{bd.cris} CRIS</span></span>}
                      {bd.planned > 0 && <span className="flex items-center gap-1"><span className={cn('w-1.5 h-1.5 rounded-sm', c('planned','dot',isLight))} /><span className={c('planned','text',isLight)}>{bd.planned} planned</span></span>}
                      {bd.np > 0 && <span className="flex items-center gap-1"><span className={cn('w-1.5 h-1.5 rounded-sm', c('np','dot',isLight))} /><span className={c('np','text',isLight)}>{bd.np} NP</span></span>}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Region detail modal */}
      {drawerRegion && (
        <RegionModal
          region={drawerRegion}
          providers={providers}
          isLight={isLight}
          onClose={() => setSelectedRegion(null)}
        />
      )}
    </>
  )
}

// --- Roadmap View ---
function RoadmapView({ upcomingLaunches, providers, regions, isLight, cardCls }) {
  const groupedByDate = useMemo(() => {
    const groups = {}
    for (const launch of upcomingLaunches) {
      if (!launch.date) continue
      const key = launch.date
      if (!groups[key]) groups[key] = { date: key, label: launch.label, items: [] }
      groups[key].items.push(launch)
    }
    return Object.values(groups).sort((a, b) => a.date.localeCompare(b.date))
  }, [upcomingLaunches])

  if (groupedByDate.length === 0) {
    return (
      <div className={cn(cardCls, 'p-8 text-center')}>
        <CalendarDays className={cn('w-12 h-12 mx-auto mb-3', isLight ? 'text-stone-300' : 'text-[#4a4d54]')} />
        <p className={cn('text-sm', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>No upcoming launches with specific dates</p>
      </div>
    )
  }

  function daysUntil(dateStr) {
    const target = new Date(dateStr)
    const now = new Date()
    const diff = Math.ceil((target - now) / (1000 * 60 * 60 * 24))
    if (diff < 0) return `${Math.abs(diff)}d ago`
    if (diff === 0) return 'Today'
    if (diff === 1) return 'Tomorrow'
    return `in ${diff}d`
  }

  return (
    <div className="space-y-4">
      {/* Timeline line */}
      <div className="relative">
        {groupedByDate.map((group, idx) => {
          const isPast = new Date(group.date) < new Date()
          return (
            <div key={group.date} className="flex gap-4 mb-4">
              {/* Timeline dot + line */}
              <div className="flex flex-col items-center w-8 flex-shrink-0">
                <div className={cn(
                  'w-3 h-3 rounded-full border-2 mt-4',
                  isPast
                    ? (isLight ? 'bg-stone-300 border-stone-400' : 'bg-[#4a4d54] border-[#6d6e72]')
                    : (isLight ? 'bg-violet-500 border-violet-600' : 'bg-violet-400 border-violet-500'),
                )} />
                {idx < groupedByDate.length - 1 && (
                  <div className={cn('w-px flex-1 mt-1',
                    isLight ? 'bg-stone-200' : 'bg-white/[0.06]'
                  )} />
                )}
              </div>

              {/* Content card */}
              <div className={cn(cardCls, 'flex-1 p-4')}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className={cn('text-lg font-bold tabular-nums',
                      isPast
                        ? (isLight ? 'text-stone-400' : 'text-[#4a4d54]')
                        : (isLight ? 'text-stone-900' : 'text-white'),
                    )}>
                      {new Date(group.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                    </span>
                    <span className={cn(
                      'text-xs px-2 py-0.5 rounded-full font-medium',
                      isPast
                        ? (isLight ? 'bg-stone-100 text-stone-400' : 'bg-white/[0.03] text-[#4a4d54]')
                        : (isLight ? 'bg-violet-100 text-violet-700' : 'bg-violet-500/20 text-violet-300'),
                    )}>
                      {daysUntil(group.date)}
                    </span>
                  </div>
                </div>

                <div className="space-y-3">
                  {/* Group by model */}
                  {Object.entries(
                    group.items.reduce((acc, item) => {
                      const key = `${item.provider}|${item.modelName}`
                      if (!acc[key]) acc[key] = { ...item, regions: [] }
                      acc[key].regions.push(item.regionCode)
                      return acc
                    }, {})
                  ).map(([key, item]) => {
                    // Count current availability for this model
                    const modelId = item.modelId
                    let currentAvail = 0
                    if (modelId && regions) {
                      for (const r of regions) {
                        const s = r.availability[modelId]
                        if (s && ['available', 'cris', 'in_region_cris', 'cris_profile'].includes(s.status)) currentAvail++
                      }
                    }
                    return (
                      <div key={key} className="space-y-1.5">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={cn(
                            'text-xs px-2 py-0.5 rounded font-medium',
                            isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/[0.05] text-[#9a9b9f]',
                          )}>{item.provider}</span>
                          <span className={cn('text-sm font-medium',
                            isLight ? 'text-stone-700' : 'text-[#e4e5e7]'
                          )}>{item.modelName}</span>
                          {currentAvail > 0 && (
                            <span className={cn('text-[10px]',
                              isLight ? 'text-stone-400' : 'text-[#4a4d54]'
                            )}>currently in {currentAvail} regions</span>
                          )}
                        </div>
                        <div className="flex items-center gap-1.5 pl-1">
                          <ArrowRight className={cn('w-3 h-3 shrink-0', isLight ? 'text-violet-400' : 'text-violet-400/60')} />
                          <div className="flex flex-wrap gap-1">
                            {item.regions.map(rc => (
                              <span key={rc} className={cn(
                                'text-xs px-1.5 py-0.5 rounded border font-mono',
                                isLight ? 'bg-violet-50 text-violet-700 border-violet-200' : 'bg-violet-500/10 text-violet-300 border-violet-500/20',
                              )}>{rc}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// --- Legend ---
function Legend({ isLight }) {
  return (
    <div className="flex flex-wrap gap-3">
      {Object.entries(STATUS_CONFIG).filter(([k]) => k !== 'other').map(([key, cfg]) => {
        const Icon = cfg.icon
        return (
          <div key={key} className="flex items-center gap-1.5">
            <div className={cn(
              'w-5 h-5 rounded border flex items-center justify-center',
              isLight ? cfg.light : cfg.dark,
            )}>
              <Icon className="w-3 h-3" />
            </div>
            <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>{cfg.label}</span>
          </div>
        )
      })}
    </div>
  )
}

// --- Main Component ---
export function RegionRoadmap() {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const user = useAuthStore((s) => s.user)

  const [view, setView] = useState('heatmap')
  const [selectedProvider, setSelectedProvider] = useState('all')
  const [search, setSearch] = useState('')

  const accent = isLight ? 'text-amber-600' : 'text-emerald-400'
  const accentBg = isLight ? 'bg-amber-600' : 'bg-emerald-500'
  const accentText = isLight ? 'text-[#faf9f5]' : 'text-[#ffffff]'

  const cardCls = cn(
    'rounded-2xl border backdrop-blur-xl transition-colors',
    isLight
      ? 'bg-white/70 border-stone-200/60 shadow-[0_2px_15px_-3px_rgba(120,113,108,0.08)]'
      : 'bg-white/[0.03] border-white/[0.06] shadow-[0_2px_15px_-3px_rgba(0,0,0,0.3)]'
  )

  const { providers, regions, summary, upcomingLaunches } = roadmapData
  const modelProviders = providers.filter(p => p.category === 'models')

  const views = [
    { id: 'heatmap', label: 'Heatmap', icon: Grid3X3 },
    { id: 'regions', label: 'By Region', icon: LayoutGrid },
    { id: 'roadmap', label: 'Roadmap', icon: CalendarDays },
  ]

  return (
    <div className="space-y-5 p-4 lg:p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            Regional Availability & Roadmap
          </h1>
          <p className={cn('text-xs mt-1', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>
            Last updated: {roadmapData.lastUpdated}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {canEditRoadmap(user) && (
            <button
              disabled
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border cursor-not-allowed opacity-50',
                isLight
                  ? 'bg-stone-100 text-stone-400 border-stone-200/60'
                  : 'bg-white/[0.04] text-[#6d6e72] border-white/[0.06]',
              )}
              title="Editing coming soon"
            >
              <Pencil className="w-3.5 h-3.5" />
              Edit
            </button>
          )}
          <Legend isLight={isLight} />
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <StatCard icon={Globe} label="Regions" value={summary.totalRegions} accent={accent} isLight={isLight} cardCls={cn(cardCls, 'p-4')} />
        <StatCard icon={Grid3X3} label="Models & Features" value={summary.totalItems} accent={accent} isLight={isLight} cardCls={cn(cardCls, 'p-4')} />
        <StatCard icon={Check} label="Available" value={summary.totalAvailable} accent={isLight ? 'text-emerald-500' : 'text-emerald-400'} isLight={isLight} cardCls={cn(cardCls, 'p-4')} />
        <StatCard icon={Globe} label="CRIS" value={summary.totalCris} accent={isLight ? 'text-amber-600' : 'text-sky-400'} isLight={isLight} cardCls={cn(cardCls, 'p-4')} />
        <StatCard icon={Calendar} label="Planned" value={summary.totalPlanned} accent={isLight ? 'text-violet-500' : 'text-violet-400'} isLight={isLight} cardCls={cn(cardCls, 'p-4')} />
      </div>

      {/* Controls */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center gap-3">
        {/* View toggle */}
        <div className={cn(
          'flex rounded-xl p-1 border',
          isLight ? 'bg-stone-100/80 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]',
        )}>
          {views.map(v => {
            const Icon = v.icon
            const active = view === v.id
            return (
              <button key={v.id} onClick={() => setView(v.id)} className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                active
                  ? cn(accentBg, accentText, 'shadow-sm')
                  : (isLight ? 'text-stone-500 hover:text-stone-700' : 'text-[#6d6e72] hover:text-[#9a9b9f]'),
              )}>
                <Icon className="w-3.5 h-3.5" />
                {v.label}
              </button>
            )
          })}
        </div>

        {/* Provider tabs (heatmap view) */}
        {view === 'heatmap' && (
          <div className="flex gap-1.5 overflow-x-auto scrollbar-none flex-1 min-w-0">
            <button onClick={() => setSelectedProvider('all')} className={cn(
              'px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all border shrink-0',
              selectedProvider === 'all'
                ? cn(accentBg, accentText, 'border-transparent')
                : (isLight ? 'text-stone-500 border-stone-200/60 hover:bg-stone-100' : 'text-[#6d6e72] border-white/[0.06] hover:bg-white/[0.04]'),
            )}>Overview</button>
            <button onClick={() => setSelectedProvider('features')} className={cn(
              'px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all border flex items-center gap-1 shrink-0',
              selectedProvider === 'features'
                ? cn(accentBg, accentText, 'border-transparent')
                : (isLight ? 'text-stone-500 border-stone-200/60 hover:bg-stone-100' : 'text-[#6d6e72] border-white/[0.06] hover:bg-white/[0.04]'),
            )}><Grid3X3 className="w-3 h-3" />Services</button>
            <div className={cn('w-px h-5 self-center mx-1 shrink-0',
              isLight ? 'bg-stone-300/60' : 'bg-white/[0.08]'
            )} />
            {modelProviders.map(p => (
              <button key={p.name} onClick={() => setSelectedProvider(p.name)} className={cn(
                'px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all border shrink-0',
                selectedProvider === p.name
                  ? cn(accentBg, accentText, 'border-transparent')
                  : (isLight ? 'text-stone-500 border-stone-200/60 hover:bg-stone-100' : 'text-[#6d6e72] border-white/[0.06] hover:bg-white/[0.04]'),
              )}>{shortProviderName(p.name)}</button>
            ))}
          </div>
        )}

        {/* Search */}
        <div className="relative lg:ml-auto">
          <Search className={cn('absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5',
            isLight ? 'text-stone-400' : 'text-[#4a4d54]'
          )} />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search regions..."
            className={cn(
              'pl-8 pr-3 py-1.5 rounded-lg text-xs border w-48',
              isLight
                ? 'bg-white/70 border-stone-200/60 text-stone-700 placeholder:text-stone-400'
                : 'bg-white/[0.03] border-white/[0.06] text-[#e4e5e7] placeholder:text-[#4a4d54]',
            )}
          />
        </div>
      </div>

      {/* Content */}
      {view === 'heatmap' && (
        <HeatmapView
          providers={providers}
          regions={regions}
          selectedProvider={selectedProvider}
          setSelectedProvider={setSelectedProvider}
          search={search}
          isLight={isLight}
          cardCls={cardCls}
        />
      )}
      {view === 'regions' && (
        <RegionCardsView
          providers={providers}
          regions={regions}
          search={search}
          isLight={isLight}
          cardCls={cardCls}
        />
      )}
      {view === 'roadmap' && (
        <RoadmapView
          upcomingLaunches={upcomingLaunches}
          providers={providers}
          regions={regions}
          isLight={isLight}
          cardCls={cardCls}
        />
      )}
    </div>
  )
}
