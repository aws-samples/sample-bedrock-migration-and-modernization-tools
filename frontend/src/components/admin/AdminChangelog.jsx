import { useState, useMemo } from 'react'
import { FileText, Search, Filter, ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'
import changelogData from '@/data/changelog-data.json'

const TYPE_CONFIG = {
  feature: {
    label: 'Feature',
    light: 'bg-emerald-100 text-emerald-700 border-emerald-200/60',
    dark: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
  },
  fix: {
    label: 'Fix',
    light: 'bg-red-100 text-red-700 border-red-200/60',
    dark: 'bg-red-500/15 text-red-400 border-red-500/20',
  },
  data: {
    label: 'Data',
    light: 'bg-blue-100 text-blue-700 border-blue-200/60',
    dark: 'bg-blue-500/15 text-blue-400 border-blue-500/20',
  },
  enhancement: {
    label: 'Enhancement',
    light: 'bg-purple-100 text-purple-700 border-purple-200/60',
    dark: 'bg-purple-500/15 text-purple-400 border-purple-500/20',
  },
  deprecation: {
    label: 'Deprecation',
    light: 'bg-amber-100 text-amber-700 border-amber-200/60',
    dark: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
  },
}

function getRelativeTime(dateStr) {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now - date
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffDays === 0) return 'today'
  if (diffDays === 1) return '1d ago'
  if (diffDays < 7) return `${diffDays}d ago`
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`
  if (diffDays < 365) return `${Math.floor(diffDays / 30)}mo ago`
  return `${Math.floor(diffDays / 365)}y ago`
}

function formatDate(dateStr) {
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

export function AdminChangelog() {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const [searchQuery, setSearchQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')

  const accent = isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
  const cardCls = cn(
    'rounded-2xl border p-5 backdrop-blur-xl transition-colors',
    isLight
      ? 'bg-white/70 border-stone-200/60 shadow-[0_2px_15px_-3px_rgba(120,113,108,0.08)]'
      : 'bg-white/[0.03] border-white/[0.06] shadow-[0_2px_15px_-3px_rgba(0,0,0,0.3)]'
  )

  const filteredEntries = useMemo(() => {
    return changelogData.entries.filter((entry) => {
      const matchesType = typeFilter === 'all' || entry.type === typeFilter
      const matchesSearch =
        searchQuery === '' ||
        entry.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        entry.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        entry.tags?.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      return matchesType && matchesSearch
    })
  }, [searchQuery, typeFilter])

  const groupedEntries = useMemo(() => {
    const groups = {}
    filteredEntries.forEach((entry) => {
      if (!groups[entry.date]) groups[entry.date] = []
      groups[entry.date].push(entry)
    })
    // Sort dates descending (newest first)
    return Object.entries(groups).sort(([a], [b]) => new Date(b) - new Date(a))
  }, [filteredEntries])

  const typeOptions = ['all', ...Object.keys(TYPE_CONFIG)]

  return (
    <div className="space-y-5 p-4 md:p-6 max-w-[1000px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <FileText className={cn('h-6 w-6', accent)} />
          <h1 className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>
            Changelog
          </h1>
          <span
            className={cn(
              'text-xs px-2 py-0.5 rounded-full',
              isLight ? 'bg-stone-100 text-stone-500' : 'bg-slate-800 text-slate-400'
            )}
          >
            {filteredEntries.length} {filteredEntries.length === 1 ? 'entry' : 'entries'}
          </span>
        </div>
        <span
          className={cn(
            'text-xs',
            isLight ? 'text-stone-400' : 'text-slate-500'
          )}
        >
          Last updated: {changelogData.lastUpdated}
        </span>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        {/* Search */}
        <div className="relative flex-1">
          <Search
            className={cn(
              'absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4',
              isLight ? 'text-stone-400' : 'text-slate-500'
            )}
          />
          <input
            type="text"
            placeholder="Search changelog..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className={cn(
              'w-full pl-10 pr-4 py-2.5 rounded-xl border text-sm transition-colors',
              isLight
                ? 'bg-white/80 border-stone-200/60 text-stone-900 placeholder:text-stone-400 focus:border-amber-300 focus:ring-1 focus:ring-amber-300'
                : 'bg-white/[0.04] border-white/[0.08] text-white placeholder:text-slate-500 focus:border-[#1A9E7A]/50 focus:ring-1 focus:ring-[#1A9E7A]/50'
            )}
          />
        </div>

        {/* Type Filter */}
        <div className="relative">
          <Filter
            className={cn(
              'absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4',
              isLight ? 'text-stone-400' : 'text-slate-500'
            )}
          />
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className={cn(
              'pl-10 pr-8 py-2.5 rounded-xl border text-sm transition-colors appearance-none cursor-pointer min-w-[140px]',
              isLight
                ? 'bg-white/80 border-stone-200/60 text-stone-900 focus:border-amber-300 focus:ring-1 focus:ring-amber-300'
                : 'bg-white/[0.04] border-white/[0.08] text-white focus:border-[#1A9E7A]/50 focus:ring-1 focus:ring-[#1A9E7A]/50'
            )}
          >
            {typeOptions.map((type) => (
              <option key={type} value={type}>
                {type === 'all' ? 'All types' : TYPE_CONFIG[type]?.label || type}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Timeline Entries */}
      {groupedEntries.length === 0 ? (
        <div
          className={cn(
            'flex items-center justify-center py-16 text-sm',
            isLight ? 'text-stone-400' : 'text-slate-500'
          )}
        >
          No entries found
        </div>
      ) : (
        <div className="relative">
          {groupedEntries.map(([date, entries], idx) => (
            <div key={date} className="flex gap-4 mb-4">
              {/* Timeline dot + line */}
              <div className="flex flex-col items-center w-8 flex-shrink-0">
                <div
                  className={cn(
                    'w-3 h-3 rounded-full border-2 mt-4',
                    isLight
                      ? 'bg-stone-300 border-stone-400'
                      : 'bg-slate-600 border-slate-500'
                  )}
                />
                {idx < groupedEntries.length - 1 && (
                  <div
                    className={cn(
                      'w-px flex-1 mt-1',
                      isLight ? 'bg-stone-200' : 'bg-white/[0.06]'
                    )}
                  />
                )}
              </div>

              {/* Content card */}
              <div className={cn(cardCls, 'flex-1 p-4')}>
                {/* Date header */}
                <div className="flex items-center gap-3 mb-4">
                  <span
                    className={cn(
                      'text-lg font-bold tabular-nums',
                      isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
                    )}
                  >
                    {formatDate(date)}
                  </span>
                  <span
                    className={cn(
                      'text-xs px-2 py-0.5 rounded-full font-medium',
                      isLight
                        ? 'bg-stone-100 text-stone-400'
                        : 'bg-white/[0.03] text-slate-500'
                    )}
                  >
                    {getRelativeTime(date)}
                  </span>
                </div>

                {/* Entries for this date */}
                <div className="space-y-4">
                  {entries.map((entry, entryIdx) => {
                    const typeConfig = TYPE_CONFIG[entry.type] || TYPE_CONFIG.feature
                    return (
                      <div key={entry.id}>
                        {/* Separator between entries */}
                        {entryIdx > 0 && (
                          <div
                            className={cn(
                              'border-t mb-4',
                              isLight ? 'border-stone-100' : 'border-white/[0.04]'
                            )}
                          />
                        )}

                        {/* Type badge + Title */}
                        <div className="flex items-center gap-2 flex-wrap mb-1.5">
                          <span
                            className={cn(
                              'text-[10px] font-bold leading-none px-2 py-1 rounded-full border uppercase',
                              isLight ? typeConfig.light : typeConfig.dark
                            )}
                          >
                            {typeConfig.label}
                          </span>
                          <h3
                            className={cn(
                              'text-sm font-semibold',
                              isLight ? 'text-stone-900' : 'text-white'
                            )}
                          >
                            {entry.title}
                          </h3>
                        </div>

                        {/* Description */}
                        <p
                          className={cn(
                            'text-sm leading-relaxed',
                            isLight ? 'text-stone-600' : 'text-slate-400'
                          )}
                        >
                          {entry.description}
                        </p>

                        {/* Tags */}
                        {entry.tags && entry.tags.length > 0 && (
                          <div className="flex items-center gap-1.5 mt-2 pl-1">
                            <ArrowRight
                              className={cn(
                                'w-3 h-3 shrink-0',
                                isLight ? 'text-amber-400' : 'text-[#1A9E7A]/60'
                              )}
                            />
                            <div className="flex flex-wrap gap-1">
                              {entry.tags.map((tag) => (
                                <span
                                  key={tag}
                                  className={cn(
                                    'text-xs px-1.5 py-0.5 rounded border font-mono',
                                    isLight
                                      ? 'bg-amber-50 text-amber-700 border-amber-200'
                                      : 'bg-[#1A9E7A]/10 text-[#1A9E7A] border-[#1A9E7A]/20'
                                  )}
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
