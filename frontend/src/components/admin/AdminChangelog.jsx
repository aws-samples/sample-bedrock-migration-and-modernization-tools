import { useState, useMemo } from 'react'
import { FileText, Search, Filter, Info, User, Tag, Calendar } from 'lucide-react'
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

      {/* Info Banner */}
      <div
        className={cn(
          'flex items-start gap-3 p-4 rounded-xl border',
          isLight
            ? 'bg-amber-50/50 border-amber-200/60 text-amber-800'
            : 'bg-[#1A9E7A]/10 border-[#1A9E7A]/20 text-[#1A9E7A]'
        )}
      >
        <Info className="h-5 w-5 flex-shrink-0 mt-0.5" />
        <div className="text-sm">
          <p className="font-medium">How to add changelog entries</p>
          <p className={cn('mt-1', isLight ? 'text-amber-700' : 'text-[#1A9E7A]/80')}>
            Edit <code className="px-1.5 py-0.5 rounded bg-black/10 font-mono text-xs">frontend/src/data/changelog-data.json</code> and commit your changes.
            Each entry should include a unique id, date, type, title, description, author, and optional tags.
          </p>
        </div>
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

      {/* Changelog Entries */}
      <div className="space-y-4">
        {filteredEntries.length === 0 ? (
          <div
            className={cn(
              'flex items-center justify-center py-16 text-sm',
              isLight ? 'text-stone-400' : 'text-slate-500'
            )}
          >
            No entries found
          </div>
        ) : (
          filteredEntries.map((entry) => {
            const typeConfig = TYPE_CONFIG[entry.type] || TYPE_CONFIG.feature
            return (
              <div key={entry.id} className={cardCls}>
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div className="flex items-center gap-2 flex-wrap">
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
                        'text-base font-semibold',
                        isLight ? 'text-stone-900' : 'text-white'
                      )}
                    >
                      {entry.title}
                    </h3>
                  </div>
                  <div
                    className={cn(
                      'flex items-center gap-1.5 text-xs flex-shrink-0',
                      isLight ? 'text-stone-400' : 'text-slate-500'
                    )}
                  >
                    <Calendar className="h-3.5 w-3.5" />
                    {entry.date}
                  </div>
                </div>

                <p
                  className={cn(
                    'text-sm leading-relaxed mb-4',
                    isLight ? 'text-stone-600' : 'text-slate-400'
                  )}
                >
                  {entry.description}
                </p>

                <div className="flex items-center justify-between flex-wrap gap-3">
                  {/* Author */}
                  <div
                    className={cn(
                      'flex items-center gap-1.5 text-xs',
                      isLight ? 'text-stone-500' : 'text-slate-500'
                    )}
                  >
                    <User className="h-3.5 w-3.5" />
                    {entry.author}
                  </div>

                  {/* Tags */}
                  {entry.tags && entry.tags.length > 0 && (
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <Tag
                        className={cn(
                          'h-3.5 w-3.5',
                          isLight ? 'text-stone-400' : 'text-slate-500'
                        )}
                      />
                      {entry.tags.map((tag) => (
                        <span
                          key={tag}
                          className={cn(
                            'text-xs px-2 py-0.5 rounded-md',
                            isLight
                              ? 'bg-stone-100 text-stone-600'
                              : 'bg-white/[0.06] text-slate-400'
                          )}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
