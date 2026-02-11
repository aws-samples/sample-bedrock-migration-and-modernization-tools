import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, ComposedChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { MapContainer, TileLayer, CircleMarker, Tooltip as LeafletTooltip } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import {
  BarChart3, Eye, Users, UserCheck, Activity, RefreshCw, Loader2, AlertCircle,
  TrendingUp, TrendingDown, Minus, Download, Globe, MousePointerClick, Star,
  GitCompare, ArrowUpRight, ArrowDownRight, Calendar, ChevronDown, Zap, Clock,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useAuthStore } from '@/stores/authStore'
import { fetchDashboardData } from '@/services/analyticsApi'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Popover, PopoverTrigger, PopoverContent } from '@/components/ui/popover'
import { Calendar as CalendarPicker } from '@/components/ui/calendar'
import { format } from 'date-fns'
import { pctChange, fmt, exportCsv, COUNTRY_COORDS, CHART_COLORS, LIGHT_CHART_COLORS } from './utils/dashboardUtils'

const PRESETS = [
  { label: 'Today', days: 1 },
  { label: '7d', days: 7 },
  { label: '30d', days: 30 },
  { label: '90d', days: 90 },
]

export function AdminDashboard() {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const accessToken = useAuthStore((s) => s.accessToken)

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [preset, setPreset] = useState(30)
  const [customRange, setCustomRange] = useState(null)
  const [calendarRange, setCalendarRange] = useState(undefined)
  const [calendarOpen, setCalendarOpen] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const opts = customRange
        ? { start: customRange.start, end: customRange.end }
        : { days: preset }
      const result = await fetchDashboardData(opts, accessToken)
      setData(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [preset, customRange, accessToken])

  useEffect(() => { loadData() }, [loadData])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(loadData, 60000)
    return () => clearInterval(id)
  }, [autoRefresh, loadData])

  const accent = isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
  const accentBg = isLight ? 'bg-amber-600' : 'bg-[#1A9E7A]'
  const accentText = isLight ? 'text-[#faf9f5]' : 'text-[#ffffff]'
  const cardCls = cn(
    'rounded-2xl border p-5 backdrop-blur-xl transition-colors',
    isLight
      ? 'bg-white/70 border-stone-200/60 shadow-[0_2px_15px_-3px_rgba(120,113,108,0.08)]'
      : 'bg-white/[0.03] border-white/[0.06] shadow-[0_2px_15px_-3px_rgba(0,0,0,0.3)]'
  )
  const axisColor = isLight ? '#78716c' : '#94a3b8'
  const gridColor = isLight ? '#d6d3d1' : '#1e293b'

  // Theme-aware chart palette
  const c1 = isLight ? '#b45309' : '#1A9E7A'    // primary: amber-700 / teal
  const c2 = isLight ? '#6366f1' : '#3B82F6'    // secondary: indigo-500 / blue-500
  const c3 = isLight ? '#ec4899' : '#F59E0B'    // tertiary: pink-500 / amber-500
  const chartColors = isLight ? LIGHT_CHART_COLORS : CHART_COLORS

  const tooltipProps = {
    contentStyle: {
      backgroundColor: isLight ? 'rgba(255,255,255,0.92)' : 'rgba(15,23,42,0.92)',
      backdropFilter: 'blur(16px)',
      WebkitBackdropFilter: 'blur(16px)',
      border: `1px solid ${isLight ? 'rgba(214,211,209,0.5)' : 'rgba(100,116,139,0.2)'}`,
      borderRadius: 12,
      fontSize: 12,
      color: isLight ? '#44403c' : '#e2e8f0',
      boxShadow: isLight
        ? '0 8px 32px -4px rgba(120,113,108,0.12)'
        : '0 8px 32px -4px rgba(0,0,0,0.5)',
      padding: '8px 12px',
    },
    labelStyle: { color: isLight ? '#292524' : '#f1f5f9', fontWeight: 600, fontSize: 12 },
    itemStyle: { color: isLight ? '#44403c' : '#cbd5e1', fontSize: 11, padding: '2px 0' },
    cursor: { fill: isLight ? 'rgba(120,113,108,0.06)' : 'rgba(148,163,184,0.06)' },
  }
  const legendStyle = { fontSize: 11, paddingTop: 8 }
  const legendFormatter = (value) => <span style={{ color: axisColor, fontSize: 11 }}>{value}</span>

  if (loading && !data) {
    return (
      <div className="flex flex-col items-center justify-center py-24">
        <Loader2 className={cn('h-8 w-8 animate-spin', accent)} />
        <p className={cn('mt-4 text-sm', isLight ? 'text-stone-500' : 'text-slate-400')}>Loading analytics...</p>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="flex flex-col items-center justify-center py-24">
        <AlertCircle className="h-8 w-8 text-red-400" />
        <p className="mt-4 text-sm text-red-400">{error}</p>
        <button onClick={loadData} className={cn('mt-4 px-4 py-2 rounded-lg text-sm font-medium', isLight ? 'bg-stone-100 text-stone-700 hover:bg-stone-200' : 'bg-slate-800 text-slate-300 hover:bg-slate-700')}>Retry</button>
      </div>
    )
  }

  const { summary = {}, previousPeriod = {}, timeSeries = [], hourlySeries = [], countries = [], regions = [], period = {} } = data || {}
  const prev = previousPeriod

  const handlePreset = (days) => {
    setCustomRange(null)
    setCalendarRange(undefined)
    setPreset(days)
    setCalendarOpen(false)
  }

  const applyCalendarRange = () => {
    if (calendarRange?.from && calendarRange?.to) {
      setCustomRange({
        start: format(calendarRange.from, 'yyyy-MM-dd'),
        end: format(calendarRange.to, 'yyyy-MM-dd'),
      })
      setPreset(null)
      setCalendarOpen(false)
    }
  }

  return (
    <div className="space-y-5 p-4 md:p-6 max-w-[1400px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <BarChart3 className={cn('h-6 w-6', accent)} />
          <h1 className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>Analytics</h1>
          {period.start && (
            <span className={cn('text-xs px-2 py-0.5 rounded-full', isLight ? 'bg-stone-100 text-stone-500' : 'bg-slate-800 text-slate-400')}>
              {period.start === period.end ? period.start : `${period.start} — ${period.end}`}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Presets */}
          <div className={cn('flex rounded-xl border overflow-hidden backdrop-blur-sm', isLight ? 'border-stone-200/60 bg-white/50' : 'border-white/[0.06] bg-white/[0.02]')}>
            {PRESETS.map((p) => (
              <button key={p.days} onClick={() => handlePreset(p.days)}
                className={cn('px-3 py-1.5 text-xs font-medium transition-colors',
                  preset === p.days && !customRange
                    ? cn(accentText, accentBg)
                    : isLight ? 'bg-white/60 text-stone-800 hover:bg-white/80' : 'bg-transparent text-slate-400 hover:bg-white/[0.06]'
                )}>
                {p.label}
              </button>
            ))}
            <Popover open={calendarOpen} onOpenChange={setCalendarOpen}>
              <PopoverTrigger asChild>
                <button
                  className={cn('px-3 py-1.5 text-xs font-medium transition-colors flex items-center gap-1',
                    customRange
                      ? cn(accentText, accentBg)
                      : isLight ? 'bg-white/60 text-stone-800 hover:bg-white/80' : 'bg-transparent text-slate-400 hover:bg-white/[0.06]'
                  )}>
                  <Calendar className="h-3 w-3" />
                  {customRange
                    ? `${customRange.start} — ${customRange.end}`
                    : 'Custom'}
                </button>
              </PopoverTrigger>
              <PopoverContent
                align="end"
                className={cn(
                  'w-auto rounded-2xl border backdrop-blur-xl',
                  isLight
                    ? 'bg-white/90 border-stone-200/60 shadow-lg'
                    : 'bg-[#1a1b1e]/95 border-white/[0.08] shadow-2xl'
                )}
              >
                <div className="p-1">
                  <CalendarPicker
                    mode="range"
                    selected={calendarRange}
                    onSelect={setCalendarRange}
                    numberOfMonths={2}
                    disabled={{ after: new Date() }}
                  />
                  <div className={cn(
                    'flex items-center justify-between px-3 py-2 border-t',
                    isLight ? 'border-stone-200/60' : 'border-white/[0.06]'
                  )}>
                    <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
                      {calendarRange?.from && calendarRange?.to
                        ? `${format(calendarRange.from, 'MMM d')} — ${format(calendarRange.to, 'MMM d, yyyy')}`
                        : 'Select a date range'}
                    </span>
                    <button
                      onClick={applyCalendarRange}
                      disabled={!calendarRange?.from || !calendarRange?.to}
                      className={cn(
                        'px-4 py-1.5 rounded-xl text-xs font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed',
                        accentText, accentBg,
                        isLight ? 'hover:bg-amber-700' : 'hover:bg-[#22b38d]'
                      )}
                    >
                      Apply
                    </button>
                  </div>
                </div>
              </PopoverContent>
            </Popover>
          </div>

          {/* Auto-refresh */}
          <button onClick={() => setAutoRefresh(!autoRefresh)}
            className={cn('px-2.5 py-1.5 rounded-xl text-xs font-medium transition-colors flex items-center gap-1 backdrop-blur-sm',
              autoRefresh ? cn(accentText, accentBg) : isLight ? 'text-stone-500 bg-stone-100/60 hover:bg-stone-200/60' : 'text-slate-400 bg-white/[0.04] hover:bg-white/[0.08]')}>
            <Zap className="h-3 w-3" /> Live
          </button>

          {/* CSV */}
          <button onClick={() => exportCsv(data)}
            className={cn('p-2 rounded-xl transition-colors backdrop-blur-sm', isLight ? 'text-stone-500 hover:bg-stone-100/60' : 'text-slate-400 hover:bg-white/[0.06]')}>
            <Download className="h-4 w-4" />
          </button>

          {/* Refresh */}
          <button onClick={loadData} disabled={loading}
            className={cn('p-2 rounded-xl transition-colors backdrop-blur-sm', isLight ? 'text-stone-500 hover:bg-stone-100/60' : 'text-slate-400 hover:bg-white/[0.06]')}>
            <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Calendar date picker is now inline via Popover above */}

      {/* Tabs */}
      <Tabs defaultValue="overview">
        <TabsList className="w-full justify-start">
          <TabsTrigger value="overview" className="text-xs">Overview</TabsTrigger>
          <TabsTrigger value="audience" className="text-xs">Audience</TabsTrigger>
          <TabsTrigger value="content" className="text-xs">Content</TabsTrigger>
          <TabsTrigger value="realtime" className="text-xs">Realtime</TabsTrigger>
        </TabsList>

        {/* ═══ OVERVIEW TAB ═══ */}
        <TabsContent value="overview" className="space-y-5 mt-4">
          {/* KPI Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
            <KpiCard icon={Eye} label="Total Views" value={summary.totalViews} prev={prev.totalViews} sparkData={timeSeries.map(d => d.views)} isLight={isLight} />
            <KpiCard icon={Users} label="Unique Users" value={summary.uniqueUsers} prev={prev.uniqueUsers} sparkData={timeSeries.map(d => d.uniqueUsers)} isLight={isLight} />
            <KpiCard icon={UserCheck} label="New Users" value={summary.newUsers} prev={prev.newUsers} isLight={isLight} />
            <KpiCard icon={TrendingUp} label="Returning" value={summary.returningUsers} prev={prev.returningUsers} isLight={isLight} />
            <KpiCard icon={Activity} label="Active Today" value={summary.activeToday} isLight={isLight} />
          </div>

          {/* Usage Over Time */}
          <div className={cardCls}>
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
              <h3 className={cn('text-sm font-semibold', isLight ? 'text-stone-700' : 'text-slate-300')}>Usage Over Time</h3>
            </div>
            {timeSeries.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={timeSeries}>
                  <defs>
                    <linearGradient id="viewsGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={c1} stopOpacity={0.25} />
                      <stop offset="95%" stopColor={c1} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: axisColor }} tickFormatter={(d) => d.slice(5)} />
                  <YAxis tick={{ fontSize: 11, fill: axisColor }} />
                  <Tooltip {...tooltipProps} />
                  <Legend iconSize={10} wrapperStyle={legendStyle} formatter={legendFormatter} />
                  <Bar dataKey="views" name="Views" fill={c1} fillOpacity={0.75} radius={[4, 4, 0, 0]} />
                  <Line type="monotone" dataKey="uniqueUsers" name="Unique Users" stroke={c2} strokeWidth={2} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            ) : <EmptyState isLight={isLight} />}
          </div>

          {/* Section + Feature Row */}
          <div className="grid md:grid-cols-2 gap-5">
            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-4', isLight ? 'text-stone-700' : 'text-slate-300')}>Section Usage</h3>
              {Object.keys(summary.sectionUsage || {}).filter(k => k !== 'admin').length > 0 ? (() => {
                const sectionData = Object.entries(summary.sectionUsage).filter(([name]) => name !== 'admin').map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value })).sort((a, b) => b.value - a.value)
                return (
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={sectionData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                      <XAxis type="number" tick={{ fontSize: 11, fill: axisColor }} />
                      <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 11, fill: axisColor }} />
                      <Tooltip {...tooltipProps} />
                      <Bar dataKey="value" name="Events" radius={[0, 4, 4, 0]}>
                        {sectionData.map((_, i) => <Cell key={i} fill={chartColors[i % chartColors.length]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                )
              })() : <EmptyState isLight={isLight} />}
            </div>

            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-4', isLight ? 'text-stone-700' : 'text-slate-300')}>Feature Breakdown</h3>
              {Object.values(summary.featureUsage || {}).some(v => v > 0) ? (
                <div className="flex items-center justify-center">
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie data={[
                        { name: 'Model Details', value: summary.featureUsage?.modelDetails || 0 },
                        { name: 'Comparisons', value: summary.featureUsage?.comparisons || 0 },
                        { name: 'Favorites', value: summary.featureUsage?.favorites || 0 },
                      ].filter(d => d.value > 0)} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value"
                        label={({ name, percent, x, y, textAnchor }) => (
                          <text x={x} y={y} textAnchor={textAnchor} fill={axisColor} fontSize={11} fontFamily="inherit">
                            {`${name} ${(percent * 100).toFixed(0)}%`}
                          </text>
                        )}>
                        {[0, 1, 2].map(i => <Cell key={i} fill={chartColors[i]} />)}
                      </Pie>
                      <Tooltip {...tooltipProps} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              ) : <EmptyState isLight={isLight} />}
            </div>
          </div>
        </TabsContent>

        {/* ═══ AUDIENCE TAB ═══ */}
        <TabsContent value="audience" className="space-y-5 mt-4">
          {/* Audience KPIs */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <KpiCard icon={Users} label="Unique Users" value={summary.uniqueUsers} prev={prev.uniqueUsers} isLight={isLight} />
            <KpiCard icon={UserCheck} label="New Users" value={summary.newUsers} prev={prev.newUsers} isLight={isLight} />
            <KpiCard icon={TrendingUp} label="Returning" value={summary.returningUsers} prev={prev.returningUsers} isLight={isLight} />
            <KpiCard icon={Activity} label="Avg Daily Views" value={Math.round(summary.avgDailyViews || 0)} prev={Math.round(prev.avgDailyViews || 0)} isLight={isLight} />
          </div>

          {/* World Map + Country List */}
          <div className="grid lg:grid-cols-5 gap-5">
            <div className={cn(cardCls, 'lg:col-span-3')}>
              <div className="flex items-center gap-2 mb-3">
                <Globe className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
                <h3 className={cn('text-sm font-semibold', isLight ? 'text-stone-700' : 'text-slate-300')}>
                  Traffic by Country
                </h3>
              </div>
              <div className="rounded-lg overflow-hidden" style={{ height: 340 }}>
                <MapContainer center={[30, 0]} zoom={2} scrollWheelZoom={true} zoomControl={true}
                  minZoom={2} maxBounds={[[-85, -180], [85, 180]]} maxBoundsViscosity={1.0}
                  style={{ height: '100%', width: '100%', background: isLight ? '#faf9f5' : 'transparent' }}
                  attributionControl={false}>
                  <TileLayer
                    url={isLight
                      ? 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png'
                      : 'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png'}
                  />
                  {(summary.countryCounts || []).map((cc) => {
                    const coords = COUNTRY_COORDS[cc.id]
                    if (!coords) return null
                    return (
                      <CircleMarker key={cc.id} center={coords}
                        radius={Math.max(5, Math.min(20, cc.count * 3))}
                        pathOptions={{
                          fillColor: c1,
                          fillOpacity: 0.6,
                          color: c1,
                          weight: 1,
                          opacity: 0.8,
                        }}>
                        <LeafletTooltip>{cc.id}: {cc.count} sessions</LeafletTooltip>
                      </CircleMarker>
                    )
                  })}
                </MapContainer>
              </div>
            </div>

            <div className={cn(cardCls, 'lg:col-span-2')}>
              <div className="flex items-center justify-between mb-3">
                <h3 className={cn('text-sm font-semibold', isLight ? 'text-stone-700' : 'text-slate-300')}>Countries</h3>
                <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-slate-500')}>{countries.length} total</span>
              </div>
              <div className="space-y-1 max-h-[310px] overflow-y-auto pr-1">
                {(summary.countryCounts || []).map((cc, i) => {
                  const maxCount = (summary.countryCounts?.[0]?.count) || 1
                  return (
                    <div key={cc.id} className={cn('flex items-center gap-2 py-1.5 px-2 rounded-lg transition-colors', isLight ? 'hover:bg-stone-100/50' : 'hover:bg-white/[0.04]')}>
                      <span className={cn('text-xs font-mono w-4 text-right', isLight ? 'text-stone-400' : 'text-slate-500')}>{i + 1}</span>
                      <span className={cn('text-sm font-medium w-8', isLight ? 'text-stone-700' : 'text-slate-300')}>{cc.id}</span>
                      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: isLight ? '#e7e5e4' : '#1e293b' }}>
                        <div className="h-full rounded-full transition-all" style={{
                          width: `${(cc.count / maxCount) * 100}%`,
                          backgroundColor: c1,
                        }} />
                      </div>
                      <span className={cn('text-xs font-medium tabular-nums w-8 text-right', isLight ? 'text-stone-600' : 'text-slate-400')}>{cc.count}</span>
                    </div>
                  )
                })}
                {(summary.countryCounts || []).length === 0 && <EmptyState isLight={isLight} />}
              </div>
            </div>
          </div>

          {/* Users Over Time + New vs Returning Donut */}
          <div className="grid md:grid-cols-3 gap-5">
            <div className={cn(cardCls, 'md:col-span-2')}>
              <h3 className={cn('text-sm font-semibold mb-4', isLight ? 'text-stone-700' : 'text-slate-300')}>Users Over Time</h3>
              {timeSeries.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={timeSeries}>
                    <defs>
                      <linearGradient id="newGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={c2} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={c2} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="retGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={c1} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={c1} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                    <XAxis dataKey="date" tick={{ fontSize: 11, fill: axisColor }} tickFormatter={(d) => d.slice(5)} />
                    <YAxis tick={{ fontSize: 11, fill: axisColor }} />
                    <Tooltip {...tooltipProps} />
                    <Legend iconSize={10} wrapperStyle={legendStyle} formatter={legendFormatter} />
                    <Area type="monotone" dataKey="returningUsers" name="Returning" stackId="1" stroke={c1} fill="url(#retGrad)" strokeWidth={2} />
                    <Area type="monotone" dataKey="newUsers" name="New" stackId="1" stroke={c2} fill="url(#newGrad)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : <EmptyState isLight={isLight} />}
            </div>

            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-4', isLight ? 'text-stone-700' : 'text-slate-300')}>New vs Returning</h3>
              {(summary.newUsers > 0 || summary.returningUsers > 0) ? (
                <div className="flex flex-col items-center">
                  <ResponsiveContainer width="100%" height={180}>
                    <PieChart>
                      <Pie data={[
                        { name: 'New', value: summary.newUsers || 0 },
                        { name: 'Returning', value: summary.returningUsers || 0 },
                      ]} cx="50%" cy="50%" innerRadius={45} outerRadius={70} dataKey="value">
                        <Cell fill={c2} />
                        <Cell fill={c1} />
                      </Pie>
                      <Tooltip {...tooltipProps} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex gap-4 mt-2">
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c2 }} />
                      <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>New {summary.newUsers || 0}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c1 }} />
                      <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Returning {summary.returningUsers || 0}</span>
                    </div>
                  </div>
                </div>
              ) : <EmptyState isLight={isLight} />}
            </div>
          </div>

          {/* Regions */}
          {regions.length > 0 && (
            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-3', isLight ? 'text-stone-700' : 'text-slate-300')}>Regions ({regions.length})</h3>
              <div className="flex flex-wrap gap-1.5">
                {regions.sort().map((r) => (
                  <span key={r} className={cn('inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-medium backdrop-blur-sm',
                    isLight ? 'bg-amber-50/80 text-amber-700 border border-amber-200/50' : 'bg-[#1A9E7A]/10 text-[#1A9E7A] border border-[#1A9E7A]/20')}>{r}</span>
                ))}
              </div>
            </div>
          )}
        </TabsContent>

        {/* ═══ CONTENT TAB ═══ */}
        <TabsContent value="content" className="space-y-5 mt-4">
          {/* Feature KPIs */}
          <div className="grid grid-cols-3 gap-3">
            <KpiCard icon={MousePointerClick} label="Detail Opens" value={summary.featureUsage?.modelDetails || 0} prev={prev.featureUsage?.modelDetails || 0} sparkData={timeSeries.map(d => d.detailOpens)} isLight={isLight} />
            <KpiCard icon={GitCompare} label="Comparisons" value={summary.featureUsage?.comparisons || 0} prev={prev.featureUsage?.comparisons || 0} sparkData={timeSeries.map(d => d.comparisonAdds)} isLight={isLight} />
            <KpiCard icon={Star} label="Favorites" value={summary.featureUsage?.favorites || 0} prev={prev.featureUsage?.favorites || 0} sparkData={timeSeries.map(d => d.favoriteToggles)} isLight={isLight} />
          </div>

          {/* Feature Usage Over Time */}
          <div className={cardCls}>
            <h3 className={cn('text-sm font-semibold mb-4', isLight ? 'text-stone-700' : 'text-slate-300')}>Feature Usage Over Time</h3>
            {timeSeries.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={timeSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: axisColor }} tickFormatter={(d) => d.slice(5)} />
                  <YAxis tick={{ fontSize: 11, fill: axisColor }} />
                  <Tooltip {...tooltipProps} />
                  <Legend iconSize={10} wrapperStyle={legendStyle} formatter={legendFormatter} />
                  <Bar dataKey="detailOpens" name="Detail Opens" stackId="a" fill={c1} radius={[0, 0, 0, 0]} />
                  <Bar dataKey="comparisonAdds" name="Comparisons" stackId="a" fill={c2} />
                  <Bar dataKey="favoriteToggles" name="Favorites" stackId="a" fill={c3} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyState isLight={isLight} />}
          </div>

          {/* Model Rankings */}
          <div className="grid md:grid-cols-3 gap-5">
            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-3', isLight ? 'text-stone-700' : 'text-slate-300')}>Top Models Viewed</h3>
              <RankedList items={summary.topModels} isLight={isLight} colors={chartColors} />
            </div>
            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-3', isLight ? 'text-stone-700' : 'text-slate-300')}>Most Compared</h3>
              <RankedList items={summary.topComparedModels} isLight={isLight} colors={chartColors} />
            </div>
            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-3', isLight ? 'text-stone-700' : 'text-slate-300')}>Most Favorited</h3>
              <RankedList items={summary.topFavoritedModels} isLight={isLight} colors={chartColors} />
            </div>
          </div>

          {/* Provider Breakdown */}
          <div className="grid md:grid-cols-2 gap-5">
            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-4', isLight ? 'text-stone-700' : 'text-slate-300')}>Comparisons by Provider</h3>
              <ProviderBar items={summary.providerComparisons} isLight={isLight} axisColor={axisColor} gridColor={gridColor} tipProps={tooltipProps} colors={chartColors} />
            </div>
            <div className={cardCls}>
              <h3 className={cn('text-sm font-semibold mb-4', isLight ? 'text-stone-700' : 'text-slate-300')}>Favorites by Provider</h3>
              <ProviderBar items={summary.providerFavorites} isLight={isLight} axisColor={axisColor} gridColor={gridColor} tipProps={tooltipProps} colors={chartColors} />
            </div>
          </div>
        </TabsContent>

        {/* ═══ REALTIME TAB ═══ */}
        <TabsContent value="realtime" className="space-y-5 mt-4">
          {/* Live indicator */}
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
            </span>
            <span className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>
              Today — {autoRefresh ? 'auto-refreshing every 60s' : 'click Live to auto-refresh'}
            </span>
          </div>

          {/* Today KPIs */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <KpiCard icon={Eye} label="Views Today" value={summary.activeToday > 0 ? (timeSeries.find(d => d.date === period.end)?.views || 0) : 0} isLight={isLight} />
            <KpiCard icon={Users} label="Users Today" value={summary.activeToday || 0} isLight={isLight} />
            <KpiCard icon={MousePointerClick} label="Detail Opens" value={timeSeries.find(d => d.date === period.end)?.detailOpens || 0} isLight={isLight} />
            <KpiCard icon={GitCompare} label="Comparisons" value={timeSeries.find(d => d.date === period.end)?.comparisonAdds || 0} isLight={isLight} />
          </div>

          {/* Hourly Activity */}
          <div className={cardCls}>
            <div className="flex items-center gap-2 mb-4">
              <Clock className={cn('h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
              <h3 className={cn('text-sm font-semibold', isLight ? 'text-stone-700' : 'text-slate-300')}>Hourly Activity (UTC)</h3>
            </div>
            {hourlySeries.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={hourlySeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                  <XAxis dataKey="hour" tick={{ fontSize: 10, fill: axisColor }} />
                  <YAxis tick={{ fontSize: 11, fill: axisColor }} />
                  <Tooltip {...tooltipProps} />
                  <Legend iconSize={10} wrapperStyle={legendStyle} formatter={legendFormatter} />
                  <Bar dataKey="events" name="Events" fill={c1} radius={[4, 4, 0, 0]} fillOpacity={0.85} />
                  <Bar dataKey="uniqueUsers" name="Users" fill={c2} radius={[4, 4, 0, 0]} fillOpacity={0.85} />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyState isLight={isLight} />}
          </div>

          {/* Today's breakdown row */}
          {(() => {
            const todayEntry = timeSeries.find(d => d.date === period.end)
            const todaySections = todayEntry?.sections || {}
            const todaySectionData = Object.entries(todaySections).filter(([k]) => k !== 'admin').map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value })).filter(d => d.value > 0)
            const todayFeatures = [
              { name: 'Detail Opens', value: todayEntry?.detailOpens || 0 },
              { name: 'Comparisons', value: todayEntry?.comparisonAdds || 0 },
              { name: 'Favorites', value: todayEntry?.favoriteToggles || 0 },
            ].filter(d => d.value > 0)
            const todayCountries = todayEntry?.countries || []

            return (
              <div className="grid md:grid-cols-3 gap-5">
                {/* Today's Section Activity */}
                <div className={cardCls}>
                  <h3 className={cn('text-sm font-semibold mb-3', isLight ? 'text-stone-700' : 'text-slate-300')}>Sections Today</h3>
                  {todaySectionData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={180}>
                      <PieChart>
                        <Pie data={todaySectionData} cx="50%" cy="50%" innerRadius={40} outerRadius={70} dataKey="value"
                          label={({ name, percent, x, y, textAnchor }) => (
                            <text x={x} y={y} textAnchor={textAnchor} fill={axisColor} fontSize={11} fontFamily="inherit">
                              {`${name} ${(percent * 100).toFixed(0)}%`}
                            </text>
                          )}>
                          {todaySectionData.map((_, i) => <Cell key={i} fill={chartColors[i % chartColors.length]} />)}
                        </Pie>
                        <Tooltip {...tooltipProps} />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : <EmptyState isLight={isLight} />}
                </div>

                {/* Today's Feature Breakdown */}
                <div className={cardCls}>
                  <h3 className={cn('text-sm font-semibold mb-3', isLight ? 'text-stone-700' : 'text-slate-300')}>Features Today</h3>
                  {todayFeatures.length > 0 ? (
                    <div className="space-y-3 pt-2">
                      {todayFeatures.map((feat, i) => {
                        const maxVal = Math.max(...todayFeatures.map(f => f.value), 1)
                        return (
                          <div key={feat.name}>
                            <div className="flex justify-between items-center mb-1">
                              <span className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>{feat.name}</span>
                              <span className={cn('text-xs font-bold tabular-nums', isLight ? 'text-stone-900' : 'text-white')}>{feat.value}</span>
                            </div>
                            <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: isLight ? '#e7e5e4' : '#1e293b' }}>
                              <div className="h-full rounded-full transition-all" style={{ width: `${(feat.value / maxVal) * 100}%`, backgroundColor: chartColors[i % chartColors.length] }} />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : <EmptyState isLight={isLight} />}
                </div>

                {/* Today's Countries */}
                <div className={cardCls}>
                  <h3 className={cn('text-sm font-semibold mb-3', isLight ? 'text-stone-700' : 'text-slate-300')}>Countries Today</h3>
                  {todayCountries.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {todayCountries.sort().map((c) => (
                        <span key={c} className={cn('inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-medium backdrop-blur-sm',
                          isLight ? 'bg-amber-50/80 text-amber-700 border border-amber-200/50' : 'bg-[#1A9E7A]/10 text-[#1A9E7A] border border-[#1A9E7A]/20')}>{c}</span>
                      ))}
                    </div>
                  ) : <EmptyState isLight={isLight} />}
                </div>
              </div>
            )
          })()}
        </TabsContent>
      </Tabs>
    </div>
  )
}


// ─── Sub-components ────────────────────────────────────────────────────────

function KpiCard({ icon: Icon, label, value, prev, sparkData, isLight }) {
  const change = prev != null ? pctChange(value || 0, prev) : null
  const up = change > 0
  const down = change < 0

  return (
    <div className={cn(
      'rounded-2xl border p-4 flex flex-col gap-1 backdrop-blur-xl transition-colors',
      isLight
        ? 'bg-white/70 border-stone-200/60 shadow-[0_2px_15px_-3px_rgba(120,113,108,0.08)]'
        : 'bg-white/[0.03] border-white/[0.06] shadow-[0_2px_15px_-3px_rgba(0,0,0,0.3)]'
    )}>
      <div className="flex items-center gap-2 mb-1">
        <Icon className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
        <span className={cn('text-xs font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>{label}</span>
      </div>
      <div className="flex items-end justify-between">
        <span className={cn('text-2xl font-bold tabular-nums', isLight ? 'text-stone-900' : 'text-white')}>
          {fmt(value || 0)}
        </span>
        {change != null && change !== 0 && (
          <span className={cn('flex items-center gap-0.5 text-xs font-medium',
            up ? 'text-emerald-600' : down ? 'text-red-500' : isLight ? 'text-stone-400' : 'text-slate-500')}>
            {up ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
            {Math.abs(change)}%
          </span>
        )}
      </div>
      {sparkData && sparkData.length > 1 && <Sparkline data={sparkData} isLight={isLight} />}
    </div>
  )
}

function Sparkline({ data, isLight }) {
  const h = 24
  const w = 100
  const max = Math.max(...data, 1)
  const min = Math.min(...data, 0)
  const range = max - min || 1
  const points = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * (h - 2) - 1}`).join(' ')

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full mt-1" style={{ height: 24 }}>
      <polyline points={points} fill="none"
        stroke={isLight ? '#b45309' : '#1A9E7A'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function RankedList({ items, isLight, colors, max = 8 }) {
  if (!items || items.length === 0) return <EmptyState isLight={isLight} />
  const palette = colors || CHART_COLORS
  const topCount = items[0]?.count || 1
  return (
    <div className="space-y-1 max-h-[280px] overflow-y-auto">
      {items.slice(0, max).map((m, i) => (
        <div key={m.id || m.modelId} className={cn('flex items-center gap-2 py-1.5 px-2 rounded-lg transition-colors', isLight ? 'hover:bg-stone-100/50' : 'hover:bg-white/[0.04]')}>
          <span className={cn('text-xs font-mono w-4 text-right flex-shrink-0', isLight ? 'text-stone-400' : 'text-slate-500')}>{i + 1}</span>
          <div className="flex-1 min-w-0">
            <span className={cn('text-xs truncate block', isLight ? 'text-stone-700' : 'text-slate-300')}>{m.id || m.modelId}</span>
            <div className="h-1.5 rounded-full mt-0.5 overflow-hidden" style={{ backgroundColor: isLight ? '#e7e5e4' : '#1e293b' }}>
              <div className="h-full rounded-full transition-all" style={{
                width: `${(m.count / topCount) * 100}%`,
                backgroundColor: palette[i % palette.length],
              }} />
            </div>
          </div>
          <span className={cn('text-xs font-medium tabular-nums flex-shrink-0', isLight ? 'text-stone-900' : 'text-white')}>{m.count}</span>
        </div>
      ))}
    </div>
  )
}

function ProviderBar({ items, isLight, axisColor, gridColor, tipProps, colors }) {
  if (!items || items.length === 0) return <EmptyState isLight={isLight} />
  const palette = colors || CHART_COLORS
  return (
    <ResponsiveContainer width="100%" height={Math.max(120, items.length * 36)}>
      <BarChart data={items.map(p => ({ name: p.id, value: p.count }))} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
        <XAxis type="number" tick={{ fontSize: 11, fill: axisColor }} />
        <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11, fill: axisColor }} />
        <Tooltip {...tipProps} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {items.map((_, i) => <Cell key={i} fill={palette[i % palette.length]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function EmptyState({ isLight }) {
  return (
    <div className={cn('flex items-center justify-center py-12 text-sm', isLight ? 'text-stone-400' : 'text-slate-500')}>
      No data available yet
    </div>
  )
}
