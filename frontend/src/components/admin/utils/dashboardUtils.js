export function pctChange(current, previous) {
  if (!previous || previous === 0) return current > 0 ? 100 : 0
  return Math.round(((current - previous) / previous) * 100)
}

export function fmt(n) {
  if (n == null) return '0'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return n.toLocaleString()
}

/**
 * Format a number as a compact percentage string.
 */
export function fmtPct(value, total) {
  if (!total || total === 0) return '0%'
  return `${Math.round((value / total) * 100)}%`
}

/**
 * Calculate engagement score (views per user per day).
 */
export function engagementScore(views, users, days) {
  if (!users || !days) return 0
  return Math.round((views / users / days) * 100) / 100
}

/**
 * Merge analytics country counts with Cognito country data.
 * Analytics = view counts per country, Cognito = registered users per country.
 * Returns combined data for the Audience tab.
 */
export function mergeCountryData(analyticsCounts = [], cognitoCountries = []) {
  const merged = new Map()

  // Analytics data (view counts)
  for (const { id, count } of analyticsCounts) {
    if (!merged.has(id)) merged.set(id, { id, views: 0, users: 0 })
    merged.get(id).views = count
  }

  // Cognito data (registered users)
  for (const { id, count } of cognitoCountries) {
    if (!merged.has(id)) merged.set(id, { id, views: 0, users: 0 })
    merged.get(id).users = count
  }

  return [...merged.values()].sort((a, b) => (b.views + b.users) - (a.views + a.users))
}

/**
 * Format region code for display (e.g., 'us-east-1' → 'US East 1').
 */
export function formatRegion(regionCode) {
  if (!regionCode) return ''
  return regionCode
    .split('-')
    .map((part, i) => i === 0 ? part.toUpperCase() : part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

/**
 * Get comparison winner display data.
 */
export function getWinnerDisplay(winner) {
  if (!winner) return null
  return {
    modelId: winner.modelId,
    // Extract short name from model ID (e.g., 'anthropic.claude-3-sonnet' → 'Claude 3 Sonnet')
    displayName: winner.modelId.split('.').pop()?.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || winner.modelId,
    count: winner.comparisons,
    total: winner.totalComparisons,
    percentage: winner.totalComparisons > 0
      ? Math.round((winner.comparisons / winner.totalComparisons) * 100)
      : 0,
  }
}

export function exportCsv(data) {
  if (!data?.timeSeries?.length) return
  const cols = ['date', 'views', 'uniqueUsers', 'newUsers', 'returningUsers', 'detailOpens', 'comparisonAdds', 'favoriteToggles']
  const header = cols.join(',')
  const rows = data.timeSeries.map(r => cols.map(c => r[c] ?? '').join(','))
  const csv = [header, ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `analytics_${data.period?.start || 'export'}_${data.period?.end || ''}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

// Country code → lat/lng for map markers
export const COUNTRY_COORDS = {
  US: [39.8, -98.5], GB: [55.3, -3.4], DE: [51.1, 10.4], FR: [46.2, 2.2],
  ES: [40.4, -3.7], IT: [41.9, 12.5], BR: [-14.2, -51.9], IN: [20.6, 78.9],
  JP: [36.2, 138.2], CN: [35.8, 104.1], AU: [-25.3, 133.8], CA: [56.1, -106.3],
  MX: [23.6, -102.5], KR: [35.9, 128.0], RU: [61.5, 105.3], ZA: [-30.6, 22.9],
  AR: [-38.4, -63.6], CO: [4.6, -74.3], CL: [-35.7, -71.5], PE: [-9.2, -75.0],
  NL: [52.1, 5.3], BE: [50.5, 4.5], SE: [60.1, 18.6], NO: [60.5, 8.5],
  FI: [61.9, 25.7], DK: [56.3, 9.5], PL: [51.9, 19.1], AT: [47.5, 14.6],
  CH: [46.8, 8.2], PT: [39.4, -8.2], IE: [53.1, -8.2], IL: [31.0, 34.9],
  AE: [23.4, 53.8], SA: [23.9, 45.1], SG: [1.4, 103.8], MY: [4.2, 101.9],
  TH: [15.9, 100.9], ID: [-0.8, 113.9], PH: [12.9, 121.8], VN: [14.1, 108.3],
  TW: [23.7, 121.0], HK: [22.4, 114.1], NZ: [-40.9, 174.9], UA: [48.4, 31.2],
  EG: [26.8, 30.8], NG: [9.1, 8.7], KE: [-0.02, 37.9], GH: [7.9, -1.0],
  TR: [39.0, 35.2], PK: [30.4, 69.3], BD: [23.7, 90.4], CZ: [49.8, 15.5],
  RO: [45.9, 24.9], HU: [47.2, 19.5], GR: [39.1, 21.8], SK: [48.7, 19.7],
  BG: [42.7, 25.5], HR: [45.1, 15.2], RS: [44.0, 21.0], LT: [55.2, 23.9],
  LV: [56.9, 24.1], EE: [58.6, 25.0], SI: [46.2, 14.9],
}

export const CHART_COLORS = ['#1A9E7A', '#3B82F6', '#F59E0B', '#8B5CF6', '#EF4444', '#EC4899', '#14B8A6', '#6366F1']
export const LIGHT_CHART_COLORS = ['#b45309', '#6366f1', '#ec4899', '#0891b2', '#16a34a', '#dc2626', '#8b5cf6', '#0d9488']
