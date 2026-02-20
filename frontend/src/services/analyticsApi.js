/**
 * Dashboard API client for admin analytics.
 */

const API_URL = import.meta.env.VITE_ANALYTICS_API_URL

/**
 * Fetch aggregated dashboard data from the analytics API.
 *
 * Dashboard API response shape:
 * {
 *   summary: { totalViews, uniqueUsers, newUsers, returningUsers, activeToday,
 *              avgDailyViews, avgDailyUsers, viewsPerUser, mostActiveSection,
 *              sectionUsage, featureUsage, topModels, topComparedModels,
 *              topFavoritedModels, providerComparisons, providerFavorites,
 *              countryCounts, regionCounts, totalRegions, comparisonWinner },
 *   previousPeriod: { ...same shape as summary },
 *   timeSeries: [{ date, views, uniqueUsers, newUsers, returningUsers, ... }],
 *   hourlySeries: [{ hour, views, events, uniqueUsers }],
 *   countries: string[],
 *   regions: string[],
 *   period: { start, end, days },
 *   cognito: { loggedInUsers, totalRegistered, newUsersInPeriod,
 *              returningUsersInPeriod, usersByCountry, dailyBreakdown }
 * }
 *
 * @param {{ days?: number, start?: string, end?: string }} options
 * @param {string|null} accessToken
 * @returns {Promise<object>}
 */
export async function fetchDashboardData(options = {}, accessToken = null) {
  if (!API_URL) {
    throw new Error('Analytics API URL not configured')
  }

  const params = new URLSearchParams()
  if (options.start && options.end) {
    params.set('start', options.start)
    params.set('end', options.end)
  } else {
    params.set('days', String(options.days || 30))
  }

  const headers = { 'Content-Type': 'application/json' }
  if (accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`
  }

  const res = await fetch(`${API_URL}/dashboard?${params}`, { headers })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error || `HTTP ${res.status}`)
  }

  return res.json()
}
