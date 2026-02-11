/**
 * Dashboard API client for admin analytics.
 */

const API_URL = import.meta.env.VITE_ANALYTICS_API_URL

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
