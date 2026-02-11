/**
 * Analytics event collection service.
 *
 * - Generates an anonymous user ID (UUID) stored in localStorage
 * - Detects country from browser timezone (no external API)
 * - Batches events and flushes every 5 min or on page unload
 * - Silent no-op when VITE_ANALYTICS_API_URL is not configured
 */

const API_URL = import.meta.env.VITE_ANALYTICS_API_URL
const FLUSH_INTERVAL_MS = 300_000  // 5 minutes
const STORAGE_KEY = 'bmp_auid'

let eventQueue = []
let flushTimer = null
let auid = null
let country = null
let region = null
let initialized = false

// --- Timezone → country mapping (covers major timezones) ---
const TZ_COUNTRY = {
  'America/New_York': 'US', 'America/Chicago': 'US', 'America/Denver': 'US',
  'America/Los_Angeles': 'US', 'America/Anchorage': 'US', 'Pacific/Honolulu': 'US',
  'America/Phoenix': 'US', 'America/Detroit': 'US', 'America/Indiana/Indianapolis': 'US',
  'America/Toronto': 'CA', 'America/Vancouver': 'CA', 'America/Edmonton': 'CA',
  'America/Winnipeg': 'CA', 'America/Halifax': 'CA', 'America/Montreal': 'CA',
  'America/Mexico_City': 'MX', 'America/Cancun': 'MX', 'America/Tijuana': 'MX',
  'America/Sao_Paulo': 'BR', 'America/Manaus': 'BR', 'America/Fortaleza': 'BR',
  'America/Bogota': 'CO', 'America/Lima': 'PE', 'America/Santiago': 'CL',
  'America/Buenos_Aires': 'AR', 'America/Argentina/Buenos_Aires': 'AR',
  'Europe/London': 'GB', 'Europe/Dublin': 'IE',
  'Europe/Paris': 'FR', 'Europe/Berlin': 'DE', 'Europe/Madrid': 'ES',
  'Europe/Rome': 'IT', 'Europe/Amsterdam': 'NL', 'Europe/Brussels': 'BE',
  'Europe/Zurich': 'CH', 'Europe/Vienna': 'AT', 'Europe/Stockholm': 'SE',
  'Europe/Oslo': 'NO', 'Europe/Copenhagen': 'DK', 'Europe/Helsinki': 'FI',
  'Europe/Warsaw': 'PL', 'Europe/Prague': 'CZ', 'Europe/Bucharest': 'RO',
  'Europe/Lisbon': 'PT', 'Europe/Athens': 'GR', 'Europe/Istanbul': 'TR',
  'Europe/Moscow': 'RU', 'Europe/Kiev': 'UA', 'Europe/Kyiv': 'UA',
  'Asia/Tokyo': 'JP', 'Asia/Seoul': 'KR', 'Asia/Shanghai': 'CN',
  'Asia/Hong_Kong': 'HK', 'Asia/Taipei': 'TW', 'Asia/Singapore': 'SG',
  'Asia/Kolkata': 'IN', 'Asia/Calcutta': 'IN', 'Asia/Mumbai': 'IN',
  'Asia/Dubai': 'AE', 'Asia/Riyadh': 'SA', 'Asia/Jakarta': 'ID',
  'Asia/Bangkok': 'TH', 'Asia/Ho_Chi_Minh': 'VN', 'Asia/Manila': 'PH',
  'Asia/Karachi': 'PK', 'Asia/Dhaka': 'BD', 'Asia/Kuala_Lumpur': 'MY',
  'Asia/Tel_Aviv': 'IL', 'Asia/Jerusalem': 'IL',
  'Australia/Sydney': 'AU', 'Australia/Melbourne': 'AU', 'Australia/Brisbane': 'AU',
  'Australia/Perth': 'AU', 'Australia/Adelaide': 'AU',
  'Pacific/Auckland': 'NZ', 'Pacific/Fiji': 'FJ',
  'Africa/Cairo': 'EG', 'Africa/Lagos': 'NG', 'Africa/Johannesburg': 'ZA',
  'Africa/Nairobi': 'KE', 'Africa/Casablanca': 'MA',
}

function detectCountry() {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone
    return TZ_COUNTRY[tz] || 'Unknown'
  } catch {
    return 'Unknown'
  }
}

function getOrCreateAuid() {
  try {
    let id = localStorage.getItem(STORAGE_KEY)
    if (!id) {
      id = crypto.randomUUID()
      localStorage.setItem(STORAGE_KEY, id)
    }
    return id
  } catch {
    return crypto.randomUUID()
  }
}

function flush() {
  if (!API_URL || eventQueue.length === 0) return

  const batch = eventQueue.splice(0)
  const payload = JSON.stringify({
    events: batch,
    auid,
    country,
    region,
  })

  // Use sendBeacon for reliability on page unload, fetch otherwise
  if (document.visibilityState === 'hidden') {
    navigator.sendBeacon(`${API_URL}/events`, payload)
  } else {
    fetch(`${API_URL}/events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
      keepalive: true,
    }).catch(() => {
      // Silently ignore — analytics should never break the app
    })
  }
}

function onVisibilityChange() {
  if (document.visibilityState === 'hidden') {
    flush()
  }
}

// --- Public API ---

export function initAnalytics() {
  if (initialized || !API_URL) return

  auid = getOrCreateAuid()
  country = detectCountry()
  initialized = true

  flushTimer = setInterval(flush, FLUSH_INTERVAL_MS)
  document.addEventListener('visibilitychange', onVisibilityChange)
}

export function trackEvent(type, meta = {}) {
  if (!API_URL || !initialized) return

  eventQueue.push({
    type,
    section: meta.section || '',
    meta,
    clientTs: Date.now(),
  })
}

/**
 * Update geo info from authenticated user profile (Midway/Cognito).
 * Overrides timezone-based detection with authoritative Federate data.
 * Only country and region are stored — no PII.
 */
export function setUserGeo({ country: authCountry, region: authRegion } = {}) {
  if (authCountry) country = authCountry
  if (authRegion) region = authRegion
}

export function shutdownAnalytics() {
  if (!initialized) return

  flush()
  if (flushTimer) clearInterval(flushTimer)
  document.removeEventListener('visibilitychange', onVisibilityChange)
  initialized = false
}
