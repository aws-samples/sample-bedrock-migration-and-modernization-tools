import { useState, useMemo, useRef, useEffect, useCallback, Fragment, memo } from 'react'
import { Search, X, Check, Minus, ChevronDown, ChevronRight, ChevronsUpDown, Zap, Globe, Globe2, Cpu, ExternalLink, AlertTriangle, AlertCircle, Info } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useModels } from '@/hooks/useModels'
import { cn } from '@/lib/utils'

// Ordered region columns grouped by business geo: NAMER, EMEA, APAC, LATAM
const REGION_COLUMNS = [
  // NAMER
  { code: 'us-east-1', short: 'USE1', label: 'N. Virginia', geo: 'NAMER' },
  { code: 'us-east-2', short: 'USE2', label: 'Ohio', geo: 'NAMER' },
  { code: 'us-west-2', short: 'USW2', label: 'Oregon', geo: 'NAMER' },
  { code: 'us-west-1', short: 'USW1', label: 'N. California', geo: 'NAMER' },
  { code: 'ca-central-1', short: 'CAC1', label: 'Montreal', geo: 'NAMER' },
  { code: 'ca-west-1', short: 'CAW1', label: 'Calgary', geo: 'NAMER' },
  // EMEA
  { code: 'eu-west-1', short: 'EUW1', label: 'Ireland', geo: 'EMEA' },
  { code: 'eu-west-2', short: 'EUW2', label: 'London', geo: 'EMEA' },
  { code: 'eu-west-3', short: 'EUW3', label: 'Paris', geo: 'EMEA' },
  { code: 'eu-central-1', short: 'EUC1', label: 'Frankfurt', geo: 'EMEA' },
  { code: 'eu-central-2', short: 'EUC2', label: 'Zurich', geo: 'EMEA' },
  { code: 'eu-north-1', short: 'EUN1', label: 'Stockholm', geo: 'EMEA' },
  { code: 'eu-south-1', short: 'EUS1', label: 'Milan', geo: 'EMEA' },
  { code: 'eu-south-2', short: 'EUS2', label: 'Spain', geo: 'EMEA' },
  { code: 'me-south-1', short: 'MES1', label: 'Bahrain', geo: 'EMEA' },
  { code: 'me-central-1', short: 'MEC1', label: 'UAE', geo: 'EMEA' },
  { code: 'af-south-1', short: 'AFS1', label: 'Cape Town', geo: 'EMEA' },
  { code: 'il-central-1', short: 'ILC1', label: 'Tel Aviv', geo: 'EMEA' },
  // APAC
  { code: 'ap-northeast-1', short: 'ANE1', label: 'Tokyo', geo: 'APAC' },
  { code: 'ap-northeast-2', short: 'ANE2', label: 'Seoul', geo: 'APAC' },
  { code: 'ap-northeast-3', short: 'ANE3', label: 'Osaka', geo: 'APAC' },
  { code: 'ap-southeast-1', short: 'ASE1', label: 'Singapore', geo: 'APAC' },
  { code: 'ap-southeast-2', short: 'ASE2', label: 'Sydney', geo: 'APAC' },
  { code: 'ap-southeast-3', short: 'ASE3', label: 'Jakarta', geo: 'APAC' },
  { code: 'ap-southeast-4', short: 'ASE4', label: 'Melbourne', geo: 'APAC' },
  { code: 'ap-southeast-5', short: 'ASE5', label: 'Malaysia', geo: 'APAC' },
  { code: 'ap-southeast-6', short: 'ASE6', label: 'Auckland', geo: 'APAC' },
  { code: 'ap-southeast-7', short: 'ASE7', label: 'Thailand', geo: 'APAC' },
  { code: 'ap-south-1', short: 'APS1', label: 'Mumbai', geo: 'APAC' },
  { code: 'ap-south-2', short: 'APS2', label: 'Hyderabad', geo: 'APAC' },
  { code: 'ap-east-1', short: 'APE1', label: 'Hong Kong', geo: 'APAC' },
  { code: 'ap-east-2', short: 'APE2', label: 'Taipei', geo: 'APAC' },
  // LATAM
  { code: 'sa-east-1', short: 'SAE1', label: 'Sao Paulo', geo: 'LATAM' },
  { code: 'mx-central-1', short: 'MXC1', label: 'Mexico City', geo: 'LATAM' },
]

const GEO_GROUPS = [
  { id: 'NAMER', label: 'NAMER' },
  { id: 'EMEA', label: 'EMEA' },
  { id: 'APAC', label: 'APAC' },
  { id: 'LATAM', label: 'LATAM' },
]

const GEO_LABELS = { NAMER: 'NAMER', EMEA: 'EMEA', APAC: 'APAC', LATAM: 'LATAM' }

// Auto-detect business geo from region code prefix
const REGION_PREFIX_GEO = {
  us: 'NAMER', ca: 'NAMER',
  eu: 'EMEA', me: 'EMEA', af: 'EMEA', il: 'EMEA',
  ap: 'APAC', cn: 'APAC', in: 'APAC',
  sa: 'LATAM', mx: 'LATAM',
}

// Lookup for known regions (fast path)
const KNOWN_REGIONS = new Map(REGION_COLUMNS.map(r => [r.code, r]))

/**
 * Build a region entry for a code not in REGION_COLUMNS.
 * Auto-detects geo from the prefix and generates a short code.
 */
function buildRegionEntry(code) {
  const prefix = code.split('-')[0]
  const geo = REGION_PREFIX_GEO[prefix] || 'EMEA'
  // Generate 4-char short: prefix uppercase + first char of middle parts + number
  const parts = code.split('-')
  const mid = parts.slice(1, -1).map(s => s[0].toUpperCase()).join('')
  const num = parts[parts.length - 1]
  const raw = parts[0].toUpperCase() + mid + num
  const short = raw.length > 4 ? raw.slice(0, 3) + num : raw
  return { code, short, label: code, geo }
}

/**
 * Normalize a CRIS profile prefix (e.g. 'us', 'eu', 'global') into a display scope.
 */
function normalizeCrisPrefix(prefix) {
  const p = prefix.toLowerCase()
  if (p === 'global') return 'Global'
  if (p === 'us') return 'US'
  if (p === 'eu') return 'EU'
  if (p === 'apac') return 'APAC'
  if (p === 'au') return 'AU'
  if (p === 'jp') return 'JP'
  if (p === 'ca') return 'CA'
  return p.toUpperCase()  // Any future prefix: just uppercase it
}

const MODEL_COL_WIDTH = 280

/**
 * Compute per-region availability for a model (on-demand + CRIS + Mantle).
 *
 * Data sources:
 * - in_region: actual ON_DEMAND availability from regional-availability Lambda
 * - cross_region_inference.source_regions: CRIS source regions
 * - mantle_inference.mantle_regions: Mantle engine regions
 */
function getRegionAvailability(model, regionCode) {
  const inRegionList = model.in_region || []
  const crisRegions = model.cross_region_inference?.source_regions || []
  const mantleRegions = model.mantle_inference?.mantle_regions || []

  // in_region is the source of truth for ON_DEMAND availability (no fallback)
  const onDemand = inRegionList.includes(regionCode)
  const cris = crisRegions.includes(regionCode)
  const mantle = mantleRegions.includes(regionCode)
  const available = onDemand || cris || mantle

  return { available, onDemand, cris, mantle }
}

/**
 * Derive color tokens for the availability cell based on type.
 * Multiple types (on-demand + CRIS/Mantle) → emerald/green (multi-availability)
 * On-Demand only → stone/neutral (standard)
 * CRIS-only → sky/blue
 * Mantle-only → violet/purple
 */
function getAvailabilityColors(onDemand, cris, mantle, isLight) {
  return {
    bg: isLight ? 'bg-emerald-100' : 'bg-emerald-500/20',
    icon: isLight ? 'text-emerald-600' : 'text-emerald-400',
  }
}

/**
 * Get lifecycle status for a model in a specific region.
 * Returns { status, legacyDate, eolDate, recommendedReplacement } or null if no lifecycle data.
 */
function getRegionLifecycleStatus(model, regionCode) {
  const lifecycle = model.model_lifecycle
  if (!lifecycle) return null
  
  // Check for regional status first
  const regionalStatus = lifecycle.regional_status?.[regionCode]
  if (regionalStatus) {
    return {
      status: regionalStatus.status || 'ACTIVE',
      legacyDate: regionalStatus.legacy_date,
      eolDate: regionalStatus.eol_date,
      recommendedReplacement: lifecycle.recommended_replacement
    }
  }
  
  // Fall back to global status if no regional data
  const globalStatus = lifecycle.global_status || lifecycle.status
  if (globalStatus && globalStatus !== 'ACTIVE' && globalStatus !== 'MIXED') {
    return {
      status: globalStatus,
      legacyDate: lifecycle.legacy_date,
      eolDate: lifecycle.eol_date,
      recommendedReplacement: lifecycle.recommended_replacement
    }
  }
  
  return null
}

/**
 * Simple availability cell — check when available, dash when not.
 * Hover tooltip shows availability breakdown based on active filter:
 * - 'all': shows all types (In Region / CRIS / Mantle)
 * - 'in_region': shows only In Region info
 * - 'cris': shows only CRIS info with scopes
 * - 'mantle': shows only Mantle info
 */
const AvailabilityCell = memo(function AvailabilityCell({ model, regionCode, regionLabel, isLight, activeView, selectedCrisScopes }) {
  const { available, onDemand, cris, mantle } = getRegionAvailability(model, regionCode)
  
  // Check if model has CRIS for this region with the selected scope(s)
  const hasCrisForSelectedScope = (model, regionCode, selectedScopes) => {
    if (!selectedScopes || selectedScopes.size === 0) {
      // No filter - check if any CRIS profile exists for this region
      return model.cross_region_inference?.source_regions?.includes(regionCode) || false
    }
    
    // Check if any profile matches both the region AND the selected scope
    const profiles = model.cross_region_inference?.profiles || []
    return profiles.some(p => {
      if (p.source_region !== regionCode) return false
      const prefix = p.profile_id?.split('.')[0]?.toLowerCase() || ''
      // Map prefix to display name for comparison
      const scopeName = prefix === 'global' ? 'Global' : prefix.toUpperCase()
      return selectedScopes.has(scopeName)
    })
  }
  
  // Get lifecycle status for this region
  const lifecycleStatus = getRegionLifecycleStatus(model, regionCode)

  // In specific views, only show relevant availability
  let isAvailable, colors
  if (activeView === 'in_region') {
    isAvailable = onDemand
    colors = isAvailable ? getAvailabilityColors(true, false, false, isLight) : null
  } else if (activeView === 'cris') {
    // Use scope-aware check when CRIS view is active
    isAvailable = hasCrisForSelectedScope(model, regionCode, selectedCrisScopes)
    colors = isAvailable ? getAvailabilityColors(false, true, false, isLight) : null
  } else if (activeView === 'mantle') {
    isAvailable = mantle
    colors = isAvailable ? getAvailabilityColors(false, false, true, isLight) : null
  } else {
    // 'all' view: show combined
    isAvailable = available
    colors = isAvailable ? getAvailabilityColors(onDemand, cris, mantle, isLight) : null
  }

  // Early return for not available (no function calls needed)
  if (!isAvailable) {
    return (
      <div className="w-4 h-4 flex items-center justify-center">
        <Minus className={cn('w-2.5 h-2.5', isLight ? 'text-stone-300' : 'text-white/10')} strokeWidth={2} />
      </div>
    )
  }

  // Check if this region has EOL or LEGACY status
  const isEol = lifecycleStatus?.status === 'EOL'
  const isLegacy = lifecycleStatus?.status === 'LEGACY'

  // Build CRIS scopes helper (used in 'all' and 'cris' views)
  // When selectedCrisScopes is provided, only return scopes that match the filter
  const getCrisScopes = () => {
    const profiles = model.cross_region_inference?.profiles || []
    const scopes = new Set()
    profiles.forEach(p => {
      if (p.source_region === regionCode) {
        const prefix = p.profile_id?.split('.')[0] || ''
        const scope = prefix.toLowerCase()
        let scopeName
        if (scope === 'global') scopeName = 'Global'
        else if (scope === 'us') scopeName = 'US'
        else if (scope === 'eu') scopeName = 'EU'
        else if (scope === 'apac') scopeName = 'APAC'
        else if (scope === 'au') scopeName = 'AU'
        else if (scope === 'jp') scopeName = 'JP'
        else if (scope === 'ca') scopeName = 'CA'
        else if (scope) scopeName = scope.toUpperCase()
        
        // Only add scope if no filter is active OR scope matches the filter
        if (scopeName && (!selectedCrisScopes || selectedCrisScopes.size === 0 || selectedCrisScopes.has(scopeName))) {
          scopes.add(scopeName)
        }
      }
    })
    
    return [...scopes].sort((a, b) => {
      const order = ['Global', 'US', 'CA', 'EU', 'APAC', 'AU', 'JP']
      return (order.indexOf(a) === -1 ? 99 : order.indexOf(a)) - (order.indexOf(b) === -1 ? 99 : order.indexOf(b))
    })
  }
  
  // Render lifecycle status in tooltip
  const renderLifecycleInfo = () => {
    if (!lifecycleStatus || lifecycleStatus.status === 'ACTIVE') return null
    
    const isLegacy = lifecycleStatus.status === 'LEGACY'
    const isEol = lifecycleStatus.status === 'EOL'
    
    return (
      <div className={cn(
        'mt-1.5 pt-1.5 border-t',
        isLight ? 'border-stone-200' : 'border-white/[0.08]'
      )}>
        <div className="flex items-center gap-1.5">
          {isEol ? (
            <AlertCircle className={cn('w-3 h-3', isLight ? 'text-red-500' : 'text-red-400')} strokeWidth={2} />
          ) : (
            <AlertTriangle className={cn('w-3 h-3', isLight ? 'text-amber-500' : 'text-amber-400')} strokeWidth={2} />
          )}
          <span className={cn(
            'font-medium',
            isEol
              ? (isLight ? 'text-red-600' : 'text-red-400')
              : (isLight ? 'text-amber-600' : 'text-amber-400')
          )}>
            {isEol ? 'End of Life' : 'Legacy'}
          </span>
        </div>
        {lifecycleStatus.legacyDate && isLegacy && (
          <div className={cn('text-[10px] mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
            Legacy: {lifecycleStatus.legacyDate}
          </div>
        )}
        {lifecycleStatus.eolDate && (isLegacy || isEol) && (
          <div className={cn('text-[10px] mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
            EOL: {lifecycleStatus.eolDate}
          </div>
        )}
        {lifecycleStatus.recommendedReplacement && (
          <div className={cn('text-[10px] mt-0.5', isLight ? 'text-blue-600' : 'text-blue-400')}>
            Suggested Replacement: {lifecycleStatus.recommendedReplacement}
          </div>
        )}
      </div>
    )
  }

  // Render tooltip content based on active view
  const renderTooltipContent = () => {
    if (activeView === 'in_region') {
      // Only show In Region info
      return (
        <div className="flex items-center gap-1.5">
          <Zap className={cn('w-3 h-3', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')} strokeWidth={2} />
          <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In Region</span>
        </div>
      )
    }
    
    if (activeView === 'cris') {
      // Only show CRIS info with scopes
      const scopeList = getCrisScopes()
      if (scopeList.length === 0) {
        return (
          <div className="flex items-center gap-1.5">
            <Globe2 className={cn('w-3 h-3', isLight ? 'text-sky-500' : 'text-sky-400')} strokeWidth={2} />
            <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Cross-Region (CRIS)</span>
          </div>
        )
      }
      return (
        <div className="flex items-center gap-1.5">
          <Globe2 className={cn('w-3 h-3 flex-shrink-0', isLight ? 'text-sky-500' : 'text-sky-400')} strokeWidth={2} />
          <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>
            CRIS ({scopeList.join(', ')})
          </span>
        </div>
      )
    }
    
    if (activeView === 'mantle') {
      // Only show Mantle info
      return (
        <div className="flex items-center gap-1.5">
          <Cpu className={cn('w-3 h-3', isLight ? 'text-violet-500' : 'text-violet-400')} strokeWidth={2} />
          <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Mantle</span>
        </div>
      )
    }
    
    // 'all' view: show all available types
    return (
      <>
        {onDemand && (
          <div className="flex items-center gap-1.5">
            <Zap className={cn('w-3 h-3', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')} strokeWidth={2} />
            <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In Region</span>
          </div>
        )}
        {cris && (() => {
          const scopeList = getCrisScopes()
          if (scopeList.length === 0) {
            return (
              <div className="flex items-center gap-1.5">
                <Globe2 className={cn('w-3 h-3', isLight ? 'text-sky-500' : 'text-sky-400')} strokeWidth={2} />
                <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Cross-Region (CRIS)</span>
              </div>
            )
          }
          return (
            <div className="flex items-center gap-1.5">
              <Globe2 className={cn('w-3 h-3 flex-shrink-0', isLight ? 'text-sky-500' : 'text-sky-400')} strokeWidth={2} />
              <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>
                CRIS ({scopeList.join(', ')})
              </span>
            </div>
          )
        })()}
        {mantle && (
          <div className="flex items-center gap-1.5">
            <Cpu className={cn('w-3 h-3', isLight ? 'text-violet-500' : 'text-violet-400')} strokeWidth={2} />
            <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Mantle</span>
          </div>
        )}
      </>
    )
  }

  // EOL means not available - show as unavailable with red X
  if (isEol) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex justify-center cursor-default relative">
            <div className={cn(
              'w-4 h-4 rounded-full flex items-center justify-center',
              isLight ? 'bg-red-100' : 'bg-red-500/20'
            )}>
              <X className={cn('w-2.5 h-2.5', isLight ? 'text-red-600' : 'text-red-400')} strokeWidth={2.5} />
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          sideOffset={6}
          className={cn(
            'rounded-lg border',
            'px-3 py-2 text-xs z-50 max-w-[220px]',
            isLight
              ? 'bg-white border-stone-200 shadow-lg'
              : 'bg-white/[0.06] backdrop-blur-xl border-white/[0.06] shadow-[0_4px_12px_rgba(0,0,0,0.3)] ring-1 ring-white/[0.03]'
          )}
        >
          <div className={cn('font-medium mb-1', isLight ? 'text-stone-700' : 'text-[#e4e5e7]')}>
            {regionLabel} ({regionCode})
          </div>
          <div className="space-y-0.5">
            {renderTooltipContent()}
          </div>
          {renderLifecycleInfo()}
        </TooltipContent>
      </Tooltip>
    )
  }

  // Main return for LEGACY and ACTIVE status
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex justify-center cursor-default relative">
          {isLegacy ? (
            // LEGACY status: show warning icon instead of checkmark
            <div className={cn(
              'w-4 h-4 rounded-full flex items-center justify-center',
              isLight ? 'bg-amber-100' : 'bg-amber-500/20'
            )}>
              <AlertTriangle className={cn('w-2.5 h-2.5', isLight ? 'text-amber-600' : 'text-amber-400')} strokeWidth={2.5} />
            </div>
          ) : (
            // Normal available: show checkmark
            <div className={cn(
              'w-4 h-4 rounded-full flex items-center justify-center',
              colors.bg
            )}>
              <Check className={cn('w-2.5 h-2.5', colors.icon)} strokeWidth={3} />
            </div>
          )}
        </div>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        sideOffset={6}
        className={cn(
          'rounded-lg border',
          'px-3 py-2 text-xs z-50 max-w-[220px]',
          isLight
            ? 'bg-white border-stone-200 shadow-lg'
            : 'bg-white/[0.06] backdrop-blur-xl border-white/[0.06] shadow-[0_4px_12px_rgba(0,0,0,0.3)] ring-1 ring-white/[0.03]'
        )}
      >
        <div className={cn('font-medium mb-1', isLight ? 'text-stone-700' : 'text-[#e4e5e7]')}>
          {regionLabel} ({regionCode})
        </div>
        <div className="space-y-0.5">
          {renderTooltipContent()}
        </div>
        {renderLifecycleInfo()}
      </TooltipContent>
    </Tooltip>
  )
})

export function RegionalAvailability() {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const { models, loading, error } = useModels()
  const [searchQuery, setSearchQuery] = useState('')
  const [collapsedProviders, setCollapsedProviders] = useState(new Set())
  
  const [hoveredRow, setHoveredRow] = useState(null)

  const [activeView, setActiveView] = useState('all')
  const [selectedGeos, setSelectedGeos] = useState(new Set())

  const tableContainerRef = useRef(null)

  // Regions with data — known ones keep their order/labels, unknown ones auto-detected
  const activeRegions = useMemo(() => {
    if (!models.length) return REGION_COLUMNS
    const usedRegions = new Set()
    models.forEach(m => {
      ;(m.in_region || []).forEach(r => usedRegions.add(r))
      ;(m.cross_region_inference?.source_regions || []).forEach(r => usedRegions.add(r))
      ;(m.mantle_inference?.mantle_regions || []).forEach(r => usedRegions.add(r))
    })

    // Known regions that appear in the data (preserves defined order)
    const known = REGION_COLUMNS.filter(r => usedRegions.has(r.code))
    const knownCodes = new Set(known.map(r => r.code))

    // Unknown regions — auto-detect geo from prefix, append at end of their geo group
    const unknown = [...usedRegions]
      .filter(code => !knownCodes.has(code))
      .map(buildRegionEntry)

    if (!unknown.length) return known

    const result = [...known]
    unknown.forEach(r => {
      // Insert after the last region of the same geo
      let insertIdx = result.length
      for (let i = result.length - 1; i >= 0; i--) {
        if (result[i].geo === r.geo) { insertIdx = i + 1; break }
      }
      result.splice(insertIdx, 0, r)
    })
    return result
  }, [models])

  // Visible regions — filtered by selected geos (multi-select) or CRIS source regions
  const visibleRegions = useMemo(() => {
    if (activeView === 'cris') {
      // For CRIS view: show source regions, filtered by selected CRIS prefixes
      const crisSourceRegions = new Set()
      models.forEach(m => {
        if (!m.cross_region_inference?.supported) return
        const profiles = m.cross_region_inference?.profiles || []
        profiles.forEach(p => {
          const prefix = p.profile_id?.split('.')[0] || ''
          const scope = normalizeCrisPrefix(prefix)
          if (selectedGeos.size === 0 || selectedGeos.has(scope)) {
            if (p.source_region) crisSourceRegions.add(p.source_region)
          }
        })
      })
      return activeRegions.filter(r => crisSourceRegions.has(r.code))
    }

    // For All/In Region/Mantle: filter by geo (multi-select)
    if (selectedGeos.size === 0) return activeRegions
    return activeRegions.filter(r => selectedGeos.has(r.geo))
  }, [activeRegions, activeView, selectedGeos, models])

  // Which geos exist in the data
  const geoIndex = useMemo(() => {
    const idx = {}
    activeRegions.forEach((r, i) => {
      if (!(r.geo in idx)) idx[r.geo] = i
    })
    return idx
  }, [activeRegions])

  const availableGeos = useMemo(() =>
    GEO_GROUPS.filter(g => g.id in geoIndex),
    [geoIndex]
  )

  // Available CRIS prefixes (computed dynamically from data)
  const availableCrisPrefixes = useMemo(() => {
    if (activeView !== 'cris') return []
    const prefixes = new Set()
    models.forEach(m => {
      (m.cross_region_inference?.profiles || []).forEach(p => {
        const prefix = p.profile_id?.split('.')[0] || ''
        prefixes.add(normalizeCrisPrefix(prefix))
      })
    })
    const order = ['Global', 'US', 'CA', 'EU', 'APAC', 'AU', 'JP']
    return order.filter(p => prefixes.has(p)).concat(
      [...prefixes].filter(p => !order.includes(p)).sort()
    )
  }, [models, activeView])

  // Geo header cells with colspan spans (based on visible regions)
  const geoHeaderCells = useMemo(() => {
    const cells = []
    let currentGeo = null
    let span = 0
    visibleRegions.forEach((r) => {
      if (r.geo !== currentGeo) {
        if (currentGeo !== null) cells.push({ geo: currentGeo, span })
        currentGeo = r.geo
        span = 1
      } else {
        span++
      }
    })
    if (currentGeo !== null) cells.push({ geo: currentGeo, span })
    return cells
  }, [visibleRegions])

  const groupedModels = useMemo(() => {
    const q = searchQuery.toLowerCase()
    const filtered = models.filter(m => {
      if (q && !(
        m.model_name?.toLowerCase().includes(q) ||
        m.model_id?.toLowerCase().includes(q) ||
        m.model_provider?.toLowerCase().includes(q)
      )) return false
      if (activeView === 'in_region') {
        // Use actual in_region availability, not declared inference_types_supported
        if (!(m.in_region?.length > 0)) return false
      }
      if (activeView === 'cris') {
        if (!m.cross_region_inference?.supported) return false
        // If a CRIS scope is selected, filter to only models with that scope
        if (selectedGeos.size > 0) {
          const profiles = m.cross_region_inference?.profiles || []
          const modelScopes = new Set(profiles.map(p => {
            const prefix = p.profile_id?.split('.')[0]?.toLowerCase() || ''
            return prefix === 'global' ? 'Global' : prefix.toUpperCase()
          }))
          // Check if model has any of the selected scopes
          const hasSelectedScope = [...selectedGeos].some(scope => modelScopes.has(scope))
          if (!hasSelectedScope) return false
        }
      }
      if (activeView === 'mantle') {
        if (!(m.mantle_inference?.supported || m.is_mantle)) return false
      }
      return true
    })

    const grouped = {}
    filtered.forEach(m => {
      const provider = m.model_provider || 'Unknown'
      if (!grouped[provider]) grouped[provider] = []
      grouped[provider].push(m)
    })

    Object.values(grouped).forEach(arr =>
      arr.sort((a, b) => (a.model_name || '').localeCompare(b.model_name || ''))
    )

    return Object.entries(grouped).sort(([a], [b]) => a.localeCompare(b))
  }, [models, searchQuery, activeView, selectedGeos])

  const totalFiltered = groupedModels.reduce((sum, [, models]) => sum + models.length, 0)

  // Per-region coverage tier for column tinting
  const regionCoverage = useMemo(() => {
    if (!totalFiltered) return {}
    const coverage = {}
    activeRegions.forEach(r => {
      let count = 0
      groupedModels.forEach(([, providerModels]) => {
        providerModels.forEach(m => {
          const { available, onDemand, cris, mantle } = getRegionAvailability(m, r.code)
          let isAvail
          if (activeView === 'in_region') isAvail = onDemand
          else if (activeView === 'cris') isAvail = cris
          else if (activeView === 'mantle') isAvail = mantle
          else isAvail = available
          if (isAvail) count++
        })
      })
      const ratio = count / totalFiltered
      coverage[r.code] = ratio >= 1 ? 'full' : ratio >= 0.5 ? 'high' : null
    })
    return coverage
  }, [groupedModels, activeRegions, totalFiltered, activeView])

  // Check if any visible models have lifecycle warnings (LEGACY or EOL)
  const hasLifecycleWarnings = useMemo(() => {
    for (const [, providerModels] of groupedModels) {
      for (const model of providerModels) {
        const lifecycle = model.model_lifecycle
        if (!lifecycle) continue
        
        // Check global status
        const globalStatus = lifecycle.global_status || lifecycle.status
        if (globalStatus === 'LEGACY' || globalStatus === 'EOL') return true
        
        // Check regional statuses
        const regionalStatus = lifecycle.regional_status
        if (regionalStatus) {
          for (const regionData of Object.values(regionalStatus)) {
            if (regionData.status === 'LEGACY' || regionData.status === 'EOL') return true
          }
        }
      }
    }
    return false
  }, [groupedModels])

  // Column tint via inset box-shadow (layers over bg without conflicting)
  const getColumnTint = useCallback((regionCode) => {
    const tier = regionCoverage[regionCode]
    if (!tier) return ''
    if (tier === 'full') {
      return isLight
        ? 'shadow-[inset_0_0_0_200px_rgb(16_185_129_/_0.07)]'
        : 'shadow-[inset_0_0_0_200px_rgb(16_185_129_/_0.05)]'
    }
    return isLight
      ? 'shadow-[inset_0_0_0_200px_rgb(245_158_11_/_0.05)]'
      : 'shadow-[inset_0_0_0_200px_rgb(245_158_11_/_0.04)]'
  }, [regionCoverage, isLight])

  const toggleProvider = (provider) => {
    setCollapsedProviders(prev => {
      const next = new Set(prev)
      next.has(provider) ? next.delete(provider) : next.add(provider)
      return next
    })
  }

  const allProviders = groupedModels.map(([provider]) => provider)
  const allCollapsed = allProviders.length > 0 && allProviders.every(p => collapsedProviders.has(p))

  const toggleAllProviders = () => {
    if (allCollapsed) {
      setCollapsedProviders(new Set())
    } else {
      setCollapsedProviders(new Set(allProviders))
    }
  }

  const handleViewChange = (viewId) => {
    setActiveView(viewId)
    setSelectedGeos(new Set())
    tableContainerRef.current?.scrollTo({ left: 0 })
  }

  const toggleGeoSelection = (id) => {
    setSelectedGeos(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
    tableContainerRef.current?.scrollTo({ left: 0 })
  }

  // Helper: is this column the first of a new geo group?
  const isGeoBreak = (i) => i > 0 && visibleRegions[i].geo !== visibleRegions[i - 1].geo

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className={cn('text-sm', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Loading models...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-sm text-red-400">Failed to load models</div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-[calc(100dvh-4.5rem)] p-4 sm:p-6 gap-4 overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 flex items-start justify-between gap-4">
        <div>
          <h1 className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-[#f0f1f3]')}>
            Regional Availability
          </h1>
          <p className={cn('text-sm mt-1', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>
            Model availability across AWS regions at a glance
          </p>
        </div>
        <a
          href="https://bedrock-pfr-onboarding.rodzanto.people.aws.dev/"
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 border whitespace-nowrap',
            isLight
              ? 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100 hover:border-amber-300'
              : 'bg-[#1A9E7A]/10 text-[#1A9E7A] border-[#1A9E7A]/20 hover:bg-[#1A9E7A]/20 hover:border-[#1A9E7A]/30'
          )}
        >
          See model onboarding and PFRs
          <ExternalLink className="w-3 h-3" />
        </a>
      </div>

      {/* Search + Legend + Geo pills */}
      <div className="flex-shrink-0 flex flex-col gap-2">
        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className={cn(
              'absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4',
              isLight ? 'text-stone-400' : 'text-[#6d6e72]'
            )} />
            <Input
              placeholder="Search models or providers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={cn(
                'h-9 pl-9 pr-8 text-sm',
                isLight
                  ? 'bg-white border-stone-200 focus:border-amber-500'
                  : 'bg-white/[0.03] border-white/[0.06] focus:border-[#1A9E7A] backdrop-blur-xl'
              )}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className={cn(
                  'absolute right-2 top-1/2 -translate-y-1/2 p-0.5 rounded-full transition-colors',
                  isLight ? 'hover:bg-stone-100' : 'hover:bg-white/[0.06]'
                )}
              >
                <X className={cn('h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-[#6d6e72]')} />
              </button>
            )}
          </div>

          <div className={cn('text-xs tabular-nums flex-shrink-0', isLight ? 'text-stone-400' : 'text-[#9a9b9f]')}>
            {totalFiltered} model{totalFiltered !== 1 ? 's' : ''} / {visibleRegions.length} regions
          </div>
        </div>

        {/* Primary view pills + Geo/CRIS prefix pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* Tier 1: Primary filter pills */}
          {[
            { id: 'all', label: 'All', icon: null },
            { id: 'in_region', label: 'In Region', icon: Zap },
            { id: 'cris', label: 'CRIS', icon: Globe },
            { id: 'mantle', label: 'Mantle', icon: Cpu },
          ].map(view => (
            <button
              key={view.id}
              onClick={() => handleViewChange(view.id)}
              className={cn(
                'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border flex items-center gap-1',
                activeView === view.id
                  ? isLight
                    ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                    : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                  : isLight
                    ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                    : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
              )}
            >
              {view.icon && <view.icon className="w-3 h-3" />}
              {view.label}
            </button>
          ))}

          {/* Divider */}
          <div className={cn('w-px h-5 mx-1', isLight ? 'bg-stone-200' : 'bg-white/[0.08]')} />

          {/* Tier 2: Geo / CRIS prefix pills (multi-select) */}
          <span className={cn('text-[10px] uppercase tracking-wider font-medium mr-1', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
            {activeView === 'cris' ? 'Scope' : 'Geo'}
          </span>

          {/* "All" pill */}
          <button
            onClick={() => { setSelectedGeos(new Set()); tableContainerRef.current?.scrollTo({ left: 0 }) }}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
              selectedGeos.size === 0
                ? isLight
                  ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                  : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                : isLight
                  ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                  : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
            )}
          >
            All
          </button>

          {/* Dynamic pills based on view */}
          {(activeView === 'cris' ? availableCrisPrefixes : availableGeos).map(item => {
            const id = typeof item === 'string' ? item : item.id
            const label = typeof item === 'string' ? item : item.label
            const isSelected = selectedGeos.has(id)
            return (
              <button
                key={id}
                onClick={() => toggleGeoSelection(id)}
                className={cn(
                  'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                  isSelected
                    ? isLight
                      ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                      : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                    : isLight
                      ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                      : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
                )}
              >
                {label}
              </button>
            )
          })}
        </div>
        
        {/* Lifecycle Legend - only show when there are models with warnings */}
        {hasLifecycleWarnings && (
          <div className={cn(
            'flex items-center gap-4 px-3 py-1.5 rounded-lg border',
            isLight
              ? 'bg-stone-50/80 border-stone-200'
              : 'bg-white/[0.02] border-white/[0.06]'
          )}>
            <div className="flex items-center gap-1.5">
              <Info className={cn('w-3 h-3', isLight ? 'text-stone-400' : 'text-slate-500')} />
              <span className={cn('text-[10px] font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                Lifecycle:
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className={cn(
                'w-4 h-4 rounded-full flex items-center justify-center',
                isLight ? 'bg-amber-100' : 'bg-amber-500/20'
              )}>
                <AlertTriangle className={cn('w-2.5 h-2.5', isLight ? 'text-amber-600' : 'text-amber-400')} strokeWidth={2.5} />
              </div>
              <span className={cn('text-[10px]', isLight ? 'text-amber-700' : 'text-amber-400')}>
                Legacy (will be deprecated)
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className={cn(
                'w-4 h-4 rounded-full flex items-center justify-center',
                isLight ? 'bg-red-100' : 'bg-red-500/20'
              )}>
                <X className={cn('w-2.5 h-2.5', isLight ? 'text-red-600' : 'text-red-400')} strokeWidth={2.5} />
              </div>
              <span className={cn('text-[10px]', isLight ? 'text-red-700' : 'text-red-400')}>
                End of Life (no longer available)
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Table container */}
      <div className="flex-1 relative min-h-0">
        <TooltipProvider delayDuration={150}>
          <div
            ref={tableContainerRef}
            className={cn(
              'h-full min-w-0 overflow-auto rounded-xl backdrop-blur-xl',
              isLight
                ? 'border border-stone-200/60 bg-white/70 shadow-[0_2px_15px_-3px_rgba(120,113,108,0.08)] ring-1 ring-stone-100/50'
                : 'border border-white/[0.06] bg-white/[0.03] shadow-[0_2px_15px_-3px_rgba(0,0,0,0.3)] ring-1 ring-white/[0.03]'
            )}
          >
          <table className="w-max min-w-full border-collapse">
            <thead>
              {/* Row 1: Geo group headers */}
              <tr>
                <th
                  rowSpan={2}
                  className={cn(
                    'sticky left-0 top-0 z-30 text-left text-[11px] font-semibold uppercase tracking-wider px-3',
                    'w-[280px] min-w-[280px] max-w-[280px]',
                    isLight
                      ? 'bg-stone-50/90 backdrop-blur-sm text-stone-500 border-b border-r border-stone-200'
                      : 'bg-[#141517]/95 backdrop-blur-xl text-[#9a9b9f] border-b border-r border-white/[0.06]'
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span>Model</span>
                    <button
                      onClick={toggleAllProviders}
                      className={cn(
                        'flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9px] font-medium transition-all duration-150',
                        allCollapsed
                          ? isLight
                            ? 'text-amber-700 hover:bg-amber-50'
                            : 'text-[#1A9E7A] hover:bg-[#1A9E7A]/10'
                          : isLight
                            ? 'text-stone-400 hover:text-stone-600 hover:bg-stone-100'
                            : 'text-[#6d6e72] hover:text-[#c0c1c5] hover:bg-white/[0.04]'
                      )}
                      title={allCollapsed ? 'Expand all providers' : 'Collapse all providers'}
                    >
                      <ChevronsUpDown className="w-2.5 h-2.5" />
                      {allCollapsed ? 'Expand' : 'Collapse'}
                    </button>
                  </div>
                </th>
                {geoHeaderCells.map((cell, gi) => (
                  <th
                    key={cell.geo}
                    colSpan={cell.span}
                    className={cn(
                      'sticky top-0 z-20 text-center px-0 py-1.5 text-[10px] font-bold uppercase tracking-widest',
                      isLight
                        ? 'bg-stone-50/90 backdrop-blur-sm border-b border-stone-200'
                        : 'bg-white/[0.04] backdrop-blur-xl border-b border-white/[0.06]',
                      gi > 0 && (isLight ? 'border-l-2 border-l-stone-300' : 'border-l-2 border-l-white/[0.12]'),
                      isLight ? 'text-amber-700' : 'text-[#1A9E7A]'
                    )}
                  >
                    {GEO_LABELS[cell.geo]}
                  </th>
                ))}
                {/* Spacer — absorbs remaining width */}
                <th
                  rowSpan={2}
                  className={cn(
                    'sticky top-0 z-20',
                    isLight
                      ? 'bg-stone-50/90 backdrop-blur-sm border-b border-stone-200'
                      : 'bg-white/[0.04] backdrop-blur-xl border-b border-white/[0.06]'
                  )}
                />
              </tr>

              {/* Row 2: Individual region columns */}
              <tr>
                {visibleRegions.map((region, i) => (
                  <th
                    key={region.code}
                    
                    
                    className={cn(
                      'sticky top-[26px] z-20 text-center px-0 py-0 w-12 min-w-12',
                      isLight
                        ? 'bg-stone-50/90 backdrop-blur-sm border-b border-stone-200'
                        : 'bg-white/[0.04] backdrop-blur-xl border-b border-white/[0.06]',
                      isGeoBreak(i) && (isLight ? 'border-l-2 border-l-stone-300' : 'border-l-2 border-l-white/[0.12]'),
                      
                      getColumnTint(region.code)
                    )}
                  >
                    <TooltipProvider delayDuration={200}>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="flex flex-col items-center py-1.5 gap-0.5 cursor-default">
                            <span className={cn(
                              'text-[11px] font-bold leading-none',
                              isLight ? 'text-stone-800' : 'text-white'
                            )}>
                              {region.short}
                            </span>
                            <span className={cn(
                              'text-[9px] leading-none max-w-[42px] truncate',
                              isLight ? 'text-stone-600' : 'text-[#9a9b9f]'
                            )}>
                              {region.label}
                            </span>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent
                          side="bottom"
                          sideOffset={4}
                          className={cn(
                            'px-2.5 py-1.5 text-xs z-50',
                            isLight
                              ? 'bg-white border-stone-200 shadow-lg'
                              : 'bg-white/[0.06] backdrop-blur-xl border-white/[0.06] shadow-[0_4px_12px_rgba(0,0,0,0.3)] ring-1 ring-white/[0.03]'
                          )}
                        >
                          <div className={cn('font-medium', isLight ? 'text-stone-700' : 'text-[#e4e5e7]')}>
                            {region.label}
                          </div>
                          <div className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
                            {region.code}
                          </div>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {groupedModels.map(([provider, providerModels]) => {
                const isCollapsed = collapsedProviders.has(provider)
                const providerRegionCoverage = visibleRegions.map(r =>
                  providerModels.some(m => {
                    const { available, onDemand, cris, mantle } = getRegionAvailability(m, r.code)
                    if (activeView === 'in_region') return onDemand
                    if (activeView === 'cris') return cris
                    if (activeView === 'mantle') return mantle
                    return available
                  })
                )

                return (
                  <Fragment key={provider}>
                    <tr
                      className={cn(
                        'cursor-pointer select-none',
                        isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.02]'
                      )}
                      onClick={() => toggleProvider(provider)}
                    >
                      <td className={cn(
                        'sticky left-0 z-10 px-3 py-2 font-semibold text-xs',
                        isLight
                          ? 'bg-stone-100/90 backdrop-blur-sm text-stone-700 border-b border-r border-stone-200'
                          : 'bg-[#1a1b1f]/95 backdrop-blur-xl text-[#e4e5e7] border-b border-r border-white/[0.06]'
                      )}>
                        <div className="flex items-center gap-2">
                          {isCollapsed
                            ? <ChevronRight className="w-3.5 h-3.5 flex-shrink-0" />
                            : <ChevronDown className="w-3.5 h-3.5 flex-shrink-0" />
                          }
                          <span>{provider}</span>
                          <Badge className={cn(
                            'ml-1 text-[10px] px-1.5 py-0 border-0 font-normal',
                            isLight ? 'bg-stone-200 text-stone-600' : 'bg-white/[0.06] text-[#9a9b9f]'
                          )}>
                            {providerModels.length}
                          </Badge>
                        </div>
                      </td>
                      {visibleRegions.map((region, i) => (
                        <td
                          key={region.code}
                          className={cn(
                            'text-center py-2',
                            isLight
                              ? 'bg-stone-100/90 border-b border-stone-200'
                              : 'bg-white/[0.06] border-b border-white/[0.06]',
                            isGeoBreak(i) && (isLight ? 'border-l-2 border-l-stone-300' : 'border-l-2 border-l-white/[0.12]'),
                            
                            getColumnTint(region.code)
                          )}
                        >
                          {providerRegionCoverage[i] && (
                            <div className={cn(
                              'w-2 h-2 rounded-full mx-auto',
                              isLight ? 'bg-stone-300' : 'bg-white/[0.15]'
                            )} />
                          )}
                        </td>
                      ))}
                      <td className={cn(
                        isLight ? 'bg-stone-100/90 border-b border-stone-200' : 'bg-white/[0.06] border-b border-white/[0.06]'
                      )} />
                    </tr>

                    {!isCollapsed && providerModels.map((model) => {
                      const regions = model.in_region || []
                      const crisRegions = model.cross_region_inference?.source_regions || []
                      const mantleRegions = model.mantle_inference?.mantle_regions || []
                      const allRegions = new Set([...regions, ...crisRegions, ...mantleRegions])
                      const regionCount = allRegions.size
                      const isHovered = hoveredRow === model.model_id

                      return (
                        <tr
                          key={model.model_id}
                          onMouseEnter={() => setHoveredRow(model.model_id)}
                          onMouseLeave={() => setHoveredRow(null)}
                          className={cn(
                            'transition-colors duration-75',
                            isHovered
                              ? isLight ? 'bg-amber-50/50' : 'bg-white/[0.03]'
                              : ''
                          )}
                        >
                          <td className={cn(
                            'sticky left-0 z-10 px-3 py-1.5',
                            isLight
                              ? 'border-b border-r border-stone-100'
                              : 'border-b border-r border-white/[0.03]',
                            isHovered
                              ? isLight ? 'bg-amber-50/80 backdrop-blur-sm' : 'bg-[#1a1b1f]/90 backdrop-blur-xl'
                              : isLight ? 'bg-white/90 backdrop-blur-sm' : 'bg-[#141517]/95 backdrop-blur-xl'
                          )}>
                            <div className="flex items-center gap-2 min-w-0">
                              <div className="min-w-0 flex-1">
                                <div className={cn(
                                  'text-xs font-medium truncate max-w-[200px]',
                                  isLight ? 'text-stone-800' : 'text-[#e4e5e7]'
                                )}>
                                  {model.model_name}
                                </div>
                                <div className={cn(
                                  'text-[10px] truncate max-w-[200px]',
                                  isLight ? 'text-stone-400' : 'text-[#6d6e72]'
                                )}>
                                  {model.model_id?.split(':')[0]}
                                </div>
                              </div>
                              <span className={cn(
                                'text-[10px] tabular-nums flex-shrink-0',
                                isLight ? 'text-stone-400' : 'text-[#6d6e72]'
                              )}>
                                {regionCount}
                              </span>
                            </div>
                          </td>
                          {visibleRegions.map((region, i) => (
                            <td
                              key={region.code}
                              className={cn(
                                'text-center py-1.5',
                                isLight ? 'border-b border-stone-100' : 'border-b border-white/[0.03]',
                                isGeoBreak(i) && (isLight ? 'border-l-2 border-l-stone-300' : 'border-l-2 border-l-white/[0.12]'),
                                
                                
                                getColumnTint(region.code)
                              )}
                            >
                              <AvailabilityCell
                                model={model}
                                regionCode={region.code}
                                regionLabel={region.label}
                                isLight={isLight}
                                activeView={activeView}
                                selectedCrisScopes={activeView === 'cris' ? selectedGeos : null}
                              />
                            </td>
                          ))}
                          <td className={cn(
                            isLight ? 'border-b border-stone-100' : 'border-b border-white/[0.03]'
                          )} />
                        </tr>
                      )
                    })}
                  </Fragment>
                )
              })}
            </tbody>
          </table>

          {groupedModels.length === 0 && (
            <div className={cn(
              'flex items-center justify-center py-16 text-sm',
              isLight ? 'text-stone-400' : 'text-[#6d6e72]'
            )}>
              No models found matching &ldquo;{searchQuery}&rdquo;
            </div>
          )}
        </div>
        </TooltipProvider>
      </div>
    </div>
  )
}
