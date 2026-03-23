import { useState, useMemo, useRef, useEffect, useCallback, Fragment, memo } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { Search, X, Check, Minus, ChevronDown, ChevronRight, ChevronsUpDown, Zap, Globe, Globe2, Cpu, AlertTriangle, AlertCircle, Info } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useTheme } from '@/components/layout/ThemeProvider'
import { useModels } from '@/hooks/useModels'
import { cn } from '@/lib/utils'
import { getRegionName, getAirportCode } from '@/utils/regionUtils'
import {
  countModelsByRouting,
} from '@/utils/filters'

/**
 * Check if a model has any pricing data.
 * Uses the same logic as ModelCard.jsx - checks if any pricing value exists
 * from the getPricingForModel function.
 * @param {Object} model - The model object
 * @param {Function} getPricingForModel - Function to get pricing for a model
 * @param {string} preferredRegion - The preferred region for pricing lookup
 * @returns {boolean} True if the model has pricing data
 */
function modelHasPricing(model, getPricingForModel, preferredRegion = 'us-east-1') {
  if (!getPricingForModel) {
    // Fallback to has_pricing field if getPricingForModel not available
    return model.has_pricing !== false
  }
  
  const pricingResult = getPricingForModel(model, preferredRegion)
  const summary = pricingResult?.summary
  
  if (!summary) return false
  
  // Check if any pricing value exists (same logic as ModelCard.jsx)
  return (
    summary.inputPrice != null ||
    summary.outputPrice != null ||
    summary.imagePrice != null ||
    summary.videoPrice != null
  )
}

// Ordered region columns grouped by business geo: NAMER, EMEA, APAC, LATAM
// Labels are derived from centralized region utilities
const REGION_COLUMNS = [
  // NAMER
  { code: 'us-east-1', short: 'USE1', label: getRegionName('us-east-1'), geo: 'NAMER' },
  { code: 'us-east-2', short: 'USE2', label: getRegionName('us-east-2'), geo: 'NAMER' },
  { code: 'us-west-2', short: 'USW2', label: getRegionName('us-west-2'), geo: 'NAMER' },
  { code: 'us-west-1', short: 'USW1', label: getRegionName('us-west-1'), geo: 'NAMER' },
  { code: 'ca-central-1', short: 'CAC1', label: getRegionName('ca-central-1'), geo: 'NAMER' },
  { code: 'ca-west-1', short: 'CAW1', label: getRegionName('ca-west-1'), geo: 'NAMER' },
  // EMEA
  { code: 'eu-west-1', short: 'EUW1', label: getRegionName('eu-west-1'), geo: 'EMEA' },
  { code: 'eu-west-2', short: 'EUW2', label: getRegionName('eu-west-2'), geo: 'EMEA' },
  { code: 'eu-west-3', short: 'EUW3', label: getRegionName('eu-west-3'), geo: 'EMEA' },
  { code: 'eu-central-1', short: 'EUC1', label: getRegionName('eu-central-1'), geo: 'EMEA' },
  { code: 'eu-central-2', short: 'EUC2', label: getRegionName('eu-central-2'), geo: 'EMEA' },
  { code: 'eu-north-1', short: 'EUN1', label: getRegionName('eu-north-1'), geo: 'EMEA' },
  { code: 'eu-south-1', short: 'EUS1', label: getRegionName('eu-south-1'), geo: 'EMEA' },
  { code: 'eu-south-2', short: 'EUS2', label: getRegionName('eu-south-2'), geo: 'EMEA' },
  { code: 'me-south-1', short: 'MES1', label: getRegionName('me-south-1'), geo: 'EMEA' },
  { code: 'me-central-1', short: 'MEC1', label: getRegionName('me-central-1'), geo: 'EMEA' },
  { code: 'af-south-1', short: 'AFS1', label: getRegionName('af-south-1'), geo: 'EMEA' },
  { code: 'il-central-1', short: 'ILC1', label: getRegionName('il-central-1'), geo: 'EMEA' },
  // APAC
  { code: 'ap-northeast-1', short: 'ANE1', label: getRegionName('ap-northeast-1'), geo: 'APAC' },
  { code: 'ap-northeast-2', short: 'ANE2', label: getRegionName('ap-northeast-2'), geo: 'APAC' },
  { code: 'ap-northeast-3', short: 'ANE3', label: getRegionName('ap-northeast-3'), geo: 'APAC' },
  { code: 'ap-southeast-1', short: 'ASE1', label: getRegionName('ap-southeast-1'), geo: 'APAC' },
  { code: 'ap-southeast-2', short: 'ASE2', label: getRegionName('ap-southeast-2'), geo: 'APAC' },
  { code: 'ap-southeast-3', short: 'ASE3', label: getRegionName('ap-southeast-3'), geo: 'APAC' },
  { code: 'ap-southeast-4', short: 'ASE4', label: getRegionName('ap-southeast-4'), geo: 'APAC' },
  { code: 'ap-southeast-5', short: 'ASE5', label: getRegionName('ap-southeast-5'), geo: 'APAC' },
  { code: 'ap-southeast-6', short: 'ASE6', label: getRegionName('ap-southeast-6'), geo: 'APAC' },
  { code: 'ap-southeast-7', short: 'ASE7', label: getRegionName('ap-southeast-7'), geo: 'APAC' },
  { code: 'ap-south-1', short: 'APS1', label: getRegionName('ap-south-1'), geo: 'APAC' },
  { code: 'ap-south-2', short: 'APS2', label: getRegionName('ap-south-2'), geo: 'APAC' },
  { code: 'ap-east-1', short: 'APE1', label: getRegionName('ap-east-1'), geo: 'APAC' },
  { code: 'ap-east-2', short: 'APE2', label: getRegionName('ap-east-2'), geo: 'APAC' },
  // LATAM
  { code: 'sa-east-1', short: 'SAE1', label: getRegionName('sa-east-1'), geo: 'LATAM' },
  { code: 'mx-central-1', short: 'MXC1', label: getRegionName('mx-central-1'), geo: 'LATAM' },
  // GOVCLOUD
  { code: 'us-gov-west-1', short: 'UGVW', label: getRegionName('us-gov-west-1'), geo: 'GOVCLOUD' },
  { code: 'us-gov-east-1', short: 'UGVE', label: getRegionName('us-gov-east-1'), geo: 'GOVCLOUD' },
]

const GEO_GROUPS = [
  { id: 'NAMER', label: 'NAMER' },
  { id: 'EMEA', label: 'EMEA' },
  { id: 'APAC', label: 'APAC' },
  { id: 'LATAM', label: 'LATAM' },
  { id: 'GOVCLOUD', label: 'GOVCLOUD' },
]

const GEO_LABELS = { NAMER: 'NAMER', EMEA: 'EMEA', APAC: 'APAC', LATAM: 'LATAM', GOVCLOUD: 'GOVCLOUD' }

// Region data by geo for dropdown selection
const REGIONS_BY_GEO = {
  NAMER: [
    { code: 'us-east-1', name: 'US East (N. Virginia)' },
    { code: 'us-east-2', name: 'US East (Ohio)' },
    { code: 'us-west-1', name: 'US West (N. California)' },
    { code: 'us-west-2', name: 'US West (Oregon)' },
    { code: 'ca-central-1', name: 'Canada (Central)' },
    { code: 'ca-west-1', name: 'Canada (Calgary)' },
  ],
  EMEA: [
    { code: 'eu-west-1', name: 'Europe (Ireland)' },
    { code: 'eu-west-2', name: 'Europe (London)' },
    { code: 'eu-west-3', name: 'Europe (Paris)' },
    { code: 'eu-central-1', name: 'Europe (Frankfurt)' },
    { code: 'eu-central-2', name: 'Europe (Zurich)' },
    { code: 'eu-north-1', name: 'Europe (Stockholm)' },
    { code: 'eu-south-1', name: 'Europe (Milan)' },
    { code: 'eu-south-2', name: 'Europe (Spain)' },
    { code: 'me-south-1', name: 'Middle East (Bahrain)' },
    { code: 'me-central-1', name: 'Middle East (UAE)' },
    { code: 'il-central-1', name: 'Israel (Tel Aviv)' },
    { code: 'af-south-1', name: 'Africa (Cape Town)' },
  ],
  APAC: [
    { code: 'ap-northeast-1', name: 'Asia Pacific (Tokyo)' },
    { code: 'ap-northeast-2', name: 'Asia Pacific (Seoul)' },
    { code: 'ap-northeast-3', name: 'Asia Pacific (Osaka)' },
    { code: 'ap-southeast-1', name: 'Asia Pacific (Singapore)' },
    { code: 'ap-southeast-2', name: 'Asia Pacific (Sydney)' },
    { code: 'ap-southeast-3', name: 'Asia Pacific (Jakarta)' },
    { code: 'ap-southeast-4', name: 'Asia Pacific (Melbourne)' },
    { code: 'ap-southeast-5', name: 'Asia Pacific (Malaysia)' },
    { code: 'ap-south-1', name: 'Asia Pacific (Mumbai)' },
    { code: 'ap-south-2', name: 'Asia Pacific (Hyderabad)' },
    { code: 'ap-east-1', name: 'Asia Pacific (Hong Kong)' },
  ],
  LATAM: [
    { code: 'sa-east-1', name: 'South America (São Paulo)' },
    { code: 'mx-central-1', name: 'Mexico (Central)' },
  ],
  GOVCLOUD: [
    { code: 'us-gov-west-1', name: 'AWS GovCloud (US-West)' },
    { code: 'us-gov-east-1', name: 'AWS GovCloud (US-East)' },
  ],
}

// Geo options for the filter
const GEO_OPTIONS = ['NAMER', 'EMEA', 'APAC', 'LATAM', 'GOVCLOUD']
const CRIS_SCOPE_LABELS = { APAC: 'APAC (Legacy)' }

// Auto-detect business geo from region code prefix
// Note: 'us-gov' is handled specially in buildRegionEntry() since it must be checked before 'us'
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
 * Uses centralized region utilities for the label.
 */
function buildRegionEntry(code) {
  // Check for us-gov prefix first (before generic 'us')
  let geo
  if (code.startsWith('us-gov-')) {
    geo = 'GOVCLOUD'
  } else {
    const prefix = code.split('-')[0]
    geo = REGION_PREFIX_GEO[prefix] || 'EMEA'
  }
  // Generate 4-char short: prefix uppercase + first char of middle parts + number
  const parts = code.split('-')
  const mid = parts.slice(1, -1).map(s => s[0].toUpperCase()).join('')
  const num = parts[parts.length - 1]
  const raw = parts[0].toUpperCase() + mid + num
  const short = raw.length > 4 ? raw.slice(0, 3) + num : raw
  // Use centralized region name if available, otherwise fall back to code
  const label = getRegionName(code)
  return { code, short, label, geo }
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
 * Compute per-region availability for a model (on-demand + CRIS + Mantle + GovCloud).
 *
 * Data sources:
 * - availability.on_demand.regions: actual ON_DEMAND availability from regional-availability Lambda
 * - availability.cross_region.regions: CRIS source regions
 * - availability.mantle.regions: Mantle engine regions
 * - availability.govcloud: GovCloud availability with inference_type (cris | in_region)
 * - availability.hide_in_region: when true, hide In-Region availability (model has both Mantle and In-Region)
 */
function getRegionAvailability(model, regionCode) {
  const hideInRegion = model.availability?.hide_in_region ?? false
  const inRegionList = hideInRegion ? [] : (model.availability?.on_demand?.regions ?? model.in_region ?? [])
  const crisRegions = model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions ?? []
  const mantleRegions = model.availability?.mantle?.regions ?? []
  
  // Check GovCloud availability
  const govcloud = model.availability?.govcloud
  const govcloudRegions = govcloud?.supported ? (govcloud.regions || []) : []
  const isGovcloudRegion = regionCode.startsWith('us-gov-')
  
  // For GovCloud regions, check govcloud availability
  // When hide_in_region is true, also hide GovCloud in_region availability
  const onDemand = hideInRegion ? false : (isGovcloudRegion 
    ? (govcloud?.inference_type === 'in_region' && govcloudRegions.includes(regionCode))
    : inRegionList.includes(regionCode))
  const cris = isGovcloudRegion
    ? (govcloud?.inference_type === 'cris' && govcloudRegions.includes(regionCode))
    : crisRegions.includes(regionCode)
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
  const lifecycle = model.lifecycle ?? model.model_lifecycle
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
    const isGovcloudRegion = regionCode.startsWith('us-gov-')
    
    // Check GovCloud CRIS availability
    if (isGovcloudRegion) {
      const govcloud = model.availability?.govcloud
      if (govcloud?.supported && govcloud?.inference_type === 'cris') {
        const govcloudRegions = govcloud.regions || []
        if (govcloudRegions.includes(regionCode)) {
          // If GovCloud scope is selected (or no filter), return true
          if (!selectedScopes || selectedScopes.size === 0 || selectedScopes.has('GOVCLOUD')) {
            return true
          }
        }
      }
      // GovCloud regions don't have regular CRIS profiles, so return false if not matched above
      return false
    }
    
    // Regular CRIS check for non-GovCloud regions
    if (!selectedScopes || selectedScopes.size === 0) {
      // No filter - check if any CRIS profile exists for this region
      const sourceRegions = model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions
      return sourceRegions?.includes(regionCode) || false
    }
    
    // Check if any profile matches both the region AND the selected scope
    const profiles = model.availability?.cross_region?.profiles ?? model.cross_region_inference?.profiles ?? []
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
  } else if (activeView === 'in_region_all') {
    // Combined view: show if either Runtime OR Mantle is available
    isAvailable = onDemand || mantle
    colors = isAvailable ? getAvailabilityColors(onDemand, false, mantle, isLight) : null
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
    const isGovcloudRegion = regionCode.startsWith('us-gov-')
    
    // For GovCloud regions, check govcloud data
    if (isGovcloudRegion) {
      const govcloud = model.availability?.govcloud
      if (govcloud?.supported && govcloud?.inference_type === 'cris') {
        const govcloudRegions = govcloud.regions || []
        if (govcloudRegions.includes(regionCode)) {
          // Only return GovCloud scope if no filter or GovCloud is selected
          if (!selectedCrisScopes || selectedCrisScopes.size === 0 || selectedCrisScopes.has('GOVCLOUD')) {
            return ['GOVCLOUD']
          }
        }
      }
      return []
    }
    
    // Regular CRIS scopes for non-GovCloud regions
    const profiles = model.availability?.cross_region?.profiles ?? model.cross_region_inference?.profiles ?? []
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
          <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In-Region (Runtime API)</span>
        </div>
      )
    }
    
    if (activeView === 'in_region_all') {
      // Show both Runtime and Mantle info when applicable
      return (
        <>
          {onDemand && (
            <div className="flex items-center gap-1.5">
              <Zap className={cn('w-3 h-3', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')} strokeWidth={2} />
              <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In-Region (Runtime API)</span>
            </div>
          )}
          {mantle && (
            <div className="flex items-center gap-1.5">
              <Cpu className={cn('w-3 h-3', isLight ? 'text-violet-500' : 'text-violet-400')} strokeWidth={2} />
              <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In-Region (Mantle API)</span>
            </div>
          )}
        </>
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
          <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In-Region (Mantle API)</span>
        </div>
      )
    }
    
    // 'all' view: show all available types
    const hideInRegion = model.availability?.hide_in_region ?? false
    return (
      <>
        {onDemand && !hideInRegion && (
          <div className="flex items-center gap-1.5">
            <Zap className={cn('w-3 h-3', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')} strokeWidth={2} />
            <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In-Region (Runtime API)</span>
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
            <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In-Region (Mantle API)</span>
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

// Geo dropdown pill component with region checkboxes
function GeoDropdownPill({ 
  geo, 
  label, 
  isGeoSelected, 
  selectedRegions,
  onToggleRegion,
  onSelectAllGeo,
  onDeselectAllGeo,
  isLight 
}) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef(null)
  
  const regions = REGIONS_BY_GEO[geo] || []
  const geoRegionCodes = regions.map(r => r.code)
  
  // Check if all regions in this geo are selected
  const allRegionsSelected = isGeoSelected || geoRegionCodes.every(code => selectedRegions.includes(code))
  // Check if some (but not all) regions are selected
  const someRegionsSelected = !allRegionsSelected && geoRegionCodes.some(code => selectedRegions.includes(code))
  // Count selected regions in this geo
  const selectedCount = geoRegionCodes.filter(code => selectedRegions.includes(code)).length
  
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const getSelectedStyle = () => isLight
    ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
    : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
  
  const getPartialStyle = () => isLight
    ? 'bg-amber-100 text-amber-800 border-amber-300'
    : 'bg-[#1A9E7A]/30 text-[#1A9E7A] border-[#1A9E7A]/50'
  
  const getUnselectedStyle = () => isLight
    ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
    : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'

  const isActive = isGeoSelected || allRegionsSelected
  const isPartial = someRegionsSelected

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border flex items-center gap-1',
          isActive ? getSelectedStyle() : isPartial ? getPartialStyle() : getUnselectedStyle()
        )}
      >
        {label}
        {selectedCount > 0 && !isGeoSelected && (
          <span className="text-[9px] opacity-70">({selectedCount})</span>
        )}
        <ChevronDown className={cn('w-3 h-3 transition-transform', isOpen && 'rotate-180')} />
      </button>

      {isOpen && (
        <div className={cn(
          'absolute z-50 mt-1 min-w-[220px] rounded-md border shadow-lg animate-slide-down',
          isLight
            ? 'bg-white border-stone-200 shadow-stone-900/10'
            : 'bg-[#25262b] border-[#373a40] shadow-black/20'
        )}>
          {/* All [GEO] option */}
          <div className={cn(
            'p-1 border-b',
            isLight ? 'border-stone-200' : 'border-[#373a40]'
          )}>
            <button
              type="button"
              onClick={() => {
                if (isGeoSelected) {
                  onDeselectAllGeo(geo)
                } else {
                  onSelectAllGeo(geo)
                }
                setIsOpen(false)
              }}
              className={cn(
                'flex w-full items-center gap-2 rounded px-2.5 py-1.5 text-xs transition-colors',
                isLight ? 'hover:bg-stone-100 text-stone-700' : 'hover:bg-[#373a40] text-[#e4e5e7]'
              )}
            >
              <div className={cn(
                'flex h-3.5 w-3.5 items-center justify-center rounded border transition-colors flex-shrink-0',
                isGeoSelected
                  ? isLight ? 'bg-amber-700 border-amber-700' : 'bg-[#1A9E7A] border-[#1A9E7A]'
                  : isLight ? 'border-stone-300' : 'border-[#4a4d54]'
              )}>
                {isGeoSelected && <Check className="h-2.5 w-2.5 text-white" />}
              </div>
              <span className="font-medium">All {label}</span>
            </button>
          </div>

          {/* Individual regions */}
          <div className="max-h-48 overflow-y-auto p-1">
            {regions.map(region => {
              const isRegionSelected = selectedRegions.includes(region.code)
              return (
                <button
                  key={region.code}
                  type="button"
                  onClick={() => onToggleRegion(region.code)}
                  className={cn(
                    'flex w-full items-center gap-2 rounded px-2.5 py-1.5 text-xs transition-colors',
                    isLight ? 'hover:bg-stone-100 text-stone-700' : 'hover:bg-[#373a40] text-[#e4e5e7]'
                  )}
                >
                  <div className={cn(
                    'flex h-3.5 w-3.5 items-center justify-center rounded border transition-colors flex-shrink-0',
                    isRegionSelected || isGeoSelected
                      ? isLight ? 'bg-amber-700 border-amber-700' : 'bg-[#1A9E7A] border-[#1A9E7A]'
                      : isLight ? 'border-stone-300' : 'border-[#4a4d54]'
                  )}>
                    {(isRegionSelected || isGeoSelected) && <Check className="h-2.5 w-2.5 text-white" />}
                  </div>
                  <span className="truncate">{region.name}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export function RegionalAvailability() {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const { models, loading, error, getPricingForModel } = useModels()
  const [searchQuery, setSearchQuery] = useState('')
  const [collapsedProviders, setCollapsedProviders] = useState(new Set())
  
  const [hoveredRow, setHoveredRow] = useState(null)

  // Nested filter state - single-select routing with conditional sub-filters
  const [selectedRouting, setSelectedRouting] = useState(null)  // null, 'in_region', or 'cris'
  const [selectedApis, setSelectedApis] = useState(null)        // null, 'runtime_api', or 'mantle' - only for in_region (single-select)
  const [selectedGeos, setSelectedGeos] = useState(new Set())   // NAMER, EMEA, etc. - only for in_region
  const [selectedEndpoints, setSelectedEndpoints] = useState(new Set())  // Global, US, EU, etc. - only for cris
  const [selectedRegions, setSelectedRegions] = useState([])  // Specific regions selected via dropdown

  // Compute activeView for backward compatibility with existing logic
  const activeView = useMemo(() => {
    if (!selectedRouting) return 'all'
    if (selectedRouting === 'cris') return 'cris'
    
    // In-Region routing - check API selection (single-select)
    if (selectedRouting === 'in_region') {
      // Only Mantle selected
      if (selectedApis === 'mantle') {
        return 'mantle'
      }
      // Only Runtime selected
      if (selectedApis === 'runtime_api') {
        return 'in_region'
      }
      // No filter (All) - show combined in-region view
      return 'in_region_all'
    }
    
    return 'all'
  }, [selectedRouting, selectedApis])

  const tableContainerRef = useRef(null)

  // Calculate routing counts
  const routingCounts = useMemo(() => {
    return countModelsByRouting(models)
  }, [models])

  // Regions with data — known ones keep their order/labels, unknown ones auto-detected
  const activeRegions = useMemo(() => {
    if (!models.length) return REGION_COLUMNS
    const usedRegions = new Set()
    models.forEach(m => {
      ;(m.availability?.on_demand?.regions ?? m.in_region ?? []).forEach(r => usedRegions.add(r))
      ;(m.availability?.cross_region?.regions ?? m.cross_region_inference?.source_regions ?? []).forEach(r => usedRegions.add(r))
      ;(m.availability?.mantle?.regions ?? []).forEach(r => usedRegions.add(r))
      ;(m.availability?.govcloud?.regions ?? []).forEach(r => usedRegions.add(r))
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

  // Visible regions — filtered by selected geos (for in_region) or CRIS source regions (for cris)
  const visibleRegions = useMemo(() => {
    // If CRIS selected, filter by CRIS source regions and selected endpoints
    if (selectedRouting === 'cris') {
      const crisSourceRegions = new Set()
      
      models.forEach(m => {
        const crisSupported = m.availability?.cross_region?.supported ?? m.cross_region_inference?.supported
        if (crisSupported) {
          const profiles = m.availability?.cross_region?.profiles ?? m.cross_region_inference?.profiles ?? []
          profiles.forEach(p => {
            const prefix = p.profile_id?.split('.')[0] || ''
            const scope = normalizeCrisPrefix(prefix)
            // If no endpoints selected, show all CRIS regions
            // If endpoints selected, filter by those endpoints
            if (selectedEndpoints.size === 0 || selectedEndpoints.has(scope)) {
              if (p.source_region) crisSourceRegions.add(p.source_region)
            }
          })
        }
      })
      
      // Add GovCloud if selected
      if (selectedEndpoints.size === 0 || selectedEndpoints.has('GOVCLOUD')) {
        models.forEach(m => {
          const govcloud = m.availability?.govcloud
          if (govcloud?.supported && govcloud?.inference_type === 'cris') {
            (govcloud.regions || []).forEach(r => crisSourceRegions.add(r))
          }
        })
      }
      
      return activeRegions.filter(r => crisSourceRegions.has(r.code))
    }

    // For In-Region: filter by API selection AND geo/regions
    if (selectedRouting === 'in_region') {
      let apiFilteredRegions = activeRegions
      
      // If specific API is selected, filter regions accordingly (single-select)
      if (selectedApis) {
        const relevantRegions = new Set()
        
        models.forEach(m => {
          const hideInRegion = m.availability?.hide_in_region ?? false
          
          if (selectedApis === 'runtime_api') {
            // For Runtime API, respect hide_in_region flag
            if (!hideInRegion) {
              const runtimeRegions = m.availability?.on_demand?.regions ?? m.in_region ?? []
              runtimeRegions.forEach(r => relevantRegions.add(r))
            }
          }
          if (selectedApis === 'mantle') {
            // For Mantle API, ignore hide_in_region - always include Mantle regions
            const mantleRegions = m.availability?.mantle?.regions ?? []
            mantleRegions.forEach(r => relevantRegions.add(r))
          }
        })
        
        apiFilteredRegions = activeRegions.filter(r => relevantRegions.has(r.code))
      }
      
      // If specific regions are selected, filter to only those
      if (selectedRegions.length > 0) {
        return apiFilteredRegions.filter(r => selectedRegions.includes(r.code))
      }
      
      // Then apply geo filter
      if (selectedGeos.size === 0) return apiFilteredRegions
      return apiFilteredRegions.filter(r => selectedGeos.has(r.geo))
    }

    // For All: filter by specific regions first, then geo
    // If specific regions are selected, filter to only those
    if (selectedRegions.length > 0) {
      return activeRegions.filter(r => selectedRegions.includes(r.code))
    }
    
    // Then apply geo filter
    if (selectedGeos.size === 0) return activeRegions
    return activeRegions.filter(r => selectedGeos.has(r.geo))
  }, [activeRegions, selectedRouting, selectedGeos, selectedEndpoints, selectedApis, selectedRegions, models])

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
    // Always compute CRIS prefixes (needed when CRIS is selected)
    const prefixes = new Set()
    let hasGovCloudCris = false
    
    models.forEach(m => {
      // Check for CRIS endpoint prefixes
      (m.availability?.cross_region?.profiles ?? m.cross_region_inference?.profiles ?? []).forEach(p => {
        const prefix = p.profile_id?.split('.')[0] || ''
        prefixes.add(normalizeCrisPrefix(prefix))
      })
      // Check for GovCloud CRIS availability
      if (m.availability?.govcloud?.supported && m.availability?.govcloud?.inference_type === 'cris') {
        hasGovCloudCris = true
      }
    })
    
    const order = ['Global', 'US', 'CA', 'EU', 'APAC', 'AU', 'JP']
    const result = order.filter(p => prefixes.has(p)).concat(
      [...prefixes].filter(p => !order.includes(p)).sort()
    )
    
    // Add GovCloud at the end if any models have GovCloud CRIS
    if (hasGovCloudCris) {
      result.push('GOVCLOUD')
    }
    
    return result
  }, [models])

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
      // Search filter
      if (q && !(
        m.model_name?.toLowerCase().includes(q) ||
        m.model_id?.toLowerCase().includes(q) ||
        m.model_provider?.toLowerCase().includes(q)
      )) return false
      
      // Routing filter
      if (selectedRouting === 'in_region') {
        const hideInRegion = m.availability?.hide_in_region ?? false
        const hasMantle = m.availability?.mantle?.supported
        const inRegionList = m.availability?.on_demand?.regions ?? m.in_region
        const hasRuntime = inRegionList?.length > 0
        
        // API filter (single-select)
        if (selectedApis) {
          let matchesApi = false
          
          if (selectedApis === 'runtime_api') {
            // For Runtime API, respect hide_in_region flag
            if (!hideInRegion && hasRuntime) matchesApi = true
          }
          if (selectedApis === 'mantle') {
            // For Mantle API, ignore hide_in_region - show if model has Mantle
            if (hasMantle) matchesApi = true
          }
          if (!matchesApi) return false
        } else {
          // No API selected (All) - show models with runtime OR mantle
          // But respect hide_in_region for runtime-only models
          if (hideInRegion) {
            // Model has hide_in_region - only show if it has Mantle
            if (!hasMantle) return false
          } else {
            // Normal case - must have either runtime or mantle
            if (!hasRuntime && !hasMantle) return false
          }
        }
      }
      
      if (selectedRouting === 'cris') {
        const crisSupported = m.availability?.cross_region?.supported ?? m.cross_region_inference?.supported
        const hasGovCloudCris = m.availability?.govcloud?.supported && 
                               m.availability?.govcloud?.inference_type === 'cris'
        if (!crisSupported && !hasGovCloudCris) return false
        
        // Endpoint filter
        if (selectedEndpoints.size > 0) {
          const profiles = m.availability?.cross_region?.profiles ?? m.cross_region_inference?.profiles ?? []
          const modelScopes = new Set(profiles.map(p => {
            const prefix = p.profile_id?.split('.')[0]?.toLowerCase() || ''
            return prefix === 'global' ? 'Global' : prefix.toUpperCase()
          }))
          
          // Add GOVCLOUD if model has it
          if (hasGovCloudCris) modelScopes.add('GOVCLOUD')
          
          const hasSelectedEndpoint = [...selectedEndpoints].some(ep => modelScopes.has(ep))
          if (!hasSelectedEndpoint) return false
        }
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
  }, [models, searchQuery, selectedRouting, selectedApis, selectedEndpoints])

  const totalFiltered = groupedModels.reduce((sum, [, models]) => sum + models.length, 0)

  // Flatten grouped models into a single array for virtualization
  // Each item is either a provider header or a model row
  const flattenedRows = useMemo(() => {
    const rows = []
    groupedModels.forEach(([provider, providerModels]) => {
      const isCollapsed = collapsedProviders.has(provider)
      // Compute provider region coverage for the header row
      const providerRegionCoverage = visibleRegions.map(r =>
        providerModels.some(m => {
          const { available, onDemand, cris, mantle } = getRegionAvailability(m, r.code)
          if (activeView === 'in_region') return onDemand
          if (activeView === 'in_region_all') return onDemand || mantle
          if (activeView === 'mantle') return mantle
          if (activeView === 'cris') return cris
          return available
        })
      )
      
      // Add provider header row
      rows.push({
        type: 'provider',
        provider,
        providerModels,
        isCollapsed,
        providerRegionCoverage,
      })
      
      // Add model rows if not collapsed
      if (!isCollapsed) {
        providerModels.forEach(model => {
          rows.push({
            type: 'model',
            provider,
            model,
          })
        })
      }
    })
    return rows
  }, [groupedModels, collapsedProviders, visibleRegions, activeView])

  // Constants for grid layout
  const REGION_COL_WIDTH = 48 // w-12 = 3rem = 48px
  const HEADER_HEIGHT = 72 // Two header rows: ~26px + ~42px

  // Set up virtualizer for table rows
  const rowVirtualizer = useVirtualizer({
    count: flattenedRows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: (index) => {
      // Provider rows are slightly taller than model rows
      return flattenedRows[index]?.type === 'provider' ? 40 : 48
    },
    overscan: 15, // Increased overscan for smoother scrolling with CSS Grid
  })

  // Total width for the scrollable area
  const totalWidth = MODEL_COL_WIDTH + (visibleRegions.length * REGION_COL_WIDTH) + 100 // +100 for spacer minimum

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
          else if (activeView === 'in_region_all') isAvail = onDemand || mantle
          else if (activeView === 'mantle') isAvail = mantle
          else if (activeView === 'cris') isAvail = cris
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
        const lifecycle = model.lifecycle ?? model.model_lifecycle
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

  // Check if any models in the current routing mode are missing pricing data
  // Uses all models (not endpoint-filtered) so the banner stays visible when drilling into a specific profile
  const hasModelsWithoutPricing = useMemo(() => {
    for (const model of models) {
      if (selectedRouting === 'cris') {
        const crisSupported = model.availability?.cross_region?.supported ?? model.cross_region_inference?.supported
        const hasGovCloudCris = model.availability?.govcloud?.supported &&
                               model.availability?.govcloud?.inference_type === 'cris'
        if (!crisSupported && !hasGovCloudCris) continue
      }
      if (selectedRouting === 'in_region') {
        const hideInRegion = model.availability?.hide_in_region ?? false
        const hasMantle = model.availability?.mantle?.supported
        const inRegionList = model.availability?.on_demand?.regions ?? model.in_region
        const hasRuntime = inRegionList?.length > 0
        if (hideInRegion && !hasMantle) continue
        if (!hasRuntime && !hasMantle) continue
      }
      if (!modelHasPricing(model, getPricingForModel)) return true
    }
    return false
  }, [models, selectedRouting, getPricingForModel])

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

  const selectRouting = (routing) => {
    if (selectedRouting === routing) {
      // Deselect - go back to "All"
      setSelectedRouting(null)
      setSelectedApis(null)
      setSelectedGeos(new Set())
      setSelectedEndpoints(new Set())
      setSelectedRegions([])
    } else {
      // Select new routing, clear other selections
      setSelectedRouting(routing)
      setSelectedApis(null)
      setSelectedGeos(new Set())
      setSelectedEndpoints(new Set())
      setSelectedRegions([])
    }
    tableContainerRef.current?.scrollTo({ left: 0 })
  }

  const selectApi = (api) => {
    // If already selected, deselect (go back to All)
    if (selectedApis === api) {
      setSelectedApis(null)
    } else {
      setSelectedApis(api)
    }
  }

  const toggleGeo = (geo) => {
    setSelectedGeos(prev => {
      const next = new Set(prev)
      if (next.has(geo)) next.delete(geo)
      else next.add(geo)
      return next
    })
    tableContainerRef.current?.scrollTo({ left: 0 })
  }

  const toggleEndpoint = (endpoint) => {
    setSelectedEndpoints(prev => {
      const next = new Set(prev)
      if (next.has(endpoint)) next.delete(endpoint)
      else next.add(endpoint)
      return next
    })
    tableContainerRef.current?.scrollTo({ left: 0 })
  }

  const clearAllFilters = () => {
    setSelectedRouting(null)
    setSelectedApis(null)
    setSelectedGeos(new Set())
    setSelectedEndpoints(new Set())
    setSelectedRegions([])
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

        {/* Filter pills row */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* Inference pills - single select */}
          <span className={cn('text-[10px] uppercase tracking-wider font-medium mr-1', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
            Inference
          </span>
          
          {/* All pill */}
          <button
            onClick={clearAllFilters}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
              !selectedRouting
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
          
          {/* In-Region pill */}
          <button
            onClick={() => selectRouting('in_region')}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border flex items-center gap-1',
              selectedRouting === 'in_region'
                ? isLight
                  ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                  : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                : isLight
                  ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                  : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
            )}
          >
            <Zap className="w-3 h-3" />
            In-Region
            <span className={cn('text-[10px]', selectedRouting === 'in_region' ? 'opacity-70' : 'opacity-50')}>
              ({routingCounts.in_region})
            </span>
          </button>
          
          {/* CRIS pill */}
          <button
            onClick={() => selectRouting('cris')}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border flex items-center gap-1',
              selectedRouting === 'cris'
                ? isLight
                  ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                  : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                : isLight
                  ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                  : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
            )}
          >
            <Globe className="w-3 h-3" />
            CRIS
            <span className={cn('text-[10px]', selectedRouting === 'cris' ? 'opacity-70' : 'opacity-50')}>
              ({routingCounts.cris})
            </span>
          </button>

          {/* API pills - only when In-Region selected */}
          {selectedRouting === 'in_region' && (
            <>
              {/* Divider */}
              <div className={cn('w-px h-5 mx-1', isLight ? 'bg-stone-200' : 'bg-white/[0.08]')} />
              
              {/* API pills */}
              <span className={cn('text-[10px] uppercase tracking-wider font-medium mr-1', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
                API
              </span>
              
              {/* All API button */}
              <button
                onClick={() => setSelectedApis(null)}
                className={cn(
                  'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                  selectedApis === null
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
              
              <button
                onClick={() => selectApi('runtime_api')}
                className={cn(
                  'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                  selectedApis === 'runtime_api'
                    ? isLight
                      ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                      : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                    : isLight
                      ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                      : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
                )}
              >
                Runtime API
              </button>
              
              <button
                onClick={() => selectApi('mantle')}
                className={cn(
                  'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                  selectedApis === 'mantle'
                    ? isLight
                      ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                      : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                    : isLight
                      ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                      : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
                )}
              >
                Mantle API
              </button>
            </>
          )}

          {/* GEO pills with region dropdowns - when All or In-Region selected */}
          {selectedRouting !== 'cris' && (
            <>
              <div className={cn('w-px h-5 mx-1', isLight ? 'bg-stone-200' : 'bg-white/[0.08]')} />
              
              <span className={cn('text-[10px] uppercase tracking-wider font-medium mr-1', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
                Geo
              </span>
              
              {/* All Geos button */}
              <button
                onClick={() => {
                  setSelectedGeos(new Set())
                  setSelectedRegions([])
                }}
                className={cn(
                  'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                  selectedGeos.size === 0 && selectedRegions.length === 0
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
              
              {/* Geo dropdown pills */}
              {GEO_OPTIONS.map(geo => (
                <GeoDropdownPill
                  key={geo}
                  geo={geo}
                  label={geo}
                  isGeoSelected={selectedGeos.has(geo)}
                  selectedRegions={selectedRegions}
                  onToggleRegion={(regionCode) => {
                    setSelectedRegions(prev => {
                      if (prev.includes(regionCode)) {
                        return prev.filter(r => r !== regionCode)
                      }
                      return [...prev, regionCode]
                    })
                  }}
                  onSelectAllGeo={(geo) => {
                    const geoRegionCodes = (REGIONS_BY_GEO[geo] || []).map(r => r.code)
                    setSelectedRegions(prev => prev.filter(r => !geoRegionCodes.includes(r)))
                    setSelectedGeos(prev => {
                      const next = new Set(prev)
                      next.add(geo)
                      return next
                    })
                  }}
                  onDeselectAllGeo={(geo) => {
                    setSelectedGeos(prev => {
                      const next = new Set(prev)
                      next.delete(geo)
                      return next
                    })
                  }}
                  isLight={isLight}
                />
              ))}
            </>
          )}

          {/* Endpoint pills - only when CRIS selected */}
          {selectedRouting === 'cris' && (
            <>
              {/* Divider */}
              <div className={cn('w-px h-5 mx-1', isLight ? 'bg-stone-200' : 'bg-white/[0.08]')} />
              
              {/* Endpoint pills */}
              <span className={cn('text-[10px] uppercase tracking-wider font-medium mr-1', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
                Endpoint
              </span>
              
              <button
                onClick={() => setSelectedEndpoints(new Set())}
                className={cn(
                  'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                  selectedEndpoints.size === 0
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
              
              {availableCrisPrefixes.map(item => {
                const id = typeof item === 'string' ? item : item.id
                const label = CRIS_SCOPE_LABELS[typeof item === 'string' ? item : item.id] || (typeof item === 'string' ? item : item.label)
                const isSelected = selectedEndpoints.has(id)
                return (
                  <button
                    key={id}
                    onClick={() => toggleEndpoint(id)}
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
            </>
          )}
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

        {/* Pricing Legend - only show when there are models without pricing */}
        {hasModelsWithoutPricing && (
          <div className={cn(
            'flex items-center gap-2 px-2.5 py-1.5 rounded-md text-[11px]',
            isLight 
              ? 'bg-amber-50 border border-amber-200 text-amber-700' 
              : 'bg-amber-900/20 border border-amber-700/30 text-amber-400'
          )}>
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
            <span>
              <strong>Verify</strong> = No pricing data. Please verify model availability in AWS console before use.
            </span>
          </div>
        )}
      </div>

      {/* Grid-based table container */}
      <div className="flex-1 relative min-h-0">
        <TooltipProvider delayDuration={150}>
          <div
            ref={tableContainerRef}
            className={cn(
              'h-full min-w-0 overflow-auto rounded-xl backdrop-blur-xl',
              'will-change-transform', // GPU acceleration for smooth scrolling
              isLight
                ? 'border border-stone-200/60 bg-white/70 shadow-[0_2px_15px_-3px_rgba(120,113,108,0.08)] ring-1 ring-stone-100/50'
                : 'border border-white/[0.06] bg-white/[0.03] shadow-[0_2px_15px_-3px_rgba(0,0,0,0.3)] ring-1 ring-white/[0.03]'
            )}
          >
            {/* Inner container with total height for scrollbar */}
            <div
              style={{
                height: rowVirtualizer.getTotalSize() + HEADER_HEIGHT,
                width: totalWidth,
                minWidth: '100%',
                position: 'relative',
              }}
            >
              {/* Sticky Header - CSS Grid for proper alignment */}
              <div
                className={cn(
                  'sticky top-0 z-20 grid',
                  isLight
                    ? 'bg-stone-50/90 backdrop-blur-sm'
                    : 'bg-[#141517]/95 backdrop-blur-xl'
                )}
                style={{
                  width: totalWidth,
                  minWidth: '100%',
                  // Grid: MODEL column + each region column + spacer
                  gridTemplateColumns: `${MODEL_COL_WIDTH}px repeat(${visibleRegions.length}, ${REGION_COL_WIDTH}px) minmax(100px, 1fr)`,
                  gridTemplateRows: 'auto auto',
                }}
              >
                {/* MODEL header - spans both rows (rowspan=2) */}
                <div
                  className={cn(
                    'sticky left-0 z-30 flex items-center text-left text-[11px] font-semibold uppercase tracking-wider px-3',
                    isLight
                      ? 'bg-stone-50/90 backdrop-blur-sm text-stone-500 border-r border-b border-stone-200'
                      : 'bg-[#141517]/95 backdrop-blur-xl text-[#9a9b9f] border-r border-b border-white/[0.06]'
                  )}
                  style={{
                    gridColumn: '1',
                    gridRow: '1 / span 2',
                  }}
                >
                  <div className="flex items-center justify-between w-full">
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
                </div>

                {/* Row 1: Geo group headers - each spans its region columns */}
                {(() => {
                  let colStart = 2 // Start after MODEL column (1-indexed for grid)
                  return geoHeaderCells.map((cell, gi) => {
                    const startCol = colStart
                    colStart += cell.span
                    return (
                      <div
                        key={cell.geo}
                        className={cn(
                          'flex items-center justify-center text-center py-1.5 text-[10px] font-bold uppercase tracking-widest',
                          gi > 0 && (isLight ? 'border-l border-l-stone-200' : 'border-l border-l-white/[0.08]'),
                          isLight ? 'text-amber-700 border-b border-stone-200' : 'text-[#1A9E7A] border-b border-white/[0.06]'
                        )}
                        style={{
                          gridColumn: `${startCol} / span ${cell.span}`,
                          gridRow: '1',
                        }}
                      >
                        {GEO_LABELS[cell.geo]}
                      </div>
                    )
                  })
                })()}

                {/* Row 1: Spacer cell (top-right corner) */}
                <div
                  className={cn(
                    isLight ? 'border-b border-stone-200' : 'border-b border-white/[0.06]'
                  )}
                  style={{
                    gridColumn: `${visibleRegions.length + 2}`,
                    gridRow: '1',
                  }}
                />

                {/* Row 2: Individual region column headers */}
                {visibleRegions.map((region, i) => (
                  <div
                    key={region.code}
                    className={cn(
                      'text-center',
                      isGeoBreak(i) && (isLight ? 'border-l border-l-stone-200' : 'border-l border-l-white/[0.08]'),
                      isLight ? 'border-b border-stone-200' : 'border-b border-white/[0.06]',
                      getColumnTint(region.code)
                    )}
                    style={{
                      gridColumn: `${i + 2}`, // +2 because MODEL is column 1
                      gridRow: '2',
                    }}
                  >
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
                          {region.label}{getAirportCode(region.code) ? ` (${getAirportCode(region.code)})` : ''}
                        </div>
                        <div className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
                          {region.code}
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                ))}

                {/* Row 2: Spacer cell (bottom-right corner) */}
                <div
                  className={cn(
                    isLight ? 'border-b border-stone-200' : 'border-b border-white/[0.06]'
                  )}
                  style={{
                    gridColumn: `${visibleRegions.length + 2}`,
                    gridRow: '2',
                  }}
                />
              </div>

              {/* Virtualized Rows - absolutely positioned */}
              {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                const row = flattenedRows[virtualRow.index]
                
                if (row.type === 'provider') {
                  // Provider header row
                  const { provider, providerModels, isCollapsed, providerRegionCoverage } = row
                  return (
                    <div
                      key={`provider-${provider}`}
                      data-index={virtualRow.index}
                      className={cn(
                        'flex cursor-pointer select-none',
                        'contain-layout contain-style', // Performance optimization
                        isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.02]'
                      )}
                      style={{
                        position: 'absolute',
                        top: virtualRow.start + HEADER_HEIGHT,
                        left: 0,
                        width: '100%',
                        height: virtualRow.size,
                      }}
                      onClick={() => toggleProvider(provider)}
                    >
                      {/* Provider name cell - sticky */}
                      <div
                        className={cn(
                          'sticky left-0 z-10 px-3 flex items-center font-semibold text-xs flex-shrink-0',
                          isLight
                            ? 'bg-stone-50/95 backdrop-blur-sm text-stone-700 border-b border-r border-stone-100'
                            : 'bg-[#1a1b1f]/95 backdrop-blur-xl text-[#e4e5e7] border-b border-r border-white/[0.06]'
                        )}
                        style={{ width: MODEL_COL_WIDTH, minWidth: MODEL_COL_WIDTH }}
                      >
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
                      </div>
                      
                      {/* Region coverage cells */}
                      {visibleRegions.map((region, i) => (
                        <div
                          key={region.code}
                          className={cn(
                            'flex items-center justify-center flex-shrink-0',
                            isLight
                              ? 'bg-stone-50/95 border-b border-stone-100'
                              : 'bg-white/[0.06] border-b border-white/[0.06]',
                            isGeoBreak(i) && (isLight ? 'border-l border-l-stone-200' : 'border-l border-l-white/[0.08]'),
                            getColumnTint(region.code)
                          )}
                          style={{ width: REGION_COL_WIDTH }}
                        >
                          {providerRegionCoverage[i] && (
                            <div className={cn(
                              'w-2 h-2 rounded-full',
                              isLight ? 'bg-stone-300' : 'bg-white/[0.15]'
                            )} />
                          )}
                        </div>
                      ))}
                      
                      {/* Spacer cell */}
                      <div className={cn(
                        'flex-1 min-w-[100px]',
                        isLight ? 'bg-stone-50/95 border-b border-stone-100' : 'bg-white/[0.06] border-b border-white/[0.06]'
                      )} />
                    </div>
                  )
                }
                
                // Model row
                const { model } = row
                const modelRegions = model.availability?.on_demand?.regions ?? model.in_region ?? []
                const crisRegions = model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions ?? []
                const mantleRegions = model.availability?.mantle?.regions ?? []
                const allModelRegions = new Set([...modelRegions, ...crisRegions, ...mantleRegions])
                const regionCount = allModelRegions.size
                const isHovered = hoveredRow === model.model_id
                const noPricing = !modelHasPricing(model, getPricingForModel)

                return (
                  <div
                    key={model.model_id}
                    data-index={virtualRow.index}
                    className={cn(
                      'flex transition-colors duration-75',
                      'contain-layout contain-style', // Performance optimization
                      // Add amber background for models without pricing
                      noPricing && (isLight 
                        ? 'bg-amber-50/50' 
                        : 'bg-amber-900/10'
                      ),
                      isHovered && !noPricing
                        ? isLight ? 'bg-amber-50/50' : 'bg-white/[0.03]'
                        : '',
                      isHovered && noPricing
                        ? isLight ? 'bg-amber-50' : 'bg-amber-900/20'
                        : ''
                    )}
                    style={{
                      position: 'absolute',
                      top: virtualRow.start + HEADER_HEIGHT,
                      left: 0,
                      width: '100%',
                      height: virtualRow.size,
                    }}
                    onMouseEnter={() => setHoveredRow(model.model_id)}
                    onMouseLeave={() => setHoveredRow(null)}
                  >
                    {/* Model name cell - sticky */}
                    <div
                      className={cn(
                        'sticky left-0 z-10 px-3 flex items-center flex-shrink-0',
                        isLight
                          ? 'border-b border-r border-stone-100'
                          : 'border-b border-r border-white/[0.03]',
                        // Background for sticky cell - match row highlight
                        noPricing
                          ? isHovered
                            ? isLight ? 'bg-amber-50 backdrop-blur-sm' : 'bg-amber-900/30 backdrop-blur-xl'
                            : isLight ? 'bg-amber-50/80 backdrop-blur-sm' : 'bg-amber-900/20 backdrop-blur-xl'
                          : isHovered
                            ? isLight ? 'bg-amber-50/80 backdrop-blur-sm' : 'bg-[#1a1b1f]/90 backdrop-blur-xl'
                            : isLight ? 'bg-white/90 backdrop-blur-sm' : 'bg-[#141517]/95 backdrop-blur-xl'
                      )}
                      style={{ width: MODEL_COL_WIDTH, minWidth: MODEL_COL_WIDTH }}
                    >
                      <div className="flex items-center gap-2 min-w-0 w-full">
                        <div className="min-w-0 flex-1">
                          <div className={cn(
                            'text-xs font-medium truncate max-w-[200px] flex items-center gap-1.5',
                            isLight ? 'text-stone-800' : 'text-[#e4e5e7]'
                          )}>
                            <span className="truncate">{model.model_name}</span>
                            {noPricing && (
                              <Tooltip delayDuration={100}>
                                <TooltipTrigger asChild>
                                  <div className={cn(
                                    'flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium flex-shrink-0',
                                    isLight 
                                      ? 'bg-amber-100 text-amber-700 border border-amber-200' 
                                      : 'bg-amber-900/30 text-amber-400 border border-amber-700/30'
                                  )}>
                                    <AlertCircle className="w-3 h-3" />
                                    <span>Verify</span>
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent 
                                  side="top" 
                                  sideOffset={4}
                                  className={cn(
                                    'max-w-[280px] p-3 z-50',
                                    isLight ? 'bg-amber-50 border-amber-200' : 'bg-amber-950 border-amber-800'
                                  )}
                                >
                                  <div className="space-y-1">
                                    <div className={cn(
                                      'flex items-center gap-1.5 font-semibold text-xs',
                                      isLight ? 'text-amber-700' : 'text-amber-400'
                                    )}>
                                      <AlertCircle className="w-3.5 h-3.5" />
                                      No pricing data available
                                    </div>
                                    <p className={cn(
                                      'text-[11px]',
                                      isLight ? 'text-amber-600' : 'text-amber-300/80'
                                    )}>
                                      This model is not listed in the AWS Pricing API. Please verify model availability before use.
                                    </p>
                                  </div>
                                </TooltipContent>
                              </Tooltip>
                            )}
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
                    </div>
                    
                    {/* Availability cells */}
                    {visibleRegions.map((region, i) => (
                      <div
                        key={region.code}
                        className={cn(
                          'flex items-center justify-center flex-shrink-0',
                          isLight ? 'border-b border-stone-100' : 'border-b border-white/[0.03]',
                          isGeoBreak(i) && (isLight ? 'border-l border-l-stone-200' : 'border-l border-l-white/[0.08]'),
                          getColumnTint(region.code)
                        )}
                        style={{ width: REGION_COL_WIDTH }}
                      >
                        <AvailabilityCell
                          model={model}
                          regionCode={region.code}
                          regionLabel={region.label}
                          isLight={isLight}
                          activeView={activeView}
                          selectedCrisScopes={selectedRouting === 'cris' ? selectedEndpoints : null}
                        />
                      </div>
                    ))}
                    
                    {/* Spacer cell */}
                    <div className={cn(
                      'flex-1 min-w-[100px]',
                      isLight ? 'border-b border-stone-100' : 'border-b border-white/[0.03]'
                    )} />
                  </div>
                )
              })}

              {/* Empty state */}
              {groupedModels.length === 0 && (
                <div 
                  className={cn(
                    'absolute left-0 right-0 flex items-center justify-center py-16 text-sm',
                    isLight ? 'text-stone-400' : 'text-[#6d6e72]'
                  )}
                  style={{ top: HEADER_HEIGHT }}
                >
                  No models found matching &ldquo;{searchQuery}&rdquo;
                </div>
              )}
            </div>
          </div>
        </TooltipProvider>
      </div>
    </div>
  )
}
