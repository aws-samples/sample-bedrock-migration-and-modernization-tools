import { useState, useMemo, useRef, memo, Fragment, useEffect } from 'react'
import { Minus, Zap, Globe, Globe2, Cpu, Check, ChevronDown, ChevronRight, MapPin, Users, Maximize2, Minimize2, ChevronsUpDown, Map, AlertTriangle, AlertCircle, X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { RegionMap } from '../RegionMap'
import { providerColorClasses } from '@/config/constants'
import { getRegionName } from '@/utils/regionUtils'

const providerColors = providerColorClasses

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

// Geo groups for filtering and collapsing
const GEO_GROUPS = [
  { id: 'NAMER', label: 'North America', geos: ['US', 'CA'] },
  { id: 'EMEA', label: 'Europe, Middle East & Africa', geos: ['EU', 'ME', 'AF'] },
  { id: 'APAC', label: 'Asia Pacific', geos: ['AP'] },
  { id: 'LATAM', label: 'Latin America', geos: ['SA'] },
  { id: 'GOVCLOUD', label: 'GovCloud (US)', geos: ['GOV'] },
]

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

// Region rows ordered by geo (NAMER, EMEA, APAC, LATAM)
const REGION_ROWS = [
  // NAMER
  { code: 'us-east-1', short: 'USE1', geo: 'NAMER' },
  { code: 'us-east-2', short: 'USE2', geo: 'NAMER' },
  { code: 'us-west-2', short: 'USW2', geo: 'NAMER' },
  { code: 'us-west-1', short: 'USW1', geo: 'NAMER' },
  { code: 'ca-central-1', short: 'CAC1', geo: 'NAMER' },
  { code: 'ca-west-1', short: 'CAW1', geo: 'NAMER' },
  // EMEA
  { code: 'eu-west-1', short: 'EUW1', geo: 'EMEA' },
  { code: 'eu-west-2', short: 'EUW2', geo: 'EMEA' },
  { code: 'eu-west-3', short: 'EUW3', geo: 'EMEA' },
  { code: 'eu-central-1', short: 'EUC1', geo: 'EMEA' },
  { code: 'eu-central-2', short: 'EUC2', geo: 'EMEA' },
  { code: 'eu-north-1', short: 'EUN1', geo: 'EMEA' },
  { code: 'eu-south-1', short: 'EUS1', geo: 'EMEA' },
  { code: 'eu-south-2', short: 'EUS2', geo: 'EMEA' },
  { code: 'me-south-1', short: 'MES1', geo: 'EMEA' },
  { code: 'me-central-1', short: 'MEC1', geo: 'EMEA' },
  { code: 'il-central-1', short: 'ILC1', geo: 'EMEA' },
  { code: 'af-south-1', short: 'AFS1', geo: 'EMEA' },
  // APAC
  { code: 'ap-northeast-1', short: 'ANE1', geo: 'APAC' },
  { code: 'ap-northeast-2', short: 'ANE2', geo: 'APAC' },
  { code: 'ap-northeast-3', short: 'ANE3', geo: 'APAC' },
  { code: 'ap-southeast-1', short: 'ASE1', geo: 'APAC' },
  { code: 'ap-southeast-2', short: 'ASE2', geo: 'APAC' },
  { code: 'ap-southeast-3', short: 'ASE3', geo: 'APAC' },
  { code: 'ap-southeast-4', short: 'ASE4', geo: 'APAC' },
  { code: 'ap-southeast-5', short: 'ASE5', geo: 'APAC' },
  { code: 'ap-southeast-6', short: 'ASE6', geo: 'APAC' },
  { code: 'ap-southeast-7', short: 'ASE7', geo: 'APAC' },
  { code: 'ap-south-1', short: 'APS1', geo: 'APAC' },
  { code: 'ap-south-2', short: 'APS2', geo: 'APAC' },
  { code: 'ap-east-1', short: 'APE1', geo: 'APAC' },
  { code: 'ap-east-2', short: 'APE2', geo: 'APAC' },
  // LATAM
  { code: 'sa-east-1', short: 'SAE1', geo: 'LATAM' },
  { code: 'mx-central-1', short: 'MXC1', geo: 'LATAM' },
  // GOVCLOUD
  { code: 'us-gov-west-1', short: 'UGVW', geo: 'GOVCLOUD' },
  { code: 'us-gov-east-1', short: 'UGVE', geo: 'GOVCLOUD' },
]

// Get regions by consumption type
function getRegionsByType(model) {
  const govcloud = model.availability?.govcloud
  const govcloudRegions = govcloud?.supported ? (govcloud.regions || []) : []
  const hideInRegion = model.availability?.hide_in_region ?? false
  
  return {
    // When hide_in_region is true, return empty array for in_region
    in_region: hideInRegion ? [] : (model.availability?.on_demand?.regions ?? model.in_region ?? model.regions_available ?? []),
    cris: model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions ?? [],
    mantle: model.availability?.mantle?.regions ?? [],
    govcloud: govcloudRegions,
    govcloud_inference_type: govcloud?.inference_type || null,
  }
}

// Get all regions for a model (combines all consumption types)
function getAllModelRegions(model) {
  const byType = getRegionsByType(model)
  return [...new Set([...byType.in_region, ...byType.cris, ...byType.mantle, ...byType.govcloud])]
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

// Get short model name for display
function getShortModelName(model) {
  const name = model.model_name || model.model_id
  // For names like "Claude Opus 4.5", return "Opus 4.5"
  // For names like "Claude 3.5 Sonnet", return "3.5 Sonnet"
  // Just return the name without the provider prefix if it starts with provider
  const provider = model.model_provider?.toLowerCase() || ''
  let displayName = name
  if (name.toLowerCase().startsWith(provider)) {
    displayName = name.slice(provider.length).trim()
  }
  // If still too long, take last 2-3 words
  if (displayName.length > 20) {
    const words = displayName.split(/\s+/)
    displayName = words.slice(-2).join(' ')
  }
  return displayName || name
}

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

// Availability cell component - checkmark style like RegionalAvailability
const AvailabilityCell = memo(function AvailabilityCell({ 
  model,
  byType, 
  regionCode, 
  regionLabel, 
  isLight, 
  activeView,
  selectedCrisScopes
}) {
  // Get lifecycle status for this region
  const lifecycleStatus = getRegionLifecycleStatus(model, regionCode)
  
  // Check availability flags
  const hideInRegion = model.availability?.hide_in_region ?? false
  const onDemand = !hideInRegion && byType.in_region?.includes(regionCode)
  const cris = byType.cris?.includes(regionCode)
  const mantle = byType.mantle?.includes(regionCode)
  const available = onDemand || cris || mantle
  
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
  
  // In specific views, only show relevant availability
  let isAvailable
  if (activeView === 'in_region') {
    isAvailable = onDemand
  } else if (activeView === 'in_region_all') {
    // Combined view: show if either Runtime OR Mantle is available
    isAvailable = onDemand || mantle
  } else if (activeView === 'cris') {
    // Use scope-aware check when CRIS view is active
    isAvailable = hasCrisForSelectedScope(model, regionCode, selectedCrisScopes)
  } else if (activeView === 'mantle') {
    isAvailable = mantle
  } else {
    // 'all' view: show combined
    isAvailable = available
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
  
  // Render lifecycle info in tooltip
  const renderLifecycleInfo = () => {
    if (!lifecycleStatus || lifecycleStatus.status === 'ACTIVE') return null
    
    const isLegacyStatus = lifecycleStatus.status === 'LEGACY'
    const isEolStatus = lifecycleStatus.status === 'EOL'
    
    return (
      <div className={cn(
        'mt-1.5 pt-1.5 border-t',
        isLight ? 'border-stone-200' : 'border-white/[0.08]'
      )}>
        <div className="flex items-center gap-1.5">
          {isEolStatus ? (
            <AlertCircle className={cn('w-3 h-3', isLight ? 'text-red-500' : 'text-red-400')} strokeWidth={2} />
          ) : (
            <AlertTriangle className={cn('w-3 h-3', isLight ? 'text-amber-500' : 'text-amber-400')} strokeWidth={2} />
          )}
          <span className={cn(
            'font-medium',
            isEolStatus
              ? (isLight ? 'text-red-600' : 'text-red-400')
              : (isLight ? 'text-amber-600' : 'text-amber-400')
          )}>
            {isEolStatus ? 'End of Life' : 'Legacy'}
          </span>
        </div>
        {lifecycleStatus.legacyDate && isLegacyStatus && (
          <div className={cn('text-[10px] mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
            Legacy: {lifecycleStatus.legacyDate}
          </div>
        )}
        {lifecycleStatus.eolDate && (isLegacyStatus || isEolStatus) && (
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
  
  // Render tooltip content based on active view (matches RegionalAvailability format)
  const renderTooltipContent = () => {
    if (activeView === 'in_region') {
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
      return (
        <div className="flex items-center gap-1.5">
          <Cpu className={cn('w-3 h-3', isLight ? 'text-violet-500' : 'text-violet-400')} strokeWidth={2} />
          <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In-Region (Mantle API)</span>
        </div>
      )
    }
    
    // 'all' view: show all available types with icons
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
  
  // Early return for not available (no function calls needed)
  if (!isAvailable) {
    return (
      <div className="flex items-center justify-center h-6">
        <Minus className={cn('w-2.5 h-2.5', isLight ? 'text-stone-300' : 'text-white/10')} strokeWidth={2} />
      </div>
    )
  }

  // EOL status: show red X icon
  if (isEol) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center justify-center h-6 cursor-default">
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
            'rounded-lg border px-3 py-2 text-xs z-50 max-w-[220px]',
            isLight
              ? 'bg-white border-stone-200 shadow-lg'
              : 'bg-white/[0.06] backdrop-blur-xl border-white/[0.06] shadow-[0_4px_12px_rgba(0,0,0,0.3)]'
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

  // LEGACY status: show amber warning icon
  if (isLegacy) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center justify-center h-6 cursor-default">
            <div className={cn(
              'w-4 h-4 rounded-full flex items-center justify-center',
              isLight ? 'bg-amber-100' : 'bg-amber-500/20'
            )}>
              <AlertTriangle className={cn('w-2.5 h-2.5', isLight ? 'text-amber-600' : 'text-amber-400')} strokeWidth={2.5} />
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          sideOffset={6}
          className={cn(
            'rounded-lg border px-3 py-2 text-xs z-50 max-w-[220px]',
            isLight
              ? 'bg-white border-stone-200 shadow-lg'
              : 'bg-white/[0.06] backdrop-blur-xl border-white/[0.06] shadow-[0_4px_12px_rgba(0,0,0,0.3)]'
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

  // ACTIVE status: show green checkmark (default)
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex items-center justify-center h-6 cursor-default">
          <div className={cn(
            'w-4 h-4 rounded-full flex items-center justify-center',
            isLight ? 'bg-emerald-100' : 'bg-emerald-500/20'
          )}>
            <Check className={cn('w-2.5 h-2.5', isLight ? 'text-emerald-600' : 'text-emerald-400')} strokeWidth={3} />
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        sideOffset={6}
        className={cn(
          'rounded-lg border px-3 py-2 text-xs z-50 max-w-[220px]',
          isLight
            ? 'bg-white border-stone-200 shadow-lg'
            : 'bg-white/[0.06] backdrop-blur-xl border-white/[0.06] shadow-[0_4px_12px_rgba(0,0,0,0.3)]'
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

export function AvailabilityTab({ selectedModels, isLight, getPricingForModel }) {
  // Nested filter state - single-select routing with conditional sub-filters (matches RegionalAvailability)
  const [selectedRouting, setSelectedRouting] = useState(null)  // null, 'in_region', or 'cris' (single-select)
  const [selectedApis, setSelectedApis] = useState(null)        // null, 'runtime_api', or 'mantle' - only for in_region (single-select)
  const [selectedGeos, setSelectedGeos] = useState(new Set())   // NAMER, EMEA, etc.
  const [selectedEndpoints, setSelectedEndpoints] = useState(new Set())  // Global, US, EU, etc. - only for cris
  const [selectedRegions, setSelectedRegions] = useState([])    // Specific region codes selected via dropdown
  const [collapsedGeos, setCollapsedGeos] = useState(new Set())
  const [isMapFullscreen, setIsMapFullscreen] = useState(false)
  const tableContainerRef = useRef(null)

  // Compute activeView for backward compatibility with existing logic (matches RegionalAvailability)
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

  // Handle Escape key to close fullscreen map
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isMapFullscreen) {
        setIsMapFullscreen(false)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isMapFullscreen])

  // Calculate model data
  const modelData = useMemo(() => {
    return selectedModels.map(({ model }) => {
      const byType = getRegionsByType(model)
      const allRegions = getAllModelRegions(model)
      return {
        model,
        byType,
        allRegions,
        count: allRegions.length,
      }
    })
  }, [selectedModels])

  // All regions across all models (all consumption types for table)
  const allRegions = useMemo(() => {
    const regions = new Set()
    selectedModels.forEach(({ model }) => {
      getAllModelRegions(model).forEach(r => regions.add(r))
    })
    return [...regions].sort()
  }, [selectedModels])

  // Common regions (available in ALL models - on_demand only)
  const commonRegions = useMemo(() => {
    const onDemandRegions = new Set()
    selectedModels.forEach(({ model }) => {
      getAllModelRegions(model).forEach(r => onDemandRegions.add(r))
    })
    return [...onDemandRegions].filter(region =>
      selectedModels.every(({ model }) => getAllModelRegions(model).includes(region))
    )
  }, [selectedModels])

  // Available CRIS prefixes (computed dynamically from data)
  const availableCrisPrefixes = useMemo(() => {
    const prefixes = new Set()
    let hasGovCloudCris = false
    
    modelData.forEach(({ model }) => {
      // Check for CRIS endpoint prefixes
      const profiles = model.availability?.cross_region?.profiles ?? model.cross_region_inference?.profiles ?? []
      profiles.forEach(p => {
        const prefix = p.profile_id?.split('.')[0]?.toLowerCase() || ''
        if (prefix === 'global') prefixes.add('Global')
        else if (prefix) prefixes.add(prefix.toUpperCase())
      })
      
      // Check for GovCloud CRIS availability
      if (model.availability?.govcloud?.supported && model.availability?.govcloud?.inference_type === 'cris') {
        hasGovCloudCris = true
      }
    })
    
    const order = ['Global', 'US', 'CA', 'EU', 'APAC', 'AU', 'JP']
    const sorted = [...prefixes].sort((a, b) => {
      const ia = order.indexOf(a), ib = order.indexOf(b)
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
    })
    
    if (hasGovCloudCris) sorted.push('GOVCLOUD')
    return sorted
  }, [modelData])

  // Filter regions based on routing strategy and location (matches RegionalAvailability)
  const visibleRegions = useMemo(() => {
    // If CRIS selected, filter by CRIS source regions and selected endpoints
    if (selectedRouting === 'cris') {
      const crisSourceRegions = new Set()
      
      modelData.forEach(({ model }) => {
        const crisSupported = model.availability?.cross_region?.supported ?? model.cross_region_inference?.supported
        if (crisSupported) {
          const profiles = model.availability?.cross_region?.profiles ?? model.cross_region_inference?.profiles ?? []
          profiles.forEach(p => {
            const prefix = p.profile_id?.split('.')[0]?.toLowerCase() || ''
            const scope = prefix === 'global' ? 'Global' : prefix.toUpperCase()
            if (selectedEndpoints.size === 0 || selectedEndpoints.has(scope)) {
              if (p.source_region) crisSourceRegions.add(p.source_region)
            }
          })
        }
        
        // Add GovCloud if selected
        if (selectedEndpoints.size === 0 || selectedEndpoints.has('GOVCLOUD')) {
          const govcloud = model.availability?.govcloud
          if (govcloud?.supported && govcloud?.inference_type === 'cris') {
            (govcloud.regions || []).forEach(r => crisSourceRegions.add(r))
          }
        }
      })
      
      return REGION_ROWS.filter(r => crisSourceRegions.has(r.code))
    }

    // For In-Region: filter by API selection AND geo/regions
    if (selectedRouting === 'in_region') {
      let apiFilteredRegions = REGION_ROWS.filter(r => 
        modelData.some(({ allRegions }) => allRegions.includes(r.code))
      )
      
      // If specific API is selected, filter regions accordingly (single-select)
      if (selectedApis) {
        const relevantRegions = new Set()
        
        modelData.forEach(({ model }) => {
          const hideInRegion = model.availability?.hide_in_region ?? false
          
          if (selectedApis === 'runtime_api') {
            if (!hideInRegion) {
              const runtimeRegions = model.availability?.on_demand?.regions ?? model.in_region ?? []
              runtimeRegions.forEach(r => relevantRegions.add(r))
            }
          }
          if (selectedApis === 'mantle') {
            const mantleRegions = model.availability?.mantle?.regions ?? []
            mantleRegions.forEach(r => relevantRegions.add(r))
          }
        })
        
        apiFilteredRegions = REGION_ROWS.filter(r => relevantRegions.has(r.code))
      }
      
      // Filter by specific regions if any are selected
      if (selectedRegions.length > 0) {
        return apiFilteredRegions.filter(r => selectedRegions.includes(r.code))
      }
      
      if (selectedGeos.size === 0) return apiFilteredRegions
      return apiFilteredRegions.filter(r => selectedGeos.has(r.geo))
    }

    // For All: filter by geo/regions
    const allRegionsFiltered = REGION_ROWS.filter(r => 
      modelData.some(({ allRegions }) => allRegions.includes(r.code))
    )
    
    // Filter by specific regions if any are selected
    if (selectedRegions.length > 0) {
      return allRegionsFiltered.filter(r => selectedRegions.includes(r.code))
    }
    
    if (selectedGeos.size === 0) return allRegionsFiltered
    return allRegionsFiltered.filter(r => selectedGeos.has(r.geo))
  }, [modelData, selectedRouting, selectedGeos, selectedEndpoints, selectedApis, selectedRegions])

  // Group visible regions by geo
  const regionsByGeo = useMemo(() => {
    const grouped = {}
    GEO_GROUPS.forEach(geo => {
      const geoRegions = visibleRegions.filter(r => r.geo === geo.id)
      if (geoRegions.length > 0) {
        grouped[geo.id] = {
          ...geo,
          regions: geoRegions,
        }
      }
    })
    return grouped
  }, [visibleRegions])

  // Total regions in table (only counts regions that are in REGION_ROWS)
  const totalRegionsInTable = useMemo(() => {
    return REGION_ROWS.filter(r => 
      modelData.some(({ allRegions }) => allRegions.includes(r.code))
    ).length
  }, [modelData])

  // Check if any selected models are missing pricing data
  // Uses the same logic as ModelCard.jsx - checks actual pricing data from getPricingForModel
  const hasModelsWithoutPricing = useMemo(() => {
    return modelData.some(({ model }) => !modelHasPricing(model, getPricingForModel))
  }, [modelData, getPricingForModel])

  // Toggle geo collapse (for table rows)
  const toggleGeoCollapse = (geoId) => {
    setCollapsedGeos(prev => {
      const next = new Set(prev)
      if (next.has(geoId)) {
        next.delete(geoId)
      } else {
        next.add(geoId)
      }
      return next
    })
  }

  // Select routing (single-select, matches RegionalAvailability)
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
  }

  // Select API (single-select, only for in_region)
  const selectApi = (api) => {
    // If already selected, deselect (go back to All)
    if (selectedApis === api) {
      setSelectedApis(null)
    } else {
      setSelectedApis(api)
    }
  }

  // Toggle geo filter (multi-select)
  const toggleGeo = (geo) => {
    setSelectedGeos(prev => {
      const next = new Set(prev)
      if (next.has(geo)) next.delete(geo)
      else next.add(geo)
      return next
    })
  }

  // Toggle endpoint filter (multi-select, only for cris)
  const toggleEndpoint = (endpoint) => {
    setSelectedEndpoints(prev => {
      const next = new Set(prev)
      if (next.has(endpoint)) next.delete(endpoint)
      else next.add(endpoint)
      return next
    })
  }

  // Clear all filters
  const clearAllFilters = () => {
    setSelectedRouting(null)
    setSelectedApis(null)
    setSelectedGeos(new Set())
    setSelectedEndpoints(new Set())
    setSelectedRegions([])
  }

  // Collapse/Expand all geos
  const allGeos = Object.keys(regionsByGeo)
  const allCollapsed = allGeos.length > 0 && allGeos.every(g => collapsedGeos.has(g))

  const toggleAllGeos = () => {
    if (allCollapsed) {
      setCollapsedGeos(new Set())
    } else {
      setCollapsedGeos(new Set(allGeos))
    }
  }

  return (
    <div className="mt-4 space-y-4">
      {/* Fullscreen Map */}
      {isMapFullscreen && (
        <div className="fixed inset-0 z-50 flex flex-col" style={{ backgroundColor: isLight ? '#fafaf9' : '#0d1117' }}>
          <div className={cn(
            'px-4 py-2.5 border-b flex items-center justify-between flex-shrink-0',
            isLight ? 'bg-white/90 border-stone-200' : 'bg-slate-950/90 border-white/[0.06] backdrop-blur-xl'
          )}>
            <div className="flex items-center gap-2">
              <Map className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <h3 className={cn('font-semibold text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                Global Availability
              </h3>
              <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                {selectedModels.length} models
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-7 gap-1.5"
              onClick={() => setIsMapFullscreen(false)}
            >
              <Minimize2 className="h-3.5 w-3.5" />
              Close
            </Button>
          </div>
          <div className="flex-1 overflow-hidden">
            <RegionMap selectedModels={selectedModels} isLight={isLight} height="calc(100vh - 45px)" />
          </div>
        </div>
      )}

      {/* Map + Highlights Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Map - takes 2 columns */}
        <div className="lg:col-span-2 relative">
          <button
            onClick={() => setIsMapFullscreen(true)}
            className={cn(
              'absolute top-2 right-2 z-10 p-1.5 rounded-md transition-colors',
              isLight 
                ? 'bg-white/80 hover:bg-white text-stone-600 hover:text-stone-900 shadow-sm'
                : 'bg-black/40 hover:bg-black/60 text-white/70 hover:text-white'
            )}
            title="View fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
          <RegionMap selectedModels={selectedModels} isLight={isLight} height="320px" />
        </div>

        {/* Highlights sidebar - takes 1 column */}
        <div className="space-y-3">
          {/* Total regions covered */}
          <div className={cn(
            'p-3 rounded-lg border',
            isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
          )}>
            <div className="flex items-center gap-2 mb-2">
              <MapPin className={cn('w-4 h-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <span className={cn('text-xs font-medium uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                Total Coverage
              </span>
            </div>
            <div className={cn('text-2xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>
              {totalRegionsInTable} <span className={cn('text-sm font-normal', isLight ? 'text-stone-500' : 'text-slate-400')}>regions</span>
            </div>
          </div>

          {/* Common regions */}
          <div className={cn(
            'p-3 rounded-lg border',
            isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
          )}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Users className={cn('w-4 h-4', isLight ? 'text-emerald-600' : 'text-emerald-400')} />
                <span className={cn('text-xs font-medium uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                  Common Regions
                </span>
              </div>
              <span className={cn('text-lg font-bold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                {commonRegions.length}
              </span>
            </div>
            {commonRegions.length > 0 ? (
              <TooltipProvider delayDuration={150}>
                <div className="max-h-24 overflow-y-auto">
                  <div className="flex flex-wrap gap-1">
                    {commonRegions.sort().map(region => (
                      <Tooltip key={region}>
                        <TooltipTrigger asChild>
                          <span className={cn(
                            'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono cursor-default',
                            isLight 
                              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          )}>
                            {region}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent side="top" className="text-xs">
                          {getRegionName(region)}
                        </TooltipContent>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              </TooltipProvider>
            ) : (
              <p className={cn('text-xs', isLight ? 'text-stone-400' : 'text-slate-500')}>
                No regions available in all selected models
              </p>
            )}
          </div>

          {/* Per-model region counts */}
          <div className={cn(
            'p-3 rounded-lg border',
            isLight ? 'bg-white/70 border-stone-200/60' : 'bg-white/[0.03] border-white/[0.06]'
          )}>
            <div className={cn('text-xs font-medium uppercase tracking-wider mb-2', isLight ? 'text-stone-500' : 'text-slate-400')}>
              Regions per Model
            </div>
            <div className="space-y-1.5">
              {modelData.map(({ model, count }) => (
                <div key={model.model_id} className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <Badge className={cn(
                      'text-[8px] px-1 py-0 flex-shrink-0',
                      'text-white',
                      providerColors[model.model_provider] || providerColors.default
                    )}>
                      {model.model_provider?.slice(0, 3).toUpperCase()}
                    </Badge>
                    <span className={cn('text-xs truncate', isLight ? 'text-stone-700' : 'text-slate-300')}>
                      {getShortModelName(model)}
                    </span>
                  </div>
                  <span className={cn('text-xs font-medium tabular-nums', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    {count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Filter Bar - matches Regional Availability style */}
      <div className="flex flex-col gap-2">
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
              {/* Divider */}
              <div className={cn('w-px h-5 mx-1', isLight ? 'bg-stone-200' : 'bg-white/[0.08]')} />
              
              {/* GEO label */}
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
                    // Select whole geo, clear specific regions for this geo
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
              
              {availableCrisPrefixes.map(prefix => {
                const isSelected = selectedEndpoints.has(prefix)
                return (
                  <button
                    key={prefix}
                    onClick={() => toggleEndpoint(prefix)}
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
                    {CRIS_SCOPE_LABELS[prefix] || prefix}
                  </button>
                )
              })}
            </>
          )}
        </div>
      </div>

      {/* Region-Model Matrix Table - Regions as ROWS, Models as COLUMNS */}
      <div className={cn(
        'rounded-xl border overflow-hidden',
        isLight
          ? 'bg-white/70 border-stone-200/60 shadow-sm'
          : 'bg-white/[0.03] border-white/[0.06]'
      )}>
        <div className={cn(
          'px-4 py-2.5 border-b flex items-center justify-between',
          isLight ? 'bg-stone-50/60 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
        )}>
          <div>
            <h4 className={cn('text-sm font-semibold', isLight ? 'text-stone-900' : 'text-white')}>
              Region Availability Matrix
            </h4>
            <p className={cn('text-xs mt-0.5', isLight ? 'text-stone-500' : 'text-slate-500')}>
              {visibleRegions.length} regions • {selectedModels.length} models
            </p>
          </div>
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

        <TooltipProvider delayDuration={150}>
          <div
            ref={tableContainerRef}
            className="overflow-auto max-h-[500px]"
          >
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  {/* Region column header - sticky left and top */}
                  <th
                    className={cn(
                      'sticky left-0 top-0 z-30 text-left text-[11px] font-semibold uppercase tracking-wider px-3 py-2',
                      'w-[220px] min-w-[220px]',
                      isLight
                        ? 'bg-stone-50/95 backdrop-blur-sm text-stone-500 border-b border-r border-stone-200'
                        : 'bg-[#141517]/95 backdrop-blur-xl text-slate-400 border-b border-r border-white/[0.06]'
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span>Region</span>
                      <button
                        onClick={toggleAllGeos}
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
                        title={allCollapsed ? 'Expand all regions' : 'Collapse all regions'}
                      >
                        <ChevronsUpDown className="w-2.5 h-2.5" />
                        {allCollapsed ? 'Expand' : 'Collapse'}
                      </button>
                    </div>
                  </th>
                  {/* Model column headers - sticky top */}
                  {modelData.map(({ model }) => (
                    <th
                      key={model.model_id}
                      className={cn(
                        'sticky top-0 z-20 text-center px-2 py-2 min-w-[100px]',
                        isLight
                          ? 'bg-stone-50/95 backdrop-blur-sm border-b border-stone-200'
                          : 'bg-[#141517]/95 backdrop-blur-xl border-b border-white/[0.06]',
                        // Add amber background for models without pricing
                        !modelHasPricing(model, getPricingForModel) && (isLight 
                          ? 'bg-amber-50/95' 
                          : 'bg-amber-900/20'
                        )
                      )}
                    >
                      <div className="flex flex-col items-center gap-1">
                        <Badge className={cn(
                          'text-[8px] px-1.5 py-0',
                          'text-white',
                          providerColors[model.model_provider] || providerColors.default
                        )}>
                          {model.model_provider?.slice(0, 3).toUpperCase()}
                        </Badge>
                        <div className="flex flex-col items-center gap-0.5">
                          <span className={cn(
                            'text-[10px] font-medium leading-tight max-w-[90px]',
                            isLight ? 'text-stone-700' : 'text-slate-200'
                          )}>
                            {getShortModelName(model)}
                          </span>
                          {!modelHasPricing(model, getPricingForModel) && (
                                <Tooltip delayDuration={100}>
                                              <TooltipTrigger asChild>
                                                <div className={cn(
                                                  'flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium',
                                                  isLight 
                                                    ? 'bg-amber-100 text-amber-700 border border-amber-200' 
                                                    : 'bg-amber-900/30 text-amber-400 border border-amber-700/30'
                                                )}>
                                  <AlertCircle className="w-3 h-3" />
                                  <span>Verify</span>
                                </div>
                              </TooltipTrigger>
                              <TooltipContent 
                                side="bottom" 
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
                      </div>
                    </th>
                  ))}
                  {/* Spacer column */}
                  <th
                    className={cn(
                      'sticky top-0 z-20',
                      isLight
                        ? 'bg-stone-50/95 backdrop-blur-sm border-b border-stone-200'
                        : 'bg-[#141517]/95 backdrop-blur-xl border-b border-white/[0.06]'
                    )}
                  />
                </tr>
              </thead>

              <tbody>
                {Object.values(regionsByGeo).map(geoGroup => {
                  const isCollapsed = collapsedGeos.has(geoGroup.id)
                  
                  return (
                    <Fragment key={geoGroup.id}>
                      {/* Geo header row - clickable to collapse */}
                      <tr
                        onClick={() => toggleGeoCollapse(geoGroup.id)}
                        className={cn(
                          'cursor-pointer select-none',
                          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.02]'
                        )}
                      >
                        <td
                          colSpan={modelData.length + 2}
                          className={cn(
                            'px-3 py-2 font-semibold text-xs',
                            isLight
                              ? 'bg-stone-100/90 text-stone-700 border-b border-stone-200'
                              : 'bg-white/[0.04] text-slate-200 border-b border-white/[0.06]'
                          )}
                        >
                          <div className="flex items-center gap-2">
                            {isCollapsed
                              ? <ChevronRight className="w-3.5 h-3.5 flex-shrink-0" />
                              : <ChevronDown className="w-3.5 h-3.5 flex-shrink-0" />
                            }
                            <span className={cn('font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
                              {geoGroup.id}
                            </span>
                            <span className={cn('font-normal', isLight ? 'text-stone-500' : 'text-slate-400')}>
                              {geoGroup.label}
                            </span>
                            <Badge className={cn(
                              'ml-1 text-[10px] px-1.5 py-0 border-0 font-normal',
                              isLight ? 'bg-stone-200 text-stone-600' : 'bg-white/[0.06] text-slate-400'
                            )}>
                              {geoGroup.regions.length}
                            </Badge>
                          </div>
                        </td>
                      </tr>

                      {/* Region rows - only show if not collapsed */}
                      {!isCollapsed && geoGroup.regions.map((region) => (
                        <tr
                          key={region.code}
                          className={cn(
                            'transition-colors',
                            isLight ? 'hover:bg-amber-50/30' : 'hover:bg-white/[0.02]'
                          )}
                        >
                          {/* Region name - sticky first column */}
                          <td className={cn(
                            'sticky left-0 z-10 px-3 py-1.5',
                            isLight
                              ? 'bg-white/95 backdrop-blur-sm border-b border-r border-stone-100'
                              : 'bg-[#141517]/95 backdrop-blur-xl border-b border-r border-white/[0.04]'
                          )}>
                            <div className="flex items-center">
                              <span className={cn('text-xs truncate', isLight ? 'text-stone-700' : 'text-slate-200')}>
                                {getRegionName(region.code)} <span className={cn('font-mono text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>({region.code})</span>
                              </span>
                            </div>
                          </td>

                          {/* Model availability cells */}
                          {modelData.map(({ model, byType }) => (
                            <td
                              key={model.model_id}
                              className={cn(
                                'text-center px-1 py-1',
                                isLight
                                  ? 'border-b border-stone-100'
                                  : 'border-b border-white/[0.04]',
                                // Add amber background for models without pricing
                                !modelHasPricing(model, getPricingForModel) && (isLight 
                                  ? 'bg-amber-50/50' 
                                  : 'bg-amber-900/10'
                                )
                              )}
                            >
                              <AvailabilityCell
                                model={model}
                                byType={byType}
                                regionCode={region.code}
                                regionLabel={getRegionName(region.code)}
                                isLight={isLight}
                                activeView={activeView}
                                selectedCrisScopes={selectedRouting === 'cris' ? selectedEndpoints : null}
                              />
                            </td>
                          ))}

                          {/* Spacer */}
                          <td className={cn(
                            isLight ? 'border-b border-stone-100' : 'border-b border-white/[0.04]'
                          )} />
                        </tr>
                      ))}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        </TooltipProvider>
      </div>
    </div>
  )
}
