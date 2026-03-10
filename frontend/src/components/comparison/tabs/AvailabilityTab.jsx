import { useState, useMemo, useRef, memo, Fragment, useEffect } from 'react'
import { Minus, Zap, Globe2, Cpu, Check, ChevronDown, ChevronRight, MapPin, Users, Maximize2, Minimize2, ChevronsUpDown, Map, AlertTriangle, AlertCircle, X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { RegionMap } from '../RegionMap'
import { providerColorClasses } from '@/config/constants'
import { getRegionName } from '@/utils/regionUtils'

const providerColors = providerColorClasses

// Consumption type definitions (matches Regional Availability)
const consumptionTypes = [
  { key: 'in_region', label: 'In Region', icon: Zap },
  { key: 'cris', label: 'CRIS', icon: Globe2 },
  { key: 'mantle', label: 'Mantle', icon: Cpu },
]

// CRIS scope prefixes for filtering
const CRIS_SCOPES = ['Global', 'US', 'CA', 'EU', 'APAC', 'AU', 'JP', 'GOVCLOUD']

// Geo groups for filtering and collapsing
const GEO_GROUPS = [
  { id: 'NAMER', label: 'North America', geos: ['US', 'CA'] },
  { id: 'EMEA', label: 'Europe, Middle East & Africa', geos: ['EU', 'ME', 'AF'] },
  { id: 'APAC', label: 'Asia Pacific', geos: ['AP'] },
  { id: 'LATAM', label: 'Latin America', geos: ['SA'] },
  { id: 'GOVCLOUD', label: 'GovCloud (US)', geos: ['GOV'] },
]

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

/**
 * Check if model has CRIS for this region with the selected scope(s).
 * Used for filtering regions when CRIS scope filters are active.
 */
function hasCrisForSelectedScope(model, regionCode, selectedScopes) {
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
    return false
  }
  
  // Regular CRIS check for non-GovCloud regions
  if (!selectedScopes || selectedScopes.size === 0) {
    // No filter - check if any CRIS profile exists for this region
    const sourceRegions = model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions ?? []
    return sourceRegions.includes(regionCode)
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

/**
 * Get CRIS scopes for a model in a specific region.
 * Returns array of scope names like ['Global', 'US', 'EU'].
 */
function getCrisScopesForRegion(model, regionCode) {
  const isGovcloudRegion = regionCode.startsWith('us-gov-')
  
  // For GovCloud regions, check govcloud data
  if (isGovcloudRegion) {
    const govcloud = model.availability?.govcloud
    if (govcloud?.supported && govcloud?.inference_type === 'cris') {
      const govcloudRegions = govcloud.regions || []
      if (govcloudRegions.includes(regionCode)) {
        return ['GOVCLOUD']
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
      
      if (scopeName) {
        scopes.add(scopeName)
      }
    }
  })
  
  return [...scopes].sort((a, b) => {
    const order = ['Global', 'US', 'CA', 'EU', 'APAC', 'AU', 'JP']
    return (order.indexOf(a) === -1 ? 99 : order.indexOf(a)) - (order.indexOf(b) === -1 ? 99 : order.indexOf(b))
  })
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

// Availability cell component - checkmark style like RegionalAvailability
const AvailabilityCell = memo(function AvailabilityCell({ 
  model,
  byType, 
  regionCode, 
  regionLabel, 
  isLight, 
  activeFilter,
  selectedScopes
}) {
  // Get available consumption types for this region
  const availableTypes = consumptionTypes.filter(ct => byType[ct.key]?.includes(regionCode))
  
  // Filter based on active filter
  const displayTypes = activeFilter === 'all' 
    ? availableTypes 
    : availableTypes.filter(ct => ct.key === activeFilter)
  
  // Get lifecycle status for this region
  const lifecycleStatus = getRegionLifecycleStatus(model, regionCode)
  const isEol = lifecycleStatus?.status === 'EOL'
  const isLegacy = lifecycleStatus?.status === 'LEGACY'
  
  // Get CRIS scopes for tooltip, filtered by selected scopes if any
  const allCrisScopes = getCrisScopesForRegion(model, regionCode)
  const crisScopes = (activeFilter === 'cris' && selectedScopes && selectedScopes.size > 0)
    ? allCrisScopes.filter(scope => selectedScopes.has(scope))
    : allCrisScopes
  
  // Check availability flags
  const hideInRegion = model.availability?.hide_in_region ?? false
  const hasInRegion = !hideInRegion && byType.in_region?.includes(regionCode)
  const hasCris = byType.cris?.includes(regionCode)
  const hasMantle = byType.mantle?.includes(regionCode)
  
  // Render lifecycle info in tooltip
  const renderLifecycleInfo = () => {
    if (!lifecycleStatus || lifecycleStatus.status === 'ACTIVE') return null
    
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
  
  // Render tooltip content based on active filter (matches RegionalAvailability format)
  const renderTooltipContent = () => {
    if (activeFilter === 'in_region') {
      return (
        <div className="flex items-center gap-1.5">
          <Zap className={cn('w-3 h-3', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')} strokeWidth={2} />
          <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In Region</span>
        </div>
      )
    }
    
    if (activeFilter === 'cris') {
      if (crisScopes.length === 0) {
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
            CRIS ({crisScopes.join(', ')})
          </span>
        </div>
      )
    }
    
    if (activeFilter === 'mantle') {
      return (
        <div className="flex items-center gap-1.5">
          <Cpu className={cn('w-3 h-3', isLight ? 'text-violet-500' : 'text-violet-400')} strokeWidth={2} />
          <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Mantle</span>
        </div>
      )
    }
    
    // 'all' view: show all available types with icons
    return (
      <>
        {hasInRegion && !hideInRegion && (
          <div className="flex items-center gap-1.5">
            <Zap className={cn('w-3 h-3', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')} strokeWidth={2} />
            <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>In Region</span>
          </div>
        )}
        {hasCris && (() => {
          if (crisScopes.length === 0) {
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
                CRIS ({crisScopes.join(', ')})
              </span>
            </div>
          )
        })()}
        {hasMantle && (
          <div className="flex items-center gap-1.5">
            <Cpu className={cn('w-3 h-3', isLight ? 'text-violet-500' : 'text-violet-400')} strokeWidth={2} />
            <span className={cn(isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Mantle</span>
          </div>
        )}
      </>
    )
  }
  
  if (displayTypes.length === 0) {
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

export function AvailabilityTab({ selectedModels, isLight }) {
  const [activeFilter, setActiveFilter] = useState('all')
  const [selectedGeos, setSelectedGeos] = useState(new Set())  // Multi-select for geo/scope
  const [collapsedGeos, setCollapsedGeos] = useState(new Set())
  const [isMapFullscreen, setIsMapFullscreen] = useState(false)
  const tableContainerRef = useRef(null)

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

  // Filter regions based on consumption type and scope/geo (multi-select)
  const visibleRegions = useMemo(() => {
    // Start with all known regions that have data
    let regions = REGION_ROWS.filter(r => {
      // Check if any model has this region for the active filter
      if (activeFilter === 'all') {
        return modelData.some(({ allRegions }) => allRegions.includes(r.code))
      }
      if (activeFilter === 'cris') {
        // For CRIS, check if any model has CRIS for this region with selected scopes
        return modelData.some(({ model }) => hasCrisForSelectedScope(model, r.code, selectedGeos))
      }
      return modelData.some(({ byType }) => byType[activeFilter]?.includes(r.code))
    })

    // For non-CRIS views, also apply geo filtering if selected
    if (activeFilter !== 'cris' && selectedGeos.size > 0) {
      regions = regions.filter(r => selectedGeos.has(r.geo))
    }

    return regions
  }, [modelData, activeFilter, selectedGeos])

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

  // Toggle geo collapse
  const toggleGeo = (geoId) => {
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
          {/* Tier 1: Primary filter pills */}
          {[
            { id: 'all', label: 'All', icon: null },
            { id: 'cris', label: 'CRIS', icon: Globe2 },
            { id: 'mantle', label: 'Mantle', icon: Cpu },
            { id: 'in_region', label: 'In Region', icon: Zap },
          ].map(view => (
            <button
              key={view.id}
              onClick={() => {
                setActiveFilter(view.id)
                setSelectedGeos(new Set())
              }}
              className={cn(
                'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border flex items-center gap-1',
                activeFilter === view.id
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

          {/* Tier 2: Geo / CRIS scope pills */}
          <span className={cn('text-[10px] uppercase tracking-wider font-medium mr-1', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
            {activeFilter === 'cris' ? 'Scope' : 'Geo'}
          </span>

          {/* "All" pill */}
          <button
            onClick={() => setSelectedGeos(new Set())}
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
          {(activeFilter === 'cris' ? CRIS_SCOPES : GEO_GROUPS.map(g => g.id)).map(id => {
            const isSelected = selectedGeos.has(id)
            return (
              <button
                key={id}
                onClick={() => {
                  setSelectedGeos(prev => {
                    const next = new Set(prev)
                    if (next.has(id)) next.delete(id)
                    else next.add(id)
                    return next
                  })
                }}
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
                {id}
              </button>
            )
          })}
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
                          : 'bg-[#141517]/95 backdrop-blur-xl border-b border-white/[0.06]'
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
                        <span className={cn(
                          'text-[10px] font-medium leading-tight max-w-[90px]',
                          isLight ? 'text-stone-700' : 'text-slate-200'
                        )}>
                          {getShortModelName(model)}
                        </span>
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
                        onClick={() => toggleGeo(geoGroup.id)}
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
                                  : 'border-b border-white/[0.04]'
                              )}
                            >
                              <AvailabilityCell
                                model={model}
                                byType={byType}
                                regionCode={region.code}
                                regionLabel={getRegionName(region.code)}
                                isLight={isLight}
                                activeFilter={activeFilter}
                                selectedScopes={selectedGeos}
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
