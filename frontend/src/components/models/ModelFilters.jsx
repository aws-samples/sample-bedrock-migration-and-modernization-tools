import { useState, useRef, useEffect, useMemo } from 'react'
import { ChevronDown, ChevronUp, Filter, X, Search, Check, Zap, Globe } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useTheme } from '@/components/layout/ThemeProvider'
import {
  modelStatusOptions,
  contextFilterOptions,
  modalityOptions,
  initialFilterState,
  countActiveFilters,
  countModelsByRouting,
  getCrisGeoScopes,
  getDisplayLabel,
} from '@/utils/filters'
import { cn } from '@/lib/utils'

// Active filter chip component
function ActiveFilterChip({ label, onRemove, isLight }) {
  return (
    <span className={cn(
      'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium',
      isLight
        ? 'bg-amber-700 text-white'
        : 'bg-[#1A9E7A]/15 text-[#1A9E7A]'
    )}>
      {label}
      <button
        onClick={onRemove}
        className={cn(
          'ml-0.5 rounded-full p-0.5 transition-colors',
          isLight ? 'hover:bg-amber-800' : 'hover:bg-[#1A9E7A]/25'
        )}
      >
        <X className="h-3 w-3" />
      </button>
    </span>
  )
}

// Multi-select dropdown component
function MultiSelectDropdown({ label, options, selected, onChange, placeholder, isLight, formatLabel }) {
  const [isOpen, setIsOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const dropdownRef = useRef(null)

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const toggleOption = (option) => {
    if (selected.includes(option)) {
      onChange(selected.filter(o => o !== option))
    } else {
      onChange([...selected, option])
    }
  }

  // Use formatter if provided, otherwise use raw value
  const getDisplayLabel = (option) => formatLabel ? formatLabel(option) : option

  const filteredOptions = options.filter(opt =>
    getDisplayLabel(opt).toLowerCase().includes(searchQuery.toLowerCase())
  )

  const displayValue = selected.length === 0
    ? placeholder
    : selected.length === 1
      ? getDisplayLabel(selected[0])
      : `${selected.length} selected`

  return (
    <div className="relative" ref={dropdownRef}>
      <p className={cn('text-[11px] mb-1 font-medium', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>{label}</p>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex items-center justify-between rounded-md border transition-colors w-full h-9 px-2.5 py-1 text-sm',
          isLight
            ? 'bg-transparent border-stone-300 text-stone-700 hover:border-stone-400'
            : 'bg-[#1a1b1e] border-[#373a40] text-[#e4e5e7] hover:border-[#4a4d54]'
        )}
      >
        <span className={cn(
          'truncate',
          selected.length === 0 && (isLight ? 'text-stone-400' : 'text-[#6d6e72]')
        )}>
          {displayValue}
        </span>
        <ChevronDown className={cn(
          'h-3.5 w-3.5 ml-1.5 transition-transform flex-shrink-0',
          isLight ? 'text-stone-400' : 'text-[#6d6e72]',
          isOpen && 'rotate-180'
        )} />
      </button>

      {isOpen && (
        <div className={cn(
          'absolute z-50 mt-1 min-w-[220px] w-full rounded-md border shadow-lg animate-slide-down',
          isLight
            ? 'bg-white border-stone-200 shadow-stone-900/10'
            : 'bg-[#25262b] border-[#373a40] shadow-black/20'
        )}>
          {options.length > 5 && (
            <div className={cn(
              'p-2 border-b',
              isLight ? 'border-stone-200' : 'border-[#373a40]'
            )}>
              <div className="relative">
                <Search className={cn(
                  'absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5',
                  isLight ? 'text-stone-500' : 'text-[#6d6e72]'
                )} />
                <Input
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-7 pl-7 text-xs"
                />
              </div>
            </div>
          )}

          <div className="max-h-48 overflow-y-auto p-1">
            {filteredOptions.length === 0 ? (
              <p className={cn(
                'px-3 py-2 text-xs',
                isLight ? 'text-stone-600' : 'text-[#6d6e72]'
              )}>
                No options found
              </p>
            ) : (
              filteredOptions.map(option => (
                <button
                  key={option}
                  type="button"
                  onClick={() => toggleOption(option)}
                  className={cn(
                    'flex w-full items-center gap-2 rounded px-2.5 py-1.5 text-xs transition-colors',
                    isLight
                      ? 'hover:bg-stone-100 text-stone-700'
                      : 'hover:bg-[#373a40] text-[#e4e5e7]'
                  )}
                >
                  <div className={cn(
                    'flex h-3.5 w-3.5 items-center justify-center rounded border transition-colors flex-shrink-0',
                    selected.includes(option)
                      ? isLight
                        ? 'bg-amber-700 border-amber-700'
                        : 'bg-[#1A9E7A] border-[#1A9E7A]'
                      : isLight
                        ? 'border-stone-300'
                        : 'border-[#4a4d54]'
                  )}>
                    {selected.includes(option) && (
                      <Check className="h-2.5 w-2.5 text-white" />
                    )}
                  </div>
                  <span className="truncate">{getDisplayLabel(option)}</span>
                </button>
              ))
            )}
          </div>

          {selected.length > 0 && (
            <div className={cn(
              'p-1.5 border-t',
              isLight ? 'border-stone-200' : 'border-[#373a40]'
            )}>
              <button
                type="button"
                onClick={() => onChange([])}
                className={cn(
                  'w-full text-[10px] py-1 rounded transition-colors',
                  isLight
                    ? 'text-stone-600 hover:bg-stone-100'
                    : 'text-[#a0a1a5] hover:bg-[#373a40]'
                )}
              >
                Clear all ({selected.length})
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function FilterSelect({ label, value, onChange, options, isLight }) {
  const selectedOption = options.find(opt => opt.value === value)
  const displayText = selectedOption?.label || value

  return (
    <div>
      <p className={cn('text-[11px] mb-1 font-medium', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>{label}</p>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-9 text-sm">
          <SelectValue>{displayText}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {options.map(opt => (
            <SelectItem key={opt.value} value={opt.value} className="text-xs">
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

// CRIS scope options for the nested filter
const CRIS_SCOPE_OPTIONS = ['Global', 'US', 'EU', 'APAC', 'JP', 'AU', 'CA']

// Geo options for the nested filter
const GEO_OPTIONS = ['NAMER', 'EMEA', 'APAC', 'LATAM', 'GOVCLOUD']

// Region data by geo for the dropdown pills
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

// Nested routing filter component - matches RegionalAvailability structure
function NestedRoutingFilter({ 
  selectedRouting, 
  selectedApi, 
  selectedCrisScopes, 
  selectedGeos,
  selectedRegions,
  onRoutingChange,
  onApiChange,
  onCrisScopesChange,
  onGeosChange,
  onRegionsChange,
  routingCounts,
  availableCrisScopes,
  isLight 
}) {
  // Pill button styles matching RegionalAvailability
  const getSelectedStyle = () => isLight
    ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
    : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
  
  const getUnselectedStyle = () => isLight
    ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
    : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'

  const selectRouting = (routing) => {
    if (selectedRouting === routing) {
      // Deselect - go back to "All"
      onRoutingChange(null)
      onApiChange(null)
      onCrisScopesChange([])
      onGeosChange([])
      onRegionsChange([])
    } else {
      // Select new routing, clear other selections
      onRoutingChange(routing)
      onApiChange(null)
      onCrisScopesChange([])
      onGeosChange([])
      onRegionsChange([])
    }
  }

  const selectApi = (api) => {
    if (selectedApi === api) {
      onApiChange(null)
    } else {
      onApiChange(api)
    }
  }

  const toggleCrisScope = (scope) => {
    if (selectedCrisScopes.includes(scope)) {
      onCrisScopesChange(selectedCrisScopes.filter(s => s !== scope))
    } else {
      onCrisScopesChange([...selectedCrisScopes, scope])
    }
  }

  const clearAllFilters = () => {
    onRoutingChange(null)
    onApiChange(null)
    onCrisScopesChange([])
    onGeosChange([])
    onRegionsChange([])
  }

  // Handler for toggling a specific region
  const handleToggleRegion = (regionCode) => {
    if (selectedRegions.includes(regionCode)) {
      onRegionsChange(selectedRegions.filter(r => r !== regionCode))
    } else {
      onRegionsChange([...selectedRegions, regionCode])
    }
  }

  // Handler for selecting all regions in a geo (selects the geo itself)
  const handleSelectAllGeo = (geo) => {
    // Clear any specific regions from this geo and add the geo to selectedGeos
    const geoRegionCodes = (REGIONS_BY_GEO[geo] || []).map(r => r.code)
    const otherRegions = selectedRegions.filter(r => !geoRegionCodes.includes(r))
    onRegionsChange(otherRegions)
    if (!selectedGeos.includes(geo)) {
      onGeosChange([...selectedGeos, geo])
    }
  }

  // Handler for deselecting all regions in a geo
  const handleDeselectAllGeo = (geo) => {
    // Remove the geo from selectedGeos and clear any specific regions from this geo
    const geoRegionCodes = (REGIONS_BY_GEO[geo] || []).map(r => r.code)
    const otherRegions = selectedRegions.filter(r => !geoRegionCodes.includes(r))
    onRegionsChange(otherRegions)
    onGeosChange(selectedGeos.filter(g => g !== geo))
  }

  // Check if any geo or region filter is active
  const hasGeoOrRegionFilter = selectedGeos.length > 0 || selectedRegions.length > 0

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {/* Inference pills - single select */}
      <span className={cn('text-[10px] uppercase tracking-wider font-medium mr-1', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
        Inference
      </span>
      
      {/* All pill */}
      <button
        type="button"
        onClick={clearAllFilters}
        className={cn(
          'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
          !selectedRouting ? getSelectedStyle() : getUnselectedStyle()
        )}
      >
        All
      </button>
      
      {/* In-Region pill */}
      <button
        type="button"
        onClick={() => selectRouting('in_region')}
        className={cn(
          'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border flex items-center gap-1',
          selectedRouting === 'in_region' ? getSelectedStyle() : getUnselectedStyle()
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
        type="button"
        onClick={() => selectRouting('cris')}
        className={cn(
          'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border flex items-center gap-1',
          selectedRouting === 'cris' ? getSelectedStyle() : getUnselectedStyle()
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
            type="button"
            onClick={() => onApiChange(null)}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
              selectedApi === null ? getSelectedStyle() : getUnselectedStyle()
            )}
          >
            All
          </button>
          
          <button
            type="button"
            onClick={() => selectApi('runtime_api')}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
              selectedApi === 'runtime_api' ? getSelectedStyle() : getUnselectedStyle()
            )}
          >
            Runtime API
          </button>
          
          <button
            type="button"
            onClick={() => selectApi('mantle')}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
              selectedApi === 'mantle' ? getSelectedStyle() : getUnselectedStyle()
            )}
          >
            Mantle API
          </button>
        </>
      )}

      {/* Scope pills - only when CRIS selected */}
      {selectedRouting === 'cris' && (
        <>
          {/* Divider */}
          <div className={cn('w-px h-5 mx-1', isLight ? 'bg-stone-200' : 'bg-white/[0.08]')} />
          
          {/* Scope pills */}
          <span className={cn('text-[10px] uppercase tracking-wider font-medium mr-1', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
            Scope
          </span>
          
          <button
            type="button"
            onClick={() => onCrisScopesChange([])}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
              selectedCrisScopes.length === 0 ? getSelectedStyle() : getUnselectedStyle()
            )}
          >
            All
          </button>
          
          {availableCrisScopes.map(scope => (
            <button
              key={scope}
              type="button"
              onClick={() => toggleCrisScope(scope)}
              className={cn(
                'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                selectedCrisScopes.includes(scope) ? getSelectedStyle() : getUnselectedStyle()
              )}
            >
              {scope}
            </button>
          ))}
        </>
      )}

      {/* GEO pills with region dropdowns - only when not CRIS */}
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
            type="button"
            onClick={() => {
              onGeosChange([])
              onRegionsChange([])
            }}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
              !hasGeoOrRegionFilter ? getSelectedStyle() : getUnselectedStyle()
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
              isGeoSelected={selectedGeos.includes(geo)}
              selectedRegions={selectedRegions}
              onToggleRegion={handleToggleRegion}
              onSelectAllGeo={handleSelectAllGeo}
              onDeselectAllGeo={handleDeselectAllGeo}
              isLight={isLight}
            />
          ))}
        </>
      )}
    </div>
  )
}

// Section header component
function SectionHeader({ title, isLight }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className={cn('text-[10px] font-bold uppercase tracking-wider', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
        {title}
      </span>
      <div className={cn('flex-1 h-px', isLight ? 'bg-stone-200' : 'bg-white/[0.06]')} />
    </div>
  )
}

// Feature toggle pills - compact multi-select for Features section
function FeatureTogglePills({ filters, updateFilter, isLight }) {
  const features = [
    { key: 'batchSupport', label: 'Batch', activeValue: 'Batch Supported' },
    { key: 'reservedSupport', label: 'Reserved', activeValue: 'Reserved Supported' },
    { key: 'streamingSupport', label: 'Streaming', activeValue: 'Streaming Supported' },
    { key: 'flexPricing', label: 'Flex Pricing', activeValue: 'Has Flex' },
    { key: 'priorityPricing', label: 'Priority Pricing', activeValue: 'Has Priority' },
  ]

  const toggleFeature = (key, activeValue) => {
    const currentValue = filters[key]
    // Toggle: if active, set to 'All Models'; if inactive, set to activeValue
    const newValue = currentValue === activeValue ? 'All Models' : activeValue
    updateFilter(key, newValue)
  }

  const getSelectedStyle = () => isLight
    ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
    : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
  
  const getUnselectedStyle = () => isLight
    ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
    : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'

  return (
    <div className="flex flex-wrap gap-1.5">
      {features.map(({ key, label, activeValue }) => {
        const isActive = filters[key] === activeValue
        return (
          <button
            key={key}
            type="button"
            onClick={() => toggleFeature(key, activeValue)}
            className={cn(
              'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
              isActive ? getSelectedStyle() : getUnselectedStyle()
            )}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}

export function ModelFilters({
  filters,
  onFiltersChange,
  availableProviders = [],
  availableCapabilities = [],
  availableUseCases = [],
  availableCustomizations = [],
  availableLanguages = [],
  models = [],
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const activeCount = countActiveFilters(filters)

  // Calculate routing counts
  const routingCounts = useMemo(() => {
    return countModelsByRouting(models)
  }, [models])

  // Calculate available CRIS scopes from model data
  const availableCrisScopes = useMemo(() => {
    const scopes = new Set()
    models.forEach(m => {
      const modelScopes = getCrisGeoScopes(m)
      modelScopes.forEach(s => scopes.add(s))
    })
    // Return in preferred order, filtering to only those that exist
    return CRIS_SCOPE_OPTIONS.filter(s => scopes.has(s))
  }, [models])

  const updateFilter = (key, value) => {
    onFiltersChange({ ...filters, [key]: value })
  }

  const resetFilters = () => {
    onFiltersChange(initialFilterState)
  }

  // Generate active filter chips
  const getActiveFilterChips = () => {
    const chips = []

    if (filters.providers.length > 0) {
      filters.providers.forEach(p => {
        chips.push({
          key: `provider-${p}`,
          label: p,
          onRemove: () => updateFilter('providers', filters.providers.filter(x => x !== p))
        })
      })
    }

    if (filters.modality !== 'All Modalities') {
      chips.push({
        key: 'modality',
        label: filters.modality,
        onRemove: () => updateFilter('modality', 'All Modalities')
      })
    }

    if (filters.modelStatus !== 'All Status') {
      chips.push({
        key: 'status',
        label: filters.modelStatus,
        onRemove: () => updateFilter('modelStatus', 'All Status')
      })
    }

    if (filters.contextFilter !== 'All Models') {
      chips.push({
        key: 'context',
        label: filters.contextFilter,
        onRemove: () => updateFilter('contextFilter', 'All Models')
      })
    }

    if (filters.capabilities.length > 0) {
      filters.capabilities.forEach(c => {
        chips.push({
          key: `cap-${c}`,
          label: c,
          onRemove: () => updateFilter('capabilities', filters.capabilities.filter(x => x !== c))
        })
      })
    }

    if (filters.useCases.length > 0) {
      filters.useCases.forEach(uc => {
        chips.push({
          key: `uc-${uc}`,
          label: uc,
          onRemove: () => updateFilter('useCases', filters.useCases.filter(x => x !== uc))
        })
      })
    }

    if (filters.streamingSupport !== 'All Models') {
      chips.push({
        key: 'streaming',
        label: `Streaming: ${filters.streamingSupport === 'Streaming Supported' ? 'Yes' : 'No'}`,
        onRemove: () => updateFilter('streamingSupport', 'All Models')
      })
    }

    if (filters.pricingFilter !== 'All Models') {
      chips.push({
        key: 'pricingFilter',
        label: `Pricing: ${filters.pricingFilter === 'Has Pricing' ? 'Yes' : 'No'}`,
        onRemove: () => updateFilter('pricingFilter', 'All Models')
      })
    }

    if (filters.customizations.length > 0) {
      filters.customizations.forEach(cu => {
        chips.push({
          key: `cust-${cu}`,
          label: cu,
          onRemove: () => updateFilter('customizations', filters.customizations.filter(x => x !== cu))
        })
      })
    }

    if (filters.languages.length > 0) {
      chips.push({
        key: 'languages',
        label: `${filters.languages.length} language${filters.languages.length > 1 ? 's' : ''}`,
        onRemove: () => updateFilter('languages', [])
      })
    }

    // New nested routing filter chips
    if (filters.selectedRouting) {
      const routingLabel = filters.selectedRouting === 'in_region' ? 'In-Region' : 'CRIS'
      chips.push({
        key: 'routing',
        label: `Routing: ${routingLabel}`,
        onRemove: () => {
          onFiltersChange({ 
            ...filters, 
            selectedRouting: null, 
            selectedApi: null, 
            selectedCrisScopes: [], 
            selectedGeos: [],
            selectedRegions: []
          })
        }
      })
    }

    if (filters.selectedApi) {
      const apiLabel = filters.selectedApi === 'runtime_api' ? 'Runtime API' : 'Mantle API'
      chips.push({
        key: 'api',
        label: `API: ${apiLabel}`,
        onRemove: () => updateFilter('selectedApi', null)
      })
    }

    if (filters.selectedCrisScopes?.length > 0) {
      filters.selectedCrisScopes.forEach(scope => {
        chips.push({
          key: `cris-scope-${scope}`,
          label: `CRIS: ${scope}`,
          onRemove: () => updateFilter('selectedCrisScopes', filters.selectedCrisScopes.filter(x => x !== scope))
        })
      })
    }

    if (filters.selectedGeos?.length > 0) {
      filters.selectedGeos.forEach(geo => {
        chips.push({
          key: `geo-${geo}`,
          label: `Geo: ${geo}`,
          onRemove: () => updateFilter('selectedGeos', filters.selectedGeos.filter(x => x !== geo))
        })
      })
    }

    // Selected regions filter chips
    if (filters.selectedRegions?.length > 0) {
      // Find region names from REGIONS_BY_GEO
      const allRegions = Object.values(REGIONS_BY_GEO).flat()
      filters.selectedRegions.forEach(regionCode => {
        const region = allRegions.find(r => r.code === regionCode)
        const regionLabel = region ? region.name : regionCode
        chips.push({
          key: `region-${regionCode}`,
          label: `Region: ${regionLabel}`,
          onRemove: () => updateFilter('selectedRegions', filters.selectedRegions.filter(x => x !== regionCode))
        })
      })
    }

    // Features filter chips
    if (filters.batchSupport !== 'All Models') {
      chips.push({
        key: 'batch',
        label: `Batch: ${filters.batchSupport === 'Batch Supported' ? 'Yes' : 'No'}`,
        onRemove: () => updateFilter('batchSupport', 'All Models')
      })
    }

    if (filters.reservedSupport !== 'All Models') {
      chips.push({
        key: 'reserved',
        label: `Reserved: ${filters.reservedSupport === 'Reserved Supported' ? 'Yes' : 'No'}`,
        onRemove: () => updateFilter('reservedSupport', 'All Models')
      })
    }

    if (filters.flexPricing !== 'All Models') {
      chips.push({
        key: 'flex',
        label: `Flex: ${filters.flexPricing === 'Has Flex' ? 'Yes' : 'No'}`,
        onRemove: () => updateFilter('flexPricing', 'All Models')
      })
    }

    if (filters.priorityPricing !== 'All Models') {
      chips.push({
        key: 'priority',
        label: `Priority: ${filters.priorityPricing === 'Has Priority' ? 'Yes' : 'No'}`,
        onRemove: () => updateFilter('priorityPricing', 'All Models')
      })
    }

    return chips
  }

  const activeChips = getActiveFilterChips()

  return (
    <div className="space-y-2">
      {/* Main filter row: Search + Region + More */}
      <div className="flex items-center gap-2">
        {/* Search bar */}
        <div className="relative flex-1">
          <Search className={cn(
            'absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4',
            isLight ? 'text-stone-400' : 'text-[#6d6e72]'
          )} />
          <Input
            placeholder="Search models..."
            value={filters.searchQuery}
            onChange={(e) => updateFilter('searchQuery', e.target.value)}
            className={cn(
              'h-9 pl-9 pr-8 text-sm',
              isLight
                ? 'bg-white border-stone-200 focus:border-amber-500'
                : 'bg-[#25262b] border-[#373a40] focus:border-[#1A9E7A]'
            )}
          />
          {filters.searchQuery && (
            <button
              onClick={() => updateFilter('searchQuery', '')}
              className={cn(
                'absolute right-2 top-1/2 -translate-y-1/2 p-0.5 rounded-full transition-colors',
                isLight ? 'hover:bg-stone-100' : 'hover:bg-[#373a40]'
              )}
            >
              <X className={cn('h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-[#6d6e72]')} />
            </button>
          )}
        </div>

        {/* More filters toggle */}
        <Button
          variant="outline"
          size="sm"
          className={cn(
            'h-9 flex-shrink-0',
            showAdvanced && (isLight ? 'bg-stone-100' : 'bg-[#2c2d32]')
          )}
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <Filter className="h-4 w-4 mr-1.5" />
          Filters
          {activeCount > 0 && (
            <Badge className={cn(
              'ml-1.5 text-[10px] px-1.5 border-0',
              isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
            )}>
              {activeCount}
            </Badge>
          )}
          {showAdvanced ? (
            <ChevronUp className="h-4 w-4 ml-1" />
          ) : (
            <ChevronDown className="h-4 w-4 ml-1" />
          )}
        </Button>
      </div>

      {/* Active filter chips */}
      {activeChips.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className={cn('text-[11px]', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>Active:</span>
          {activeChips.map(chip => (
            <ActiveFilterChip
              key={chip.key}
              label={chip.label}
              onRemove={chip.onRemove}
              isLight={isLight}
            />
          ))}
          <button
            onClick={resetFilters}
            className={cn(
              'text-[11px] font-medium ml-1 transition-colors',
              isLight
                ? 'text-amber-700 hover:text-amber-800'
                : 'text-[#1A9E7A] hover:text-[#22b38d]'
            )}
          >
            Clear all
          </button>
        </div>
      )}

      {/* Advanced filters panel */}
      {showAdvanced && (
        <div className={cn(
          'border rounded-lg p-3 animate-slide-down',
          isLight
            ? 'bg-stone-50/50 border-stone-200'
            : 'bg-[#1a1b1e]/50 border-[#373a40]'
        )}>
          {/* Section: Availability */}
          <SectionHeader title="Availability" isLight={isLight} />
          <NestedRoutingFilter
            selectedRouting={filters.selectedRouting}
            selectedApi={filters.selectedApi}
            selectedCrisScopes={filters.selectedCrisScopes || []}
            selectedGeos={filters.selectedGeos || []}
            selectedRegions={filters.selectedRegions || []}
            onRoutingChange={(v) => updateFilter('selectedRouting', v)}
            onApiChange={(v) => updateFilter('selectedApi', v)}
            onCrisScopesChange={(v) => updateFilter('selectedCrisScopes', v)}
            onGeosChange={(v) => updateFilter('selectedGeos', v)}
            onRegionsChange={(v) => updateFilter('selectedRegions', v)}
            routingCounts={routingCounts}
            availableCrisScopes={availableCrisScopes}
            isLight={isLight}
          />

          {/* Divider */}
          <div className={cn('my-2.5 border-t', isLight ? 'border-stone-200/60' : 'border-[#2c2d32]/60')} />

          {/* Section: Pricing */}
          <SectionHeader title="Pricing" isLight={isLight} />
          <div className="flex flex-wrap gap-1.5 items-center">
            <button
              type="button"
              onClick={() => updateFilter('pricingFilter', 'All Models')}
              className={cn(
                'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                filters.pricingFilter === 'All Models'
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
              type="button"
              onClick={() => updateFilter('pricingFilter', filters.pricingFilter === 'Has Pricing' ? 'All Models' : 'Has Pricing')}
              className={cn(
                'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                filters.pricingFilter === 'Has Pricing'
                  ? isLight
                    ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                    : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                  : isLight
                    ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                    : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
              )}
            >
              Has Pricing
            </button>
            <button
              type="button"
              onClick={() => updateFilter('pricingFilter', filters.pricingFilter === 'No Pricing' ? 'All Models' : 'No Pricing')}
              className={cn(
                'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 border',
                filters.pricingFilter === 'No Pricing'
                  ? isLight
                    ? 'bg-amber-700 text-[#faf9f5] border-amber-700 shadow-sm'
                    : 'bg-[#1A9E7A] text-white border-[#1A9E7A] shadow-sm shadow-[#1A9E7A]/20'
                  : isLight
                    ? 'bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700'
                    : 'bg-white/[0.03] text-[#9a9b9f] border-white/[0.06] hover:bg-white/[0.06] hover:text-[#c0c1c5] hover:border-white/[0.12]'
              )}
            >
              No Pricing
            </button>
          </div>

          {/* Divider */}
          <div className={cn('my-2.5 border-t', isLight ? 'border-stone-200/60' : 'border-[#2c2d32]/60')} />

          {/* Section: Model */}
          <SectionHeader title="Model" isLight={isLight} />
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-3 gap-y-2.5">
            {/* Row 1: Provider, Modality, Status, Context Window */}
            {availableProviders.length > 0 && (
              <MultiSelectDropdown
                label="Provider"
                options={availableProviders}
                selected={filters.providers}
                onChange={(v) => updateFilter('providers', v)}
                placeholder="All providers"
                isLight={isLight}
              />
            )}

            <FilterSelect
              label="Modality"
              value={filters.modality}
              onChange={(v) => updateFilter('modality', v)}
              options={modalityOptions}
              isLight={isLight}
            />

            <FilterSelect
              label="Status"
              value={filters.modelStatus}
              onChange={(v) => updateFilter('modelStatus', v)}
              options={modelStatusOptions}
              isLight={isLight}
            />

            <FilterSelect
              label="Context Window"
              value={filters.contextFilter}
              onChange={(v) => updateFilter('contextFilter', v)}
              options={contextFilterOptions}
              isLight={isLight}
            />

            {/* Row 2: Use Cases, Capabilities, Languages, Customization */}
            {availableUseCases.length > 0 && (
              <MultiSelectDropdown
                label="Use Cases"
                options={availableUseCases}
                selected={filters.useCases}
                onChange={(v) => updateFilter('useCases', v)}
                placeholder="All use cases"
                isLight={isLight}
                formatLabel={(v) => getDisplayLabel(v, 'useCase')}
              />
            )}

            {availableCapabilities.length > 0 && (
              <MultiSelectDropdown
                label="Capabilities"
                options={availableCapabilities}
                selected={filters.capabilities}
                onChange={(v) => updateFilter('capabilities', v)}
                placeholder="All capabilities"
                isLight={isLight}
                formatLabel={(v) => getDisplayLabel(v, 'capability')}
              />
            )}

            {availableLanguages.length > 0 && (
              <MultiSelectDropdown
                label="Languages"
                options={availableLanguages}
                selected={filters.languages}
                onChange={(v) => updateFilter('languages', v)}
                placeholder="All languages"
                isLight={isLight}
                formatLabel={(v) => getDisplayLabel(v, 'language')}
              />
            )}

            {availableCustomizations.length > 0 && (
              <MultiSelectDropdown
                label="Customization"
                options={availableCustomizations}
                selected={filters.customizations}
                onChange={(v) => updateFilter('customizations', v)}
                placeholder="All types"
                isLight={isLight}
                formatLabel={(v) => getDisplayLabel(v, 'customization')}
              />
            )}
          </div>

          {/* Divider */}
          <div className={cn('my-2.5 border-t', isLight ? 'border-stone-200/60' : 'border-[#2c2d32]/60')} />

          {/* Section: Features */}
          <SectionHeader title="Features" isLight={isLight} />
          <FeatureTogglePills
            filters={filters}
            updateFilter={updateFilter}
            isLight={isLight}
          />
        </div>
      )}
    </div>
  )
}
