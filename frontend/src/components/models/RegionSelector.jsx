import * as React from 'react'
import { useState, useMemo, useRef, useEffect } from 'react'
import { Globe, MapPin, Search, Layers } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { awsRegions, isGeoSelection } from '@/utils/filters'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/layout/ThemeProvider'

// Group regions by geography
const regionsByGeo = {
  US: awsRegions.filter(r => r.geo === 'US'),
  EU: awsRegions.filter(r => r.geo === 'EU'),
  AP: awsRegions.filter(r => r.geo === 'AP'),
  CA: awsRegions.filter(r => r.geo === 'CA'),
  SA: awsRegions.filter(r => r.geo === 'SA'),
}

const geoLabels = {
  US: 'United States',
  EU: 'Europe',
  AP: 'Asia Pacific',
  CA: 'Canada',
  SA: 'South America',
}

// GEO-level options for filtering by entire geographic areas
const geoOptions = [
  { value: 'geo:US', label: 'All US Regions', geo: 'US' },
  { value: 'geo:EU', label: 'All Europe Regions', geo: 'EU' },
  { value: 'geo:AP', label: 'All Asia Pacific', geo: 'AP' },
  { value: 'geo:CA', label: 'All Canada Regions', geo: 'CA' },
  { value: 'geo:SA', label: 'All South America', geo: 'SA' },
]

// Get display label for a value
function getDisplayLabel(value) {
  if (!value || value === 'all') return 'All Regions'
  if (isGeoSelection(value)) {
    const geoOption = geoOptions.find(g => g.value === value)
    return geoOption?.label || value
  }
  const region = awsRegions.find(r => r.value === value)
  return region?.label || value
}

export function RegionSelector({ value, onChange, className }) {
  const [searchQuery, setSearchQuery] = useState('')
  const searchInputRef = useRef(null)
  const { theme } = useTheme()
  const isLight = theme === 'light'

  // Radix Select steals focus when items re-render; force it back to the search input
  useEffect(() => {
    if (searchQuery && searchInputRef.current) {
      requestAnimationFrame(() => {
        searchInputRef.current?.focus()
      })
    }
  }, [searchQuery])

  // Filter regions based on search
  const filteredRegionsByGeo = useMemo(() => {
    if (!searchQuery) return regionsByGeo
    const query = searchQuery.toLowerCase()
    const filtered = {}
    for (const [geo, regions] of Object.entries(regionsByGeo)) {
      const matchingRegions = regions.filter(r =>
        r.label.toLowerCase().includes(query) ||
        r.value.toLowerCase().includes(query) ||
        geoLabels[geo].toLowerCase().includes(query)
      )
      if (matchingRegions.length > 0) {
        filtered[geo] = matchingRegions
      }
    }
    return filtered
  }, [searchQuery])

  // Filter geo options based on search
  const filteredGeoOptions = useMemo(() => {
    if (!searchQuery) return geoOptions
    const query = searchQuery.toLowerCase()
    return geoOptions.filter(opt =>
      opt.label.toLowerCase().includes(query) ||
      geoLabels[opt.geo]?.toLowerCase().includes(query)
    )
  }, [searchQuery])

  const showAllRegions = !searchQuery || 'all regions'.includes(searchQuery.toLowerCase())

  return (
    <Select value={value || 'all'} onValueChange={onChange}>
      <SelectTrigger className={className}>
        <div className="flex items-center gap-2">
          {!value || value === 'all' ? (
            <Layers className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          ) : isGeoSelection(value) ? (
            <MapPin className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          ) : (
            <Globe className="h-4 w-4 text-blue-500" />
          )}
          <SelectValue placeholder="Select region">
            {getDisplayLabel(value)}
          </SelectValue>
        </div>
      </SelectTrigger>
      <SelectContent className="max-h-[400px]">
        {/* Search input */}
        <div className="px-2 pb-2">
          <div className="relative">
            <Search className={cn('absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-[#6d6e72]')} />
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search regions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={cn(
                'flex w-full rounded-md border px-8 h-8 text-sm ring-offset-background',
                'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]'
              )}
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => e.stopPropagation()}
              onFocus={(e) => e.stopPropagation()}
              autoComplete="off"
            />
          </div>
        </div>

        {/* All Regions option */}
        {showAllRegions && (
          <>
            <SelectItem value="all">
              <span className="flex items-center gap-2 font-medium">
                <Layers className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
                All Regions
              </span>
            </SelectItem>
            <SelectSeparator className={cn('my-2', isLight ? 'bg-stone-300' : 'bg-[#4a4d54]')} />
          </>
        )}

        {/* GEO-level options */}
        {filteredGeoOptions.length > 0 && (
          <>
            <SelectGroup>
              <SelectLabel className={cn(
                'text-[10px] uppercase tracking-wider font-bold pb-1',
                isLight ? 'text-stone-500' : 'text-[#1A9E7A]'
              )}>
                Filter by Area
              </SelectLabel>
              {filteredGeoOptions.map(option => (
                regionsByGeo[option.geo]?.length > 0 && (
                  <SelectItem key={option.value} value={option.value}>
                    <span className="flex items-center gap-2 font-medium">
                      <MapPin className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
                      {option.label}
                    </span>
                  </SelectItem>
                )
              ))}
            </SelectGroup>
            <SelectSeparator className={cn('my-2', isLight ? 'bg-stone-300' : 'bg-[#4a4d54]')} />
          </>
        )}

        {/* Individual regions by geo */}
        {Object.entries(filteredRegionsByGeo).map(([geo, regions], index) => (
          regions.length > 0 && (
            <React.Fragment key={geo}>
              {index > 0 && <SelectSeparator className={cn('my-1', isLight ? 'bg-stone-200' : 'bg-[#373a40]')} />}
              <SelectGroup>
                <SelectLabel className={cn(
                  'text-[10px] uppercase tracking-wider font-bold pb-1',
                  isLight ? 'text-stone-500' : 'text-[#9a9b9f]'
                )}>
                  {geoLabels[geo]}
                </SelectLabel>
                {regions.map(region => (
                  <SelectItem key={region.value} value={region.value} className="font-normal">
                    {region.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </React.Fragment>
          )
        ))}

        {/* No results */}
        {searchQuery && Object.keys(filteredRegionsByGeo).length === 0 && filteredGeoOptions.length === 0 && (
          <div className={cn('py-4 text-center text-sm', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>
            No regions found
          </div>
        )}
      </SelectContent>
    </Select>
  )
}
