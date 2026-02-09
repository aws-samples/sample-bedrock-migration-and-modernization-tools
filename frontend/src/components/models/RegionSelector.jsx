import * as React from 'react'
import { Globe, MapPin } from 'lucide-react'
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
  if (isGeoSelection(value)) {
    const geoOption = geoOptions.find(g => g.value === value)
    return geoOption?.label || value
  }
  const region = awsRegions.find(r => r.value === value)
  return region?.label || value
}

export function RegionSelector({ value, onChange, className }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className={className}>
        <div className="flex items-center gap-2">
          {isGeoSelection(value) ? (
            <MapPin className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          ) : (
            <Globe className="h-4 w-4 text-blue-500" />
          )}
          <SelectValue placeholder="Select region">
            {getDisplayLabel(value)}
          </SelectValue>
        </div>
      </SelectTrigger>
      <SelectContent>
        {/* GEO-level options first */}
        <SelectGroup>
          <SelectLabel className={cn(
            'text-[10px] uppercase tracking-wider font-bold pb-1',
            isLight ? 'text-stone-500' : 'text-[#1A9E7A]'
          )}>
            Filter by Area
          </SelectLabel>
          {geoOptions.map(option => (
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

        {/* Individual regions by geo */}
        {Object.entries(regionsByGeo).map(([geo, regions], index) => (
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
      </SelectContent>
    </Select>
  )
}
