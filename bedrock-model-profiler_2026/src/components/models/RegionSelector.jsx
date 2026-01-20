import { Globe } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { awsRegions } from '@/utils/filters'

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

export function RegionSelector({ value, onChange, className }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className={className}>
        <div className="flex items-center gap-2">
          <Globe className="h-4 w-4 text-blue-500" />
          <SelectValue placeholder="Select region" />
        </div>
      </SelectTrigger>
      <SelectContent>
        {Object.entries(regionsByGeo).map(([geo, regions]) => (
          regions.length > 0 && (
            <SelectGroup key={geo}>
              <SelectLabel className="text-xs uppercase tracking-wider">
                {geoLabels[geo]}
              </SelectLabel>
              {regions.map(region => (
                <SelectItem key={region.value} value={region.value}>
                  {region.label}
                </SelectItem>
              ))}
            </SelectGroup>
          )
        ))}
      </SelectContent>
    </Select>
  )
}
