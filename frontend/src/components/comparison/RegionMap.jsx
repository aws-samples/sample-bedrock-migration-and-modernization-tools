import { useEffect, useMemo } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import MarkerClusterGroup from 'react-leaflet-cluster'
import L from 'leaflet'
import { useTheme } from '@/components/layout/ThemeProvider'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'

// Import Leaflet CSS
import 'leaflet/dist/leaflet.css'

// AWS Region coordinates
const regionCoordinates = {
  // US
  'us-east-1': { lat: 38.9519, lng: -77.4480, name: 'N. Virginia', geo: 'US' },
  'us-east-2': { lat: 39.9612, lng: -82.9988, name: 'Ohio', geo: 'US' },
  'us-west-1': { lat: 37.3541, lng: -121.9552, name: 'N. California', geo: 'US' },
  'us-west-2': { lat: 45.8696, lng: -119.6880, name: 'Oregon', geo: 'US' },
  // EU
  'eu-west-1': { lat: 53.3498, lng: -6.2603, name: 'Ireland', geo: 'EU' },
  'eu-west-2': { lat: 51.5074, lng: -0.1278, name: 'London', geo: 'EU' },
  'eu-west-3': { lat: 48.8566, lng: 2.3522, name: 'Paris', geo: 'EU' },
  'eu-central-1': { lat: 50.1109, lng: 8.6821, name: 'Frankfurt', geo: 'EU' },
  'eu-central-2': { lat: 47.3769, lng: 8.5417, name: 'Zurich', geo: 'EU' },
  'eu-north-1': { lat: 59.3293, lng: 18.0686, name: 'Stockholm', geo: 'EU' },
  'eu-south-1': { lat: 45.4642, lng: 9.1900, name: 'Milan', geo: 'EU' },
  'eu-south-2': { lat: 40.4168, lng: -3.7038, name: 'Spain', geo: 'EU' },
  // Asia Pacific
  'ap-south-1': { lat: 19.0760, lng: 72.8777, name: 'Mumbai', geo: 'AP' },
  'ap-south-2': { lat: 17.3850, lng: 78.4867, name: 'Hyderabad', geo: 'AP' },
  'ap-northeast-1': { lat: 35.6762, lng: 139.6503, name: 'Tokyo', geo: 'AP' },
  'ap-northeast-2': { lat: 37.5665, lng: 126.9780, name: 'Seoul', geo: 'AP' },
  'ap-northeast-3': { lat: 34.6937, lng: 135.5023, name: 'Osaka', geo: 'AP' },
  'ap-southeast-1': { lat: 1.3521, lng: 103.8198, name: 'Singapore', geo: 'AP' },
  'ap-southeast-2': { lat: -33.8688, lng: 151.2093, name: 'Sydney', geo: 'AP' },
  'ap-southeast-3': { lat: -6.2088, lng: 106.8456, name: 'Jakarta', geo: 'AP' },
  'ap-southeast-4': { lat: -37.8136, lng: 144.9631, name: 'Melbourne', geo: 'AP' },
  'ap-southeast-5': { lat: 3.1390, lng: 101.6869, name: 'Malaysia', geo: 'AP' },
  'ap-east-1': { lat: 22.3193, lng: 114.1694, name: 'Hong Kong', geo: 'AP' },
  // Canada
  'ca-central-1': { lat: 45.5017, lng: -73.5673, name: 'Montreal', geo: 'CA' },
  'ca-west-1': { lat: 51.0447, lng: -114.0719, name: 'Calgary', geo: 'CA' },
  // South America
  'sa-east-1': { lat: -23.5505, lng: -46.6333, name: 'São Paulo', geo: 'SA' },
  // Middle East
  'me-south-1': { lat: 26.0667, lng: 50.5577, name: 'Bahrain', geo: 'ME' },
  'me-central-1': { lat: 24.4539, lng: 54.3773, name: 'UAE', geo: 'ME' },
  'il-central-1': { lat: 32.0853, lng: 34.7818, name: 'Tel Aviv', geo: 'ME' },
  // Africa
  'af-south-1': { lat: -33.9249, lng: 18.4241, name: 'Cape Town', geo: 'AF' },
}

// Provider colors for markers
const providerColors = {
  Amazon: '#FF9900',
  Anthropic: '#D4A27F',
  Meta: '#0082FB',
  Mistral: '#F54E42',
  Cohere: '#39594D',
  'AI21 Labs': '#6C5CE7',
  AI21: '#6C5CE7',
  'Stability AI': '#7C5CFF',
  Stability: '#7C5CFF',
  Luma: '#6366F1',
  default: '#64748b',
}

// Create custom marker icon
function createMarkerIcon(color, count = null) {
  const size = count ? 36 : 28
  const html = count
    ? `<div style="
        background: ${color};
        width: ${size}px;
        height: ${size}px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      ">${count}</div>`
    : `<div style="
        background: ${color};
        width: ${size}px;
        height: ${size}px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      "></div>`

  return L.divIcon({
    html,
    className: 'custom-marker',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  })
}

// Create cluster icon
function createClusterIcon(cluster) {
  const count = cluster.getChildCount()
  const size = count > 20 ? 50 : count > 10 ? 44 : 38

  return L.divIcon({
    html: `<div style="
      background: linear-gradient(135deg, #1A9E7A 0%, #158567 100%);
      width: ${size}px;
      height: ${size}px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: bold;
      font-size: ${count > 20 ? 14 : 12}px;
      border: 3px solid white;
      box-shadow: 0 4px 12px rgba(26, 158, 122, 0.4);
    ">${count}</div>`,
    className: 'custom-cluster',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  })
}

// Map bounds updater component
function MapBoundsUpdater({ markers }) {
  const map = useMap()

  useEffect(() => {
    if (markers.length > 0) {
      const bounds = L.latLngBounds(markers.map(m => [m.lat, m.lng]))
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 5 })
    }
  }, [markers, map])

  return null
}

export function RegionMap({ selectedModels, isLight }) {
  // Build markers data: for each region, list which models are available
  const markersData = useMemo(() => {
    const regionModels = {}

    selectedModels.forEach(({ model }) => {
      const regions = model.regions_available || []
      regions.forEach(regionCode => {
        if (!regionCoordinates[regionCode]) return

        if (!regionModels[regionCode]) {
          regionModels[regionCode] = {
            ...regionCoordinates[regionCode],
            code: regionCode,
            models: [],
          }
        }
        regionModels[regionCode].models.push(model)
      })
    })

    return Object.values(regionModels)
  }, [selectedModels])

  // Get common regions (available in all models)
  const commonRegions = useMemo(() => {
    if (selectedModels.length === 0) return new Set()

    const allRegions = selectedModels[0].model.regions_available || []
    return new Set(
      allRegions.filter(region =>
        selectedModels.every(({ model }) =>
          (model.regions_available || []).includes(region)
        )
      )
    )
  }, [selectedModels])

  // Tile layer URL based on theme
  const tileUrl = isLight
    ? 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'

  if (markersData.length === 0) {
    return (
      <div className={cn(
        'h-[400px] rounded-lg border flex items-center justify-center',
        isLight
          ? 'bg-stone-100 border-stone-200 text-stone-500'
          : 'bg-slate-800 border-slate-700 text-slate-400'
      )}>
        No region data available
      </div>
    )
  }

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight ? 'border-stone-200' : 'border-slate-700'
    )}>
      <MapContainer
        center={[20, 0]}
        zoom={2}
        style={{ height: '400px', width: '100%' }}
        className="z-0"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url={tileUrl}
        />

        <MapBoundsUpdater markers={markersData} />

        <MarkerClusterGroup
          chunkedLoading
          iconCreateFunction={createClusterIcon}
          maxClusterRadius={60}
          spiderfyOnMaxZoom={true}
          showCoverageOnHover={false}
          zoomToBoundsOnClick={true}
          disableClusteringAtZoom={6}
        >
          {markersData.map(marker => {
            const isCommon = commonRegions.has(marker.code)
            const modelCount = marker.models.length
            const color = isCommon ? '#10b981' : '#1A9E7A'

            return (
              <Marker
                key={marker.code}
                position={[marker.lat, marker.lng]}
                icon={createMarkerIcon(color, modelCount > 1 ? modelCount : null)}
              >
                <Popup>
                  <div className="min-w-[200px]">
                    <div className="font-bold text-sm mb-1">{marker.name}</div>
                    <div className="text-xs text-gray-500 mb-2 font-mono">{marker.code}</div>

                    {isCommon && (
                      <div className="mb-2 px-2 py-1 bg-emerald-100 text-emerald-700 text-xs rounded">
                        Available in all selected models
                      </div>
                    )}

                    <div className="text-xs font-medium mb-1">
                      {marker.models.length} model{marker.models.length > 1 ? 's' : ''} available:
                    </div>
                    <div className="space-y-1 max-h-[150px] overflow-y-auto">
                      {marker.models.map(model => (
                        <div
                          key={model.model_id}
                          className="flex items-center gap-1.5 text-xs"
                        >
                          <span
                            className="w-2 h-2 rounded-full flex-shrink-0"
                            style={{
                              backgroundColor: providerColors[model.model_provider] || providerColors.default
                            }}
                          />
                          <span className="truncate">
                            {model.model_name || model.model_id}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </Popup>
              </Marker>
            )
          })}
        </MarkerClusterGroup>
      </MapContainer>

      {/* Legend */}
      <div className={cn(
        'px-4 py-2 border-t flex items-center gap-4 text-xs',
        isLight
          ? 'bg-stone-50 border-stone-200 text-stone-600'
          : 'bg-slate-800/50 border-slate-700 text-slate-400'
      )}>
        <div className="flex items-center gap-1.5">
          <span
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: '#10b981' }}
          />
          <span>Available in all models</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: '#1A9E7A' }}
          />
          <span>Available in some models</span>
        </div>
        <div className={cn(
          'ml-auto',
          isLight ? 'text-stone-500' : 'text-slate-500'
        )}>
          Click clusters to zoom in, click markers for details
        </div>
      </div>
    </div>
  )
}
