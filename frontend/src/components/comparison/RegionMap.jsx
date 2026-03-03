import { useEffect, useMemo } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import MarkerClusterGroup from 'react-leaflet-cluster'
import L from 'leaflet'
import { useTheme } from '@/components/layout/ThemeProvider'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { regionCoordinates, providerColors } from '@/config/constants'

// Import Leaflet CSS
import 'leaflet/dist/leaflet.css'

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

// Helper to get all regions for a model (on-demand + CRIS + Mantle)
function getAllModelRegions(model) {
  const onDemand = model.availability?.on_demand?.regions || []
  const cris = model.availability?.cross_region?.regions ?? []
  const mantle = model.availability?.mantle?.regions || []
  return [...new Set([...onDemand, ...cris, ...mantle])]
}

export function RegionMap({ selectedModels, isLight, height = '350px' }) {
  // Build markers data: for each region, list which models are available
  const markersData = useMemo(() => {
    const regionModels = {}

    selectedModels.forEach(({ model }) => {
      const regions = getAllModelRegions(model)
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

    const allRegions = getAllModelRegions(selectedModels[0].model)
    return new Set(
      allRegions.filter(region =>
        selectedModels.every(({ model }) =>
          getAllModelRegions(model).includes(region)
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
        'rounded-lg border flex items-center justify-center',
        isLight
          ? 'bg-stone-100 border-stone-200 text-stone-500'
          : 'bg-white/[0.03] border-white/[0.06] text-slate-400'
      )}>
        No region data available
      </div>
    )
  }

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight ? 'border-stone-200' : 'border-white/[0.06]'
    )}>
      <MapContainer
        key={isLight ? 'light' : 'dark'}
        center={[20, 0]}
        zoom={2}
        minZoom={2}
        maxBounds={[[-85, -180], [85, 180]]}
        maxBoundsViscosity={1.0}
        style={{ height, width: '100%', background: isLight ? '#faf9f5' : 'transparent' }}
        className="z-0"
        attributionControl={false}
      >
        <TileLayer
          url={tileUrl}
          noWrap={true}
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
          : 'bg-white/[0.02] border-white/[0.06] text-slate-400'
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
