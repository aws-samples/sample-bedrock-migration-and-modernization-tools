import { useState } from 'react'
import { Star, Globe, Zap, MessageSquare, Image, FileText, Video, Mic, Check, X, ChevronDown, ChevronRight, Search, Database, Languages, Cpu, Layers, Package, Server, ExternalLink } from 'lucide-react'
import { useTheme } from '@/components/layout/ThemeProvider'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'

// Provider color mapping - using actual brand colors
const providerColors = {
  Amazon: 'bg-[#FF9900]',        // Amazon Orange
  Anthropic: 'bg-[#D4A27F]',     // Anthropic Tan/Clay
  Meta: 'bg-[#0082FB]',          // Meta Blue
  Mistral: 'bg-[#F54E42]',       // Mistral Orange-Red
  Cohere: 'bg-[#39594D]',        // Cohere Dark Green
  'AI21 Labs': 'bg-[#6C5CE7]',   // AI21 Purple
  AI21: 'bg-[#6C5CE7]',          // AI21 Purple (alternate name)
  'Stability AI': 'bg-[#7C5CFF]', // Stability Purple
  Stability: 'bg-[#7C5CFF]',     // Stability Purple (alternate name)
  Luma: 'bg-[#6366F1]',          // Luma Indigo
  default: 'bg-slate-500',
}

// Modality icons
const modalityIcons = {
  TEXT: MessageSquare,
  IMAGE: Image,
  DOCUMENT: FileText,
  VIDEO: Video,
  AUDIO: Mic,
  SPEECH: Mic,
}

function formatNumber(num) {
  if (typeof num !== 'number' || isNaN(num)) return 'N/A'
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(0)}K`
  return num.toString()
}

function getProviderColor(provider) {
  return providerColors[provider] || providerColors.default
}

// Collapsible section component
function CollapsibleSection({ title, icon: Icon, children, defaultExpanded = false }) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <div className={cn(
      'rounded-lg overflow-hidden border',
      isLight
        ? 'bg-stone-50/80 border-stone-200/80 backdrop-blur-sm'
        : 'bg-white/5 border-white/10 backdrop-blur-sm'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-3 transition-colors',
          isLight ? 'hover:bg-stone-100/80' : 'hover:bg-white/5'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Icon className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{title}</span>
        </div>
        {isExpanded ? (
          <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
        ) : (
          <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
        )}
      </button>
      {isExpanded && (
        <div className={cn(
          'px-3 pb-3 pt-3 border-t',
          isLight
            ? 'border-stone-200/80 bg-white/60 backdrop-blur-sm'
            : 'border-white/10 bg-slate-900/30 backdrop-blur-sm'
        )}>
          {children}
        </div>
      )}
    </div>
  )
}

// Region display name mapping
const regionDisplayNames = {
  'us-east-1': 'N. Virginia',
  'us-east-2': 'Ohio',
  'us-west-1': 'N. California',
  'us-west-2': 'Oregon',
  'us-gov-west-1': 'US-GovCloud West',
  'eu-west-1': 'Ireland',
  'eu-west-2': 'London',
  'eu-west-3': 'Paris',
  'eu-central-1': 'Frankfurt',
  'eu-central-2': 'Zurich',
  'eu-north-1': 'Stockholm',
  'eu-south-1': 'Milan',
  'eu-south-2': 'Spain',
  'ap-east-1': 'Hong Kong',
  'ap-east-2': 'Hong Kong 2',
  'ap-northeast-1': 'Tokyo',
  'ap-northeast-2': 'Seoul',
  'ap-northeast-3': 'Osaka',
  'ap-southeast-1': 'Singapore',
  'ap-southeast-2': 'Sydney',
  'ap-southeast-3': 'Jakarta',
  'ap-southeast-4': 'Melbourne',
  'ap-southeast-5': 'Malaysia',
  'ap-southeast-7': 'Thailand',
  'ap-south-1': 'Mumbai',
  'ap-south-2': 'Hyderabad',
  'ca-central-1': 'Canada Central',
  'ca-west-1': 'Calgary',
  'sa-east-1': 'São Paulo',
  'me-south-1': 'Bahrain',
  'me-central-1': 'UAE',
  'mx-central-1': 'Mexico',
  'af-south-1': 'Cape Town',
  'il-central-1': 'Israel',
}

// Geographic groupings
const geoGroups = {
  'US': { name: 'United States', icon: '🇺🇸', prefixes: ['us-'] },
  'EU': { name: 'Europe', icon: '🇪🇺', prefixes: ['eu-'] },
  'APAC': { name: 'Asia Pacific', icon: '🌏', prefixes: ['ap-'] },
  'CA': { name: 'Canada', icon: '🇨🇦', prefixes: ['ca-'] },
  'SA': { name: 'South America', icon: '🌎', prefixes: ['sa-'] },
  'MX': { name: 'Mexico', icon: '🇲🇽', prefixes: ['mx-'] },
  'ME': { name: 'Middle East', icon: '🏜️', prefixes: ['me-', 'il-'] },
  'AF': { name: 'Africa', icon: '🌍', prefixes: ['af-'] },
}

function groupRegionsByGeo(regions) {
  const grouped = {}
  for (const region of regions) {
    let foundGroup = 'Other'
    for (const [groupKey, groupInfo] of Object.entries(geoGroups)) {
      if (groupInfo.prefixes.some(prefix => region.startsWith(prefix))) {
        foundGroup = groupKey
        break
      }
    }
    if (!grouped[foundGroup]) grouped[foundGroup] = []
    grouped[foundGroup].push(region)
  }
  return grouped
}

// Regional Availability grouped by geography
function RegionalAvailabilityGrouped({ regions }) {
  const [expandedGroups, setExpandedGroups] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const grouped = groupRegionsByGeo(regions)

  const toggleGroup = (group) => {
    setExpandedGroups(prev => ({ ...prev, [group]: !prev[group] }))
  }

  return (
    <div className="space-y-2">
      <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-400')}>
        Available in {regions.length} regions across {Object.keys(grouped).length} geographic areas
      </p>
      {Object.entries(grouped).map(([groupKey, groupRegions]) => {
        const groupInfo = geoGroups[groupKey] || { name: groupKey, icon: '🌐' }
        const isExpanded = expandedGroups[groupKey]

        return (
          <div key={groupKey} className={cn(
            'rounded-lg border overflow-hidden',
            isLight ? 'bg-white border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
          )}>
            <button
              className={cn(
                'w-full flex items-center justify-between p-2 transition-colors',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-slate-800/50'
              )}
              onClick={() => toggleGroup(groupKey)}
            >
              <div className="flex items-center gap-2">
                <span>{groupInfo.icon}</span>
                <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{groupInfo.name}</span>
                <Badge variant="secondary" className="text-xs">{groupRegions.length} regions</Badge>
              </div>
              {isExpanded ? (
                <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
              ) : (
                <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
              )}
            </button>
            {isExpanded && (
              <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-slate-700')}>
                <div className="flex flex-wrap gap-1.5 pt-2">
                  {groupRegions.sort().map(region => (
                    <Badge key={region} variant="outline" className="text-xs">
                      {regionDisplayNames[region] || region} <span className={cn('font-mono ml-1', isLight ? 'text-stone-500' : 'text-slate-400')}>({region})</span>
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// Cross-Region Inference Section with profiles grouped by source region
function CrossRegionInferenceSection({ crisData }) {
  const [expandedRegions, setExpandedRegions] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'

  // Calculate stats
  const profiles = crisData.profiles || []
  const sourceRegions = crisData.source_regions || []

  // Separate global endpoints from regional ones
  const globalProfilesMap = new Map() // Map profile_id -> { profile, regions: [] }
  const regionalProfiles = []

  for (const profile of profiles) {
    // Check if this is a global endpoint (profile name contains 'global')
    const isGlobal = profile.profile_name?.toLowerCase().includes('global') ||
                     profile.type?.toLowerCase() === 'global'
    if (isGlobal) {
      // Group by profile_id to deduplicate, collecting all regions
      const existingEntry = globalProfilesMap.get(profile.profile_id)
      if (existingEntry) {
        if (profile.source_region && !existingEntry.regions.includes(profile.source_region)) {
          existingEntry.regions.push(profile.source_region)
        }
      } else {
        globalProfilesMap.set(profile.profile_id, {
          profile,
          regions: profile.source_region ? [profile.source_region] : []
        })
      }
    } else {
      regionalProfiles.push(profile)
    }
  }

  // Convert map to array for rendering
  const globalProfiles = Array.from(globalProfilesMap.values())

  // Group regional profiles by source_region
  const profilesByRegion = {}
  for (const profile of regionalProfiles) {
    const region = profile.source_region
    if (!profilesByRegion[region]) {
      profilesByRegion[region] = []
    }
    profilesByRegion[region].push(profile)
  }

  // Get unique profile count
  const uniqueProfileIds = new Set(profiles.map(p => p.profile_id))

  // Group regions by geography
  const regionsByGeo = { 'US': [], 'EU': [], 'APAC': [], 'Other': [] }
  for (const region of Object.keys(profilesByRegion)) {
    if (region.startsWith('us-')) regionsByGeo['US'].push(region)
    else if (region.startsWith('eu-')) regionsByGeo['EU'].push(region)
    else if (region.startsWith('ap-')) regionsByGeo['APAC'].push(region)
    else regionsByGeo['Other'].push(region)
  }

  const toggleRegion = (key) => {
    setExpandedRegions(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const geoInfo = { 'US': '🇺🇸', 'EU': '🇪🇺', 'APAC': '🌏', 'Other': '📍' }

  return (
    <div className="space-y-3">
      {/* Status metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {crisData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Supported</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-500')}>Not Supported</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Total Profiles</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{crisData.profiles_count || profiles.length}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Source Regions</p>
          <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{Object.keys(profilesByRegion).length}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Unique Endpoints</p>
          <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{uniqueProfileIds.size}</p>
        </div>
      </div>

      {/* CRIS Endpoints grouped by source region */}
      {crisData.supported && profiles.length > 0 && (
        <div className="space-y-3">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>CRIS Endpoints by Source Region</p>

          {/* Global Endpoints Group */}
          {globalProfiles.length > 0 && (
            <div className={cn(
              'rounded-lg border overflow-hidden',
              isLight ? 'bg-white border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
            )}>
              <button
                className={cn(
                  'w-full flex items-center justify-between p-3 transition-colors',
                  isLight ? 'hover:bg-stone-50' : 'hover:bg-slate-800/50'
                )}
                onClick={() => toggleRegion('geo_Global')}
              >
                <div className="flex items-center gap-2">
                  <span>🌐</span>
                  <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>Global Endpoints</span>
                  <Badge variant="info" className="text-xs">{globalProfiles.length} endpoints</Badge>
                </div>
                {expandedRegions['geo_Global'] ? (
                  <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                ) : (
                  <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                )}
              </button>
              {expandedRegions['geo_Global'] && (
                <div className={cn('px-3 pb-3 pt-3 border-t space-y-2', isLight ? 'border-stone-200' : 'border-slate-700')}>
                  {globalProfiles.map(({ profile, regions }, idx) => (
                    <div key={`${profile.profile_id}-${idx}`} className={cn(
                      'rounded p-2',
                      isLight ? 'bg-stone-50 border border-stone-200' : 'bg-[#161d26] border border-slate-600/40'
                    )}>
                      <p className={cn('text-sm font-medium', isLight ? 'text-stone-900' : 'text-white')}>
                        {profile.profile_name}
                      </p>
                      <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                        {profile.profile_id}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="secondary" className="text-[10px]">{profile.type || 'inference'}</Badge>
                      </div>
                      {profile.description && (
                        <p className={cn('text-xs mt-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>
                          {profile.description}
                        </p>
                      )}
                      {regions.length > 0 && (
                        <div className="mt-2">
                          <p className={cn('text-[10px] mb-1', isLight ? 'text-stone-500' : 'text-slate-400')}>
                            Available in {regions.length} regions:
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {regions.sort().map(region => (
                              <Tooltip key={region} delayDuration={200}>
                                <TooltipTrigger asChild>
                                  <Badge variant="outline" className="text-[10px] cursor-default">
                                    {regionDisplayNames[region] || region}
                                  </Badge>
                                </TooltipTrigger>
                                <TooltipContent side="bottom" sideOffset={4}>
                                  <p className="font-mono text-xs">{region}</p>
                                </TooltipContent>
                              </Tooltip>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Regional Endpoints by Geography */}
          {['US', 'EU', 'APAC', 'Other'].map(geoKey => {
            const geoRegions = regionsByGeo[geoKey]
            if (geoRegions.length === 0) return null
            const isGeoExpanded = expandedRegions[`geo_${geoKey}`]

            // Group endpoints by profile_id within this geo, collecting all regions
            const geoEndpointsMap = new Map()
            for (const region of geoRegions) {
              const regionProfiles = profilesByRegion[region] || []
              for (const profile of regionProfiles) {
                const existing = geoEndpointsMap.get(profile.profile_id)
                if (existing) {
                  if (!existing.regions.includes(region)) {
                    existing.regions.push(region)
                  }
                } else {
                  geoEndpointsMap.set(profile.profile_id, {
                    profile,
                    regions: [region]
                  })
                }
              }
            }
            const geoEndpoints = Array.from(geoEndpointsMap.values())

            return (
              <div key={geoKey} className={cn(
                'rounded-lg border overflow-hidden',
                isLight ? 'bg-white border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
              )}>
                <button
                  className={cn(
                    'w-full flex items-center justify-between p-3 transition-colors',
                    isLight ? 'hover:bg-stone-50' : 'hover:bg-slate-800/50'
                  )}
                  onClick={() => toggleRegion(`geo_${geoKey}`)}
                >
                  <div className="flex items-center gap-2">
                    <span>{geoInfo[geoKey]}</span>
                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{geoKey} Regions</span>
                    <Badge variant="secondary" className="text-xs">{geoRegions.length} regions</Badge>
                    <Badge variant="info" className="text-xs">{geoEndpoints.length} endpoints</Badge>
                  </div>
                  {isGeoExpanded ? (
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                  )}
                </button>
                {isGeoExpanded && (
                  <div className={cn('px-3 pb-3 pt-3 border-t space-y-2', isLight ? 'border-stone-200' : 'border-slate-700')}>
                    {geoEndpoints.map(({ profile, regions }, idx) => (
                      <div key={`${profile.profile_id}-${idx}`} className={cn(
                        'rounded p-2',
                        isLight ? 'bg-stone-50 border border-stone-200' : 'bg-[#161d26] border border-slate-600/40'
                      )}>
                        <p className={cn('text-sm font-medium', isLight ? 'text-stone-900' : 'text-white')}>
                          {profile.profile_name}
                        </p>
                        <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                          {profile.profile_id}
                        </p>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="secondary" className="text-[10px]">{profile.type || 'inference'}</Badge>
                        </div>
                        {profile.description && (
                          <p className={cn('text-xs mt-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>
                            {profile.description}
                          </p>
                        )}
                        {regions.length > 0 && (
                          <div className="mt-2">
                            <p className={cn('text-[10px] mb-1', isLight ? 'text-stone-500' : 'text-slate-400')}>
                              Available in {regions.length} regions:
                            </p>
                            <div className="flex flex-wrap gap-1">
                              {regions.sort().map(region => (
                                <Tooltip key={region} delayDuration={200}>
                                  <TooltipTrigger asChild>
                                    <Badge variant="outline" className="text-[10px] cursor-default">
                                      {regionDisplayNames[region] || region}
                                    </Badge>
                                  </TooltipTrigger>
                                  <TooltipContent side="bottom" sideOffset={4}>
                                    <p className="font-mono text-xs">{region}</p>
                                  </TooltipContent>
                                </Tooltip>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// Batch Inference Section with grouped regions
function BatchInferenceSection({ batchData }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const regions = batchData.supported_regions || []
  const grouped = groupRegionsByGeo(regions)

  return (
    <div className="space-y-3">
      {/* Status metrics */}
      <div className="grid grid-cols-3 gap-2">
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {batchData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Supported</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-500')}>Not Supported</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Regions</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{regions.length}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Coverage</p>
          <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{batchData.coverage_percentage?.toFixed(0) || 0}%</p>
        </div>
      </div>

      {/* Regions grouped by geography */}
      {batchData.supported && regions.length > 0 && (
        <div className={cn(
          'rounded-lg border overflow-hidden',
          isLight ? 'bg-white border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between p-2 transition-colors',
              isLight ? 'hover:bg-stone-50' : 'hover:bg-slate-800/50'
            )}
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <div className="flex items-center gap-2">
              <Globe className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>Available Regions</span>
              <Badge variant="info" className="text-xs">{regions.length} regions</Badge>
            </div>
            {isExpanded ? (
              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
            ) : (
              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
            )}
          </button>
          {isExpanded && (
            <div className={cn('px-2 pb-2 pt-2 space-y-2 border-t', isLight ? 'border-stone-200' : 'border-slate-700')}>
              {Object.entries(grouped).map(([geoKey, geoRegions]) => {
                const geoInfo = geoGroups[geoKey] || { name: geoKey, icon: '🌐' }
                return (
                  <div key={geoKey}>
                    <p className={cn('text-xs mb-1', isLight ? 'text-stone-600' : 'text-slate-400')}>
                      {geoInfo.icon} {geoInfo.name} ({geoRegions.length})
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {geoRegions.sort().map(region => (
                        <Badge key={region} variant="outline" className="text-xs">
                          {regionDisplayNames[region] || region}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SpecsTab({ model }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const contextWindow = model.converse_data?.context_window
  const maxOutput = model.converse_data?.max_output_tokens
  const inputModalities = model.model_modalities?.input_modalities || []
  const outputModalities = model.model_modalities?.output_modalities || []
  const capabilities = model.model_capabilities || []
  const useCases = model.model_use_cases || []
  const documentationLinks = model.documentation_links || {}
  const regions = model.regions_available || []
  const streamingSupported = model.streaming_supported
  const crisData = model.cross_region_inference || {}
  const batchData = model.batch_inference_supported || {}
  const consumptionOptions = model.consumption_options || []
  const inferenceTypes = model.inference_types_supported || []
  const customizations = model.customization?.customization_supported || []
  const lifecycleStatus = model.model_lifecycle?.status || model.model_status || 'Unknown'

  return (
    <ScrollArea className="h-[500px]">
      <div className="space-y-4 pr-4">
        {/* Metrics Banner */}
        <div className="grid grid-cols-4 gap-3 items-stretch">
          <div className={cn(
            'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[80px]',
            isLight
              ? 'bg-amber-50/80 border-amber-100/50 backdrop-blur-sm'
              : 'bg-white/5 border-white/10 backdrop-blur-sm'
          )}>
            <Globe className={cn('h-5 w-5 mx-auto mb-1', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
            <p className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>{regions.length}</p>
            <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Regions</p>
          </div>
          <div className={cn(
            'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[80px]',
            isLight
              ? 'bg-purple-50/80 border-purple-100/50 backdrop-blur-sm'
              : 'bg-white/5 border-white/10 backdrop-blur-sm'
          )}>
            <FileText className="h-5 w-5 text-purple-500 mx-auto mb-1" />
            <p className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>{Object.keys(documentationLinks).length}</p>
            <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Doc Links</p>
          </div>
          <div className={cn(
            'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[80px]',
            isLight
              ? 'bg-emerald-50/80 border-emerald-100/50 backdrop-blur-sm'
              : 'bg-white/5 border-white/10 backdrop-blur-sm'
          )}>
            <Cpu className="h-5 w-5 text-emerald-500 mx-auto mb-1" />
            <p className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>{capabilities.length}</p>
            <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Capabilities</p>
          </div>
          <div className={cn(
            'rounded-lg p-3 text-center border flex flex-col justify-center items-center min-h-[80px]',
            isLight
              ? 'bg-stone-50/80 border-stone-100/50 backdrop-blur-sm'
              : 'bg-white/5 border-white/10 backdrop-blur-sm'
          )}>
            <Badge variant={lifecycleStatus === 'ACTIVE' ? 'success' : 'warning'} className="text-xs mb-1">
              {lifecycleStatus}
            </Badge>
            <p className={cn('text-xs mt-1', isLight ? 'text-stone-600' : 'text-slate-400')}>Status</p>
          </div>
        </div>

        {/* 1. Core Model Information */}
        <CollapsibleSection title="Core Model Information" icon={Database} defaultExpanded={true}>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Model ID</p>
              <p className={cn('text-sm font-mono truncate', isLight ? 'text-stone-900' : 'text-white')}>{model.model_id}</p>
            </div>
            <div>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Provider</p>
              <p className={cn('text-sm font-medium', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{model.model_provider}</p>
            </div>
            <div>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Context Window</p>
              <p className={cn('text-sm font-semibold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
                {contextWindow ? contextWindow.toLocaleString() : 'N/A'} tokens
              </p>
            </div>
            <div>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Max Output</p>
              <p className="text-sm font-semibold text-purple-600 dark:text-purple-400">
                {maxOutput ? maxOutput.toLocaleString() : 'N/A'} tokens
              </p>
            </div>
            <div>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Streaming Support</p>
              <div className="flex items-center gap-1">
                {streamingSupported ? (
                  <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm text-emerald-600 dark:text-emerald-400">Supported</span></>
                ) : (
                  <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-500')}>Not Supported</span></>
                )}
              </div>
            </div>
            <div>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Lifecycle Status</p>
              <Badge variant={lifecycleStatus === 'ACTIVE' ? 'success' : 'warning'} className="text-xs">
                {lifecycleStatus}
              </Badge>
            </div>
          </div>
        </CollapsibleSection>

        {/* 2. Input & Output Modalities */}
        <CollapsibleSection title="Input & Output Modalities" icon={Layers}>
          <div className="space-y-3">
            <div>
              <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-400')}>Input Modalities</p>
              <div className="flex flex-wrap gap-2">
                {inputModalities.length > 0 ? inputModalities.map(mod => {
                  const Icon = modalityIcons[mod] || MessageSquare
                  return (
                    <Badge key={mod} className={cn(isLight ? 'text-[#faf9f5] bg-amber-700' : 'text-white bg-[#1A9E7A]')}>
                      <Icon className="h-3 w-3 mr-1" />{mod}
                    </Badge>
                  )
                }) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-500')}>None specified</span>}
              </div>
            </div>
            <div>
              <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-400')}>Output Modalities</p>
              <div className="flex flex-wrap gap-2">
                {outputModalities.length > 0 ? outputModalities.map(mod => {
                  const Icon = modalityIcons[mod] || MessageSquare
                  return (
                    <Badge key={mod} className={cn('bg-emerald-600', isLight ? 'text-[#faf9f5]' : 'text-white')}>
                      <Icon className="h-3 w-3 mr-1" />{mod}
                    </Badge>
                  )
                }) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-500')}>None specified</span>}
              </div>
            </div>
          </div>
        </CollapsibleSection>

        {/* 3. Capabilities & Use Cases */}
        <CollapsibleSection title="Capabilities & Use Cases" icon={Cpu}>
          <div className="space-y-3">
            <div>
              <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-400')}>Capabilities</p>
              <div className="flex flex-wrap gap-1.5">
                {capabilities.length > 0 ? capabilities.map(cap => (
                  <Badge key={cap} variant="secondary" className="text-xs">{cap}</Badge>
                )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-500')}>None specified</span>}
              </div>
            </div>
            <div>
              <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-400')}>Use Cases</p>
              <div className="flex flex-wrap gap-1.5">
                {useCases.length > 0 ? useCases.map(uc => (
                  <Badge key={uc} variant="outline" className="text-xs">{uc}</Badge>
                )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-500')}>None specified</span>}
              </div>
            </div>
          </div>
        </CollapsibleSection>

        {/* 4. Documentation & Resources */}
        <CollapsibleSection title="Documentation & Resources" icon={FileText}>
          <div className="space-y-2">
            {Object.keys(documentationLinks).length > 0 ? (
              <div className="flex flex-col gap-2">
                {documentationLinks.model_docs && (
                  <a href={documentationLinks.model_docs} target="_blank" rel="noopener noreferrer"
                     className={cn('flex items-center gap-2 text-sm hover:underline', isLight ? 'text-blue-600' : 'text-blue-400')}>
                    <ExternalLink className="h-3.5 w-3.5" />
                    Model Documentation
                  </a>
                )}
                {documentationLinks.languages && (
                  <a href={documentationLinks.languages} target="_blank" rel="noopener noreferrer"
                     className={cn('flex items-center gap-2 text-sm hover:underline', isLight ? 'text-blue-600' : 'text-blue-400')}>
                    <ExternalLink className="h-3.5 w-3.5" />
                    Supported Languages
                  </a>
                )}
                {documentationLinks.api_reference && (
                  <a href={documentationLinks.api_reference} target="_blank" rel="noopener noreferrer"
                     className={cn('flex items-center gap-2 text-sm hover:underline', isLight ? 'text-blue-600' : 'text-blue-400')}>
                    <ExternalLink className="h-3.5 w-3.5" />
                    API Reference
                  </a>
                )}
                {documentationLinks.bedrock_console && (
                  <a href={documentationLinks.bedrock_console} target="_blank" rel="noopener noreferrer"
                     className={cn('flex items-center gap-2 text-sm hover:underline', isLight ? 'text-blue-600' : 'text-blue-400')}>
                    <ExternalLink className="h-3.5 w-3.5" />
                    Bedrock Console
                  </a>
                )}
              </div>
            ) : (
              <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-500')}>No documentation links available</span>
            )}
          </div>
        </CollapsibleSection>

        {/* 5. Regional Availability - Grouped by Geography */}
        <CollapsibleSection title="Regional Availability" icon={Globe}>
          <RegionalAvailabilityGrouped regions={regions} />
        </CollapsibleSection>

        {/* 6. Consumption & Deployment Options */}
        <CollapsibleSection title="Consumption & Deployment Options" icon={Server}>
          <div className="space-y-3">
            <div>
              <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-400')}>Consumption Options</p>
              <div className="flex flex-wrap gap-1.5">
                {consumptionOptions.length > 0 ? consumptionOptions.map(opt => (
                  <Badge key={opt} variant="info" className="text-xs">
                    {opt === 'on_demand' ? 'On-Demand' : opt === 'provisioned' ? 'Provisioned' : opt === 'batch' ? 'Batch' : opt}
                  </Badge>
                )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-500')}>Not specified</span>}
              </div>
            </div>
            <div>
              <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-400')}>Inference Types</p>
              <div className="flex flex-wrap gap-1.5">
                {inferenceTypes.length > 0 ? inferenceTypes.map(type => (
                  <Badge key={type} variant="secondary" className="text-xs">{type}</Badge>
                )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-500')}>Not specified</span>}
              </div>
            </div>
            {customizations.length > 0 && (
              <div>
                <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-400')}>Customizations Supported</p>
                <div className="flex flex-wrap gap-1.5">
                  {customizations.map(custom => (
                    <Badge key={custom} variant="outline" className="text-xs">{custom}</Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CollapsibleSection>

        {/* 7. Cross-Region Inference */}
        <CollapsibleSection title="Cross-Region Inference" icon={Globe}>
          <CrossRegionInferenceSection crisData={crisData} />
        </CollapsibleSection>

        {/* 8. Batch Inference Support */}
        <CollapsibleSection title="Batch Inference Support" icon={Package}>
          <BatchInferenceSection batchData={batchData} />
        </CollapsibleSection>
      </div>
    </ScrollArea>
  )
}

function CollapsibleRegion({ region, quotas, defaultExpanded = false, showAdjustable = false }) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const regionQuotas = quotas || []

  return (
    <div className={cn(
      'rounded-lg overflow-hidden border',
      isLight ? 'bg-white border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-2 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-slate-800/50'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{regionDisplayNames[region] || region}</span>
          <span className={cn('text-xs font-mono', isLight ? 'text-stone-600' : 'text-slate-400')}>({region})</span>
          <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>- {Array.isArray(regionQuotas) ? regionQuotas.length : 0} quotas</span>
        </div>
        {isExpanded ? (
          <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
        ) : (
          <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
        )}
      </button>
      {isExpanded && (
        <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-slate-700')}>
          {Array.isArray(regionQuotas) && regionQuotas.length > 0 ? (
            <div className="space-y-1.5 pt-2">
              {regionQuotas.map((quota, idx) => (
                <div key={idx} className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
                  <div className="flex justify-between items-start gap-3">
                    <div className="flex-1 min-w-0">
                      <p className={cn('text-xs leading-relaxed', isLight ? 'text-stone-800' : 'text-slate-200')}>
                        {quota.quota_name || 'Unknown quota'}
                      </p>
                      <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                        {quota.quota_code || ''}
                      </p>
                    </div>
                    <div className="text-right flex-shrink-0 min-w-[80px]">
                      <p className={cn('text-sm font-semibold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                        {formatNumber(quota.value)}
                      </p>
                      {showAdjustable && (
                        <p className={cn('text-[10px] mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                          {quota.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className={cn('text-sm pt-2', isLight ? 'text-stone-600' : 'text-slate-400')}>No quotas defined</p>
          )}
        </div>
      )}
    </div>
  )
}

// Quota category definitions
const quotaCategories = {
  on_demand: { name: 'On-Demand Inference', icon: '🚀', color: 'text-emerald-500' },
  cross_region: { name: 'Cross-Region Inference', icon: '🌍', color: 'text-blue-500' },
  batch: { name: 'Batch Inference', icon: '📦', color: 'text-purple-500' },
  provisioned: { name: 'Provisioned Throughput', icon: '⚡', color: 'text-amber-500' },
  customization: { name: 'Model Customization', icon: '🎯', color: 'text-red-500' },
  general: { name: 'General Limits', icon: '⚙️', color: 'text-slate-500' },
}

function categorizeQuota(quotaName) {
  const name = quotaName.toLowerCase()
  if (name.includes('on-demand') || name.includes('on demand')) return 'on_demand'
  if (name.includes('cross-region') || name.includes('cross region')) return 'cross_region'
  if (name.includes('batch')) return 'batch'
  if (name.includes('provisioned') || name.includes('model units')) return 'provisioned'
  if (name.includes('customization') || name.includes('fine-tuning') || name.includes('training')) return 'customization'
  return 'general'
}

function QuotasTab({ model }) {
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedCategories, setExpandedCategories] = useState({})
  const [expandedGeos, setExpandedGeos] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const quotas = model.model_service_quotas || {}
  const allRegions = Object.keys(quotas)

  const toggleGeo = (key) => {
    setExpandedGeos(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const geoInfo = {
    'Global': { icon: '🌐', name: 'Global (Same across all regions)' },
    'US': { icon: '🇺🇸', name: 'United States' },
    'EU': { icon: '🇪🇺', name: 'Europe' },
    'APAC': { icon: '🌏', name: 'Asia Pacific' },
    'CA': { icon: '🇨🇦', name: 'Canada' },
    'SA': { icon: '🌎', name: 'South America' },
    'ME': { icon: '🏜️', name: 'Middle East' },
    'AF': { icon: '🌍', name: 'Africa' },
    'Other': { icon: '📍', name: 'Other' }
  }

  const getGeoForRegion = (region) => {
    if (region.startsWith('us-')) return 'US'
    if (region.startsWith('eu-')) return 'EU'
    if (region.startsWith('ap-')) return 'APAC'
    if (region.startsWith('ca-')) return 'CA'
    if (region.startsWith('sa-')) return 'SA'
    if (region.startsWith('me-') || region.startsWith('il-')) return 'ME'
    if (region.startsWith('af-')) return 'AF'
    return 'Other'
  }

  // Calculate statistics
  let totalQuotas = 0
  let adjustableCount = 0
  const categorizedQuotas = {}

  // Process all quotas
  for (const region of allRegions) {
    const regionQuotas = quotas[region] || []
    for (const quota of regionQuotas) {
      totalQuotas++
      if (quota.adjustable) adjustableCount++

      const category = categorizeQuota(quota.quota_name || '')
      if (!categorizedQuotas[category]) {
        categorizedQuotas[category] = {}
      }
      if (!categorizedQuotas[category][region]) {
        categorizedQuotas[category][region] = []
      }
      categorizedQuotas[category][region].push(quota)
    }
  }

  const categoriesWithData = Object.keys(categorizedQuotas).filter(cat => Object.keys(categorizedQuotas[cat]).length > 0)

  // Filter by search (region name, region code, geo, quota name)
  const filterQuotasBySearch = (categoryQuotas) => {
    if (!searchQuery) return categoryQuotas
    const query = searchQuery.toLowerCase()
    const filtered = {}

    for (const [region, regionQuotas] of Object.entries(categoryQuotas)) {
      const regionName = (regionDisplayNames[region] || '').toLowerCase()
      const regionCode = region.toLowerCase()
      const geo = getGeoForRegion(region)
      const geoName = geoInfo[geo]?.name?.toLowerCase() || ''

      // Check if region code, region name, or geo matches
      const regionMatches = regionCode.includes(query) ||
                           regionName.includes(query) ||
                           geo.toLowerCase().includes(query) ||
                           geoName.includes(query)

      if (regionMatches) {
        // Include all quotas for matching region/geo
        filtered[region] = regionQuotas
      } else {
        // Check individual quota names
        const matchingQuotas = regionQuotas.filter(q =>
          q.quota_name?.toLowerCase().includes(query) ||
          q.quota_code?.toLowerCase().includes(query)
        )
        if (matchingQuotas.length > 0) {
          filtered[region] = matchingQuotas
        }
      }
    }
    return filtered
  }

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({ ...prev, [category]: !prev[category] }))
  }

  if (allRegions.length === 0) {
    return (
      <div className={cn('text-center py-8', isLight ? 'text-stone-600' : 'text-slate-400')}>
        <p>No quota information available</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Metrics Banner */}
      <div className="grid grid-cols-4 gap-2 items-stretch">
        <div className={cn(
          'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[60px]',
          isLight
            ? 'bg-amber-50/80 border-amber-100/50 backdrop-blur-sm'
            : 'bg-white/5 border-white/10 backdrop-blur-sm'
        )}>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{totalQuotas}</p>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Total Quotas</p>
        </div>
        <div className={cn(
          'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[60px]',
          isLight
            ? 'bg-emerald-50/80 border-emerald-100/50 backdrop-blur-sm'
            : 'bg-white/5 border-white/10 backdrop-blur-sm'
        )}>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{adjustableCount}/{totalQuotas}</p>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Adjustable</p>
        </div>
        <div className={cn(
          'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[60px]',
          isLight
            ? 'bg-blue-50/80 border-blue-100/50 backdrop-blur-sm'
            : 'bg-white/5 border-white/10 backdrop-blur-sm'
        )}>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{allRegions.length}</p>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Regions</p>
        </div>
        <div className={cn(
          'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[60px]',
          isLight
            ? 'bg-purple-50/80 border-purple-100/50 backdrop-blur-sm'
            : 'bg-white/5 border-white/10 backdrop-blur-sm'
        )}>
          <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{categoriesWithData.length}</p>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Categories</p>
        </div>
      </div>

      {/* Search bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
        <Input
          placeholder="Search by region, geo (US, Europe...), or quota name..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* Scrollable category list */}
      <ScrollArea className="h-[400px]">
        <div className="space-y-3 pr-4">
          {['on_demand', 'cross_region', 'batch', 'provisioned', 'customization', 'general'].map(categoryKey => {
            const categoryData = categorizedQuotas[categoryKey]
            if (!categoryData || Object.keys(categoryData).length === 0) return null

            const filteredData = filterQuotasBySearch(categoryData)
            if (Object.keys(filteredData).length === 0) return null

            const catInfo = quotaCategories[categoryKey]
            const regionCount = Object.keys(filteredData).length
            const isExpanded = expandedCategories[categoryKey]

            return (
              <div key={categoryKey} className={cn('rounded-lg overflow-hidden', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
                <button
                  className={cn(
                    'w-full flex items-center justify-between p-3 transition-colors',
                    isLight ? 'hover:bg-stone-200' : 'hover:bg-slate-700/50'
                  )}
                  onClick={() => toggleCategory(categoryKey)}
                >
                  <div className="flex items-center gap-2">
                    <span>{catInfo.icon}</span>
                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{catInfo.name}</span>
                    <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>({regionCount} regions)</span>
                  </div>
                  {isExpanded ? (
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                  )}
                </button>
                {isExpanded && (
                  <div className={cn('px-3 pb-3 pt-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-slate-700')}>
                    {(() => {
                      // Identify global quotas (quota name contains "global")
                      const globalQuotasMap = new Map()
                      const regionalQuotasByRegion = {}

                      // Process all quotas - separate global from regional
                      for (const [region, regionQuotas] of Object.entries(filteredData)) {
                        for (const quota of regionQuotas) {
                          const quotaName = (quota.quota_name || '').toLowerCase()
                          const isGlobalQuota = quotaName.includes('global')

                          if (isGlobalQuota) {
                            // Group global quotas by name, collecting regions
                            const key = quota.quota_name
                            const existing = globalQuotasMap.get(key)
                            if (existing) {
                              if (!existing.availableRegions.includes(region)) {
                                existing.availableRegions.push(region)
                              }
                            } else {
                              globalQuotasMap.set(key, {
                                ...quota,
                                availableRegions: [region]
                              })
                            }
                          } else {
                            // Keep as regional
                            if (!regionalQuotasByRegion[region]) regionalQuotasByRegion[region] = []
                            regionalQuotasByRegion[region].push(quota)
                          }
                        }
                      }

                      const globalQuotas = Array.from(globalQuotasMap.values())

                      // Group remaining regional quotas by geo
                      const regionsByGeo = {}
                      for (const [region, regionQuotas] of Object.entries(regionalQuotasByRegion)) {
                        const geo = getGeoForRegion(region)
                        if (!regionsByGeo[geo]) regionsByGeo[geo] = {}
                        regionsByGeo[geo][region] = regionQuotas
                      }

                      return (
                        <>
                          {/* Global quotas group */}
                          {globalQuotas.length > 0 && (
                            <div className={cn(
                              'rounded-lg border overflow-hidden',
                              isLight ? 'bg-stone-50 border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
                            )}>
                              <button
                                className={cn(
                                  'w-full flex items-center justify-between p-2 transition-colors',
                                  isLight ? 'hover:bg-stone-100' : 'hover:bg-slate-700/50'
                                )}
                                onClick={() => toggleGeo(`${categoryKey}_Global`)}
                              >
                                <div className="flex items-center gap-2">
                                  <span>🌐</span>
                                  <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>Global</span>
                                  <Badge variant="info" className="text-xs">{globalQuotas.length} quotas</Badge>
                                </div>
                                {expandedGeos[`${categoryKey}_Global`] ? (
                                  <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                                ) : (
                                  <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                                )}
                              </button>
                              {expandedGeos[`${categoryKey}_Global`] && (
                                <div className={cn('px-2 pb-2 pt-2 border-t space-y-1.5', isLight ? 'border-stone-200' : 'border-slate-700')}>
                                  {globalQuotas.map((quota, idx) => (
                                    <div key={idx} className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
                                      <div className="flex justify-between items-start gap-3">
                                        <div className="flex-1 min-w-0">
                                          <p className={cn('text-xs leading-relaxed', isLight ? 'text-stone-800' : 'text-slate-200')}>
                                            {quota.quota_name || 'Unknown quota'}
                                          </p>
                                          <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                                            {quota.quota_code || ''}
                                          </p>
                                          <div className="flex flex-wrap gap-1 mt-1.5">
                                            {quota.availableRegions.sort().map(region => (
                                              <Tooltip key={region} delayDuration={200}>
                                                <TooltipTrigger asChild>
                                                  <Badge variant="outline" className="text-[10px] cursor-default">
                                                    {regionDisplayNames[region] || region}
                                                  </Badge>
                                                </TooltipTrigger>
                                                <TooltipContent side="bottom" sideOffset={4}>
                                                  <p className="font-mono text-xs">{region}</p>
                                                </TooltipContent>
                                              </Tooltip>
                                            ))}
                                          </div>
                                        </div>
                                        <div className="text-right flex-shrink-0 min-w-[80px]">
                                          <p className={cn('text-sm font-semibold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                                            {formatNumber(quota.value)}
                                          </p>
                                          <p className={cn('text-[10px] mt-0.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
                                            {quota.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                                          </p>
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}

                          {/* Regional quotas by geo */}
                          {['US', 'EU', 'APAC', 'CA', 'SA', 'ME', 'AF', 'Other'].map(geoKey => {
                            const geoRegions = regionsByGeo[geoKey]
                            if (!geoRegions || Object.keys(geoRegions).length === 0) return null

                            const geo = geoInfo[geoKey]
                            const geoExpandKey = `${categoryKey}_${geoKey}`
                            const isGeoExpanded = expandedGeos[geoExpandKey]
                            const regionCount = Object.keys(geoRegions).length

                            return (
                              <div key={geoKey} className={cn(
                                'rounded-lg border overflow-hidden',
                                isLight ? 'bg-stone-50 border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
                              )}>
                                <button
                                  className={cn(
                                    'w-full flex items-center justify-between p-2 transition-colors',
                                    isLight ? 'hover:bg-stone-100' : 'hover:bg-slate-700/50'
                                  )}
                                  onClick={() => toggleGeo(geoExpandKey)}
                                >
                                  <div className="flex items-center gap-2">
                                    <span>{geo.icon}</span>
                                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{geo.name}</span>
                                    <Badge variant="secondary" className="text-xs">{regionCount} regions</Badge>
                                  </div>
                                  {isGeoExpanded ? (
                                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                                  ) : (
                                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                                  )}
                                </button>
                                {isGeoExpanded && (
                                  <div className={cn('px-2 pb-2 pt-2 border-t space-y-2', isLight ? 'border-stone-200' : 'border-slate-700')}>
                                    {Object.entries(geoRegions).sort().map(([region, regionQuotas]) => (
                                      <CollapsibleRegion
                                        key={region}
                                        region={region}
                                        quotas={regionQuotas}
                                        defaultExpanded={false}
                                        showAdjustable={true}
                                      />
                                    ))}
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </>
                      )
                    })()}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}

// Extract pricing from various structures
function extractRegionPricing(regionPricing) {
  if (!regionPricing) return { onDemand: [], provisioned: [] }

  const onDemand = []
  const provisioned = []

  // Handle agreement_offers structure
  if (regionPricing.on_demand) {
    const od = regionPricing.on_demand
    if (od.input_tokens) {
      od.input_tokens.forEach(p => {
        onDemand.push({
          type: 'input',
          description: p.description || 'Input Tokens',
          price: parseFloat(p.price),
          unit: p.unit || 'Per 1K tokens'
        })
      })
    }
    if (od.output_tokens) {
      od.output_tokens.forEach(p => {
        onDemand.push({
          type: 'output',
          description: p.description || 'Output Tokens',
          price: parseFloat(p.price),
          unit: p.unit || 'Per 1K tokens'
        })
      })
    }
  }

  // Handle provisioned throughput
  if (regionPricing.provisioned_throughput) {
    regionPricing.provisioned_throughput.forEach(p => {
      provisioned.push({
        description: p.description || 'Provisioned Throughput',
        price: parseFloat(p.price),
        unit: p.unit || 'Per Hour'
      })
    })
  }

  // Handle simple structure (input_per_1k_tokens, output_per_1k_tokens)
  if (regionPricing.input_per_1k_tokens !== undefined) {
    onDemand.push({
      type: 'input',
      description: 'Input Tokens',
      price: regionPricing.input_per_1k_tokens,
      unit: 'Per 1K tokens'
    })
  }
  if (regionPricing.output_per_1k_tokens !== undefined) {
    onDemand.push({
      type: 'output',
      description: 'Output Tokens',
      price: regionPricing.output_per_1k_tokens,
      unit: 'Per 1K tokens'
    })
  }

  // Handle text sub-object
  if (regionPricing.text) {
    if (regionPricing.text.input_per_1k_tokens !== undefined) {
      onDemand.push({
        type: 'input',
        description: 'Input Tokens',
        price: regionPricing.text.input_per_1k_tokens,
        unit: 'Per 1K tokens'
      })
    }
    if (regionPricing.text.output_per_1k_tokens !== undefined) {
      onDemand.push({
        type: 'output',
        description: 'Output Tokens',
        price: regionPricing.text.output_per_1k_tokens,
        unit: 'Per 1K tokens'
      })
    }
  }

  return { onDemand, provisioned }
}

function CollapsiblePricingRegion({ region, pricing, category, defaultExpanded = false }) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const { onDemand, provisioned } = extractRegionPricing(pricing)
  const pricingItems = category === 'on_demand' ? onDemand : provisioned

  // Quick summary
  const inputItem = pricingItems.find(p => p.type === 'input')
  const outputItem = pricingItems.find(p => p.type === 'output')

  return (
    <div className={cn(
      'rounded-lg overflow-hidden border',
      isLight ? 'bg-white border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-2 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-slate-800/50'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{regionDisplayNames[region] || region}</span>
          <span className={cn('text-xs font-mono', isLight ? 'text-stone-600' : 'text-slate-400')}>({region})</span>
        </div>
        <div className="flex items-center gap-3">
          {inputItem && outputItem && (
            <span className="text-xs text-emerald-600 dark:text-emerald-400">
              ${inputItem.price.toFixed(4)} / ${outputItem.price.toFixed(4)}
            </span>
          )}
          {isExpanded ? (
            <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
          ) : (
            <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
          )}
        </div>
      </button>
      {isExpanded && (
        <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-slate-700')}>
          <div className="space-y-1.5 pt-2">
            {pricingItems.length > 0 ? pricingItems.map((item, idx) => (
              <div key={idx} className={cn('rounded p-2 flex justify-between items-center', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
                <div>
                  <p className={cn('text-xs', isLight ? 'text-stone-800' : 'text-slate-200')}>{item.description}</p>
                  <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>{item.unit}</p>
                </div>
                <p className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">
                  ${typeof item.price === 'number' ? item.price.toFixed(6) : item.price}
                </p>
              </div>
            )) : (
              <p className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>No pricing available</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// Pricing group icons and labels
const pricingGroupInfo = {
  'On-Demand': { icon: '🚀', label: 'On-Demand' },
  'On-Demand Global': { icon: '🌐', label: 'On-Demand Global' },
  'On-Demand Long Context': { icon: '📚', label: 'On-Demand Long Context' },
  'Batch': { icon: '📦', label: 'Batch' },
  'Batch Global': { icon: '🌍', label: 'Batch Global' },
  'Batch Long Context': { icon: '📖', label: 'Batch Long Context' },
  'Batch Long Context Global': { icon: '🌏', label: 'Batch Long Context Global' },
  'Provisioned Throughput': { icon: '⚡', label: 'Provisioned Throughput' },
  'Custom Model': { icon: '🎯', label: 'Custom Model' },
}

function PricingTab({ model, getPricingForModel, preferredRegion = 'us-east-1' }) {
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedCategories, setExpandedCategories] = useState({ 'On-Demand': true })
  const [expandedGeos, setExpandedGeos] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'

  // Get pricing from new source (pass full model object for pricing_file_reference matching)
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
  const fullPricing = pricingResult?.fullPricing

  // Fallback to model's embedded pricing if no external pricing
  const legacyPricing = model.model_pricing || model.comprehensive_pricing || {}
  const legacyByRegion = legacyPricing.by_region || {}

  // Process new pricing structure
  const pricingByGroup = {}
  let allRegions = []

  if (fullPricing?.regions) {
    allRegions = Object.keys(fullPricing.regions)

    for (const [region, regionData] of Object.entries(fullPricing.regions)) {
      if (!regionData?.pricing_groups) continue

      for (const [groupName, items] of Object.entries(regionData.pricing_groups)) {
        if (!pricingByGroup[groupName]) {
          pricingByGroup[groupName] = {}
        }
        pricingByGroup[groupName][region] = items
      }
    }
  } else if (Object.keys(legacyByRegion).length > 0) {
    // Use legacy pricing
    allRegions = Object.keys(legacyByRegion)
    for (const region of allRegions) {
      const regionData = legacyByRegion[region]
      const { onDemand, provisioned } = extractRegionPricing(regionData)
      if (onDemand.length > 0) {
        if (!pricingByGroup['On-Demand']) pricingByGroup['On-Demand'] = {}
        pricingByGroup['On-Demand'][region] = onDemand.map(p => ({
          description: p.description,
          price_per_thousand: p.price,
          unit: p.unit
        }))
      }
      if (provisioned.length > 0) {
        if (!pricingByGroup['Provisioned Throughput']) pricingByGroup['Provisioned Throughput'] = {}
        pricingByGroup['Provisioned Throughput'][region] = provisioned.map(p => ({
          description: p.description,
          price_per_thousand: p.price,
          unit: p.unit
        }))
      }
    }
  }

  const pricingGroups = Object.keys(pricingByGroup)
  const consumptionOptions = model.consumption_options || []

  const getGeoForRegion = (region) => {
    if (region.startsWith('us-')) return 'US'
    if (region.startsWith('eu-')) return 'EU'
    if (region.startsWith('ap-')) return 'APAC'
    if (region.startsWith('ca-')) return 'CA'
    if (region.startsWith('sa-')) return 'SA'
    if (region.startsWith('me-') || region.startsWith('il-')) return 'ME'
    if (region.startsWith('af-')) return 'AF'
    return 'Other'
  }

  const geoInfo = {
    'US': { icon: '🇺🇸', name: 'United States' },
    'EU': { icon: '🇪🇺', name: 'Europe' },
    'APAC': { icon: '🌏', name: 'Asia Pacific' },
    'CA': { icon: '🇨🇦', name: 'Canada' },
    'SA': { icon: '🌎', name: 'South America' },
    'ME': { icon: '🏜️', name: 'Middle East' },
    'AF': { icon: '🌍', name: 'Africa' },
    'Other': { icon: '📍', name: 'Other' }
  }

  // Filter regions by search
  const filterRegions = (regions) => {
    if (!searchQuery) return regions
    const query = searchQuery.toLowerCase()
    return regions.filter(region =>
      region.toLowerCase().includes(query) ||
      (regionDisplayNames[region] || '').toLowerCase().includes(query) ||
      getGeoForRegion(region).toLowerCase().includes(query) ||
      (geoInfo[getGeoForRegion(region)]?.name || '').toLowerCase().includes(query)
    )
  }

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({ ...prev, [category]: !prev[category] }))
  }

  const toggleGeo = (key) => {
    setExpandedGeos(prev => ({ ...prev, [key]: !prev[key] }))
  }

  if (allRegions.length === 0) {
    return (
      <div className={cn('text-center py-8', isLight ? 'text-stone-600' : 'text-slate-400')}>
        <p>No pricing information available</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Metrics Banner */}
      <div className="grid grid-cols-3 gap-3 items-stretch">
        <div className={cn(
          'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[60px]',
          isLight
            ? 'bg-amber-50/80 border-amber-100/50 backdrop-blur-sm'
            : 'bg-white/5 border-white/10 backdrop-blur-sm'
        )}>
          <p className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>{pricingGroups.length}</p>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Pricing Types</p>
        </div>
        <div className={cn(
          'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[60px]',
          isLight
            ? 'bg-emerald-50/80 border-emerald-100/50 backdrop-blur-sm'
            : 'bg-white/5 border-white/10 backdrop-blur-sm'
        )}>
          <p className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>{allRegions.length}</p>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Regions</p>
        </div>
        <div className={cn(
          'rounded-lg p-3 text-center border flex flex-col justify-center min-h-[60px]',
          isLight
            ? 'bg-purple-50/80 border-purple-100/50 backdrop-blur-sm'
            : 'bg-white/5 border-white/10 backdrop-blur-sm'
        )}>
          <p className={cn('text-xl font-bold', isLight ? 'text-stone-900' : 'text-white')}>{consumptionOptions.length || pricingGroups.length}</p>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Options</p>
        </div>
      </div>

      {/* Search bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
        <Input
          placeholder="Search by region, geo, or pricing type..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* Scrollable pricing groups */}
      <ScrollArea className="h-[400px]">
        <div className="space-y-3 pr-4">
          {pricingGroups.map(groupName => {
            const groupRegions = Object.keys(pricingByGroup[groupName])
            const filteredRegions = filterRegions(groupRegions)
            if (filteredRegions.length === 0) return null

            const info = pricingGroupInfo[groupName] || { icon: '💰', label: groupName }
            const isExpanded = expandedCategories[groupName]

            // Group regions by geo
            const regionsByGeo = {}
            for (const region of filteredRegions) {
              const geo = getGeoForRegion(region)
              if (!regionsByGeo[geo]) regionsByGeo[geo] = []
              regionsByGeo[geo].push(region)
            }

            return (
              <div key={groupName} className={cn('rounded-lg overflow-hidden', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
                <button
                  className={cn(
                    'w-full flex items-center justify-between p-3 transition-colors',
                    isLight ? 'hover:bg-stone-200' : 'hover:bg-slate-700/50'
                  )}
                  onClick={() => toggleCategory(groupName)}
                >
                  <div className="flex items-center gap-2">
                    <span>{info.icon}</span>
                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{info.label}</span>
                    <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>
                      ({filteredRegions.length} regions)
                    </span>
                  </div>
                  {isExpanded ? (
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                  )}
                </button>
                {isExpanded && (
                  <div className={cn('px-3 pb-3 pt-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-slate-700')}>
                    {['US', 'EU', 'APAC', 'CA', 'SA', 'ME', 'AF', 'Other'].map(geoKey => {
                      const geoRegions = regionsByGeo[geoKey]
                      if (!geoRegions || geoRegions.length === 0) return null

                      const geo = geoInfo[geoKey]
                      const geoExpandKey = `${groupName}_${geoKey}`
                      const isGeoExpanded = expandedGeos[geoExpandKey]

                      return (
                        <div key={geoKey} className={cn(
                          'rounded-lg border overflow-hidden',
                          isLight ? 'bg-stone-50 border-stone-200' : 'bg-[#1a2330] border-slate-600/40'
                        )}>
                          <button
                            className={cn(
                              'w-full flex items-center justify-between p-2 transition-colors',
                              isLight ? 'hover:bg-stone-100' : 'hover:bg-slate-700/50'
                            )}
                            onClick={() => toggleGeo(geoExpandKey)}
                          >
                            <div className="flex items-center gap-2">
                              <span>{geo.icon}</span>
                              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{geo.name}</span>
                              <Badge variant="secondary" className="text-xs">{geoRegions.length} regions</Badge>
                            </div>
                            {isGeoExpanded ? (
                              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                            ) : (
                              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-400')} />
                            )}
                          </button>
                          {isGeoExpanded && (
                            <div className={cn('px-2 pb-2 pt-2 border-t space-y-2', isLight ? 'border-stone-200' : 'border-slate-700')}>
                              {geoRegions.sort().map(region => {
                                const items = pricingByGroup[groupName][region] || []
                                return (
                                  <div key={region} className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#161d26] border border-slate-600/40')}>
                                    <div className="flex items-center gap-2 mb-2">
                                      <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
                                      <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                                        {regionDisplayNames[region] || region}
                                      </span>
                                      <span className={cn('text-xs font-mono', isLight ? 'text-stone-500' : 'text-slate-400')}>
                                        ({region})
                                      </span>
                                    </div>
                                    <div className="space-y-1">
                                      {items.map((item, idx) => {
                                        // Use price_per_thousand for token pricing, price_per_unit for image/other pricing
                                        const price = item.price_per_thousand ?? item.price_per_unit
                                        const unitLabel = item.unit_label || (item.price_per_thousand != null ? '/1K tokens' : `/${item.unit || 'unit'}`)
                                        return (
                                          <div key={idx} className="flex justify-between items-center text-xs">
                                            <span className={cn(isLight ? 'text-stone-600' : 'text-slate-400')}>
                                              {item.description || item.dimension || 'Price'}
                                            </span>
                                            <span className={cn('font-semibold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                                              ${typeof price === 'number' ? price.toFixed(6) : price ?? 'N/A'}
                                              <span className={cn('font-normal ml-1', isLight ? 'text-stone-500' : 'text-slate-500')}>
                                                {unitLabel}
                                              </span>
                                            </span>
                                          </div>
                                        )
                                      })}
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}

export function ModelCardExpanded({
  model,
  open,
  onOpenChange,
  onToggleFavorite,
  isFavorite = false,
  getPricingForModel,
  preferredRegion = 'us-east-1',
}) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  if (!model) return null

  const isActive = model.model_status === 'ACTIVE'

  return (
    <TooltipProvider>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-6xl max-h-[90vh]">
          <DialogHeader>
            <div className="flex items-center gap-2">
              <Badge className={cn('text-xs font-medium px-2 py-0.5', isLight ? 'text-[#faf9f5]' : 'text-white', getProviderColor(model.model_provider))}>
                {model.model_provider}
              </Badge>
              <Badge variant={isActive ? 'success' : 'warning'} className="text-xs px-2 py-0.5">
                {isActive ? 'Active' : 'Legacy'}
              </Badge>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <DialogTitle className="text-xl flex-1">
                {model.model_name || model.model_id}
              </DialogTitle>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 shrink-0"
                onClick={() => onToggleFavorite?.(model.model_id)}
              >
                <Star
                  className={cn(
                    'h-5 w-5',
                    isFavorite ? 'fill-yellow-500 text-yellow-500' : 'text-slate-400'
                  )}
                />
              </Button>
            </div>
            <DialogDescription className="font-mono">
              {model.model_id}
            </DialogDescription>
          </DialogHeader>

          <Separator />

          <Tabs defaultValue="specs" className="mt-2">
            <TabsList className="w-full justify-start">
              <TabsTrigger value="specs" className="flex-1">Technical Specs</TabsTrigger>
              <TabsTrigger value="quotas" className="flex-1">Service Quotas</TabsTrigger>
              <TabsTrigger value="pricing" className="flex-1">Pricing</TabsTrigger>
            </TabsList>

            <TabsContent value="specs" className="mt-4">
              <SpecsTab model={model} />
            </TabsContent>

            <TabsContent value="quotas" className="mt-4">
              <QuotasTab model={model} />
            </TabsContent>

            <TabsContent value="pricing" className="mt-4">
              <PricingTab model={model} getPricingForModel={getPricingForModel} preferredRegion={preferredRegion} />
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  )
}
