import { useState } from 'react'
import { Star, Globe, Zap, MessageSquare, Image, FileText, Video, Mic, Check, X, ChevronDown, ChevronRight, Search, Database, Languages, Cpu, Layers, Package, Server, ExternalLink, Copy, DollarSign, GitCompareArrows } from 'lucide-react'
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
  default: 'bg-[#6d6e72]',
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

// Copyable model ID for expanded view
function CopyableModelIdExpanded({ modelId, isLight }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async (e) => {
    e.stopPropagation()
    await navigator.clipboard.writeText(modelId)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          onClick={handleCopy}
          className={cn(
            'flex items-center gap-1.5 text-xs font-mono transition-colors group',
            isLight
              ? 'text-stone-500 hover:text-stone-700'
              : 'text-[#9a9b9f] hover:text-[#c0c1c5]'
          )}
        >
          <span>{modelId}</span>
          {copied ? (
            <Check className="h-3.5 w-3.5 text-emerald-500" />
          ) : (
            <Copy className={cn(
              'h-3.5 w-3.5 opacity-0 group-hover:opacity-100 transition-opacity',
              isLight ? 'text-stone-400' : 'text-[#6d6e72]'
            )} />
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent>
        <p>{copied ? 'Copied!' : 'Click to copy model ID'}</p>
      </TooltipContent>
    </Tooltip>
  )
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
          <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
        ) : (
          <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
        )}
      </button>
      {isExpanded && (
        <div className={cn(
          'px-3 pb-3 pt-3 border-t',
          isLight
            ? 'border-stone-200/80 bg-white/60 backdrop-blur-sm'
            : 'border-[#373a40] bg-[#25262b]/50 backdrop-blur-sm'
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
      <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>
        Available in {regions.length} regions across {Object.keys(grouped).length} geographic areas
      </p>
      {Object.entries(grouped).map(([groupKey, groupRegions]) => {
        const groupInfo = geoGroups[groupKey] || { name: groupKey, icon: '🌐' }
        const isExpanded = expandedGroups[groupKey]

        return (
          <div key={groupKey} className={cn(
            'rounded-lg border overflow-hidden',
            isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]'
          )}>
            <button
              className={cn(
                'w-full flex items-center justify-between p-2 transition-colors',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-[#2c2d32]'
              )}
              onClick={() => toggleGroup(groupKey)}
            >
              <div className="flex items-center gap-2">
                <span>{groupInfo.icon}</span>
                <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{groupInfo.name}</span>
                <Badge variant="secondary" className="text-xs">{groupRegions.length} regions</Badge>
              </div>
              {isExpanded ? (
                <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
              ) : (
                <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
              )}
            </button>
            {isExpanded && (
              <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-[#373a40]')}>
                <div className="flex flex-wrap gap-1.5 pt-2">
                  {groupRegions.sort().map(region => (
                    <Badge key={region} variant="outline" className="text-xs">
                      {regionDisplayNames[region] || region} <span className={cn('font-mono ml-1', isLight ? 'text-stone-500' : 'text-[#c0c1c5]')}>({region})</span>
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
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {crisData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Supported</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-[#b0b1b5]')}>Not Supported</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Total Profiles</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{crisData.profiles_count || profiles.length}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Source Regions</p>
          <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{sourceRegions.length || Object.keys(profilesByRegion).length}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Unique Endpoints</p>
          <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{uniqueProfileIds.size}</p>
        </div>
      </div>

      {/* CRIS Endpoints grouped by source region */}
      {crisData.supported && profiles.length > 0 && (
        <div className="space-y-3">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>CRIS Endpoints by Source Region</p>

          {/* Global Endpoints Group */}
          {globalProfiles.length > 0 && (
            <div className={cn(
              'rounded-lg border overflow-hidden',
              isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]'
            )}>
              <button
                className={cn(
                  'w-full flex items-center justify-between p-3 transition-colors',
                  isLight ? 'hover:bg-stone-50' : 'hover:bg-[#2c2d32]'
                )}
                onClick={() => toggleRegion('geo_Global')}
              >
                <div className="flex items-center gap-2">
                  <span>🌐</span>
                  <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>Global Endpoints</span>
                  <Badge variant="info" className="text-xs">{globalProfiles.length} endpoints</Badge>
                </div>
                {expandedRegions['geo_Global'] ? (
                  <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
                ) : (
                  <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
                )}
              </button>
              {expandedRegions['geo_Global'] && (
                <div className={cn('px-3 pb-3 pt-3 border-t space-y-2', isLight ? 'border-stone-200' : 'border-[#373a40]')}>
                  {globalProfiles.map(({ profile, regions }, idx) => (
                    <div key={`${profile.profile_id}-${idx}`} className={cn(
                      'rounded p-2',
                      isLight ? 'bg-stone-50 border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]'
                    )}>
                      <p className={cn('text-sm font-medium', isLight ? 'text-stone-900' : 'text-white')}>
                        {profile.profile_name}
                      </p>
                      <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-[#c0c1c5]')}>
                        {profile.profile_id}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="secondary" className="text-[10px]">{profile.type || 'inference'}</Badge>
                      </div>
                      {profile.description && (
                        <p className={cn('text-xs mt-1.5', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>
                          {profile.description}
                        </p>
                      )}
                      {regions.length > 0 && (
                        <div className="mt-2">
                          <div className="flex flex-wrap gap-1">
                            {regions.sort().map(region => (
                              <Tooltip key={region} delayDuration={200}>
                                <TooltipTrigger asChild>
                                  <Badge variant="secondary" className="text-[10px] cursor-default">
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
                isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]'
              )}>
                <button
                  className={cn(
                    'w-full flex items-center justify-between p-3 transition-colors',
                    isLight ? 'hover:bg-stone-50' : 'hover:bg-[#2c2d32]'
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
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
                  )}
                </button>
                {isGeoExpanded && (
                  <div className={cn('px-3 pb-3 pt-3 border-t space-y-2', isLight ? 'border-stone-200' : 'border-[#373a40]')}>
                    {geoEndpoints.map(({ profile, regions }, idx) => (
                      <div key={`${profile.profile_id}-${idx}`} className={cn(
                        'rounded p-2',
                        isLight ? 'bg-stone-50 border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]'
                      )}>
                        <p className={cn('text-sm font-medium', isLight ? 'text-stone-900' : 'text-white')}>
                          {profile.profile_name}
                        </p>
                        <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-[#c0c1c5]')}>
                          {profile.profile_id}
                        </p>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="secondary" className="text-[10px]">{profile.type || 'inference'}</Badge>
                        </div>
                        {profile.description && (
                          <p className={cn('text-xs mt-1.5', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>
                            {profile.description}
                          </p>
                        )}
                        {regions.length > 0 && (
                          <div className="mt-2">
                            <div className="flex flex-wrap gap-1">
                              {regions.sort().map(region => (
                                <Tooltip key={region} delayDuration={200}>
                                  <TooltipTrigger asChild>
                                    <Badge variant="secondary" className="text-[10px] cursor-default">
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
      <div className="grid grid-cols-2 gap-2">
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {batchData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Supported</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-[#b0b1b5]')}>Not Supported</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Regions</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{regions.length}</p>
        </div>
      </div>

      {/* Regions grouped by geography */}
      {batchData.supported && regions.length > 0 && (
        <div className={cn(
          'rounded-lg border overflow-hidden',
          isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]'
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between p-2 transition-colors',
              isLight ? 'hover:bg-stone-50' : 'hover:bg-[#2c2d32]'
            )}
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <div className="flex items-center gap-2">
              <Globe className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>Available Regions</span>
              <Badge variant="info" className="text-xs">{regions.length} regions</Badge>
            </div>
            {isExpanded ? (
              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
            ) : (
              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
            )}
          </button>
          {isExpanded && (
            <div className={cn('px-2 pb-2 pt-2 space-y-2 border-t', isLight ? 'border-stone-200' : 'border-[#373a40]')}>
              {Object.entries(grouped).map(([geoKey, geoRegions]) => {
                const geoInfo = geoGroups[geoKey] || { name: geoKey, icon: '🌐' }
                return (
                  <div key={geoKey}>
                    <p className={cn('text-xs mb-1', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>
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
    <ScrollArea className="h-full">
      <div className="p-6">
        {/* Two-column grid layout for better use of space */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-4">
            {/* Input & Output Modalities */}
            <CollapsibleSection title="Input & Output Modalities" icon={Layers} defaultExpanded={true}>
              <div className="space-y-3">
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Input Modalities</p>
                  <div className="flex flex-wrap gap-2">
                    {inputModalities.length > 0 ? inputModalities.map(mod => {
                      const Icon = modalityIcons[mod] || MessageSquare
                      return (
                        <Badge key={mod} className={cn(isLight ? 'text-[#faf9f5] bg-amber-700' : 'text-white bg-[#1A9E7A]')}>
                          <Icon className="h-3 w-3 mr-1" />{mod}
                        </Badge>
                      )
                    }) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-[#b0b1b5]')}>None specified</span>}
                  </div>
                </div>
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Output Modalities</p>
                  <div className="flex flex-wrap gap-2">
                    {outputModalities.length > 0 ? outputModalities.map(mod => {
                      const Icon = modalityIcons[mod] || MessageSquare
                      return (
                        <Badge key={mod} className={cn('bg-emerald-600', isLight ? 'text-[#faf9f5]' : 'text-white')}>
                          <Icon className="h-3 w-3 mr-1" />{mod}
                        </Badge>
                      )
                    }) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-[#b0b1b5]')}>None specified</span>}
                  </div>
                </div>
              </div>
            </CollapsibleSection>

            {/* Capabilities & Use Cases */}
            <CollapsibleSection title="Capabilities & Use Cases" icon={Cpu} defaultExpanded={true}>
              <div className="space-y-3">
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Capabilities</p>
                  <div className="flex flex-wrap gap-1.5">
                    {capabilities.length > 0 ? capabilities.map(cap => (
                      <Badge key={cap} variant="secondary" className="text-xs">{cap}</Badge>
                    )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-[#b0b1b5]')}>None specified</span>}
                  </div>
                </div>
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Use Cases</p>
                  <div className="flex flex-wrap gap-1.5">
                    {useCases.length > 0 ? useCases.map(uc => (
                      <Badge key={uc} variant="outline" className="text-xs">{uc}</Badge>
                    )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-[#b0b1b5]')}>None specified</span>}
                  </div>
                </div>
              </div>
            </CollapsibleSection>

            {/* Documentation & Resources */}
            <CollapsibleSection title="Documentation & Resources" icon={FileText} defaultExpanded={true}>
              <div className="space-y-2">
                {Object.keys(documentationLinks).length > 0 ? (
                  <div className="flex flex-col gap-2">
                    {documentationLinks.aws_bedrock_guide && (
                      <a href={documentationLinks.aws_bedrock_guide} target="_blank" rel="noopener noreferrer"
                         className={cn('flex items-center gap-2 text-sm hover:underline', isLight ? 'text-blue-600' : 'text-blue-400')}>
                        <ExternalLink className="h-3.5 w-3.5" />
                        AWS Bedrock Guide
                      </a>
                    )}
                    {documentationLinks.pricing_guide && (
                      <a href={documentationLinks.pricing_guide} target="_blank" rel="noopener noreferrer"
                         className={cn('flex items-center gap-2 text-sm hover:underline', isLight ? 'text-blue-600' : 'text-blue-400')}>
                        <ExternalLink className="h-3.5 w-3.5" />
                        Pricing Guide
                      </a>
                    )}
                    {documentationLinks.provider_guide && (
                      <a href={documentationLinks.provider_guide} target="_blank" rel="noopener noreferrer"
                         className={cn('flex items-center gap-2 text-sm hover:underline', isLight ? 'text-blue-600' : 'text-blue-400')}>
                        <ExternalLink className="h-3.5 w-3.5" />
                        Provider Documentation
                      </a>
                    )}
                  </div>
                ) : (
                  <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-[#b0b1b5]')}>No documentation links available</span>
                )}
              </div>
            </CollapsibleSection>

            {/* Consumption & Deployment Options */}
            <CollapsibleSection title="Consumption & Deployment" icon={Server} defaultExpanded={true}>
              <div className="space-y-3">
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Consumption Options</p>
                  <div className="flex flex-wrap gap-1.5">
                    {consumptionOptions.length > 0 ? consumptionOptions.map(opt => {
                      const labels = {
                        'on_demand': 'On-Demand',
                        'batch': 'Batch',
                        'provisioned': 'Provisioned',
                        'provisioned_throughput': 'Provisioned Throughput',
                        'cross_region_inference': 'Cross-Region Inference'
                      }
                      return (
                        <Badge key={opt} variant="info" className="text-xs">
                          {labels[opt] || opt}
                        </Badge>
                      )
                    }) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-[#b0b1b5]')}>Not specified</span>}
                  </div>
                </div>
                {inferenceTypes.length > 0 && (
                  <div>
                    <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Inference Types</p>
                    <div className="flex flex-wrap gap-1.5">
                      {inferenceTypes.map(type => (
                        <Badge key={type} variant="secondary" className="text-xs">{type}</Badge>
                      ))}
                    </div>
                  </div>
                )}
                {customizations.length > 0 && (
                  <div>
                    <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>Customizations</p>
                    <div className="flex flex-wrap gap-1.5">
                      {customizations.map(custom => (
                        <Badge key={custom} variant="outline" className="text-xs">{custom}</Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CollapsibleSection>
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            {/* Regional Availability */}
            <CollapsibleSection title="Regional Availability" icon={Globe} defaultExpanded={true}>
              <RegionalAvailabilityGrouped regions={regions} />
            </CollapsibleSection>

            {/* Cross-Region Inference */}
            <CollapsibleSection title="Cross-Region Inference" icon={Globe} defaultExpanded={true}>
              <CrossRegionInferenceSection crisData={crisData} />
            </CollapsibleSection>

            {/* Batch Inference Support */}
            <CollapsibleSection title="Batch Inference Support" icon={Package} defaultExpanded={true}>
              <BatchInferenceSection batchData={batchData} />
            </CollapsibleSection>
          </div>
        </div>
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
      isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-2 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-[#2c2d32]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{regionDisplayNames[region] || region}</span>
          <span className={cn('text-xs font-mono', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>({region})</span>
          <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#c0c1c5]')}>- {Array.isArray(regionQuotas) ? regionQuotas.length : 0} quotas</span>
        </div>
        {isExpanded ? (
          <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
        ) : (
          <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
        )}
      </button>
      {isExpanded && (
        <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-[#373a40]')}>
          {Array.isArray(regionQuotas) && regionQuotas.length > 0 ? (
            <div className="space-y-1.5 pt-2">
              {regionQuotas.map((quota, idx) => (
                <div key={idx} className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                  <div className="flex justify-between items-start gap-3">
                    <div className="flex-1 min-w-0">
                      <p className={cn('text-xs leading-relaxed', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>
                        {quota.quota_name || 'Unknown quota'}
                      </p>
                      <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-[#c0c1c5]')}>
                        {quota.quota_code || ''}
                      </p>
                    </div>
                    <div className="text-right flex-shrink-0 min-w-[80px]">
                      <p className={cn('text-sm font-semibold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                        {formatNumber(quota.value)}
                      </p>
                      {showAdjustable && (
                        <p className={cn('text-[10px] mt-0.5', isLight ? 'text-stone-500' : 'text-[#c0c1c5]')}>
                          {quota.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className={cn('text-sm pt-2', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>No quotas defined</p>
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
  general: { name: 'General Limits', icon: '⚙️', color: 'text-[#b0b1b5]' },
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
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const quotas = model.model_service_quotas || {}
  const allRegions = Object.keys(quotas)

  const geoInfo = {
    'US': { icon: '🇺🇸', name: 'United States' },
    'EU': { icon: '🇪🇺', name: 'Europe' },
    'APAC': { icon: '🌏', name: 'Asia Pacific' },
    'CA': { icon: '🇨🇦', name: 'Canada' },
    'SA': { icon: '🌎', name: 'South America' },
    'ME': { icon: '🏜️', name: 'Middle East' },
    'Other': { icon: '📍', name: 'Other' }
  }

  const getGeoForRegion = (region) => {
    if (region.startsWith('us-')) return 'US'
    if (region.startsWith('eu-')) return 'EU'
    if (region.startsWith('ap-')) return 'APAC'
    if (region.startsWith('ca-')) return 'CA'
    if (region.startsWith('sa-')) return 'SA'
    if (region.startsWith('me-') || region.startsWith('il-')) return 'ME'
    return 'Other'
  }

  // Calculate statistics and categorize quotas
  const categorizedQuotas = {}

  for (const region of allRegions) {
    const regionQuotas = quotas[region] || []
    for (const quota of regionQuotas) {
      const category = categorizeQuota(quota.quota_name || '')
      if (!categorizedQuotas[category]) categorizedQuotas[category] = {}
      if (!categorizedQuotas[category][region]) categorizedQuotas[category][region] = []
      categorizedQuotas[category][region].push(quota)
    }
  }

  if (allRegions.length === 0) {
    return (
      <div className={cn('text-center py-8', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>
        <p>No quota information available</p>
      </div>
    )
  }

  // Get unique quota types for a category (across all regions), with search filter
  const getUniqueQuotaTypes = (categoryData) => {
    const quotaTypes = new Map()
    const query = searchQuery.toLowerCase()
    for (const [region, regionQuotas] of Object.entries(categoryData)) {
      for (const quota of regionQuotas) {
        // Apply search filter
        if (query) {
          const regionName = (regionDisplayNames[region] || '').toLowerCase()
          const geo = getGeoForRegion(region).toLowerCase()
          const geoName = (geoInfo[getGeoForRegion(region)]?.name || '').toLowerCase()
          const quotaName = (quota.quota_name || '').toLowerCase()
          const quotaCode = (quota.quota_code || '').toLowerCase()
          const matches = region.toLowerCase().includes(query) ||
                         regionName.includes(query) ||
                         geo.includes(query) ||
                         geoName.includes(query) ||
                         quotaName.includes(query) ||
                         quotaCode.includes(query)
          if (!matches) continue
        }
        const key = quota.quota_name
        if (!quotaTypes.has(key)) {
          quotaTypes.set(key, {
            name: quota.quota_name,
            code: quota.quota_code,
            adjustable: quota.adjustable,
            regions: []
          })
        }
        quotaTypes.get(key).regions.push({ region, value: quota.value })
      }
    }
    return Array.from(quotaTypes.values())
  }

  // Group regions by geo
  const groupRegionsByGeo = (regions) => {
    const byGeo = {}
    for (const r of regions) {
      const geo = getGeoForRegion(r.region)
      if (!byGeo[geo]) byGeo[geo] = []
      byGeo[geo].push(r)
    }
    return byGeo
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-6">
        {/* Search Bar */}
        <div className="mb-6">
          <div className="relative max-w-md">
            <Search className={cn('absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4', isLight ? 'text-stone-400' : 'text-[#6d6e72]')} />
            <Input
              placeholder="Search by region, geo (US, Europe...), or quota code..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>

        {/* Two-column grid layout matching Technical Specs */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-4">
            {/* On-Demand Quotas */}
            {categorizedQuotas['on_demand'] && Object.keys(categorizedQuotas['on_demand']).length > 0 && (
              <CollapsibleSection title="On-Demand Inference" icon={Zap} defaultExpanded={true}>
                <div className="space-y-2">
                  {getUniqueQuotaTypes(categorizedQuotas['on_demand']).map((quotaType, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <div className="flex-1 min-w-0">
                          <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{quotaType.name}</p>
                          <p className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>{quotaType.code}</p>
                        </div>
                        <Badge variant={quotaType.adjustable ? 'success' : 'secondary'} className="text-[10px]">
                          {quotaType.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(quotaType.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">{formatNumber(r.value)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Batch Quotas */}
            {categorizedQuotas['batch'] && Object.keys(categorizedQuotas['batch']).length > 0 && (
              <CollapsibleSection title="Batch Inference" icon={Layers} defaultExpanded={true}>
                <div className="space-y-2">
                  {getUniqueQuotaTypes(categorizedQuotas['batch']).map((quotaType, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <div className="flex-1 min-w-0">
                          <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{quotaType.name}</p>
                          <p className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>{quotaType.code}</p>
                        </div>
                        <Badge variant={quotaType.adjustable ? 'success' : 'secondary'} className="text-[10px]">
                          {quotaType.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(quotaType.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">{formatNumber(r.value)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            {/* Cross-Region Quotas */}
            {categorizedQuotas['cross_region'] && Object.keys(categorizedQuotas['cross_region']).length > 0 && (
              <CollapsibleSection title="Cross-Region Inference" icon={Globe} defaultExpanded={true}>
                <div className="space-y-2">
                  {getUniqueQuotaTypes(categorizedQuotas['cross_region']).map((quotaType, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <div className="flex-1 min-w-0">
                          <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{quotaType.name}</p>
                          <p className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>{quotaType.code}</p>
                        </div>
                        <Badge variant={quotaType.adjustable ? 'success' : 'secondary'} className="text-[10px]">
                          {quotaType.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(quotaType.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">{formatNumber(r.value)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Provisioned Quotas */}
            {categorizedQuotas['provisioned'] && Object.keys(categorizedQuotas['provisioned']).length > 0 && (
              <CollapsibleSection title="Provisioned Throughput" icon={Server} defaultExpanded={true}>
                <div className="space-y-2">
                  {getUniqueQuotaTypes(categorizedQuotas['provisioned']).map((quotaType, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <div className="flex-1 min-w-0">
                          <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{quotaType.name}</p>
                          <p className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>{quotaType.code}</p>
                        </div>
                        <Badge variant={quotaType.adjustable ? 'success' : 'secondary'} className="text-[10px]">
                          {quotaType.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(quotaType.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">{formatNumber(r.value)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Customization Quotas */}
            {categorizedQuotas['customization'] && Object.keys(categorizedQuotas['customization']).length > 0 && (
              <CollapsibleSection title="Customization" icon={Cpu} defaultExpanded={true}>
                <div className="space-y-2">
                  {getUniqueQuotaTypes(categorizedQuotas['customization']).map((quotaType, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <div className="flex-1 min-w-0">
                          <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{quotaType.name}</p>
                          <p className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>{quotaType.code}</p>
                        </div>
                        <Badge variant={quotaType.adjustable ? 'success' : 'secondary'} className="text-[10px]">
                          {quotaType.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(quotaType.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">{formatNumber(r.value)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* General Quotas */}
            {categorizedQuotas['general'] && Object.keys(categorizedQuotas['general']).length > 0 && (
              <CollapsibleSection title="General" icon={FileText} defaultExpanded={true}>
                <div className="space-y-2">
                  {getUniqueQuotaTypes(categorizedQuotas['general']).map((quotaType, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <div className="flex-1 min-w-0">
                          <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{quotaType.name}</p>
                          <p className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>{quotaType.code}</p>
                        </div>
                        <Badge variant={quotaType.adjustable ? 'success' : 'secondary'} className="text-[10px]">
                          {quotaType.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(quotaType.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">{formatNumber(r.value)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}
          </div>
        </div>
      </div>
    </ScrollArea>
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
      isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-2 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-[#2c2d32]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{regionDisplayNames[region] || region}</span>
          <span className={cn('text-xs font-mono', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>({region})</span>
        </div>
        <div className="flex items-center gap-3">
          {inputItem && outputItem && (
            <span className="text-xs text-emerald-600 dark:text-emerald-400">
              ${inputItem.price.toFixed(4)} / ${outputItem.price.toFixed(4)}
            </span>
          )}
          {isExpanded ? (
            <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
          ) : (
            <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')} />
          )}
        </div>
      </button>
      {isExpanded && (
        <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-[#373a40]')}>
          <div className="space-y-1.5 pt-2">
            {pricingItems.length > 0 ? pricingItems.map((item, idx) => (
              <div key={idx} className={cn('rounded p-2 flex justify-between items-center', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                <div>
                  <p className={cn('text-xs', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{item.description}</p>
                  <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#c0c1c5]')}>{item.unit}</p>
                </div>
                <p className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">
                  ${typeof item.price === 'number' ? item.price.toFixed(6) : item.price}
                </p>
              </div>
            )) : (
              <p className={cn('text-sm', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>No pricing available</p>
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
  const { theme } = useTheme()
  const isLight = theme === 'light'

  // Get pricing from new source
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
  const fullPricing = pricingResult?.fullPricing

  // Fallback to model's embedded pricing
  const legacyPricing = model.model_pricing || model.comprehensive_pricing || {}
  const legacyByRegion = legacyPricing.by_region || {}

  // Process pricing structure
  const pricingByGroup = {}
  let allRegions = []

  if (fullPricing?.regions) {
    allRegions = Object.keys(fullPricing.regions)
    for (const [region, regionData] of Object.entries(fullPricing.regions)) {
      if (!regionData?.pricing_groups) continue
      for (const [groupName, items] of Object.entries(regionData.pricing_groups)) {
        if (!pricingByGroup[groupName]) pricingByGroup[groupName] = {}
        pricingByGroup[groupName][region] = items
      }
    }
  } else if (Object.keys(legacyByRegion).length > 0) {
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

  const pricingGroupOrder = ['On-Demand', 'On-Demand Long Context', 'On-Demand Global', 'Batch', 'Batch Long Context', 'Batch Global', 'Provisioned Throughput', 'Custom Model']
  const pricingGroups = Object.keys(pricingByGroup).sort((a, b) => {
    const indexA = pricingGroupOrder.indexOf(a)
    const indexB = pricingGroupOrder.indexOf(b)
    if (indexA !== -1 && indexB !== -1) return indexA - indexB
    if (indexA !== -1) return -1
    if (indexB !== -1) return 1
    return a.localeCompare(b)
  })

  const geoInfo = {
    'US': { icon: '🇺🇸', name: 'United States' },
    'EU': { icon: '🇪🇺', name: 'Europe' },
    'APAC': { icon: '🌏', name: 'Asia Pacific' },
    'CA': { icon: '🇨🇦', name: 'Canada' },
    'SA': { icon: '🌎', name: 'South America' },
    'ME': { icon: '🏜️', name: 'Middle East' },
    'Other': { icon: '📍', name: 'Other' }
  }

  const getGeoForRegion = (region) => {
    if (region.startsWith('us-')) return 'US'
    if (region.startsWith('eu-')) return 'EU'
    if (region.startsWith('ap-')) return 'APAC'
    if (region.startsWith('ca-')) return 'CA'
    if (region.startsWith('sa-')) return 'SA'
    if (region.startsWith('me-') || region.startsWith('il-')) return 'ME'
    return 'Other'
  }

  // Group pricing items by description across regions, with search filter
  const getPricingItems = (groupData) => {
    const items = new Map()
    const query = searchQuery.toLowerCase()
    for (const [region, regionItems] of Object.entries(groupData)) {
      for (const item of regionItems) {
        // Apply search filter
        if (query) {
          const regionName = (regionDisplayNames[region] || '').toLowerCase()
          const geo = getGeoForRegion(region).toLowerCase()
          const geoName = (geoInfo[getGeoForRegion(region)]?.name || '').toLowerCase()
          const description = (item.description || item.dimension || '').toLowerCase()
          const matches = region.toLowerCase().includes(query) ||
                         regionName.includes(query) ||
                         geo.includes(query) ||
                         geoName.includes(query) ||
                         description.includes(query)
          if (!matches) continue
        }
        const key = item.description || item.dimension || 'Price'
        if (!items.has(key)) {
          items.set(key, { description: key, regions: [] })
        }
        items.get(key).regions.push({
          region,
          price: item.price_per_thousand ?? item.price_per_unit,
          unit: item.unit_label || (item.price_per_thousand != null ? '/1K tokens' : `/${item.unit || 'unit'}`)
        })
      }
    }
    return Array.from(items.values())
  }

  // Group regions by geo
  const groupRegionsByGeo = (regions) => {
    const byGeo = {}
    for (const r of regions) {
      const geo = getGeoForRegion(r.region)
      if (!byGeo[geo]) byGeo[geo] = []
      byGeo[geo].push(r)
    }
    return byGeo
  }

  if (allRegions.length === 0) {
    return (
      <div className={cn('text-center py-8', isLight ? 'text-stone-600' : 'text-[#c0c1c5]')}>
        <p>No pricing information available</p>
      </div>
    )
  }

  // Categorize pricing groups
  const onDemandGroups = pricingGroups.filter(g => g.toLowerCase().includes('on-demand'))
  const batchGroups = pricingGroups.filter(g => g.toLowerCase().includes('batch'))
  const otherGroups = pricingGroups.filter(g => !g.toLowerCase().includes('on-demand') && !g.toLowerCase().includes('batch'))

  return (
    <ScrollArea className="h-full">
      <div className="p-6">
        {/* Search Bar */}
        <div className="mb-6">
          <div className="relative max-w-md">
            <Search className={cn('absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4', isLight ? 'text-stone-400' : 'text-[#6d6e72]')} />
            <Input
              placeholder="Search by region, geo (US, Europe...), or pricing type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>

        {/* Two-column grid layout matching Technical Specs */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-4">
            {/* On-Demand Pricing */}
            {onDemandGroups.map(groupName => (
              <CollapsibleSection key={groupName} title={pricingGroupInfo[groupName]?.label || groupName} icon={Zap} defaultExpanded={groupName === 'On-Demand'}>
                <div className="space-y-2">
                  {getPricingItems(pricingByGroup[groupName]).map((item, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{item.description}</p>
                        <p className={cn('text-xs font-semibold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                          ${item.regions[0]?.price?.toFixed(6) || 'N/A'}
                          <span className={cn('font-normal ml-1', isLight ? 'text-stone-500' : 'text-[#b0b1b5]')}>{item.regions[0]?.unit}</span>
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(item.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">${r.price?.toFixed(6)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            ))}
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            {/* Batch Pricing */}
            {batchGroups.map(groupName => (
              <CollapsibleSection key={groupName} title={pricingGroupInfo[groupName]?.label || groupName} icon={Layers} defaultExpanded={true}>
                <div className="space-y-2">
                  {getPricingItems(pricingByGroup[groupName]).map((item, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{item.description}</p>
                        <p className={cn('text-xs font-semibold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                          ${item.regions[0]?.price?.toFixed(6) || 'N/A'}
                          <span className={cn('font-normal ml-1', isLight ? 'text-stone-500' : 'text-[#b0b1b5]')}>{item.regions[0]?.unit}</span>
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(item.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">${r.price?.toFixed(6)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            ))}

            {/* Other Pricing (Provisioned, Custom, etc.) */}
            {otherGroups.map(groupName => (
              <CollapsibleSection key={groupName} title={pricingGroupInfo[groupName]?.label || groupName} icon={Server} defaultExpanded={true}>
                <div className="space-y-2">
                  {getPricingItems(pricingByGroup[groupName]).map((item, idx) => (
                    <div key={idx} className={cn('rounded-lg p-2.5', isLight ? 'bg-white border border-stone-200' : 'bg-[#1a1b1e] border border-[#373a40]')}>
                      <div className="flex justify-between items-start gap-2 mb-2">
                        <p className={cn('text-xs font-medium', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>{item.description}</p>
                        <p className={cn('text-xs font-semibold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                          ${item.regions[0]?.price?.toFixed(6) || 'N/A'}
                          <span className={cn('font-normal ml-1', isLight ? 'text-stone-500' : 'text-[#b0b1b5]')}>{item.regions[0]?.unit}</span>
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(groupRegionsByGeo(item.regions)).map(([geo, regions]) => (
                          <Tooltip key={geo} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] cursor-default">
                                {geoInfo[geo]?.icon} {regions.length} regions
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-xs">
                              <div className="space-y-1">
                                {regions.map(r => (
                                  <p key={r.region} className="text-xs">
                                    <span className="font-mono">{r.region}</span>: <span className="font-semibold text-emerald-400">${r.price?.toFixed(6)}</span>
                                  </p>
                                ))}
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            ))}
          </div>
        </div>
      </div>
    </ScrollArea>
  )
}

export function ModelCardExpanded({
  model,
  open,
  onOpenChange,
  onToggleFavorite,
  isFavorite = false,
  onToggleCompare,
  isInComparison = false,
  getPricingForModel,
  preferredRegion = 'us-east-1',
}) {
  const [activeTab, setActiveTab] = useState('specs')
  const { theme } = useTheme()
  const isLight = theme === 'light'

  if (!model) return null

  const isActive = model.model_status === 'ACTIVE'

  const contextWindow = model.converse_data?.context_window
  const maxOutput = model.converse_data?.max_output_tokens
  const regions = model.regions_available || []
  const capabilities = model.model_capabilities || []
  const streamingSupported = model.streaming_supported
  const crisSupported = model.cross_region_inference?.supported
  const inputModalities = model.model_modalities?.input_modalities || []
  const outputModalities = model.model_modalities?.output_modalities || []

  // Compute quota stats
  const quotas = model.model_service_quotas || {}
  const quotaRegions = Object.keys(quotas)
  let totalQuotas = 0
  let adjustableQuotas = 0
  const quotaCategories = new Set()
  for (const region of quotaRegions) {
    const regionQuotas = quotas[region] || []
    for (const quota of regionQuotas) {
      totalQuotas++
      if (quota.adjustable) adjustableQuotas++
      quotaCategories.add(categorizeQuota(quota.quota_name || ''))
    }
  }

  // Compute pricing stats
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
  const fullPricing = pricingResult?.fullPricing
  const legacyPricing = model.model_pricing || model.comprehensive_pricing || {}
  const legacyByRegion = legacyPricing.by_region || {}
  let pricingRegions = []
  let pricingTypes = 0
  const consumptionOptions = model.consumption_options || []

  if (fullPricing?.regions) {
    pricingRegions = Object.keys(fullPricing.regions)
    const pricingGroups = new Set()
    for (const regionData of Object.values(fullPricing.regions)) {
      if (regionData?.pricing_groups) {
        for (const groupName of Object.keys(regionData.pricing_groups)) {
          pricingGroups.add(groupName)
        }
      }
    }
    pricingTypes = pricingGroups.size
  } else if (Object.keys(legacyByRegion).length > 0) {
    pricingRegions = Object.keys(legacyByRegion)
    pricingTypes = 2 // On-Demand and Provisioned
  }

  return (
    <TooltipProvider>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-[95vw] w-full max-h-[95vh] h-[95vh] p-0 gap-0 flex flex-col">
          {/* Compact Header */}
          <div className={cn(
            'flex items-center justify-between px-6 py-4 border-b flex-shrink-0',
            isLight ? 'border-stone-200' : 'border-[#373a40]'
          )}>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Badge className={cn('text-xs font-medium px-2 py-0.5', isLight ? 'text-[#faf9f5]' : 'text-white', getProviderColor(model.model_provider))}>
                  {model.model_provider}
                </Badge>
                <Badge variant={isActive ? 'success' : 'warning'} className="text-xs px-2 py-0.5">
                  {isActive ? 'Active' : 'Legacy'}
                </Badge>
              </div>
              <div>
                <h2 className={cn('text-lg font-semibold', isLight ? 'text-stone-900' : 'text-white')}>
                  {model.model_name || model.model_id}
                </h2>
                <CopyableModelIdExpanded modelId={model.model_id.split(':')[0]} isLight={isLight} />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => onToggleCompare?.(model)}
                  >
                    <GitCompareArrows className={cn('h-5 w-5', isInComparison ? 'text-[#1A9E7A]' : 'text-[#c0c1c5]')} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{isInComparison ? 'Remove from comparison' : 'Add to comparison'}</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => onToggleFavorite?.(model.model_id)}
                  >
                    <Star className={cn('h-5 w-5', isFavorite ? 'fill-yellow-500 text-yellow-500' : 'text-[#c0c1c5]')} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{isFavorite ? 'Remove from favorites' : 'Add to favorites'}</p>
                </TooltipContent>
              </Tooltip>
            </div>
          </div>

          {/* Main Content - Two Column Layout */}
          <div className="flex flex-1 min-h-0">
            {/* Left Sidebar - Key Stats */}
            <div className={cn(
              'w-64 flex-shrink-0 border-r p-4 flex flex-col gap-4 overflow-y-auto',
              isLight ? 'bg-stone-50 border-stone-200' : 'bg-[#1a1b1e] border-[#373a40]'
            )}>
              {/* Token Limits - Always shown */}
              <div className="space-y-3">
                <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>
                  Token Limits
                </h3>
                <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                  <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Context Window</p>
                  <p className={cn('text-xl font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
                    {contextWindow ? (contextWindow >= 1000000 ? `${(contextWindow/1000000).toFixed(1)}M` : contextWindow >= 1000 ? `${(contextWindow/1000).toFixed(0)}K` : contextWindow) : 'N/A'}
                  </p>
                </div>
                <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                  <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Max Output</p>
                  <p className={cn('text-xl font-bold', isLight ? 'text-purple-700' : 'text-purple-400')}>
                    {maxOutput ? (maxOutput >= 1000 ? `${(maxOutput/1000).toFixed(0)}K` : maxOutput) : 'N/A'}
                  </p>
                </div>
              </div>

              {/* Tab-specific stats */}
              {activeTab === 'specs' && (
                <>
                  {/* Availability */}
                  <div className="space-y-3">
                    <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>
                      Availability
                    </h3>
                    <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                      <div className="flex items-center justify-between">
                        <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Regions</p>
                        <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{regions.length}</p>
                      </div>
                    </div>
                    <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                      <div className="flex items-center justify-between">
                        <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Capabilities</p>
                        <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{capabilities.length}</p>
                      </div>
                    </div>
                  </div>

                  {/* Features */}
                  <div className="space-y-3">
                    <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>
                      Features
                    </h3>
                    <div className="space-y-2">
                      <div className={cn('flex items-center gap-2 text-sm', streamingSupported ? 'text-emerald-500' : isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
                        {streamingSupported ? <Check className="h-4 w-4" /> : <X className="h-4 w-4" />}
                        <span>Streaming</span>
                      </div>
                      <div className={cn('flex items-center gap-2 text-sm', crisSupported ? 'text-emerald-500' : isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>
                        {crisSupported ? <Check className="h-4 w-4" /> : <X className="h-4 w-4" />}
                        <span>Cross-Region</span>
                      </div>
                    </div>
                  </div>

                  {/* Modalities */}
                  <div className="space-y-3">
                    <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>
                      Modalities
                    </h3>
                    <div className="space-y-2">
                      <div>
                        <p className={cn('text-xs mb-1', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Input</p>
                        <div className="flex flex-wrap gap-1">
                          {inputModalities.map(mod => (
                            <Badge key={mod} variant="secondary" className="text-[10px] px-1.5 py-0">{mod}</Badge>
                          ))}
                          {inputModalities.length === 0 && <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>None</span>}
                        </div>
                      </div>
                      <div>
                        <p className={cn('text-xs mb-1', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Output</p>
                        <div className="flex flex-wrap gap-1">
                          {outputModalities.map(mod => (
                            <Badge key={mod} variant="secondary" className="text-[10px] px-1.5 py-0">{mod}</Badge>
                          ))}
                          {outputModalities.length === 0 && <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>None</span>}
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {activeTab === 'quotas' && (
                <div className="space-y-3">
                  <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>
                    Quota Summary
                  </h3>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Total Quotas</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{totalQuotas}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Adjustable</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-emerald-700' : 'text-emerald-400')}>{adjustableQuotas}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Regions</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-blue-700' : 'text-blue-400')}>{quotaRegions.length}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Categories</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-purple-700' : 'text-purple-400')}>{quotaCategories.size}</p>
                  </div>
                </div>
              )}

              {activeTab === 'pricing' && (
                <div className="space-y-3">
                  <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>
                    Pricing Summary
                  </h3>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Pricing Types</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{pricingTypes}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Regions</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-emerald-700' : 'text-emerald-400')}>{pricingRegions.length}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-[#25262b] border-[#373a40]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Options</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-purple-700' : 'text-purple-400')}>{consumptionOptions.length || pricingTypes}</p>
                  </div>
                  {consumptionOptions.length > 0 && (
                    <div className="pt-2">
                      <p className={cn('text-xs mb-2', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>Consumption</p>
                      <div className="flex flex-wrap gap-1">
                        {consumptionOptions.map(opt => {
                          const labels = { 'on_demand': 'On-Demand', 'batch': 'Batch', 'provisioned': 'Provisioned', 'cross_region_inference': 'Cross-Region' }
                          return <Badge key={opt} variant="info" className="text-[10px]">{labels[opt] || opt}</Badge>
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right Content - Tabs */}
            <div className="flex-1 flex flex-col min-w-0 min-h-0">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
                <TabsList className={cn(
                  'w-full justify-start rounded-none border-b flex-shrink-0 h-auto p-0',
                  isLight ? 'bg-transparent border-stone-200' : 'bg-transparent border-[#373a40]'
                )}>
                  <TabsTrigger value="specs" className="rounded-none border-b-2 border-transparent data-[state=active]:border-current px-6 py-3">
                    Technical Specs
                  </TabsTrigger>
                  <TabsTrigger value="quotas" className="rounded-none border-b-2 border-transparent data-[state=active]:border-current px-6 py-3">
                    Service Quotas
                  </TabsTrigger>
                  <TabsTrigger value="pricing" className="rounded-none border-b-2 border-transparent data-[state=active]:border-current px-6 py-3">
                    Pricing
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="specs" className="flex-1 mt-0 min-h-0 overflow-hidden">
                  <SpecsTab model={model} />
                </TabsContent>

                <TabsContent value="quotas" className="flex-1 mt-0 min-h-0 overflow-hidden">
                  <QuotasTab model={model} />
                </TabsContent>

                <TabsContent value="pricing" className="flex-1 mt-0 min-h-0 overflow-hidden">
                  <PricingTab model={model} getPricingForModel={getPricingForModel} preferredRegion={preferredRegion} />
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  )
}
