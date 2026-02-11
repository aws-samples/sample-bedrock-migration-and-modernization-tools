import { useState } from 'react'
import { Star, Globe, Zap, MessageSquare, Image, FileText, Video, Mic, Check, X, ChevronDown, ChevronRight, Search, Database, Languages, Cpu, Layers, Package, Server, ExternalLink, Copy, DollarSign, GitCompareArrows, Radio } from 'lucide-react'
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

// Provider color mapping - using actual brand colors (Tailwind classes)
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

// Hex colors for inline styles (provider badge contrast)
const providerHexColors = {
  Amazon: '#FF9900',
  Anthropic: '#D4A27F',
  Meta: '#0082FB',
  Mistral: '#F54E42',
  'Mistral AI': '#F54E42',
  Cohere: '#39594D',
  'AI21 Labs': '#6C5CE7',
  AI21: '#6C5CE7',
  'Stability AI': '#7C5CFF',
  Stability: '#7C5CFF',
  Luma: '#6366F1',
  'Luma AI': '#6366F1',
  Writer: '#4A90D9',
  NVIDIA: '#76B900',
  DeepSeek: '#4A90D9',
  Qwen: '#6366F1',
  Google: '#4285F4',
  OpenAI: '#10A37F',
  TwelveLabs: '#6366F1',
  MiniMax: '#6366F1',
  'Moonshot AI': '#6366F1',
  default: '#6d6e72',
}

function getProviderHexColor(provider) {
  return providerHexColors[provider] || providerHexColors.default
}

// Returns '#ffffff' or '#000000' based on background luminance for readable contrast
function getContrastColor(hexColor) {
  if (!hexColor) return '#ffffff'
  const hex = hexColor.replace('#', '')
  const r = parseInt(hex.substring(0, 2), 16)
  const g = parseInt(hex.substring(2, 4), 16)
  const b = parseInt(hex.substring(4, 6), 16)
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  return luminance > 0.75 ? '#000000' : '#ffffff'
}

// Modality icons and labels
const modalityIcons = {
  TEXT: MessageSquare,
  IMAGE: Image,
  DOCUMENT: FileText,
  VIDEO: Video,
  AUDIO: Mic,
  SPEECH: Mic,
}

const modalityLabels = {
  TEXT: 'Text',
  IMAGE: 'Image',
  DOCUMENT: 'Doc',
  VIDEO: 'Video',
  AUDIO: 'Audio',
  SPEECH: 'Speech',
}

// Format snake_case identifiers to Title Case (e.g. "complex_analysis" → "Complex Analysis")
function formatLabel(str) {
  if (!str) return str
  if (str.includes('_')) return str.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  return str
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
    <button
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Click to copy model ID'}
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
          'h-3.5 w-3.5 transition-colors',
          isLight ? 'text-stone-400 group-hover:text-stone-600' : 'text-[#6d6e72] group-hover:text-[#c0c1c5]'
        )} />
      )}
    </button>
  )
}

function CopyableText({ text, isLight, className: extraClass }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async (e) => {
    e.stopPropagation()
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button onClick={handleCopy} className={cn('flex items-center gap-1.5 group', extraClass)}>
          <span className="truncate">{text}</span>
          {copied ? (
            <Check className="h-3 w-3 text-emerald-500 flex-shrink-0" />
          ) : (
            <Copy className={cn(
              'h-3 w-3 flex-shrink-0 transition-colors',
              isLight ? 'text-stone-400 group-hover:text-stone-600' : 'text-[#6d6e72] group-hover:text-[#c0c1c5]'
            )} />
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent><p>{copied ? 'Copied!' : 'Click to copy'}</p></TooltipContent>
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
          <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
        ) : (
          <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
        )}
      </button>
      {isExpanded && (
        <div className={cn(
          'px-3 pb-3 pt-3 border-t',
          isLight
            ? 'border-stone-200/80 bg-white/60 backdrop-blur-sm'
            : 'border-white/[0.06] bg-white/[0.03] backdrop-blur-xl'
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
      <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>
        Available in {regions.length} regions across {Object.keys(grouped).length} geographic areas
      </p>
      {Object.entries(grouped).map(([groupKey, groupRegions]) => {
        const groupInfo = geoGroups[groupKey] || { name: groupKey, icon: '🌐' }
        const isExpanded = expandedGroups[groupKey]

        return (
          <div key={groupKey} className={cn(
            'rounded-lg border overflow-hidden',
            isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
          )}>
            <button
              className={cn(
                'w-full flex items-center justify-between p-2 transition-colors',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
              )}
              onClick={() => toggleGroup(groupKey)}
            >
              <div className="flex items-center gap-2">
                <span>{groupInfo.icon}</span>
                <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{groupInfo.name}</span>
                <Badge variant="secondary" className="text-xs">{groupRegions.length} regions</Badge>
              </div>
              {isExpanded ? (
                <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
              ) : (
                <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
              )}
            </button>
            {isExpanded && (
              <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                <div className="flex flex-wrap gap-1.5 pt-2">
                  {groupRegions.sort().map(region => (
                    <Badge key={region} variant="outline" className="text-xs">
                      {regionDisplayNames[region] || region} <span className={cn('font-mono ml-1', isLight ? 'text-stone-500' : 'text-slate-300')}>({region})</span>
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
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {crisData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Supported</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>Not Supported</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Total Profiles</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{crisData.profiles_count || profiles.length}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Source Regions</p>
          <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{sourceRegions.length || Object.keys(profilesByRegion).length}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Unique Endpoints</p>
          <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{uniqueProfileIds.size}</p>
        </div>
      </div>

      {/* CRIS Endpoints grouped by source region */}
      {crisData.supported && profiles.length > 0 && (
        <div className="space-y-3">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>CRIS Endpoints by Source Region</p>

          {/* Global Endpoints Group */}
          {globalProfiles.length > 0 && (
            <div className={cn(
              'rounded-lg border overflow-hidden',
              isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
            )}>
              <button
                className={cn(
                  'w-full flex items-center justify-between p-3 transition-colors',
                  isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
                )}
                onClick={() => toggleRegion('geo_Global')}
              >
                <div className="flex items-center gap-2">
                  <span>🌐</span>
                  <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>Global Endpoints</span>
                  <Badge variant="info" className="text-xs">{globalProfiles.length} endpoints</Badge>
                </div>
                {expandedRegions['geo_Global'] ? (
                  <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                ) : (
                  <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                )}
              </button>
              {expandedRegions['geo_Global'] && (
                <div className={cn('px-3 pb-3 pt-3 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                  {globalProfiles.map(({ profile, regions }, idx) => (
                    <div key={`${profile.profile_id}-${idx}`} className={cn(
                      'rounded p-2',
                      isLight ? 'bg-stone-50 border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
                    )}>
                      <p className={cn('text-sm font-medium', isLight ? 'text-stone-900' : 'text-white')}>
                        {profile.profile_name}
                      </p>
                      <CopyableText
                        text={profile.profile_id}
                        isLight={isLight}
                        className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500 hover:text-stone-700' : 'text-[#c0c1c5] hover:text-white')}
                      />
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="secondary" className="text-[10px]">{profile.type || 'inference'}</Badge>
                      </div>
                      {profile.description && (
                        <p className={cn('text-xs mt-1.5', isLight ? 'text-stone-600' : 'text-slate-300')}>
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
                isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
              )}>
                <button
                  className={cn(
                    'w-full flex items-center justify-between p-3 transition-colors',
                    isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
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
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                  )}
                </button>
                {isGeoExpanded && (
                  <div className={cn('px-3 pb-3 pt-3 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                    {geoEndpoints.map(({ profile, regions }, idx) => (
                      <div key={`${profile.profile_id}-${idx}`} className={cn(
                        'rounded p-2',
                        isLight ? 'bg-stone-50 border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
                      )}>
                        <p className={cn('text-sm font-medium', isLight ? 'text-stone-900' : 'text-white')}>
                          {profile.profile_name}
                        </p>
                        <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-slate-300')}>
                          {profile.profile_id}
                        </p>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="secondary" className="text-[10px]">{profile.type || 'inference'}</Badge>
                        </div>
                        {profile.description && (
                          <p className={cn('text-xs mt-1.5', isLight ? 'text-stone-600' : 'text-slate-300')}>
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
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {batchData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Supported</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>Not Supported</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Regions</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{regions.length}</p>
        </div>
      </div>

      {/* Regions grouped by geography */}
      {batchData.supported && regions.length > 0 && (
        <div className={cn(
          'rounded-lg border overflow-hidden',
          isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between p-2 transition-colors',
              isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
            )}
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <div className="flex items-center gap-2">
              <Globe className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>Available Regions</span>
              <Badge variant="info" className="text-xs">{regions.length} regions</Badge>
            </div>
            {isExpanded ? (
              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            ) : (
              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            )}
          </button>
          {isExpanded && (
            <div className={cn('px-2 pb-2 pt-2 space-y-2 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
              {Object.entries(grouped).map(([geoKey, geoRegions]) => {
                const geoInfo = geoGroups[geoKey] || { name: geoKey, icon: '🌐' }
                return (
                  <div key={geoKey}>
                    <p className={cn('text-xs mb-1', isLight ? 'text-stone-600' : 'text-slate-300')}>
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
  const languages = model.languages_supported || []
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
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Input Modalities</p>
                  <div className="flex flex-wrap gap-2">
                    {inputModalities.length > 0 ? inputModalities.map(mod => {
                      const Icon = modalityIcons[mod] || MessageSquare
                      return (
                        <Badge key={mod} className={cn(isLight ? 'text-[#faf9f5] bg-amber-700' : 'text-white bg-[#1A9E7A]')}>
                          <Icon className="h-3 w-3 mr-1" />{modalityLabels[mod] || mod}
                        </Badge>
                      )
                    }) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>None specified</span>}
                  </div>
                </div>
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Output Modalities</p>
                  <div className="flex flex-wrap gap-2">
                    {outputModalities.length > 0 ? outputModalities.map(mod => {
                      const Icon = modalityIcons[mod] || MessageSquare
                      return (
                        <Badge key={mod} className={cn('bg-emerald-600', isLight ? 'text-[#faf9f5]' : 'text-white')}>
                          <Icon className="h-3 w-3 mr-1" />{modalityLabels[mod] || mod}
                        </Badge>
                      )
                    }) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>None specified</span>}
                  </div>
                </div>
              </div>
            </CollapsibleSection>

            {/* Capabilities & Use Cases */}
            <CollapsibleSection title="Capabilities & Use Cases" icon={Cpu} defaultExpanded={true}>
              <div className="space-y-3">
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Capabilities</p>
                  <div className="flex flex-wrap gap-1.5">
                    {capabilities.length > 0 ? capabilities.map(cap => (
                      <Badge key={cap} variant="secondary" className="text-xs">{formatLabel(cap)}</Badge>
                    )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>None specified</span>}
                  </div>
                </div>
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Use Cases</p>
                  <div className="flex flex-wrap gap-1.5">
                    {useCases.length > 0 ? useCases.map(uc => (
                      <Badge key={uc} variant="outline" className="text-xs">{formatLabel(uc)}</Badge>
                    )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>None specified</span>}
                  </div>
                </div>
              </div>
            </CollapsibleSection>

            {/* Languages */}
            <CollapsibleSection title="Languages" icon={Languages} defaultExpanded={true}>
              <div className="flex flex-wrap gap-1.5">
                {languages.length > 0 ? languages.map(lang => (
                  <Badge key={lang} variant="secondary" className="text-xs">{lang}</Badge>
                )) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>Not specified</span>}
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
                  <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>No documentation links available</span>
                )}
              </div>
            </CollapsibleSection>

            {/* Consumption & Deployment Options */}
            <CollapsibleSection title="Consumption & Deployment" icon={Server} defaultExpanded={true}>
              <div className="space-y-3">
                <div>
                  <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Consumption Options</p>
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
                    }) : <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>Not specified</span>}
                  </div>
                </div>
                {inferenceTypes.length > 0 && (
                  <div>
                    <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Inference Types</p>
                    <div className="flex flex-wrap gap-1.5">
                      {inferenceTypes.map(type => (
                        <Badge key={type} variant="secondary" className="text-xs">{type}</Badge>
                      ))}
                    </div>
                  </div>
                )}
                {customizations.length > 0 && (
                  <div>
                    <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Customizations</p>
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
      isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-2 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{regionDisplayNames[region] || region}</span>
          <span className={cn('text-xs font-mono', isLight ? 'text-stone-600' : 'text-slate-300')}>({region})</span>
          <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-300')}>- {Array.isArray(regionQuotas) ? regionQuotas.length : 0} quotas</span>
        </div>
        {isExpanded ? (
          <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
        ) : (
          <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
        )}
      </button>
      {isExpanded && (
        <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
          {Array.isArray(regionQuotas) && regionQuotas.length > 0 ? (
            <div className="space-y-1.5 pt-2">
              {regionQuotas.map((quota, idx) => (
                <div key={idx} className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
                  <div className="flex justify-between items-start gap-3">
                    <div className="flex-1 min-w-0">
                      <p className={cn('text-xs leading-relaxed', isLight ? 'text-stone-800' : 'text-[#e4e5e7]')}>
                        {quota.quota_name || 'Unknown quota'}
                      </p>
                      <p className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500' : 'text-slate-300')}>
                        {quota.quota_code || ''}
                      </p>
                    </div>
                    <div className="text-right flex-shrink-0 min-w-[80px]">
                      <p className={cn('text-sm font-semibold', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                        {formatNumber(quota.value)}
                      </p>
                      {showAdjustable && (
                        <p className={cn('text-[10px] mt-0.5', isLight ? 'text-stone-500' : 'text-slate-300')}>
                          {quota.adjustable ? '🔧 Adjustable' : '🔒 Fixed'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className={cn('text-sm pt-2', isLight ? 'text-stone-600' : 'text-slate-300')}>No quotas defined</p>
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
  general: { name: 'General Limits', icon: '⚙️', color: 'text-slate-400' },
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

function simplifyQuotaName(quotaName) {
  if (!quotaName) return 'Unknown quota'
  let name = quotaName.trim()
  const qualifiers = []

  // Extract "(doubled for cross-region calls)" qualifier
  if (/\(doubled for cross-region/i.test(name)) {
    qualifiers.push('2x cross-region')
    name = name.replace(/\s*\(doubled for[^)]*\)/i, '')
  }

  // Strip "(Model customization)" prefix
  name = name.replace(/^\(Model customization\)\s*/i, '')

  // Extract context length qualifier from model ref (e.g. "1M Context Length", "200K Context Length")
  const ctxMatch = name.match(/\b(\d+[kKmM])\s+Context\s+Length/i)
  if (ctxMatch) {
    qualifiers.push(ctxMatch[1].toUpperCase().replace('K', 'K').replace('M', 'M') + ' context')
  }

  // Check for "global" prefix
  if (/^global\s/i.test(name)) {
    qualifiers.push('global')
    name = name.replace(/^global\s+/i, '')
  }

  // Split on "for" and take the first part (the metric description)
  const forParts = name.split(/\bfor\b/i)
  let metric = forParts[0].trim()

  // Strip category prefixes to get the core metric
  metric = metric
    .replace(/^cross[- ]region\s+model\s+inference\s*/i, '')
    .replace(/^on[- ]demand\s+model\s+inference\s*/i, '')
    .replace(/^model\s+invocation\s*/i, '')
    .replace(/^batch\s+inference\s*/i, '')
    .replace(/^no[- ]commitment\s+/i, (() => { qualifiers.push('no commitment'); return '' })())

  // Simplify common metric patterns
  const metricLower = metric.toLowerCase().trim()

  // Tokens per minute/day
  if (/^(max\s+)?tokens\s+per\s+minute$/i.test(metric.trim())) {
    metric = metricLower.startsWith('max') ? 'Max tokens/min' : 'Tokens/min'
  } else if (/^(max\s+)?tokens\s+per\s+day$/i.test(metric.trim())) {
    metric = metricLower.startsWith('max') ? 'Max tokens/day' : 'Tokens/day'
  }
  // Requests per minute
  else if (/^requests\s+per\s+minute$/i.test(metric.trim())) {
    metric = 'Requests/min'
  }
  // Job size (in GB)
  else if (/^job\s+size\s*\(in\s+GB\)/i.test(metric.trim())) {
    metric = 'Job size (GB)'
  }
  // Input file size (in GB)
  else if (/^input\s+file\s+size\s*\(in\s+GB\)/i.test(metric.trim())) {
    metric = 'Input file size (GB)'
  }
  // Records per job / per input file
  else if (/^records\s+per\s+input\s+file\s+per\s.*job$/i.test(metric.trim())) {
    metric = 'Records/input file'
  } else if (/^records\s+per\s.*job$/i.test(metric.trim())) {
    metric = 'Records/job'
  }
  // Min records
  else if (/^minimum\s+(number\s+of\s+)?records\s+per\s.*job/i.test(metric.trim())) {
    metric = 'Min records/job'
  }
  // Concurrent jobs
  else if (/^sum\s+of\s+in-progress/i.test(metric.trim())) {
    metric = 'Concurrent jobs'
  }
  // Model units
  else if (/^model\s+units/i.test(metric.trim())) {
    metric = 'Model units'
  }
  // Active fine-tuning jobs
  else if (/active\s+fine[- ]?tuning\s+jobs/i.test(metric.trim())) {
    metric = 'Active fine-tuning jobs'
  }
  // Custom model count
  else if (/custom\s+models?$/i.test(metric.trim())) {
    metric = 'Custom models'
  }
  // Fallback: capitalize first letter, trim
  else {
    metric = metric.trim()
    if (metric.length > 0) {
      metric = metric.charAt(0).toUpperCase() + metric.slice(1)
    }
  }

  // Remove trailing punctuation
  metric = metric.replace(/[.,;]+$/, '').trim()

  if (qualifiers.length > 0) {
    return `${metric} · ${qualifiers.join(' · ')}`
  }
  return metric || quotaName
}

function QuotaItemsList({ items, isLight }) {
  const [expandedIdx, setExpandedIdx] = useState(null)
  const [copiedIdx, setCopiedIdx] = useState(null)

  const sorted = [...items].sort((a, b) => {
    if (a.adjustable !== b.adjustable) return a.adjustable ? -1 : 1
    return (b.value || 0) - (a.value || 0)
  })

  const handleCopyValue = async (e, quota, idx) => {
    e.stopPropagation()
    const text = `${quota.quota_name}: ${formatNumber(quota.value)}`
    await navigator.clipboard.writeText(text)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(null), 1500)
  }

  return (
    <div>
      {sorted.map((quota, idx) => {
        const isExpanded = expandedIdx === idx
        const label = simplifyQuotaName(quota.quota_name)
        const isAdjustable = quota.adjustable

        return (
          <div key={idx}>
            <button
              onClick={() => setExpandedIdx(isExpanded ? null : idx)}
              className={cn(
                'w-full flex items-center justify-between px-2 py-1.5 text-left transition-colors rounded',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
              )}
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className={cn(
                  'inline-block w-1.5 h-1.5 rounded-full flex-shrink-0',
                  isAdjustable ? 'bg-amber-500' : 'bg-slate-400'
                )} />
                <span className={cn(
                  'text-xs truncate',
                  isLight ? 'text-stone-700' : 'text-[#e4e5e7]'
                )}>{label}</span>
              </div>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                <span className={cn(
                  'text-xs font-mono font-semibold tabular-nums',
                  isLight ? 'text-stone-900' : 'text-emerald-400'
                )}>
                  {formatNumber(quota.value)}
                </span>
                <span className={cn(
                  'text-[9px] px-1.5 py-0.5 rounded-full font-medium',
                  isAdjustable
                    ? (isLight ? 'bg-amber-100 text-amber-700' : 'bg-amber-500/15 text-amber-400')
                    : (isLight ? 'bg-stone-100 text-stone-500' : 'bg-white/[0.06] text-slate-400')
                )}>
                  {isAdjustable ? 'Adjustable' : 'Fixed'}
                </span>
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => handleCopyValue(e, quota, idx)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleCopyValue(e, quota, idx) }}
                  className={cn(
                    'p-0.5 rounded transition-colors',
                    isLight ? 'hover:bg-stone-200' : 'hover:bg-white/[0.08]'
                  )}
                >
                  {copiedIdx === idx ? (
                    <Check className="h-3 w-3 text-emerald-500" />
                  ) : (
                    <Copy className={cn('h-3 w-3', isLight ? 'text-stone-400' : 'text-slate-400')} />
                  )}
                </span>
              </div>
            </button>
            {isExpanded && (
              <div className={cn(
                'mx-2 mb-1.5 px-3 py-2 rounded text-[10px] font-mono leading-relaxed space-y-0.5',
                isLight ? 'text-stone-500 bg-stone-50' : 'text-slate-300 bg-white/[0.02]'
              )}>
                <p className="break-all">
                  <span className={isLight ? 'text-stone-400' : 'text-slate-500'}>name </span>
                  {quota.quota_name}
                </p>
                <p>
                  <span className={isLight ? 'text-stone-400' : 'text-slate-500'}>code </span>
                  {quota.quota_code}
                </p>
                <p>
                  <span className={isLight ? 'text-stone-400' : 'text-slate-500'}>adj{'  '}</span>
                  {isAdjustable ? 'Yes \u2014 can request increase' : 'No \u2014 fixed limit'}
                </p>
                <p>
                  <span className={isLight ? 'text-stone-400' : 'text-slate-500'}>val{'  '}</span>
                  {typeof quota.value === 'number' ? quota.value.toLocaleString() : 'N/A'}
                  {quota.unit && quota.unit !== 'None' ? ` ${quota.unit}` : ''}
                </p>
                {quota.period && Object.keys(quota.period).length > 0 && (
                  <p>
                    <span className={isLight ? 'text-stone-400' : 'text-slate-500'}>per{'  '}</span>
                    {quota.period.value} {quota.period.unit}
                  </p>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function QuotasTab({ model }) {
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedGeos, setExpandedGeos] = useState({})
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

  const geoOrder = ['US', 'EU', 'APAC', 'CA', 'SA', 'ME', 'Other']

  const getGeoForRegion = (region) => {
    if (region.startsWith('us-')) return 'US'
    if (region.startsWith('eu-')) return 'EU'
    if (region.startsWith('ap-')) return 'APAC'
    if (region.startsWith('ca-')) return 'CA'
    if (region.startsWith('sa-')) return 'SA'
    if (region.startsWith('me-') || region.startsWith('il-')) return 'ME'
    return 'Other'
  }

  const toggleGeo = (key) => setExpandedGeos(prev => ({ ...prev, [key]: !prev[key] }))

  // Calculate statistics and categorize quotas by category -> geo -> region
  const categorizedQuotas = {}

  for (const region of allRegions) {
    const regionQuotas = quotas[region] || []
    for (const quota of regionQuotas) {
      const category = categorizeQuota(quota.quota_name || '')
      const geo = getGeoForRegion(region)
      if (!categorizedQuotas[category]) categorizedQuotas[category] = {}
      if (!categorizedQuotas[category][geo]) categorizedQuotas[category][geo] = {}
      if (!categorizedQuotas[category][geo][region]) categorizedQuotas[category][geo][region] = []
      categorizedQuotas[category][geo][region].push(quota)
    }
  }

  if (allRegions.length === 0) {
    return (
      <div className={cn('text-center py-8', isLight ? 'text-stone-600' : 'text-slate-300')}>
        <p>No quota information available</p>
      </div>
    )
  }

  // Filter quotas by search
  const filterQuotas = (geoData) => {
    if (!searchQuery) return geoData
    const query = searchQuery.toLowerCase()
    const filtered = {}
    for (const [geo, regions] of Object.entries(geoData)) {
      const geoName = (geoInfo[geo]?.name || '').toLowerCase()
      for (const [region, regionQuotas] of Object.entries(regions)) {
        const regionName = (regionDisplayNames[region] || '').toLowerCase()
        const matchingQuotas = regionQuotas.filter(q => {
          const quotaName = (q.quota_name || '').toLowerCase()
          const quotaCode = (q.quota_code || '').toLowerCase()
          return region.toLowerCase().includes(query) ||
                 regionName.includes(query) ||
                 geo.toLowerCase().includes(query) ||
                 geoName.includes(query) ||
                 quotaName.includes(query) ||
                 quotaCode.includes(query)
        })
        if (matchingQuotas.length > 0) {
          if (!filtered[geo]) filtered[geo] = {}
          filtered[geo][region] = matchingQuotas
        }
      }
    }
    return filtered
  }

  // Render a category section with geo grouping
  const renderCategorySection = (categoryKey, title, icon) => {
    const categoryData = categorizedQuotas[categoryKey]
    if (!categoryData || Object.keys(categoryData).length === 0) return null

    const filteredData = filterQuotas(categoryData)
    if (Object.keys(filteredData).length === 0) return null

    return (
      <CollapsibleSection title={title} icon={icon} defaultExpanded={true}>
        <div className="space-y-2">
          {geoOrder.map(geo => {
            const geoData = filteredData[geo]
            if (!geoData || Object.keys(geoData).length === 0) return null

            const regionCount = Object.keys(geoData).length
            const quotaCount = Object.values(geoData).flat().length
            const geoKey = `${categoryKey}_${geo}`
            const isGeoExpanded = expandedGeos[geoKey]

            return (
              <div key={geo} className={cn('rounded-lg border overflow-hidden', isLight ? 'bg-stone-50/50 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]')}>
                <button
                  className={cn('w-full flex items-center justify-between p-2.5 transition-colors', isLight ? 'hover:bg-stone-100' : 'hover:bg-white/[0.08]')}
                  onClick={() => toggleGeo(geoKey)}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{geoInfo[geo]?.icon}</span>
                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-800' : 'text-white')}>{geoInfo[geo]?.name}</span>
                    <Badge variant="secondary" className="text-[10px]">{regionCount} regions</Badge>
                    <Badge variant="outline" className="text-[10px]">{quotaCount} quotas</Badge>
                  </div>
                  {isGeoExpanded ? (
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
                  )}
                </button>
                {isGeoExpanded && (
                  <div className={cn('px-2.5 pb-2.5 space-y-2 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                    {Object.entries(geoData).sort().map(([region, regionQuotas]) => (
                      <div key={region} className={cn('rounded-lg p-2 mt-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
                        <div className="flex items-center gap-2 mb-2">
                          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
                          <span className={cn('font-medium text-xs', isLight ? 'text-stone-800' : 'text-white')}>{regionDisplayNames[region] || region}</span>
                          <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-slate-400')}>({region})</span>
                        </div>
                        <QuotaItemsList items={regionQuotas} isLight={isLight} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </CollapsibleSection>
    )
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-6">
        {/* Search Bar */}
        <div className="mb-6">
          <div className="relative max-w-md">
            <Search className={cn('absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
            <Input
              placeholder="Search by region, geo (US, Europe...), or quota code..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>

        {/* Two-column grid layout */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-4">
            {renderCategorySection('on_demand', 'On-Demand Inference', Zap)}
            {renderCategorySection('batch', 'Batch Inference', Layers)}
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            {renderCategorySection('cross_region', 'Cross-Region Inference', Globe)}
            {renderCategorySection('provisioned', 'Provisioned Throughput', Server)}
            {renderCategorySection('customization', 'Customization', Cpu)}
            {renderCategorySection('general', 'General', FileText)}
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

// Simplify verbose AWS pricing descriptions into clean labels.
// Handles all formats: 1P dollar-prefixed, 3P marketplace pipe-delimited,
// dimension strings, and already-clean labels. All matching is keyword-based
// so it scales to any model without hardcoding.
// Returns { label, type } where type is 'input' | 'output' | 'other'
function simplifyPricingDescription(desc, dimension) {
  if (!desc && !dimension) return { label: 'Price', type: 'other' }

  // Combine description + dimension for robust keyword extraction
  const combined = `${dimension || ''} ${desc || ''}`.toLowerCase()
  const dl = (desc || '').toLowerCase()

  // --- Determine direction ---
  // No word boundaries: catches both "input-tokens" and camelCase "InputTokenCount"
  const isOutput = /output|response/i.test(combined)
  const isInput = /input/i.test(combined) && !isOutput
  const type = isOutput ? 'output' : isInput ? 'input' : 'other'
  const direction = isOutput ? 'Output' : isInput ? 'Input' : null

  // --- Non-token billing (no input/output direction) ---
  // Detect billing unit type from "per <unit>" patterns in description
  if (!direction) {
    if (/\bper hour\b|provisioned.?throughput/i.test(combined)) {
      const commit = /no.?commit/i.test(combined) ? ' (No Commit)' :
                     /6.?month/i.test(combined) ? ' (6 Month)' :
                     /1.?month/i.test(combined) ? ' (1 Month)' : ''
      return { label: `Provisioned Throughput${commit}`, type: 'other' }
    }
    if (/\bper\s+(?:\d+\s+)?image\b|\bimages?\s*processed\b/i.test(dl)) return { label: 'Per Image', type: 'other' }
    if (/\bper\s+(?:\d+\s+)?(second|secs)\b|\bvideo.*(second|sec)\b/i.test(dl)) return { label: 'Per Second', type: 'other' }
    if (/\bper video\b/i.test(dl)) return { label: 'Per Video', type: 'other' }
    if (/\bper.*request/i.test(dl)) return { label: 'Per Request', type: 'other' }
    if (/\bmodel.month\b|\bstorage\b/i.test(dl)) return { label: 'Model Storage', type: 'other' }
    if (/\bper (page|pages?\s*processed)\b/i.test(dl)) return { label: 'Per Page', type: 'other' }
    if (/\bper (minute|minutes?\s*processed)\b/i.test(dl)) return { label: 'Per Minute', type: 'other' }
    if (/\btext.?unit/i.test(dl)) return { label: 'Text Units', type: 'other' }
    if (/\bsearch.?unit/i.test(dl)) return { label: 'Search Units', type: 'other' }
    if (/\bnode.?transition/i.test(dl)) return { label: 'Node Transitions', type: 'other' }
    if (/\btpm.?hour\b|\btokens?.per.minute\b/i.test(combined)) return { label: 'Reserved TPM', type: 'other' }
    if (/\bcustom.?model\b/i.test(dl)) return { label: 'Custom Model', type: 'other' }
    if (/\bfield/i.test(dl)) return { label: 'Per Field', type: 'other' }
    if (/\bgrounding\b/i.test(combined)) return { label: 'Grounding', type: 'other' }
    if (/\bcustomiz/i.test(dl)) return { label: 'Customization', type: 'other' }

    // Fallback for no-direction items: strip known prefixes/suffixes
    let label = (desc || dimension || 'Price')
    label = label.replace(/^\$?[\d.,]+\s*(?:USD\s+)?per\s+.+?\s+for\s+/i, '')
    label = label.replace(/\s+in\s+(?:US|EU|Asia|Canada|South|Middle|Africa).+$/i, '')
    label = label.replace(/^AWS Marketplace software usage[^|]*\|/i, '')
    // For multi-pipe marketplace (e.g. "region|metric"): take last segment
    if (label.includes('|')) label = label.split('|').pop().trim()
    label = label.replace(/^Million\s+/i, '')
    label = label.replace(/\s+(Regional|Global)$/i, '')
    label = label.replace(/^Price per \d+\s*/i, '')
    label = label.replace(/^[A-Z]{2,4}\d?-/, '')
    return { label: label.trim() || 'Price', type: 'other' }
  }

  // --- Reserved capacity with direction (TPM pricing) ---
  if (/\btpm\b|\breserved\b|\btokens?.per.minute\b/i.test(combined)) {
    const commitMatch = combined.match(/(\d+)[- ]?month/)
    const months = commitMatch ? ` (${commitMatch[1]}M)` : ''
    return { label: `Reserved ${direction}${months}`, type }
  }

  // --- Token-based pricing with direction ---
  // Detect modality
  const modality = /\bimage\b/i.test(combined) ? 'Image' :
                   /\bvideo\b/i.test(combined) ? 'Video' :
                   /\baudio\b/i.test(combined) ? 'Audio' :
                   /\bspeech\b/i.test(combined) ? 'Speech' : null

  // Detect qualifier (cache tier, pricing tier)
  const qualifier =
    /cache[- ]?read/i.test(combined) ? 'Cache Read' :
    /1[- ]?h(?:our)?\s*cache/i.test(combined) ? '1h Cache' :
    /cache[- ]?write/i.test(combined) ? 'Cache Write' :
    /\bflex\b/i.test(combined) ? 'Flex' :
    /\bpriority\b/i.test(combined) ? 'Priority' : null

  // Build label: [Modality] Direction [(Qualifier)]
  const parts = []
  if (modality) parts.push(modality)
  parts.push(direction)
  if (qualifier) parts.push(`(${qualifier})`)
  return { label: parts.join(' '), type }
}

// Shared component for rendering pricing item rows with click-to-reveal details
function PricingItemsList({ items, isLight }) {
  const [expandedIdx, setExpandedIdx] = useState(null)
  const [copiedIdx, setCopiedIdx] = useState(null)

  const handleCopyPrice = async (e, price, unit, idx) => {
    e.stopPropagation()
    const text = `$${typeof price === 'number' ? price.toFixed(6) : price} ${unit}`
    await navigator.clipboard.writeText(text)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(null), 1500)
  }

  return (
    <div>
      {items.map((item, idx) => {
        const priceStr = typeof item._price === 'number' ? item._price.toFixed(6) : item._price || 'N/A'
        const isExpanded = expandedIdx === idx
        return (
          <div key={idx}>
            <button
              onClick={() => setExpandedIdx(isExpanded ? null : idx)}
              className={cn(
                'w-full flex items-center justify-between px-2 py-1.5 text-left transition-colors rounded',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
              )}
            >
              <div className="flex items-center gap-2">
                <span className={cn(
                  'inline-block w-1.5 h-1.5 rounded-full flex-shrink-0',
                  item._type === 'input' ? 'bg-blue-500' : item._type === 'output' ? 'bg-emerald-500' : 'bg-[#6d6e72]'
                )} />
                <span className={cn('text-xs', isLight ? 'text-stone-700' : 'text-[#e4e5e7]')}>{item._label}</span>
              </div>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                <span className={cn('text-xs font-mono font-semibold tabular-nums', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                  ${priceStr}
                  <span className={cn('font-normal ml-1', isLight ? 'text-stone-400' : 'text-slate-400')}>{item._unit}</span>
                </span>
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => handleCopyPrice(e, item._price, item._unit, idx)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleCopyPrice(e, item._price, item._unit, idx) }}
                  className={cn('p-0.5 rounded transition-colors', isLight ? 'hover:bg-stone-200' : 'hover:bg-white/[0.08]')}
                >
                  {copiedIdx === idx ? (
                    <Check className="h-3 w-3 text-emerald-500" />
                  ) : (
                    <Copy className={cn('h-3 w-3', isLight ? 'text-stone-400' : 'text-slate-400')} />
                  )}
                </span>
              </div>
            </button>
            {isExpanded && (
              <div className={cn(
                'mx-2 mb-1.5 px-3 py-2 rounded text-[10px] font-mono leading-relaxed space-y-0.5',
                isLight ? 'text-stone-500 bg-stone-50' : 'text-slate-300 bg-white/[0.02]'
              )}>
                {item._raw && <p className="break-all"><span className={isLight ? 'text-stone-400' : 'text-slate-500'}>desc </span>{item._raw}</p>}
                {item.dimension && <p className="break-all"><span className={isLight ? 'text-stone-400' : 'text-slate-500'}>dim  </span>{item.dimension}</p>}
                {item.original_price != null && <p><span className={isLight ? 'text-stone-400' : 'text-slate-500'}>raw  </span>${item.original_price} {item.unit || ''}</p>}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
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
      isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-2 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{regionDisplayNames[region] || region}</span>
          <span className={cn('text-xs font-mono', isLight ? 'text-stone-600' : 'text-slate-300')}>({region})</span>
        </div>
        <div className="flex items-center gap-3">
          {inputItem && outputItem && (
            <span className="text-xs text-emerald-600 dark:text-emerald-400">
              ${inputItem.price.toFixed(4)} / ${outputItem.price.toFixed(4)}
            </span>
          )}
          {isExpanded ? (
            <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
          ) : (
            <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
          )}
        </div>
      </button>
      {isExpanded && (
        <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
          <div className="pt-2">
            {pricingItems.length > 0 ? (
              <PricingItemsList items={pricingItems.map(item => ({
                ...item,
                _label: simplifyPricingDescription(item.description).label,
                _type: item.type,
                _price: item.price,
                _unit: item.unit || 'per 1K tokens',
                _raw: item.description,
              }))} isLight={isLight} />
            ) : (
              <p className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-300')}>No pricing available</p>
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
  const [expandedGeos, setExpandedGeos] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const geoInfo = {
    'US': { icon: '🇺🇸', name: 'United States' },
    'EU': { icon: '🇪🇺', name: 'Europe' },
    'APAC': { icon: '🌏', name: 'Asia Pacific' },
    'CA': { icon: '🇨🇦', name: 'Canada' },
    'SA': { icon: '🌎', name: 'South America' },
    'ME': { icon: '🏜️', name: 'Middle East' },
    'Other': { icon: '📍', name: 'Other' }
  }

  const geoOrder = ['US', 'EU', 'APAC', 'CA', 'SA', 'ME', 'Other']

  const getGeoForRegion = (region) => {
    if (region.startsWith('us-')) return 'US'
    if (region.startsWith('eu-')) return 'EU'
    if (region.startsWith('ap-')) return 'APAC'
    if (region.startsWith('ca-')) return 'CA'
    if (region.startsWith('sa-')) return 'SA'
    if (region.startsWith('me-') || region.startsWith('il-')) return 'ME'
    return 'Other'
  }

  const toggleGeo = (key) => setExpandedGeos(prev => ({ ...prev, [key]: !prev[key] }))

  // Get pricing from new source
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
  const fullPricing = pricingResult?.fullPricing

  // Fallback to model's embedded pricing
  const legacyPricing = model.model_pricing || model.comprehensive_pricing || {}
  const legacyByRegion = legacyPricing.by_region || {}

  // Process pricing structure: group -> geo -> region -> items
  const pricingByGroupGeoRegion = {}
  let allRegions = []

  if (fullPricing?.regions) {
    allRegions = Object.keys(fullPricing.regions)
    for (const [region, regionData] of Object.entries(fullPricing.regions)) {
      if (!regionData?.pricing_groups) continue
      const geo = getGeoForRegion(region)
      for (const [groupName, items] of Object.entries(regionData.pricing_groups)) {
        if (!pricingByGroupGeoRegion[groupName]) pricingByGroupGeoRegion[groupName] = {}
        if (!pricingByGroupGeoRegion[groupName][geo]) pricingByGroupGeoRegion[groupName][geo] = {}
        pricingByGroupGeoRegion[groupName][geo][region] = items
      }
    }
  } else if (Object.keys(legacyByRegion).length > 0) {
    allRegions = Object.keys(legacyByRegion)
    for (const region of allRegions) {
      const regionData = legacyByRegion[region]
      const { onDemand, provisioned } = extractRegionPricing(regionData)
      const geo = getGeoForRegion(region)
      if (onDemand.length > 0) {
        if (!pricingByGroupGeoRegion['On-Demand']) pricingByGroupGeoRegion['On-Demand'] = {}
        if (!pricingByGroupGeoRegion['On-Demand'][geo]) pricingByGroupGeoRegion['On-Demand'][geo] = {}
        pricingByGroupGeoRegion['On-Demand'][geo][region] = onDemand.map(p => ({
          description: p.description,
          price_per_thousand: p.price,
          unit: p.unit
        }))
      }
      if (provisioned.length > 0) {
        if (!pricingByGroupGeoRegion['Provisioned Throughput']) pricingByGroupGeoRegion['Provisioned Throughput'] = {}
        if (!pricingByGroupGeoRegion['Provisioned Throughput'][geo]) pricingByGroupGeoRegion['Provisioned Throughput'][geo] = {}
        pricingByGroupGeoRegion['Provisioned Throughput'][geo][region] = provisioned.map(p => ({
          description: p.description,
          price_per_thousand: p.price,
          unit: p.unit
        }))
      }
    }
  }

  const pricingGroupOrder = ['On-Demand', 'On-Demand Long Context', 'On-Demand Global', 'Batch', 'Batch Long Context', 'Batch Global', 'Provisioned Throughput', 'Custom Model']
  const pricingGroups = Object.keys(pricingByGroupGeoRegion).sort((a, b) => {
    const indexA = pricingGroupOrder.indexOf(a)
    const indexB = pricingGroupOrder.indexOf(b)
    if (indexA !== -1 && indexB !== -1) return indexA - indexB
    if (indexA !== -1) return -1
    if (indexB !== -1) return 1
    return a.localeCompare(b)
  })

  if (allRegions.length === 0) {
    return (
      <div className={cn('text-center py-8', isLight ? 'text-stone-600' : 'text-slate-300')}>
        <p>No pricing information available</p>
      </div>
    )
  }

  // Filter pricing by search
  const filterPricing = (geoData) => {
    if (!searchQuery) return geoData
    const query = searchQuery.toLowerCase()
    const filtered = {}
    for (const [geo, regions] of Object.entries(geoData)) {
      const geoName = (geoInfo[geo]?.name || '').toLowerCase()
      for (const [region, regionItems] of Object.entries(regions)) {
        const regionName = (regionDisplayNames[region] || '').toLowerCase()
        const matchingItems = regionItems.filter(item => {
          const description = (item.description || item.dimension || '').toLowerCase()
          return region.toLowerCase().includes(query) ||
                 regionName.includes(query) ||
                 geo.toLowerCase().includes(query) ||
                 geoName.includes(query) ||
                 description.includes(query)
        })
        if (matchingItems.length > 0) {
          if (!filtered[geo]) filtered[geo] = {}
          filtered[geo][region] = matchingItems
        }
      }
    }
    return filtered
  }

  // Render a pricing group section with geo grouping
  const renderPricingGroupSection = (groupName, icon) => {
    const groupData = pricingByGroupGeoRegion[groupName]
    if (!groupData || Object.keys(groupData).length === 0) return null

    const filteredData = filterPricing(groupData)
    if (Object.keys(filteredData).length === 0) return null

    return (
      <CollapsibleSection title={pricingGroupInfo[groupName]?.label || groupName} icon={icon} defaultExpanded={groupName === 'On-Demand'}>
        <div className="space-y-2">
          {geoOrder.map(geo => {
            const geoData = filteredData[geo]
            if (!geoData || Object.keys(geoData).length === 0) return null

            const regionCount = Object.keys(geoData).length
            const itemCount = Object.values(geoData).reduce((sum, items) => sum + items.length, 0)
            const geoKey = `${groupName}_${geo}`
            const isGeoExpanded = expandedGeos[geoKey]

            return (
              <div key={geo} className={cn('rounded-lg border overflow-hidden', isLight ? 'bg-stone-50/50 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]')}>
                <button
                  className={cn('w-full flex items-center justify-between p-2.5 transition-colors', isLight ? 'hover:bg-stone-100' : 'hover:bg-white/[0.08]')}
                  onClick={() => toggleGeo(geoKey)}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{geoInfo[geo]?.icon}</span>
                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-800' : 'text-white')}>{geoInfo[geo]?.name}</span>
                    <Badge variant="secondary" className="text-[10px]">{regionCount} regions</Badge>
                    <Badge variant="outline" className="text-[10px]">{itemCount} items</Badge>
                  </div>
                  {isGeoExpanded ? (
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
                  )}
                </button>
                {isGeoExpanded && (
                  <div className={cn('px-2.5 pb-2.5 space-y-2 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                    {Object.entries(geoData).sort().map(([region, regionItems]) => (
                      <div key={region} className={cn('rounded-lg p-2 mt-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
                        <div className="flex items-center gap-2 mb-2">
                          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
                          <span className={cn('font-medium text-xs', isLight ? 'text-stone-800' : 'text-white')}>{regionDisplayNames[region] || region}</span>
                          <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-slate-400')}>({region})</span>
                        </div>
                        <PricingItemsList items={regionItems.map(item => {
                          const { label, type } = simplifyPricingDescription(item.description, item.dimension)
                          return {
                            ...item,
                            _label: label,
                            _type: type,
                            _price: item.price_per_thousand ?? item.price_per_unit,
                            _unit: item.unit_label || `per ${item.unit || 'unit'}`,
                            _raw: item.description || item.dimension,
                          }
                        })} isLight={isLight} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </CollapsibleSection>
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
            <Search className={cn('absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
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
          {/* Left Column - On-Demand Pricing */}
          <div className="space-y-4">
            {onDemandGroups.map(groupName => (
              <div key={groupName}>
                {renderPricingGroupSection(groupName, Zap)}
              </div>
            ))}
          </div>

          {/* Right Column - Batch and Other Pricing */}
          <div className="space-y-4">
            {batchGroups.map(groupName => (
              <div key={groupName}>
                {renderPricingGroupSection(groupName, Layers)}
              </div>
            ))}
            {otherGroups.map(groupName => (
              <div key={groupName}>
                {renderPricingGroupSection(groupName, Server)}
              </div>
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

  const isActive = model.model_lifecycle?.status === 'ACTIVE' || model.model_status === 'ACTIVE'

  const contextWindow = model.converse_data?.context_window
  const extendedContext = model.converse_data?.extended_context
  const hasExtendedContext = model.converse_data?.has_extended_context
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
            isLight ? 'border-stone-200' : 'border-white/[0.06]'
          )}>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Badge
                  className="text-xs font-medium px-2 py-0.5"
                  style={{ backgroundColor: getProviderHexColor(model.model_provider), color: getContrastColor(getProviderHexColor(model.model_provider)) }}
                >
                  {model.model_provider}
                </Badge>
                <span className={cn(
                  'px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide',
                  isActive
                    ? isLight
                      ? 'bg-emerald-100 text-emerald-700'
                      : 'bg-emerald-500/15 text-emerald-400'
                    : isLight
                      ? 'bg-amber-100 text-amber-700'
                      : 'bg-amber-500/15 text-amber-400'
                )}>
                  {isActive ? 'Active' : 'Legacy'}
                </span>
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
                    <GitCompareArrows className={cn('h-5 w-5', isInComparison ? 'text-[#1A9E7A]' : 'text-slate-300')} />
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
                    <Star className={cn('h-5 w-5', isFavorite ? 'fill-yellow-500 text-yellow-500' : 'text-slate-300')} />
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
              isLight ? 'bg-stone-50 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
            )}>
              {/* Token Limits - Always shown */}
              <div className="space-y-3">
                <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                  Token Limits
                </h3>
                <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                  <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Context Window</p>
                  <p className={cn('text-xl font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>
                    {contextWindow ? (contextWindow >= 1000000 ? `${(contextWindow/1000000).toFixed(1)}M` : contextWindow >= 1000 ? `${(contextWindow/1000).toFixed(0)}K` : contextWindow) : 'N/A'}
                    {hasExtendedContext && extendedContext && (
                      <span className={cn('text-sm font-normal ml-1', isLight ? 'text-amber-500' : 'text-emerald-400')}>
                        / {extendedContext >= 1000000 ? `${(extendedContext/1000000).toFixed(0)}M` : `${(extendedContext/1000).toFixed(0)}K`}
                      </span>
                    )}
                  </p>
                  {hasExtendedContext && (
                    <p className={cn('text-[10px] mt-1', isLight ? 'text-amber-600' : 'text-emerald-500')}>Extended context (beta)</p>
                  )}
                </div>
                <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                  <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Max Output</p>
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
                    <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      Availability
                    </h3>
                    <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                      <div className="flex items-center justify-between">
                        <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Regions</p>
                        <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{regions.length}</p>
                      </div>
                    </div>
                    <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                      <div className="flex items-center justify-between">
                        <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Capabilities</p>
                        <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{capabilities.length}</p>
                      </div>
                    </div>
                  </div>

                  {/* Features */}
                  <div className="space-y-3">
                    <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      Features
                    </h3>
                    <div className="flex flex-wrap gap-1.5">
                      <span className={cn(
                        'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
                        streamingSupported
                          ? isLight ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : isLight ? 'bg-stone-100 text-stone-400 border border-stone-200' : 'bg-white/[0.03] text-slate-500 border border-white/[0.06]'
                      )}>
                        <Radio className="h-3.5 w-3.5" />
                        Stream
                      </span>
                      <span className={cn(
                        'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
                        crisSupported
                          ? isLight ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : isLight ? 'bg-stone-100 text-stone-400 border border-stone-200' : 'bg-white/[0.03] text-slate-500 border border-white/[0.06]'
                      )}>
                        <Globe className="h-3.5 w-3.5" />
                        CRIS
                      </span>
                    </div>
                  </div>

                  {/* Modalities */}
                  <div className="space-y-3">
                    <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      Modalities
                    </h3>
                    <div className="space-y-2">
                      <div>
                        <p className={cn('text-xs mb-1', isLight ? 'text-stone-500' : 'text-slate-400')}>Input</p>
                        <div className="flex flex-wrap gap-1">
                          {inputModalities.map(mod => {
                            const Icon = modalityIcons[mod] || MessageSquare
                            return (
                              <span key={mod} className={cn(
                                'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium',
                                isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/5 text-slate-400'
                              )}>
                                <Icon className="h-3 w-3" />
                                {modalityLabels[mod] || mod}
                              </span>
                            )
                          })}
                          {inputModalities.length === 0 && <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-slate-500')}>None</span>}
                        </div>
                      </div>
                      <div>
                        <p className={cn('text-xs mb-1', isLight ? 'text-stone-500' : 'text-slate-400')}>Output</p>
                        <div className="flex flex-wrap gap-1">
                          {outputModalities.map(mod => {
                            const Icon = modalityIcons[mod] || MessageSquare
                            return (
                              <span key={mod} className={cn(
                                'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium',
                                isLight ? 'bg-blue-50 text-blue-600' : 'bg-blue-500/10 text-blue-400'
                              )}>
                                <Icon className="h-3 w-3" />
                                {modalityLabels[mod] || mod}
                              </span>
                            )
                          })}
                          {outputModalities.length === 0 && <span className={cn('text-xs', isLight ? 'text-stone-400' : 'text-slate-500')}>None</span>}
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {activeTab === 'quotas' && (
                <div className="space-y-3">
                  <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    Quota Summary
                  </h3>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Total Quotas</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{totalQuotas}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Adjustable</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-emerald-700' : 'text-emerald-400')}>{adjustableQuotas}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Regions</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-blue-700' : 'text-blue-400')}>{quotaRegions.length}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Categories</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-purple-700' : 'text-purple-400')}>{quotaCategories.size}</p>
                  </div>
                </div>
              )}

              {activeTab === 'pricing' && (
                <div className="space-y-3">
                  <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    Pricing Summary
                  </h3>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Pricing Types</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{pricingTypes}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Regions</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-emerald-700' : 'text-emerald-400')}>{pricingRegions.length}</p>
                  </div>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Options</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-purple-700' : 'text-purple-400')}>{consumptionOptions.length || pricingTypes}</p>
                  </div>
                  {consumptionOptions.length > 0 && (
                    <div className="pt-2">
                      <p className={cn('text-xs mb-2', isLight ? 'text-stone-500' : 'text-slate-400')}>Consumption</p>
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
                  isLight ? 'bg-transparent border-stone-200' : 'bg-transparent border-white/[0.06]'
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
