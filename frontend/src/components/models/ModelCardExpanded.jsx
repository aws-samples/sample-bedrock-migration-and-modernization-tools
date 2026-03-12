import { useState } from 'react'
import { Star, Globe, Zap, MessageSquare, Image, FileText, Video, Mic, Check, X, ChevronDown, ChevronRight, Search, Database, Languages, Cpu, Layers, Package, Server, ExternalLink, Copy, DollarSign, GitCompareArrows, Radio, Info, Bot, BookOpen, Workflow, Shield, Clock, Route, Wrench, AlertTriangle, AlertCircle, MapPin, Split, Calculator } from 'lucide-react'
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
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'
import { useAuthStore } from '@/stores/authStore'
import { canViewQuotas, canViewProvisionedPricing, getSectionBadge } from '@/config/admin'
import { getRegionName, getRegionInfo, geoGroups as regionGeoGroups } from '@/utils/regionUtils'

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
function CollapsibleSection({ title, icon: Icon, children, defaultExpanded = false, infoLink = null, dataSource = null }) {
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
          {infoLink && (
            <a
              href={infoLink}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className={cn('ml-1 transition-colors', isLight ? 'text-blue-600 hover:text-blue-700' : 'text-blue-400 hover:text-blue-300')}
              title="Learn more"
            >
              <Info className="h-3.5 w-3.5" />
            </a>
          )}
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
          {dataSource && (
            <div className={cn(
              'mt-3 pt-2 border-t flex items-start gap-2 text-xs',
              isLight 
                ? 'border-stone-200 text-stone-600' 
                : 'border-white/[0.06] text-slate-400'
            )}>
              <Info className={cn('h-3.5 w-3.5 flex-shrink-0 mt-0.5', isLight ? 'text-stone-400' : 'text-slate-500')} />
              <span>{dataSource}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Region display name - uses centralized region utilities
const getRegionDisplayName = (regionCode) => getRegionName(regionCode)

// Geographic groupings - uses centralized geoGroups with additional UI-specific data
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

// Regional Availability grouped by geography - CRIS style
function RegionalAvailabilityGrouped({ regions }) {
  const [expandedGroups, setExpandedGroups] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const grouped = groupRegionsByGeo(regions)

  const toggleGroup = (group) => {
    setExpandedGroups(prev => ({ ...prev, [group]: !prev[group] }))
  }

  // Geo display names without emojis (CRIS style)
  const geoDisplayNames = {
    'US': 'United States',
    'EU': 'Europe', 
    'APAC': 'Asia Pacific',
    'CA': 'Canada',
    'SA': 'South America',
    'MX': 'Mexico',
    'ME': 'Middle East',
    'AF': 'Africa',
    'GOV': 'GovCloud'
  }

  return (
    <div className="space-y-2">
      {Object.entries(grouped).map(([groupKey, groupRegions]) => {
        const displayName = geoDisplayNames[groupKey] || groupKey
        const isExpanded = expandedGroups[groupKey]

        return (
          <div key={groupKey} className={cn(
            'rounded-lg border overflow-hidden',
            isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
          )}>
            <button
              className={cn(
                'w-full flex items-center justify-between p-3 transition-colors',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
              )}
              onClick={() => toggleGroup(groupKey)}
            >
              <div className="flex items-center gap-2">
                <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                  {displayName}
                </span>
                <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                  {groupRegions.length} regions
                </span>
              </div>
              {isExpanded ? (
                <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
              ) : (
                <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
              )}
            </button>
            {isExpanded && (
              <div className={cn('px-3 pb-3 pt-3 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                <div className="flex flex-wrap gap-1">
                  {groupRegions.sort().map(region => (
                    <Tooltip key={region} delayDuration={200}>
                      <TooltipTrigger asChild>
                        <Badge variant="secondary" className="text-[10px] cursor-default">
                          {getRegionDisplayName(region)}
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
        )
      })}
    </div>
  )
}

// On-Demand Availability section with model ID highlight, stats, and geo-grouped regions
function OnDemandAvailabilitySection({ model, govcloudData }) {
  const [regionsExpanded, setRegionsExpanded] = useState(false)
  const [govcloudExpanded, setGovcloudExpanded] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const regions = model.availability?.on_demand?.regions ?? model.in_region ?? []
  const grouped = groupRegionsByGeo(regions)
  const geoCount = Object.keys(grouped).length
  const modelId = model.model_id
  const isMantleOnly = model.availability?.mantle?.only

  // Check if GovCloud should be shown in On-Demand section (in_region inference type)
  const govcloudRegions = govcloudData?.regions || []
  const hasGovCloudInRegion = govcloudData?.supported && govcloudRegions.length > 0 && govcloudData?.inference_type === 'in_region'

  const geoDisplayNames = {
    'US': 'United States', 'EU': 'Europe', 'APAC': 'Asia Pacific',
    'CA': 'Canada', 'SA': 'South America', 'MX': 'Mexico', 'ME': 'Middle East',
    'AF': 'Africa', 'GOV': 'GovCloud'
  }

  return (
    <div className="space-y-3">
      {/* Mantle-only notice */}
      {isMantleOnly && (
        <div className={cn(
          'rounded-lg p-3 border',
          isLight
            ? 'bg-violet-50 border-violet-200'
            : 'bg-violet-500/10 border border-violet-500/20'
        )}>
          <p className={cn('text-sm font-medium', isLight ? 'text-violet-700' : 'text-violet-400')}>
            Available via Mantle Inference only
          </p>
          <p className={cn('text-xs mt-1', isLight ? 'text-violet-600' : 'text-violet-300')}>
            This model is not available for direct in-region invocation. Use Mantle Inference to access this model.
          </p>
        </div>
      )}

      {/* Model ID highlight bar - only show when commercial regions are available */}
      {modelId && !isMantleOnly && regions.length > 0 && (
        <div className={cn(
          'rounded-lg p-3 border',
          isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
        )}>
          <div className="flex items-center justify-between">
            <CopyableText
              text={modelId}
              isLight={isLight}
              className={cn('text-sm font-mono', isLight ? 'text-stone-700 hover:text-stone-900' : 'text-[#c0c1c5] hover:text-white')}
            />
          </div>
          <div className="flex items-center gap-2 mt-2">
            <Badge variant="secondary" className="text-[10px]">IN_REGION</Badge>
          </div>
          <p className={cn('text-xs mt-1.5', isLight ? 'text-stone-500' : 'text-slate-400')}>
            Direct model invocation in supported regions.
          </p>
        </div>
      )}

      {/* Summary stats bar */}
      <div className={cn('grid gap-2', regions.length > 0 ? 'grid-cols-3' : 'grid-cols-1')}>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {(regions.length > 0 || hasGovCloudInRegion) ? (
              <>
                <Check className="h-4 w-4 text-emerald-500" />
                <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">
                  {regions.length > 0 ? 'Supported' : 'GovCloud Only'}
                </span>
              </>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>Not Available</span></>
            )}
          </div>
        </div>
        {regions.length > 0 && (
          <>
            <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Total Regions</p>
              <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{regions.length}</p>
            </div>
            <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Geographies</p>
              <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{geoCount}</p>
            </div>
          </>
        )}
      </div>

      {/* Single collapsible for all regions - matches Batch style */}
      {regions.length > 0 && (
        <div className="space-y-2">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>Regions by Geo</p>
          <div className={cn(
            'rounded-lg border overflow-hidden',
            isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
          )}>
            <button
              className={cn(
                'w-full flex items-center justify-between p-3 transition-colors',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
              )}
              onClick={() => setRegionsExpanded(!regionsExpanded)}
            >
              <div className="flex items-center gap-2">
                <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                  Available Regions
                </span>
                <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                  {regions.length} regions
                </span>
              </div>
              {regionsExpanded ? (
                <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
              ) : (
                <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
              )}
            </button>
            {regionsExpanded && (
              <div className={cn('px-3 pb-3 pt-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                {Object.entries(grouped).map(([geoKey, geoRegions]) => (
                  <div key={geoKey}>
                    <p className={cn('text-[10px] mb-2 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      {geoDisplayNames[geoKey] || geoKey} ({geoRegions.length})
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {geoRegions.sort().map(region => (
                        <Tooltip key={region} delayDuration={200}>
                          <TooltipTrigger asChild>
                            <Badge variant="secondary" className="text-[10px] cursor-default">
                              {getRegionDisplayName(region)}
                            </Badge>
                          </TooltipTrigger>
                          <TooltipContent side="bottom" sideOffset={4}>
                            <p className="font-mono text-xs">{region}</p>
                          </TooltipContent>
                        </Tooltip>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* GovCloud section for in_region inference type */}
      {hasGovCloudInRegion && (
        <div className={cn(
          'rounded-lg border overflow-hidden',
          isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between p-3 transition-colors',
              isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
            )}
            onClick={() => setGovcloudExpanded(!govcloudExpanded)}
          >
            <div className="flex items-center gap-2">
              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                GovCloud
              </span>
              <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                {govcloudRegions.length} regions
              </span>
            </div>
            {govcloudExpanded ? (
              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            ) : (
              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            )}
          </button>
          {govcloudExpanded && (
            <div className={cn('px-3 pb-3 pt-3 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
              <p className={cn('text-[10px] mb-2 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                Regions ({govcloudRegions.length})
              </p>
              <div className="flex flex-wrap gap-1">
                {govcloudRegions.sort().map(region => (
                  <Tooltip key={region} delayDuration={200}>
                    <TooltipTrigger asChild>
                      <Badge variant="secondary" className="text-[10px] cursor-default">
                        {getRegionDisplayName(region)}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" sideOffset={4}>
                      <p className="font-mono text-xs">{region}</p>
                    </TooltipContent>
                  </Tooltip>
                ))}
              </div>
              <p className={cn('text-[10px] mt-2 italic', isLight ? 'text-stone-400' : 'text-slate-500')}>
                Available in US GovCloud regions via in-region inference
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Regions that support Application Inference Profiles (user-created profiles for tagging/routing)
const APP_PROFILE_SUPPORTED_REGIONS = [
  'ap-northeast-1', 'ap-northeast-2', 'ap-south-1', 'ap-southeast-1', 'ap-southeast-2',
  'ca-central-1',
  'eu-central-1', 'eu-west-1', 'eu-west-2', 'eu-west-3',
  'sa-east-1',
  'us-east-1', 'us-east-2', 'us-gov-east-1', 'us-west-2'
]

// Application Inference Profile Section
function ApplicationInferenceProfileSection({ regionsAvailable }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const [showAllRegions, setShowAllRegions] = useState(false)

  // Calculate which regions support app profiles for this model
  const modelRegions = regionsAvailable || []
  const supportedRegions = modelRegions.filter(r => APP_PROFILE_SUPPORTED_REGIONS.includes(r))
  const isSupported = supportedRegions.length > 0

  // Group supported regions by geography
  const regionsByGeo = { 'US': [], 'EU': [], 'APAC': [], 'Other': [] }
  for (const region of supportedRegions) {
    if (region.startsWith('us-')) regionsByGeo['US'].push(region)
    else if (region.startsWith('eu-')) regionsByGeo['EU'].push(region)
    else if (region.startsWith('ap-')) regionsByGeo['APAC'].push(region)
    else if (region.startsWith('ca-') || region.startsWith('sa-')) regionsByGeo['Other'].push(region)
    else regionsByGeo['Other'].push(region)
  }

  const geoIcons = { 'US': '🇺🇸', 'EU': '🇪🇺', 'APAC': '🌏', 'Other': '🌎' }

  return (
    <div className="space-y-3">
      {/* Status and count */}
      <div className="grid grid-cols-2 gap-2">
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {isSupported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Available</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>Not Available</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Supported Regions</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{supportedRegions.length}</p>
        </div>
      </div>

      {/* Description */}
      <div className={cn('rounded p-3', isLight ? 'bg-amber-50 border border-amber-200' : 'bg-[#1A9E7A]/10 border border-[#1A9E7A]/20')}>
        <p className={cn('text-xs', isLight ? 'text-amber-800' : 'text-[#1A9E7A]')}>
          Application inference profiles let you create custom profiles for this model to:
        </p>
        <ul className={cn('text-xs mt-1.5 space-y-0.5', isLight ? 'text-amber-700' : 'text-[#1A9E7A]/80')}>
          <li>• Tag requests for cost allocation & tracking</li>
          <li>• Organize usage by application or team</li>
          <li>• Apply custom routing rules</li>
        </ul>
      </div>

      {/* Supported regions by geography */}
      {isSupported && (
        <div className="space-y-2">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>
            Available in {supportedRegions.length} regions
          </p>
          <div className="flex flex-wrap gap-2">
            {['US', 'EU', 'APAC', 'Other'].map(geoKey => {
              const geoRegions = regionsByGeo[geoKey]
              if (geoRegions.length === 0) return null
              return (
                <div
                  key={geoKey}
                  className={cn(
                    'inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs',
                    isLight ? 'bg-stone-100 border border-stone-200' : 'bg-white/[0.06] border border-white/[0.06]'
                  )}
                >
                  <span>{geoIcons[geoKey]}</span>
                  <span className={cn('font-medium', isLight ? 'text-stone-700' : 'text-white')}>{geoKey}</span>
                  <span className={cn(isLight ? 'text-stone-500' : 'text-slate-400')}>({geoRegions.length})</span>
                </div>
              )
            })}
          </div>

          {/* Expandable region list */}
          <button
            onClick={() => setShowAllRegions(!showAllRegions)}
            className={cn(
              'text-xs flex items-center gap-1 transition-colors',
              isLight ? 'text-amber-600 hover:text-amber-700' : 'text-[#1A9E7A] hover:text-[#1A9E7A]/80'
            )}
          >
            {showAllRegions ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            {showAllRegions ? 'Hide region details' : 'Show all regions'}
          </button>

          {showAllRegions && (
            <div className="flex flex-wrap gap-1.5 pt-1">
              {supportedRegions.sort().map(region => (
                <Badge key={region} variant="secondary" className="text-[10px]">
                  {getRegionDisplayName(region)}
                  <span className={cn('ml-1 font-mono', isLight ? 'text-stone-400' : 'text-slate-500')}>
                    ({region})
                  </span>
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Not supported message */}
      {!isSupported && (
        <p className={cn('text-sm', isLight ? 'text-stone-500' : 'text-slate-400')}>
          This model is not available in any regions that support application inference profiles.
        </p>
      )}
    </div>
  )
}

// Compact Application Inference Profile info banner (supplementary, not an inference type)
function ApplicationInferenceProfileBanner() {
  const [isExpanded, setIsExpanded] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'

  return (
    <div>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          'w-full flex items-center justify-between px-3 py-2 rounded-md transition-all text-xs',
          isLight
            ? 'bg-blue-50/60 border border-blue-200/50 hover:bg-blue-50'
            : 'bg-blue-500/[0.06] border border-blue-500/15 hover:bg-blue-500/[0.10]',
          isExpanded && 'rounded-b-none'
        )}
      >
        <div className="flex items-center gap-2">
          <Info className={cn('h-3.5 w-3.5', isLight ? 'text-blue-500' : 'text-blue-400')} />
          <span className={cn('font-medium', isLight ? 'text-stone-700' : 'text-slate-300')}>
            App Inference Profiles
          </span>
        </div>
        {isExpanded
          ? <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-blue-500' : 'text-blue-400')} />
          : <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-blue-500' : 'text-blue-400')} />
        }
      </button>

      {isExpanded && (
        <div className={cn(
          'px-3 py-3 rounded-b-md border border-t-0 text-xs space-y-3',
          isLight
            ? 'bg-blue-50/30 border-blue-200/50'
            : 'bg-blue-500/[0.03] border-blue-500/15'
        )}>
          <p className={cn(isLight ? 'text-stone-600' : 'text-slate-400')}>
            User-created inference profiles for cost tracking, team organization, and custom routing.
            Can wrap a single-region model or a CRIS profile for cross-region routing.
          </p>

          <div>
            <p className={cn('font-medium mb-1.5', isLight ? 'text-stone-700' : 'text-slate-300')}>Use for</p>
            <div className="flex flex-wrap gap-1.5">
              {['Cost allocation', 'Usage tagging', 'Team organization', 'Custom routing'].map(tag => (
                <span key={tag} className={cn(
                  'px-1.5 py-0.5 rounded text-[10px]',
                  isLight ? 'bg-blue-100/70 text-blue-700 border border-blue-200/60' : 'bg-blue-500/10 text-blue-300 border border-blue-500/20'
                )}>{tag}</span>
              ))}
            </div>
          </div>

          <a
            href="https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html"
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              'inline-flex items-center gap-1 font-medium transition-colors',
              isLight ? 'text-blue-600 hover:text-blue-700' : 'text-blue-400 hover:text-blue-300'
            )}
          >
            Learn more
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      )}
    </div>
  )
}

// Cross-Region Inference Section grouped by endpoint prefix
function CrossRegionInferenceSection({ crisData, govcloudData }) {
  const [expandedScopes, setExpandedScopes] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const profiles = crisData.profiles || []
  const sourceRegions = crisData.regions || []

  // Helper to get profile fields (handles both field naming conventions)
  const getProfileId = (p) => p.profile_id || p.inference_profile_id
  const getSourceRegion = (p) => p.source_region || p.region

  // Extract prefix from profile ID (e.g., "us" from "us.anthropic.claude...")
  const getScopePrefix = (profile) => {
    const profileId = getProfileId(profile)
    return profileId?.split('.')[0]?.toLowerCase() || null
  }

  // Group profiles by profile_id, collecting all source regions for each endpoint
  const profilesMap = new Map()
  for (const profile of profiles) {
    const profileId = getProfileId(profile)
    if (!profileId) continue

    const existing = profilesMap.get(profileId)
    const sourceRegion = getSourceRegion(profile)

    if (existing) {
      // Add source region to the set
      if (sourceRegion) {
        existing.sourceRegions.add(sourceRegion)
      }
    } else {
      profilesMap.set(profileId, {
        profile,
        scopePrefix: getScopePrefix(profile),
        sourceRegions: sourceRegion ? new Set([sourceRegion]) : new Set()
      })
    }
  }

  // Group by scope prefix (exact match, no normalization)
  const profilesByScope = {}
  for (const [, data] of profilesMap) {
    const scope = data.scopePrefix?.toUpperCase() || 'UNKNOWN'
    if (!profilesByScope[scope]) {
      profilesByScope[scope] = []
    }
    profilesByScope[scope].push(data)
  }

  // Check if GovCloud has regions AND uses CRIS inference type
  const govcloudRegions = govcloudData?.regions || []
  const hasGovCloud = govcloudData?.supported && govcloudRegions.length > 0 && govcloudData?.inference_type === 'cris'

  // Get available scopes sorted (global first, then alphabetically, GovCloud last)
  const availableScopes = Object.keys(profilesByScope).sort((a, b) => {
    if (a === 'GLOBAL') return -1
    if (b === 'GLOBAL') return 1
    return a.localeCompare(b)
  })

  // Calculate total prefix groups (including GovCloud if it has regions)
  const totalPrefixGroups = availableScopes.length + (hasGovCloud ? 1 : 0)

  const toggleScope = (key) => {
    setExpandedScopes(prev => ({ ...prev, [key]: !prev[key] }))
  }

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
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Endpoint Prefixes</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{totalPrefixGroups}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Source Regions</p>
          <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{sourceRegions.length}</p>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Unique Endpoints</p>
          <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{profilesMap.size}</p>
        </div>
      </div>

      {/* CRIS Endpoints grouped by prefix */}
      {crisData.supported && (availableScopes.length > 0 || hasGovCloud) && (
        <div className="space-y-3">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>CRIS Endpoints by Prefix</p>

          {availableScopes.map(scopeKey => {
            const scopeProfiles = profilesByScope[scopeKey]
            const isExpanded = expandedScopes[scopeKey]
            // Count total source regions across all profiles in this scope
            const totalSourceRegions = scopeProfiles.reduce((acc, p) => acc + p.sourceRegions.size, 0)

            return (
              <div key={scopeKey} className={cn(
                'rounded-lg border overflow-hidden',
                isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
              )}>
                <button
                  className={cn(
                    'w-full flex items-center justify-between p-3 transition-colors',
                    isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
                  )}
                  onClick={() => toggleScope(scopeKey)}
                >
                  <div className="flex items-center gap-2">
                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                      {scopeKey === 'GLOBAL' ? 'Global' : scopeKey}
                    </span>
                    <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      {totalSourceRegions} regions
                    </span>
                  </div>
                  {isExpanded ? (
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                  )}
                </button>
                {isExpanded && (
                  <div className={cn('px-3 pb-3 pt-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                    {scopeProfiles.map(({ profile, sourceRegions: profileSourceRegions }, idx) => {
                      const sourceRegionsList = [...profileSourceRegions].sort()
                      return (
                        <div key={`${getProfileId(profile)}-${idx}`}>
                          {/* Endpoint ID */}
                          <div className="mb-3">
                            <p className={cn('text-[10px] mb-1 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                              Endpoint ID
                            </p>
                            <CopyableText
                              text={getProfileId(profile)}
                              isLight={isLight}
                              className={cn('text-xs font-mono', isLight ? 'text-stone-700 hover:text-stone-900' : 'text-[#c0c1c5] hover:text-white')}
                            />
                          </div>

                          {profile.status && profile.status !== 'ACTIVE' && (
                            <div className="flex items-center gap-2 mb-3">
                              <Badge variant="destructive" className="text-[10px]">{profile.status}</Badge>
                            </div>
                          )}

                          {/* Source regions as badges */}
                          {sourceRegionsList.length > 0 && (
                            <div>
                              <p className={cn('text-[10px] mb-2 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                                Source Regions ({sourceRegionsList.length})
                              </p>
                              <div className="flex flex-wrap gap-1">
                                {sourceRegionsList.map(sourceRegion => (
                                  <Tooltip key={sourceRegion} delayDuration={200}>
                                    <TooltipTrigger asChild>
                                      <Badge variant="secondary" className="text-[10px] cursor-default">
                                        {getRegionDisplayName(sourceRegion)}
                                      </Badge>
                                    </TooltipTrigger>
                                    <TooltipContent side="bottom" sideOffset={4}>
                                      <p className="font-mono text-xs">{sourceRegion}</p>
                                    </TooltipContent>
                                  </Tooltip>
                                ))}
                              </div>
                              <p className={cn('text-[10px] mt-2 italic', isLight ? 'text-stone-400' : 'text-slate-500')}>
                                These regions can be used as source regions for this endpoint
                              </p>
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

          {/* GovCloud group at the end */}
          {hasGovCloud && (
            <div className={cn(
              'rounded-lg border overflow-hidden',
              isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
            )}>
              <button
                className={cn(
                  'w-full flex items-center justify-between p-3 transition-colors',
                  isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
                )}
                onClick={() => toggleScope('GOVCLOUD')}
              >
                <div className="flex items-center gap-2">
                  <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                    GovCloud
                  </span>
                  <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    {govcloudRegions.length} regions
                  </span>
                </div>
                {expandedScopes['GOVCLOUD'] ? (
                  <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                ) : (
                  <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                )}
              </button>
              {expandedScopes['GOVCLOUD'] && (
                <div className={cn('px-3 pb-3 pt-3 border-t', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                  <p className={cn('text-[10px] mb-2 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    Regions ({govcloudRegions.length})
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {govcloudRegions.sort().map(region => (
                      <Tooltip key={region} delayDuration={200}>
                        <TooltipTrigger asChild>
                          <Badge variant="secondary" className="text-[10px] cursor-default">
                            {getRegionDisplayName(region)}
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent side="bottom" sideOffset={4}>
                          <p className="font-mono text-xs">{region}</p>
                        </TooltipContent>
                      </Tooltip>
                    ))}
                  </div>
                  <p className={cn('text-[10px] mt-2 italic', isLight ? 'text-stone-400' : 'text-slate-500')}>
                    Available in US GovCloud regions
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Provisioned Throughput Section with grouped regions
function ProvisionedThroughputSection({ provisionedData }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const regions = provisionedData.regions ?? provisionedData.provisioned_regions ?? []
  const grouped = groupRegionsByGeo(regions)
  const geoCount = Object.keys(grouped).length

  return (
    <div className="space-y-3">
      {/* Summary stats bar */}
      <div className={cn('grid gap-2', regions.length > 0 ? 'grid-cols-3' : 'grid-cols-1')}>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {provisionedData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Supported</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>Not Available</span></>
            )}
          </div>
        </div>
        {regions.length > 0 && (
          <>
            <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Total Regions</p>
              <p className={cn('text-lg font-bold', isLight ? 'text-amber-600' : 'text-amber-400')}>{regions.length}</p>
            </div>
            <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
              <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Geographies</p>
              <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{geoCount}</p>
            </div>
          </>
        )}
      </div>

      {/* Geo-grouped region display */}
      {regions.length > 0 && (
        <div>
          <p className={cn('text-xs font-medium mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Regions by Geo</p>
          <RegionalAvailabilityGrouped regions={regions} />
        </div>
      )}
    </div>
  )
}

// Batch Inference Section with CRIS-style grouped regions
function BatchInferenceSection({ batchData, crisData }) {
  const [expandedGroups, setExpandedGroups] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'
  
  // Get batch regions from different sources
  const inRegionBatchRegions = batchData.regions ?? batchData.supported_regions ?? []
  
  // Check if CRIS supports batch (look for Batch Global/Geo pricing groups)
  const crisBatchRegions = crisData?.regions || []
  const hasCrisBatch = crisData?.supported && crisBatchRegions.length > 0
  
  const toggleGroup = (key) => {
    setExpandedGroups(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const totalRegions = new Set([...inRegionBatchRegions, ...(hasCrisBatch ? crisBatchRegions : [])]).size

  // Group regions by geo for display
  const groupByGeo = (regions) => {
    const grouped = {}
    regions.forEach(r => {
      let geo = 'Other'
      if (r.startsWith('us-gov-')) geo = 'GOV'
      else if (r.startsWith('us-')) geo = 'US'
      else if (r.startsWith('eu-')) geo = 'EU'
      else if (r.startsWith('ap-')) geo = 'AP'
      else if (r.startsWith('ca-')) geo = 'CA'
      else if (r.startsWith('sa-')) geo = 'SA'
      else if (r.startsWith('me-') || r.startsWith('il-')) geo = 'ME'
      else if (r.startsWith('af-')) geo = 'AF'
      if (!grouped[geo]) grouped[geo] = []
      grouped[geo].push(r)
    })
    return grouped
  }

  const geoDisplayNames = {
    'US': 'United States', 'EU': 'Europe', 'AP': 'Asia Pacific',
    'CA': 'Canada', 'SA': 'South America', 'ME': 'Middle East',
    'AF': 'Africa', 'GOV': 'GovCloud', 'Other': 'Other'
  }

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
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Total Regions</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{totalRegions}</p>
        </div>
      </div>

      {/* Batch by consumption type */}
      {batchData.supported && (
        <div className="space-y-2">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>Batch Availability</p>
          
          {/* In-Region Batch */}
          {inRegionBatchRegions.length > 0 && (
            <div className={cn(
              'rounded-lg border overflow-hidden',
              isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
            )}>
              <button
                className={cn(
                  'w-full flex items-center justify-between p-3 transition-colors',
                  isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
                )}
                onClick={() => toggleGroup('in_region')}
              >
                <div className="flex items-center gap-2">
                  <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                    In-Region
                  </span>
                  <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    {inRegionBatchRegions.length} regions
                  </span>
                </div>
                {expandedGroups['in_region'] ? (
                  <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                ) : (
                  <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                )}
              </button>
              {expandedGroups['in_region'] && (
                <div className={cn('px-3 pb-3 pt-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                  {Object.entries(groupByGeo(inRegionBatchRegions)).map(([geoKey, geoRegions]) => (
                    <div key={geoKey}>
                      <p className={cn('text-[10px] mb-2 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                        {geoDisplayNames[geoKey]} ({geoRegions.length})
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {geoRegions.sort().map(region => (
                          <Tooltip key={region} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="secondary" className="text-[10px] cursor-default">
                                {getRegionDisplayName(region)}
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" sideOffset={4}>
                              <p className="font-mono text-xs">{region}</p>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* CRIS Batch */}
          {hasCrisBatch && (
            <div className={cn(
              'rounded-lg border overflow-hidden',
              isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
            )}>
              <button
                className={cn(
                  'w-full flex items-center justify-between p-3 transition-colors',
                  isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
                )}
                onClick={() => toggleGroup('cris')}
              >
                <div className="flex items-center gap-2">
                  <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                    Cross-Region (CRIS)
                  </span>
                  <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    {crisBatchRegions.length} source regions
                  </span>
                </div>
                {expandedGroups['cris'] ? (
                  <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                ) : (
                  <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                )}
              </button>
              {expandedGroups['cris'] && (
                <div className={cn('px-3 pb-3 pt-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                  {Object.entries(groupByGeo(crisBatchRegions)).map(([geoKey, geoRegions]) => (
                    <div key={geoKey}>
                      <p className={cn('text-[10px] mb-2 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                        {geoDisplayNames[geoKey]} ({geoRegions.length})
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {geoRegions.sort().map(region => (
                          <Tooltip key={region} delayDuration={200}>
                            <TooltipTrigger asChild>
                              <Badge variant="secondary" className="text-[10px] cursor-default">
                                {getRegionDisplayName(region)}
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" sideOffset={4}>
                              <p className="font-mono text-xs">{region}</p>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                  <p className={cn('text-[10px] italic', isLight ? 'text-stone-400' : 'text-slate-500')}>
                    Batch jobs can be submitted from these source regions via CRIS
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Mantle Inference Section with CRIS-style grouped regions
function MantleInferenceSection({ mantleData }) {
  const [regionsExpanded, setRegionsExpanded] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const regions = mantleData.regions || []
  const grouped = groupRegionsByGeo(regions)

  const geoDisplayNames = {
    'US': 'United States', 'EU': 'Europe', 'APAC': 'Asia Pacific',
    'CA': 'Canada', 'SA': 'South America', 'MX': 'Mexico', 'ME': 'Middle East',
    'AF': 'Africa', 'GOV': 'GovCloud', 'Other': 'Other'
  }

  return (
    <div className="space-y-3">
      {/* Status metrics */}
      <div className="grid grid-cols-2 gap-2">
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {mantleData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Supported</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>Not Supported</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Total Regions</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-violet-700' : 'text-violet-400')}>{regions.length}</p>
        </div>
      </div>

      {/* Mantle Availability - single collapsible like Batch */}
      {mantleData.supported && regions.length > 0 && (
        <div className="space-y-2">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>Mantle Availability</p>
          <div className={cn(
            'rounded-lg border overflow-hidden',
            isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
          )}>
            <button
              className={cn(
                'w-full flex items-center justify-between p-3 transition-colors',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
              )}
              onClick={() => setRegionsExpanded(!regionsExpanded)}
            >
              <div className="flex items-center gap-2">
                <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                  Available Regions
                </span>
                <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                  {regions.length} regions
                </span>
              </div>
              {regionsExpanded ? (
                <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
              ) : (
                <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
              )}
            </button>
            {regionsExpanded && (
              <div className={cn('px-3 pb-3 pt-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                {Object.entries(grouped).map(([geoKey, geoRegions]) => (
                  <div key={geoKey}>
                    <p className={cn('text-[10px] mb-2 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      {geoDisplayNames[geoKey] || geoKey} ({geoRegions.length})
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {geoRegions.sort().map(region => (
                        <Tooltip key={region} delayDuration={200}>
                          <TooltipTrigger asChild>
                            <Badge variant="secondary" className="text-[10px] cursor-default">
                              {getRegionDisplayName(region)}
                            </Badge>
                          </TooltipTrigger>
                          <TooltipContent side="bottom" sideOffset={4}>
                            <p className="font-mono text-xs">{region}</p>
                          </TooltipContent>
                        </Tooltip>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Endpoint pattern info */}
      {mantleData.supported && (
        <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
          Endpoint: <code className={cn('font-mono px-1 py-0.5 rounded', isLight ? 'bg-stone-100' : 'bg-white/[0.06]')}>{mantleData.mantle_endpoint_pattern ?? 'bedrock-mantle.{region}.api.aws'}</code>
        </p>
      )}
    </div>
  )
}

// Reserved Capacity Section with commitment terms and available geos (simplified - no region details)
function ReservedCapacitySection({ reservedData }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const regions = reservedData.regions || []
  const commitments = reservedData.commitments || []
  const grouped = groupRegionsByGeo(regions)
  const geos = Object.keys(grouped)

  // Format commitment term for display (e.g., "1_month" -> "1 Month")
  const formatCommitment = (term) => {
    return term.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())
  }

  // Geo display names
  const geoDisplayNames = {
    'US': 'United States',
    'EU': 'Europe',
    'APAC': 'Asia Pacific',
    'CA': 'Canada',
    'SA': 'South America',
    'MX': 'Mexico',
    'ME': 'Middle East',
    'AF': 'Africa',
    'GOV': 'GovCloud',
    'Other': 'Other'
  }

  return (
    <div className="space-y-3">
      {/* Summary stats bar - only Status and Geographies for Reserved */}
      <div className="grid grid-cols-2 gap-2">
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Status</p>
          <div className="flex items-center gap-1 mt-1">
            {reservedData.supported ? (
              <><Check className="h-4 w-4 text-emerald-500" /><span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Available</span></>
            ) : (
              <><X className="h-4 w-4 text-red-400" /><span className={cn('text-sm font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>Not Available</span></>
            )}
          </div>
        </div>
        <div className={cn('rounded p-2', isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Geographies</p>
          <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{geos.length}</p>
        </div>
      </div>

      {/* Commitment terms */}
      {commitments.length > 0 && (
        <div>
          <p className={cn('text-xs font-medium mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Commitment Terms</p>
          <div className="flex flex-wrap gap-1.5">
            {commitments.map(term => (
              <Badge key={term} variant="secondary" className={cn(
                'text-xs',
                isLight ? 'bg-amber-100 text-amber-800 border-amber-200' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
              )}>
                {formatCommitment(term)}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Available Geos - simple tags, no region details */}
      {geos.length > 0 && (
        <div>
          <p className={cn('text-xs font-medium mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Available Geos</p>
          <div className="flex flex-wrap gap-1.5">
            {geos.sort().map(geo => (
              <Badge key={geo} variant="secondary" className="text-xs">
                {geoDisplayNames[geo] || geo}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Info note */}
      {reservedData.supported && (
        <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
          Reserved capacity requires a commitment term. See pricing tab for rates.
        </p>
      )}
    </div>
  )
}

// Consumption option explanations for info popover
const consumptionExplanations = {
  'In-Region': 'Data stays in one region. Use bedrock-runtime endpoint for standard inference or bedrock-mantle for managed endpoints.',
  'Cross-Region (CRIS)': 'Cross-Region Inference Service routes requests to available capacity across regions for higher throughput.',
  'Provisioned Throughput': 'Dedicated capacity with guaranteed throughput. Pay for reserved model units.',
  'Reserved Tiers': 'Committed capacity with 1-month or 3-month terms. Lower per-token cost in exchange for upfront commitment.',
}

// CRIS geo prefix display names and icons
const crisPrefixInfo = {
  'global': { name: 'Global', icon: Globe, description: 'Routes worldwide for maximum availability' },
  'us': { name: 'US', icon: MapPin, description: 'Routes within United States regions' },
  'eu': { name: 'EU', icon: MapPin, description: 'Routes within European regions' },
  'apac': { name: 'APAC', icon: MapPin, description: 'Routes within Asia Pacific regions' },
  'jp': { name: 'Japan', icon: MapPin, description: 'Routes within Japan regions' },
  'au': { name: 'Australia', icon: MapPin, description: 'Routes within Australia regions' },
}

// Helper to parse CRIS profile prefix
function getCrisProfilePrefix(profileId) {
  if (!profileId) return null
  const prefix = profileId.split('.')[0]?.toLowerCase()
  return prefix || null
}

// Helper to check if prefix is global vs geographic
function isGlobalPrefix(prefix) {
  return prefix === 'global'
}

// Collapsible region list component for nested sections
function CollapsibleRegionList({ label, regions, defaultExpanded = false, isLight, variant = 'default' }) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const grouped = groupRegionsByGeo(regions)
  
  const geoDisplayNames = {
    'US': 'United States', 'EU': 'Europe', 'APAC': 'Asia Pacific',
    'CA': 'Canada', 'SA': 'South America', 'MX': 'Mexico', 'ME': 'Middle East',
    'AF': 'Africa', 'GOV': 'GovCloud', 'Other': 'Other'
  }

  const variantStyles = {
    default: isLight 
      ? 'bg-white border-stone-200 hover:bg-stone-50' 
      : 'bg-white/[0.03] border-white/[0.06] hover:bg-white/[0.06]',
    nested: isLight
      ? 'bg-stone-50/50 border-stone-200/60 hover:bg-stone-100/50'
      : 'bg-white/[0.02] border-white/[0.04] hover:bg-white/[0.04]',
  }

  return (
    <div className={cn('rounded-lg border overflow-hidden', variantStyles[variant])}>
      <button
        className={cn('w-full flex items-center justify-between p-2.5 transition-colors')}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <span className={cn('font-medium text-xs', isLight ? 'text-stone-700' : 'text-slate-200')}>
            {label}
          </span>
          <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-400')}>
            {regions.length} {regions.length === 1 ? 'region' : 'regions'}
          </span>
        </div>
        {isExpanded ? (
          <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
        ) : (
          <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
        )}
      </button>
      {isExpanded && (
        <div className={cn('px-2.5 pb-2.5 pt-2 border-t space-y-2', isLight ? 'border-stone-200/60' : 'border-white/[0.04]')}>
          {Object.entries(grouped).map(([geoKey, geoRegions]) => (
            <div key={geoKey}>
              <p className={cn('text-[10px] mb-1.5 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>
                {geoDisplayNames[geoKey] || geoKey} ({geoRegions.length})
              </p>
              <div className="flex flex-wrap gap-1">
                {geoRegions.sort().map(region => (
                  <Tooltip key={region} delayDuration={200}>
                    <TooltipTrigger asChild>
                      <Badge variant="secondary" className="text-[10px] cursor-default">
                        {getRegionDisplayName(region)}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" sideOffset={4}>
                      <p className="font-mono text-xs">{region}</p>
                    </TooltipContent>
                  </Tooltip>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// In-Region Runtime API sub-section (On-Demand + Batch)
function InRegionRuntimeSection({ onDemandRegions, batchRegions, modelId, govcloudData, isLight }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const hasOnDemand = onDemandRegions.length > 0
  const hasBatch = batchRegions.length > 0
  const hasGovCloud = govcloudData?.supported && govcloudData?.inference_type === 'in_region' && (govcloudData?.regions?.length ?? 0) > 0
  const govcloudRegions = govcloudData?.regions || []
  
  const totalRegions = new Set([...onDemandRegions, ...batchRegions, ...(hasGovCloud ? govcloudRegions : [])]).size
  
  if (!hasOnDemand && !hasBatch && !hasGovCloud) return null

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-3 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Server className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-800' : 'text-white')}>
            Runtime API
          </span>
          <Tooltip delayDuration={200}>
            <TooltipTrigger asChild onClick={(e) => e.stopPropagation()}>
              <code className={cn(
                'text-[10px] px-1.5 py-0.5 rounded font-mono cursor-help',
                isLight ? 'bg-stone-100 text-stone-600 border border-stone-200 hover:bg-stone-200' : 'bg-white/[0.06] text-slate-400 border border-white/[0.08] hover:bg-white/[0.1]'
              )}>
                bedrock-runtime
              </code>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-[280px] text-xs space-y-1.5 p-3">
              <p className="font-mono text-[10px] opacity-70">bedrock-runtime.&#123;region&#125;.amazonaws.com</p>
              <p className="font-medium">Supported APIs:</p>
              <ul className="list-disc pl-3.5 space-y-0.5">
                <li>InvokeModel / InvokeModelWithResponseStream</li>
                <li>Converse / ConverseStream</li>
                <li>Chat Completions (v1/chat/completions)</li>
              </ul>
            </TooltipContent>
          </Tooltip>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-slate-400')}>
            {totalRegions} {totalRegions === 1 ? 'region' : 'regions'}
          </span>
          {isExpanded ? (
            <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
          ) : (
            <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
          )}
        </div>
      </button>
      {isExpanded && (
        <div className={cn('px-3 pb-3 pt-2 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
          {/* Model ID */}
          {modelId && (
            <div className={cn('rounded p-2 mb-2', isLight ? 'bg-stone-50 border border-stone-200' : 'bg-white/[0.02] border border-white/[0.04]')}>
              <CopyableText
                text={modelId}
                isLight={isLight}
                className={cn('text-xs font-mono', isLight ? 'text-stone-700 hover:text-stone-900' : 'text-[#c0c1c5] hover:text-white')}
              />
            </div>
          )}
          
          {/* On-Demand */}
          {hasOnDemand && (
            <CollapsibleRegionList 
              label="On-Demand" 
              regions={onDemandRegions} 
              isLight={isLight}
              variant="nested"
            />
          )}
          
          {/* Batch */}
          {hasBatch && (
            <CollapsibleRegionList 
              label="Batch" 
              regions={batchRegions} 
              isLight={isLight}
              variant="nested"
            />
          )}
          
          {/* GovCloud */}
          {hasGovCloud && (
            <CollapsibleRegionList 
              label="GovCloud" 
              regions={govcloudRegions} 
              isLight={isLight}
              variant="nested"
            />
          )}
        </div>
      )}
    </div>
  )
}

// In-Region Mantle API sub-section
function InRegionMantleSection({ mantleData, isLight }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const regions = mantleData?.regions || []
  
  if (!mantleData?.supported || regions.length === 0) return null

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-3 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Layers className={cn('h-3.5 w-3.5', isLight ? 'text-violet-600' : 'text-violet-400')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-800' : 'text-white')}>
            Mantle API
          </span>
          <Tooltip delayDuration={200}>
            <TooltipTrigger asChild onClick={(e) => e.stopPropagation()}>
              <code className={cn(
                'text-[10px] px-1.5 py-0.5 rounded font-mono cursor-help',
                isLight ? 'bg-violet-50 text-violet-600 border border-violet-200 hover:bg-violet-100' : 'bg-violet-500/10 text-violet-400 border border-violet-500/20 hover:bg-violet-500/20'
              )}>
                bedrock-mantle
              </code>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-[280px] text-xs space-y-1.5 p-3">
              <p className="font-mono text-[10px] opacity-70">bedrock-mantle.&#123;region&#125;.api.aws</p>
              <p className="font-medium">Supported APIs:</p>
              <ul className="list-disc pl-3.5 space-y-0.5">
                <li>Responses API (invoke / invoke-with-response-stream)</li>
                <li>Chat Completions API (v1/chat/completions)</li>
              </ul>
              <p className="opacity-70 italic">OpenAI-compatible managed endpoints</p>
            </TooltipContent>
          </Tooltip>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-slate-400')}>
            {regions.length} {regions.length === 1 ? 'region' : 'regions'}
          </span>
          {isExpanded ? (
            <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
          ) : (
            <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
          )}
        </div>
      </button>
      {isExpanded && (
        <div className={cn('px-3 pb-3 pt-2 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
          <CollapsibleRegionList 
            label="On-Demand" 
            regions={regions} 
            isLight={isLight}
            variant="nested"
          />
        </div>
      )}
    </div>
  )
}

// CRIS Global endpoint section
function CRISGlobalSection({ profiles, batchSupported, isLight }) {
  const [isExpanded, setIsExpanded] = useState(false)
  
  // Collect all source regions from global profiles
  const sourceRegions = new Set()
  const profileIds = []
  
  for (const { profile, sourceRegions: profileSourceRegions } of profiles) {
    const profileId = profile.profile_id || profile.inference_profile_id
    if (profileId) profileIds.push(profileId)
    for (const region of profileSourceRegions) {
      sourceRegions.add(region)
    }
  }
  
  const regionsList = [...sourceRegions].sort()
  
  if (regionsList.length === 0) return null

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-3 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-blue-600' : 'text-blue-400')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-800' : 'text-white')}>
            Global
          </span>
          <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-400')}>
            routes worldwide
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-slate-400')}>
            {regionsList.length} source {regionsList.length === 1 ? 'region' : 'regions'}
          </span>
          {isExpanded ? (
            <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
          ) : (
            <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
          )}
        </div>
      </button>
      {isExpanded && (
        <div className={cn('px-3 pb-3 pt-2 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
          {/* Endpoint ID */}
          {profileIds.length > 0 && (
            <div className={cn('rounded p-2 mb-2', isLight ? 'bg-stone-50 border border-stone-200' : 'bg-white/[0.02] border border-white/[0.04]')}>
              <p className={cn('text-[10px] mb-1 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>Endpoint ID</p>
              <CopyableText
                text={profileIds[0]}
                isLight={isLight}
                className={cn('text-xs font-mono', isLight ? 'text-stone-700 hover:text-stone-900' : 'text-[#c0c1c5] hover:text-white')}
              />
            </div>
          )}
          
          {/* On-Demand source regions */}
          <CollapsibleRegionList 
            label="On-Demand" 
            regions={regionsList} 
            isLight={isLight}
            variant="nested"
            defaultExpanded={true}
          />
          
          {/* Batch (if supported) */}
          {batchSupported && (
            <div className={cn(
              'rounded-lg border p-2',
              isLight ? 'bg-stone-50/50 border-stone-200/60' : 'bg-white/[0.02] border-white/[0.04]'
            )}>
              <div className="flex items-center gap-2">
                <span className={cn('font-medium text-xs', isLight ? 'text-stone-700' : 'text-slate-200')}>
                  Batch
                </span>
                <Badge variant="secondary" className="text-[9px]">Available</Badge>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// CRIS Geographic endpoint section (for a single geo like US, EU, etc.)
function CRISGeoSection({ geoKey, profiles, batchSupported, isLight }) {
  const [isExpanded, setIsExpanded] = useState(false)
  
  const info = crisPrefixInfo[geoKey] || { name: geoKey.toUpperCase(), icon: MapPin }
  const Icon = info.icon
  
  // Collect all source regions from this geo's profiles
  const sourceRegions = new Set()
  const profileIds = []
  
  for (const { profile, sourceRegions: profileSourceRegions } of profiles) {
    const profileId = profile.profile_id || profile.inference_profile_id
    if (profileId) profileIds.push(profileId)
    for (const region of profileSourceRegions) {
      sourceRegions.add(region)
    }
  }
  
  const regionsList = [...sourceRegions].sort()
  
  if (regionsList.length === 0) return null

  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-3 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Icon className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-amber-400')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-800' : 'text-white')}>
            {info.name}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-slate-400')}>
            {regionsList.length} source {regionsList.length === 1 ? 'region' : 'regions'}
          </span>
          {isExpanded ? (
            <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
          ) : (
            <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-500' : 'text-slate-400')} />
          )}
        </div>
      </button>
      {isExpanded && (
        <div className={cn('px-3 pb-3 pt-2 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
          {/* Endpoint ID */}
          {profileIds.length > 0 && (
            <div className={cn('rounded p-2 mb-2', isLight ? 'bg-stone-50 border border-stone-200' : 'bg-white/[0.02] border border-white/[0.04]')}>
              <p className={cn('text-[10px] mb-1 font-medium', isLight ? 'text-stone-500' : 'text-slate-400')}>Endpoint ID</p>
              <CopyableText
                text={profileIds[0]}
                isLight={isLight}
                className={cn('text-xs font-mono', isLight ? 'text-stone-700 hover:text-stone-900' : 'text-[#c0c1c5] hover:text-white')}
              />
            </div>
          )}
          
          {/* Source regions */}
          <CollapsibleRegionList 
            label="Source Regions" 
            regions={regionsList} 
            isLight={isLight}
            variant="nested"
            defaultExpanded={true}
          />
          
          {/* Batch (if supported) */}
          {batchSupported && (
            <div className={cn(
              'rounded-lg border p-2',
              isLight ? 'bg-stone-50/50 border-stone-200/60' : 'bg-white/[0.02] border-white/[0.04]'
            )}>
              <div className="flex items-center gap-2">
                <span className={cn('font-medium text-xs', isLight ? 'text-stone-700' : 'text-slate-200')}>
                  Batch
                </span>
                <Badge variant="secondary" className="text-[9px]">Available</Badge>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Reserved Tiers Section with expandable geographic endpoints
function ReservedTiersSection({ reservedData, isLight }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const regions = reservedData.regions || []
  const commitments = reservedData.commitments || []
  
  // Group regions by geo to determine available geos
  const grouped = groupRegionsByGeo(regions)
  const availableGeos = Object.keys(grouped)
  
  // Geo display names
  const geoDisplayNames = {
    'US': 'United States',
    'EU': 'Europe',
    'APAC': 'Asia Pacific',
    'CA': 'Canada',
    'SA': 'South America',
    'MX': 'Mexico',
    'ME': 'Middle East',
    'AF': 'Africa',
    'GOV': 'GovCloud',
    'Other': 'Other'
  }
  
  // Format commitment term for display (e.g., "1_month" -> "1 Month")
  const formatCommitment = (term) => {
    return term.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())
  }
  
  // Determine if Global is available (reserved pricing is available for CRIS endpoints)
  // Global is available if there are any regions
  const hasGlobal = regions.length > 0
  
  return (
    <div className={cn(
      'rounded-lg border overflow-hidden border-l-2',
      isLight
        ? 'bg-white border-stone-200 border-l-purple-500'
        : 'bg-white/[0.02] border-white/[0.06] border-l-purple-500'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-3 transition-colors',
          isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Clock className={cn('h-3.5 w-3.5', isLight ? 'text-purple-600' : 'text-purple-400')} />
          <span className={cn('font-medium text-xs', isLight ? 'text-stone-800' : 'text-white')}>
            Reserved Tiers
          </span>
          {commitments.length > 0 && (
            <div className="flex gap-1">
              {commitments.slice(0, 2).map(term => (
                <Badge key={term} variant="secondary" className="text-[9px]">
                  {formatCommitment(term)}
                </Badge>
              ))}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-600' : 'text-slate-400')}>
            {regions.length} {regions.length === 1 ? 'region' : 'regions'}
          </span>
          {isExpanded ? (
            <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
          ) : (
            <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
          )}
        </div>
      </button>
      {isExpanded && (
        <div className={cn('px-3 pb-3 pt-1 border-t space-y-3', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
          {/* Global endpoint */}
          {hasGlobal && (
            <div className={cn(
              'rounded-lg border p-3',
              isLight ? 'bg-stone-50/50 border-stone-200' : 'bg-white/[0.02] border-white/[0.04]'
            )}>
              <div className="flex items-center gap-2 mb-2">
                <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-blue-600' : 'text-blue-400')} />
                <span className={cn('font-medium text-sm', isLight ? 'text-stone-800' : 'text-white')}>
                  Global
                </span>
                <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-400')}>
                  same price from any source region
                </span>
              </div>
              <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                Reserved capacity available from {regions.length} source {regions.length === 1 ? 'region' : 'regions'}
              </p>
            </div>
          )}
          
          {/* Geographic endpoints */}
          {availableGeos.length > 0 && (
            <div className="space-y-2">
              <p className={cn('text-[10px] font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>
                Geographic Endpoints
              </p>
              <div className="flex flex-wrap gap-2">
                {availableGeos.sort().map(geoKey => {
                  const geoRegions = grouped[geoKey]
                  return (
                    <Tooltip key={geoKey} delayDuration={200}>
                      <TooltipTrigger asChild>
                        <div className={cn(
                          'inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs cursor-default',
                          isLight ? 'bg-stone-100 border border-stone-200' : 'bg-white/[0.04] border border-white/[0.06]'
                        )}>
                          <MapPin className={cn('h-3 w-3', isLight ? 'text-purple-600' : 'text-purple-400')} />
                          <span className={cn('font-medium', isLight ? 'text-stone-700' : 'text-white')}>
                            {geoDisplayNames[geoKey] || geoKey}
                          </span>
                          <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-400')}>
                            ({geoRegions.length})
                          </span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="bottom" className="max-w-xs">
                        <div className="space-y-1">
                          <p className="font-medium text-xs">{geoDisplayNames[geoKey] || geoKey}</p>
                          <div className="flex flex-wrap gap-1">
                            {geoRegions.sort().map(region => (
                              <span key={region} className="text-[10px] px-1.5 py-0.5 rounded bg-white/10">
                                {getRegionDisplayName(region)}
                              </span>
                            ))}
                          </div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  )
                })}
              </div>
            </div>
          )}
          
          {/* Info note */}
          <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
            Reserved capacity requires a commitment term. See pricing tab for rates.
          </p>
        </div>
      )}
    </div>
  )
}

// Main restructured Availability Summary component
function AvailabilitySummary({ model, getPricingForModel, preferredRegion = 'us-east-1' }) {
  const [expandedSections, setExpandedSections] = useState({ inRegion: true, cris: true })
  const { theme } = useTheme()
  const isLight = theme === 'light'

  // Derive CRIS-specific batch support from pricing data
  // BUG: model.availability.batch.supported is a general flag that doesn't distinguish
  // in-region batch from CRIS batch. Until the backend separates these, we check
  // pricing groups directly. See: "Batch Global"/"Batch Geo" = CRIS batch,
  // "Batch" = in-region batch.
  const hasCrisBatch = (() => {
    const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
    const fullPricing = pricingResult?.fullPricing
    if (!fullPricing?.regions) return false
    for (const regionData of Object.values(fullPricing.regions)) {
      if (!regionData?.pricing_groups) continue
      for (const groupName of Object.keys(regionData.pricing_groups)) {
        if (groupName.startsWith('Batch') && (groupName.includes('Global') || groupName.includes('Geo'))) {
          return true
        }
      }
    }
    return false
  })()

  // Extract all availability data
  const isMantleOnly = model.availability?.mantle?.only
  const hideInRegion = model.availability?.hide_in_region ?? false
  const onDemandRegions = model.availability?.on_demand?.regions ?? model.in_region ?? []
  const crisData = model.availability?.cross_region ?? model.cross_region_inference ?? {}
  const batchData = model.availability?.batch ?? model.batch_inference_supported ?? {}
  const mantleData = model.availability?.mantle ?? {}
  const govcloudData = model.availability?.govcloud ?? {}
  const reservedData = model.availability?.reserved ?? {}
  const provisionedData = model.availability?.provisioned ?? {}
  
  // In-region batch regions
  const inRegionBatchRegions = batchData.regions ?? batchData.supported_regions ?? []
  
  // Process CRIS profiles - group by prefix
  const profiles = crisData.profiles || []
  const profilesMap = new Map()
  
  for (const profile of profiles) {
    const profileId = profile.profile_id || profile.inference_profile_id
    if (!profileId) continue
    
    const prefix = getCrisProfilePrefix(profileId)
    const sourceRegion = profile.source_region || profile.region
    
    const existing = profilesMap.get(profileId)
    if (existing) {
      if (sourceRegion) existing.sourceRegions.add(sourceRegion)
    } else {
      profilesMap.set(profileId, {
        profile,
        prefix,
        sourceRegions: sourceRegion ? new Set([sourceRegion]) : new Set()
      })
    }
  }
  
  // Group profiles by prefix type (global vs geo)
  const globalProfiles = []
  const geoProfiles = {} // { us: [], eu: [], ... }
  
  for (const [, data] of profilesMap) {
    if (isGlobalPrefix(data.prefix)) {
      globalProfiles.push(data)
    } else if (data.prefix) {
      if (!geoProfiles[data.prefix]) geoProfiles[data.prefix] = []
      geoProfiles[data.prefix].push(data)
    }
  }
  
  // Check if GovCloud uses CRIS
  const hasGovCloudCris = govcloudData?.supported && govcloudData?.inference_type === 'cris' && (govcloudData?.regions?.length ?? 0) > 0
  
  // Determine what sections to show
  const showInRegion = !isMantleOnly && !hideInRegion && (onDemandRegions.length > 0 || inRegionBatchRegions.length > 0 || mantleData?.supported)
  const showCris = !isMantleOnly && crisData.supported && (globalProfiles.length > 0 || Object.keys(geoProfiles).length > 0 || hasGovCloudCris)
  const showProvisioned = !isMantleOnly && provisionedData?.supported
  const showReserved = !isMantleOnly && reservedData?.supported
  
  // Calculate totals for display
  const inRegionTotal = new Set([...onDemandRegions, ...inRegionBatchRegions, ...(mantleData?.regions || [])]).size
  const crisSourceRegions = crisData.regions?.length ?? 0

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  return (
    <div className="space-y-2">
      {/* Header with info button and doc link */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>
            Consumption Options
          </span>
          <a
            href="https://docs.aws.amazon.com/bedrock/latest/userguide/model-availability-compatibility.html"
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              'text-[10px] flex items-center gap-0.5 transition-colors',
              isLight ? 'text-blue-600 hover:text-blue-700' : 'text-blue-400 hover:text-blue-300'
            )}
          >
            <ExternalLink className="h-2.5 w-2.5" />
            Docs
          </a>
        </div>
        <Popover>
          <PopoverTrigger asChild>
            <button
              className={cn(
                'p-1 rounded-md transition-colors',
                isLight
                  ? 'hover:bg-stone-100 text-stone-400 hover:text-stone-600'
                  : 'hover:bg-white/[0.06] text-slate-500 hover:text-slate-300'
              )}
            >
              <Info className="h-3.5 w-3.5" />
            </button>
          </PopoverTrigger>
          <PopoverContent
            side="left"
            align="start"
            className={cn(
              'w-72 p-3',
              isLight ? 'bg-white border-stone-200' : 'bg-[#1c1d1f] border-white/[0.08]'
            )}
          >
            <div className="space-y-2">
              <h4 className={cn('text-xs font-semibold', isLight ? 'text-stone-700' : 'text-white')}>
                Consumption Options
              </h4>
              {Object.entries(consumptionExplanations).map(([label, explanation]) => (
                <div key={label} className="space-y-0.5">
                  <div className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>
                    {label}
                  </div>
                  <div className={cn('text-[11px]', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    {explanation}
                  </div>
                </div>
              ))}
            </div>
          </PopoverContent>
        </Popover>
      </div>

      {/* Routing legend */}
      <div className={cn(
        'flex flex-wrap gap-x-4 gap-y-1 px-2.5 py-1.5 rounded-md text-[10px]',
        isLight ? 'bg-stone-50 border border-stone-200/60' : 'bg-white/[0.02] border border-white/[0.04]'
      )}>
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" />
          <span className={isLight ? 'text-stone-600' : 'text-slate-400'}>In-Region — data stays in one region</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
          <span className={isLight ? 'text-stone-600' : 'text-slate-400'}>Cross-Region — routes across regions</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />
          <span className={isLight ? 'text-stone-600' : 'text-slate-400'}>Provisioned — dedicated capacity</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
          <span className={isLight ? 'text-stone-600' : 'text-slate-400'}>Reserved — committed terms</span>
        </span>
      </div>

      {/* Mantle-only notice */}
      {isMantleOnly && (
        <div className={cn(
          'px-2.5 py-1.5 rounded-md text-xs mb-2',
          isLight
            ? 'bg-violet-50 text-violet-700 border border-violet-200'
            : 'bg-violet-500/10 text-violet-400 border border-violet-500/20'
        )}>
          This model is available exclusively via Mantle Inference
        </div>
      )}

      {/* ===== IN-REGION SECTION ===== */}
      {(showInRegion || mantleData?.supported) && (
        <div className={cn(
          'rounded-lg border overflow-hidden border-l-2',
          isLight
            ? 'bg-white border-stone-200 border-l-emerald-500'
            : 'bg-white/[0.02] border-white/[0.06] border-l-emerald-500'
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between p-3 transition-colors',
              isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
            )}
            onClick={() => toggleSection('inRegion')}
          >
            <div className="flex items-center gap-2">
              <MapPin className={cn('h-3.5 w-3.5', isLight ? 'text-emerald-600' : 'text-emerald-400')} />
              <span className={cn('font-medium text-xs', isLight ? 'text-stone-800' : 'text-white')}>
                In-Region
              </span>
            </div>
            <div className="flex items-center gap-2">
              {inRegionBatchRegions.length > 0 && (
                <span className={cn(
                  'text-[9px] px-1.5 py-0.5 rounded font-medium',
                  isLight ? 'bg-indigo-50 text-indigo-600 border border-indigo-200' : 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                )}>
                  Batch
                </span>
              )}
              {inRegionTotal > 0 && (
                <span className={cn('text-[10px] font-mono font-semibold', isLight ? 'text-stone-600' : 'text-slate-400')}>
                  {inRegionTotal} {inRegionTotal === 1 ? 'region' : 'regions'}
                </span>
              )}
              {expandedSections.inRegion ? (
                <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
              ) : (
                <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
              )}
            </div>
          </button>
          {expandedSections.inRegion && (
            <div className={cn('px-3 pb-3 pt-1 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
              {/* Doc link for in-region */}
              <a
                href="https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html"
                target="_blank"
                rel="noopener noreferrer"
                className={cn(
                  'text-[10px] flex items-center gap-1 mb-2 transition-colors',
                  isLight ? 'text-blue-600 hover:text-blue-700' : 'text-blue-400 hover:text-blue-300'
                )}
              >
                <ExternalLink className="h-2.5 w-2.5" />
                Endpoint availability docs
              </a>
              
              {/* Runtime API (On-Demand + Batch) */}
              {!hideInRegion && !isMantleOnly && (onDemandRegions.length > 0 || inRegionBatchRegions.length > 0) && (
                <InRegionRuntimeSection
                  onDemandRegions={onDemandRegions}
                  batchRegions={inRegionBatchRegions}
                  modelId={model.model_id}
                  govcloudData={govcloudData}
                  isLight={isLight}
                />
              )}
              
              {/* Mantle API */}
              {mantleData?.supported && (
                <InRegionMantleSection mantleData={mantleData} isLight={isLight} />
              )}
            </div>
          )}
        </div>
      )}

      {/* ===== CROSS-REGION INFERENCE (CRIS) SECTION ===== */}
      {showCris && (
        <div className={cn(
          'rounded-lg border overflow-hidden border-l-2',
          isLight
            ? 'bg-white border-stone-200 border-l-blue-500'
            : 'bg-white/[0.02] border-white/[0.06] border-l-blue-500'
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between p-3 transition-colors',
              isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.06]'
            )}
            onClick={() => toggleSection('cris')}
          >
            <div className="flex items-center gap-2">
              <Split className={cn('h-3.5 w-3.5 rotate-90', isLight ? 'text-blue-600' : 'text-blue-400')} />
              <span className={cn('font-medium text-xs', isLight ? 'text-stone-800' : 'text-white')}>
                Cross-Region Inference
              </span>
            </div>
            <div className="flex items-center gap-2">
              {hasCrisBatch && (
                <span className={cn(
                  'text-[9px] px-1.5 py-0.5 rounded font-medium',
                  isLight ? 'bg-indigo-50 text-indigo-600 border border-indigo-200' : 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                )}>
                  Batch
                </span>
              )}
              {crisSourceRegions > 0 && (
                <span className={cn('text-[10px] font-mono font-semibold', isLight ? 'text-stone-600' : 'text-slate-400')}>
                  {crisSourceRegions} source {crisSourceRegions === 1 ? 'region' : 'regions'}
                </span>
              )}
              {expandedSections.cris ? (
                <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
              ) : (
                <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
              )}
            </div>
          </button>
          {expandedSections.cris && (
            <div className={cn('px-3 pb-3 pt-1 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
              {/* Global endpoints */}
              {globalProfiles.length > 0 && (
                <CRISGlobalSection
                  profiles={globalProfiles}
                  batchSupported={hasCrisBatch}
                  isLight={isLight}
                />
              )}
              
              {/* Geographic endpoints */}
              {Object.keys(geoProfiles).length > 0 && (
                <div className="space-y-2">
                  <p className={cn('text-[10px] font-medium', isLight ? 'text-blue-700' : 'text-blue-300')}>
                    Geographic Endpoints
                  </p>
                  {Object.entries(geoProfiles)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([geoKey, geoProfilesList]) => (
                      <CRISGeoSection
                        key={geoKey}
                        geoKey={geoKey}
                        profiles={geoProfilesList}
                        batchSupported={hasCrisBatch}
                        isLight={isLight}
                      />
                    ))
                  }
                </div>
              )}
              
              {/* GovCloud CRIS */}
              {hasGovCloudCris && (
                <CollapsibleRegionList 
                  label="GovCloud" 
                  regions={govcloudData.regions} 
                  isLight={isLight}
                />
              )}
            </div>
          )}
        </div>
      )}

      {/* ===== PROVISIONED THROUGHPUT ===== */}
      {showProvisioned && (
        <div className={cn(
          'flex items-center justify-between px-3 py-2 rounded-lg border border-l-2',
          isLight
            ? 'bg-white border-stone-200 border-l-amber-500'
            : 'bg-white/[0.02] border-white/[0.06] border-l-amber-500'
        )}>
          <div className="flex items-center gap-2">
            <Zap className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-amber-400')} />
            <span className={cn('font-medium text-sm', isLight ? 'text-stone-800' : 'text-white')}>
              Provisioned Throughput
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-600' : 'text-slate-400')}>
              {provisionedData.regions?.length ?? 0} {(provisionedData.regions?.length ?? 0) === 1 ? 'region' : 'regions'}
            </span>
            <span className={cn(
              'inline-flex items-center justify-center w-[18px] h-[18px] rounded-full',
              isLight ? 'bg-emerald-500 text-white' : 'bg-emerald-400 text-emerald-950'
            )}>
              <Check className="h-3 w-3 stroke-[2.5]" />
            </span>
          </div>
        </div>
      )}

      {/* ===== RESERVED TIERS ===== */}
      {showReserved && (
        <ReservedTiersSection reservedData={reservedData} isLight={isLight} />
      )}

      {/* App Inference Profiles — supplementary info banner */}
      <ApplicationInferenceProfileBanner />
    </div>
  )
}

// Expandable tag list component - shows first N items with "+X more" button
// Detects long-form capabilities (title: description) vs short tags automatically
function ExpandableTagList({ label, items, maxVisible = 10, isLight }) {
  const [expanded, setExpanded] = useState(false)

  if (!items || items.length === 0) {
    return (
      <div>
        <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>{label}</p>
        <span className={cn('text-sm', isLight ? 'text-stone-600' : 'text-slate-400')}>None specified</span>
      </div>
    )
  }

  // Detect if items are long descriptions (title: description format)
  const isLongFormat = items.some(item => item.length > 80)

  if (isLongFormat) {
    // Parse "Title: Description" entries from items
    // Items might be already split or could be one big string split on commas
    const entries = []
    const rawText = items.join(', ')

    // Split on ". " followed by uppercase letter (sentence/category boundary)
    const sentences = rawText.split(/\.\s+(?=[A-Z])/)
    for (const sentence of sentences) {
      const clean = sentence.trim().replace(/\.$/, '')
      if (!clean) continue
      const colonIdx = clean.indexOf(':')
      if (colonIdx > 0 && colonIdx < 80) {
        entries.push({
          title: clean.slice(0, colonIdx).trim(),
          description: clean.slice(colonIdx + 1).trim()
        })
      } else if (clean.length <= 80) {
        entries.push({ title: clean, description: '' })
      } else {
        entries.push({ title: clean.slice(0, 80) + '...', description: '' })
      }
    }

    const visibleEntries = expanded ? entries : entries.slice(0, 4)
    const remainingCount = entries.length - 4

    return (
      <div>
        <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>{label}</p>
        <div className="space-y-2">
          {visibleEntries.map((entry, i) => (
            <div key={i} className={cn(
              'rounded-md px-3 py-2 text-xs',
              isLight ? 'bg-stone-100/80' : 'bg-white/[0.04]'
            )}>
              <span className={cn('font-medium', isLight ? 'text-stone-800' : 'text-slate-200')}>
                {entry.title}
              </span>
              {entry.description && (
                <span className={cn('ml-1', isLight ? 'text-stone-500' : 'text-slate-400')}>
                  — {entry.description}
                </span>
              )}
            </div>
          ))}
          {remainingCount > 0 && !expanded && (
            <button
              onClick={() => setExpanded(true)}
              className={cn(
                'text-xs font-medium px-3 py-1 rounded-md transition-colors',
                isLight
                  ? 'text-amber-700 hover:bg-amber-50'
                  : 'text-[#1A9E7A] hover:bg-[#1A9E7A]/10'
              )}
            >
              +{remainingCount} more
            </button>
          )}
          {expanded && entries.length > 4 && (
            <button
              onClick={() => setExpanded(false)}
              className={cn(
                'text-xs font-medium px-3 py-1 rounded-md transition-colors',
                isLight
                  ? 'text-stone-500 hover:bg-stone-100'
                  : 'text-slate-500 hover:bg-white/5'
              )}
            >
              Show less
            </button>
          )}
        </div>
      </div>
    )
  }

  // Short tags — original badge layout
  const visibleItems = expanded ? items : items.slice(0, maxVisible)
  const remainingCount = items.length - maxVisible

  return (
    <div>
      <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {visibleItems.map(item => (
          <Badge key={item} variant="secondary" className="text-xs">{formatLabel(item)}</Badge>
        ))}
        {remainingCount > 0 && !expanded && (
          <button
            onClick={() => setExpanded(true)}
            className={cn(
              'inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium transition-colors',
              isLight
                ? 'bg-amber-100 text-amber-700 hover:bg-amber-200 border border-amber-200'
                : 'bg-[#1A9E7A]/20 text-[#1A9E7A] hover:bg-[#1A9E7A]/30 border border-[#1A9E7A]/30'
            )}
          >
            +{remainingCount} more
          </button>
        )}
        {expanded && items.length > maxVisible && (
          <button
            onClick={() => setExpanded(false)}
            className={cn(
              'inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium transition-colors',
              isLight
                ? 'bg-stone-100 text-stone-500 hover:bg-stone-200 border border-stone-200'
                : 'bg-white/5 text-slate-400 hover:bg-white/10 border border-white/10'
            )}
          >
            Show less
          </button>
        )}
      </div>
    </div>
  )
}

// Bedrock Features Section Component - dynamically displays all features from data
function BedrockFeaturesSection({ featureSupport }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  // Guard against missing or invalid feature_support
  if (!featureSupport || typeof featureSupport !== 'object') {
    return (
      <p className={cn('text-sm', isLight ? 'text-stone-500' : 'text-slate-400')}>
        No feature support data available
      </p>
    )
  }

  // Display labels for known feature keys (unknown keys will be auto-formatted)
  const featureLabels = {
    agent: 'Agents',
    flow: 'Flows',
    knowledge_base: 'Knowledge Base',
    guardrails: 'Guardrails',
    prompt_caching: 'Prompt Caching',
    batch_inference: 'Batch Inference',
    intelligent_routing: 'Intelligent Routing',
    model_evaluation: 'Model Evaluation',
    prompt_management: 'Prompt Management',
    latency_optimized: 'Latency Optimized',
    system_tools: 'System Tools',
  }

  // Display labels for sub-feature keys
  const subFeatureLabels = {
    isSupported: 'Supported',
    isStreamingSupported: 'Streaming',
    isParsingSupported: 'Parsing',
    isExternalSourcesSupported: 'External Sources',
    baseModelSupported: 'Base Model',
    crossRegionSupported: 'Cross-Region',
    customModelSupported: 'Custom Model',
  }

  // Auto-format unknown keys: snake_case -> Title Case
  const formatKey = (key) => {
    return key
      .replace(/_/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/^./, str => str.toUpperCase())
      .replace(/\bis\b/gi, '')
      .replace(/\bSupported\b/gi, '')
      .trim()
  }

  // Get all feature keys from the data
  const featureKeys = Object.keys(featureSupport).filter(key => {
    const data = featureSupport[key]
    // Include if it's an object with properties, an array with items, or a boolean true
    if (Array.isArray(data)) return data.length > 0
    if (typeof data === 'object' && data !== null) return Object.keys(data).length > 0
    return data === true
  })

  if (featureKeys.length === 0) {
    return (
      <p className={cn('text-sm', isLight ? 'text-stone-500' : 'text-slate-400')}>
        No feature support data available
      </p>
    )
  }

  // Helper to check if feature is supported
  const isSupported = (data) => {
    if (data === true) return true
    if (Array.isArray(data)) return data.length > 0
    if (!data || typeof data !== 'object') return false
    if (data.isSupported === true) return true
    if (data.baseModelSupported === true) return true
    return Object.values(data).some(v => v === true)
  }

  // Helper to get supported sub-features (boolean properties that are true)
  const getSupportedSubFeatures = (data) => {
    if (!data || typeof data !== 'object' || Array.isArray(data)) return []
    return Object.entries(data)
      .filter(([key, value]) => value === true && key !== 'isSupported')
      .map(([key]) => key)
  }

  // Sort features: supported first, then alphabetically
  const sortedFeatureKeys = [...featureKeys].sort((a, b) => {
    const aSupported = isSupported(featureSupport[a])
    const bSupported = isSupported(featureSupport[b])
    if (aSupported && !bSupported) return -1
    if (!aSupported && bSupported) return 1
    return a.localeCompare(b)
  })

  return (
    <div className="flex flex-wrap gap-2">
      {sortedFeatureKeys.map(featureKey => {
        const data = featureSupport[featureKey]
        const supported = isSupported(data)
        const label = featureLabels[featureKey] || formatKey(featureKey)

        // Handle array features (like system_tools)
        const isArrayFeature = Array.isArray(data)
        const arrayItems = isArrayFeature ? data : []

        // Get sub-features for object features
        const supportedSubFeatures = getSupportedSubFeatures(data)
        const hasDetails = supportedSubFeatures.length > 0 || arrayItems.length > 0

        const badge = (
          <div
            className={cn(
              'inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium cursor-default',
              supported
                ? isLight
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                : isLight
                  ? 'bg-stone-100 text-stone-400 border border-stone-200'
                  : 'bg-white/[0.06] text-slate-500 border border-white/[0.06]'
            )}
          >
            {supported ? <Check className="h-3 w-3" /> : <X className="h-3 w-3" />}
            <span>{label}</span>
            {hasDetails && (
              <span className={cn(
                'text-[10px]',
                isLight ? 'text-emerald-600' : 'text-emerald-500'
              )}>
                +{isArrayFeature ? arrayItems.length : supportedSubFeatures.length}
              </span>
            )}
          </div>
        )

        // Wrap with tooltip if there are details to show
        if (hasDetails) {
          return (
            <Tooltip key={featureKey}>
              <TooltipTrigger asChild>
                {badge}
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs">
                <div className="space-y-1">
                  <p className="font-medium text-xs">{label}</p>
                  <div className="flex flex-wrap gap-1">
                    {isArrayFeature ? (
                      // Show array items (e.g., system tools)
                      arrayItems.map((item, idx) => (
                        <span
                          key={idx}
                          className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300"
                        >
                          {typeof item === 'string' ? item : JSON.stringify(item)}
                        </span>
                      ))
                    ) : (
                      // Show sub-features
                      supportedSubFeatures.map(subKey => (
                        <span
                          key={subKey}
                          className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300"
                        >
                          <Check className="h-2 w-2" />
                          {subFeatureLabels[subKey] || formatKey(subKey)}
                        </span>
                      ))
                    )}
                  </div>
                </div>
              </TooltipContent>
            </Tooltip>
          )
        }

        return <div key={featureKey}>{badge}</div>
      })}
    </div>
  )
}

// Lifecycle Details Section Component
function LifecycleDetailsSection({ model, isLight }) {
  const [regionalDetailsExpanded, setRegionalDetailsExpanded] = useState(false)
  
  const lifecycle = model.lifecycle ?? model.model_lifecycle ?? {}
  const status = lifecycle.status || model.model_status || 'ACTIVE'
  const globalStatus = lifecycle.global_status
  const primaryStatus = lifecycle.primary_status
  const regionalStatus = lifecycle.regional_status
  const statusSummary = lifecycle.status_summary
  const releaseDate = lifecycle.release_date
  const eolDate = lifecycle.eol_date
  const legacyDate = lifecycle.legacy_date
  const extendedAccessDate = lifecycle.extended_access_date
  const recommendedReplacement = lifecycle.recommended_replacement
  const recommendedModelId = lifecycle.recommended_model_id
  
  // Check if we have regional data
  const hasRegionalData = globalStatus === 'MIXED' && regionalStatus && Object.keys(regionalStatus).length > 0
  
  // Helper to format timestamp to readable date
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return null
    // If it's a number (Unix timestamp), convert to date
    if (typeof timestamp === 'number') {
      const date = new Date(timestamp * 1000)
      return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
    }
    // If it's an ISO string, parse and format
    if (typeof timestamp === 'string') {
      const date = new Date(timestamp)
      if (!isNaN(date.getTime())) {
        return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
      }
    }
    return timestamp
  }
  
  // Helper to format date for regional display (shorter format)
  const formatShortDate = (dateStr) => {
    if (!dateStr) return null
    // If it's a timestamp number
    if (typeof dateStr === 'number') {
      const date = new Date(dateStr * 1000)
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    }
    // Try to parse and reformat string dates
    try {
      const date = new Date(dateStr)
      if (!isNaN(date.getTime())) {
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      }
    } catch {
      // If parsing fails, return as-is
    }
    return dateStr
  }
  
  // Helper to get status styles
  const getStatusStyles = (statusValue) => {
    const normalizedStatus = (statusValue || 'ACTIVE').toUpperCase()
    switch (normalizedStatus) {
      case 'ACTIVE':
        return isLight
          ? 'bg-emerald-100 text-emerald-700 border border-emerald-200'
          : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
      case 'LEGACY':
        return isLight
          ? 'bg-amber-100 text-amber-700 border border-amber-200'
          : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
      case 'EOL':
        return isLight
          ? 'bg-red-100 text-red-700 border border-red-200'
          : 'bg-red-500/20 text-red-400 border border-red-500/30'
      case 'MIXED':
        return isLight
          ? 'bg-purple-100 text-purple-700 border border-purple-200'
          : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
      default:
        return isLight
          ? 'bg-stone-100 text-stone-700 border border-stone-200'
          : 'bg-white/10 text-slate-400 border border-white/20'
    }
  }
  
  const getStatusLabel = (statusValue) => {
    const normalizedStatus = (statusValue || 'ACTIVE').toUpperCase()
    switch (normalizedStatus) {
      case 'ACTIVE': return 'Active'
      case 'LEGACY': return 'Legacy'
      case 'EOL': return 'End of Life'
      case 'MIXED': return 'Mixed'
      default: return normalizedStatus
    }
  }
  
  // Get status badge colors for summary badges
  const getStatusBadgeStyles = (statusValue) => {
    const normalizedStatus = (statusValue || 'ACTIVE').toUpperCase()
    switch (normalizedStatus) {
      case 'ACTIVE':
        return isLight
          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
          : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
      case 'LEGACY':
        return isLight
          ? 'bg-amber-50 text-amber-700 border-amber-200'
          : 'bg-amber-500/15 text-amber-400 border-amber-500/30'
      case 'EOL':
        return isLight
          ? 'bg-red-50 text-red-700 border-red-200'
          : 'bg-red-500/15 text-red-400 border-red-500/30'
      default:
        return isLight
          ? 'bg-stone-50 text-stone-600 border-stone-200'
          : 'bg-white/5 text-slate-400 border-white/10'
    }
  }
  
  const formattedReleaseDate = formatTimestamp(releaseDate)
  
  // Get status order for display (LEGACY, ACTIVE, EOL)
  const statusOrder = ['LEGACY', 'ACTIVE', 'EOL']
  
  return (
    <div className="space-y-3">
      {/* Status Section */}
      <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
        <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Status</p>
        
        {hasRegionalData ? (
          // Mixed status with compact inline summary
          <div className="space-y-2">
            {/* Compact inline status summary */}
            <div className="flex items-center gap-3 flex-wrap">
              {statusOrder.map(statusKey => {
                const regions = statusSummary?.[statusKey] || []
                if (regions.length === 0) return null
                
                // Get status indicator color
                const getStatusDotColor = (status) => {
                  switch (status) {
                    case 'ACTIVE': return isLight ? 'bg-emerald-500' : 'bg-emerald-400'
                    case 'LEGACY': return isLight ? 'bg-amber-500' : 'bg-amber-400'
                    case 'EOL': return isLight ? 'bg-red-500' : 'bg-red-400'
                    default: return isLight ? 'bg-stone-400' : 'bg-slate-400'
                  }
                }
                
                const getStatusTextColor = (status) => {
                  switch (status) {
                    case 'ACTIVE': return isLight ? 'text-emerald-700' : 'text-emerald-400'
                    case 'LEGACY': return isLight ? 'text-amber-700' : 'text-amber-400'
                    case 'EOL': return isLight ? 'text-red-700' : 'text-red-400'
                    default: return isLight ? 'text-stone-600' : 'text-slate-400'
                  }
                }
                
                const StatusIcon = statusKey === 'LEGACY' ? AlertTriangle : statusKey === 'EOL' ? AlertCircle : null
                
                return (
                  <div key={statusKey} className="flex items-center gap-1.5">
                    {StatusIcon ? (
                      <StatusIcon className={cn('h-3 w-3', getStatusTextColor(statusKey))} />
                    ) : (
                      <span className={cn('w-2 h-2 rounded-full', getStatusDotColor(statusKey))} />
                    )}
                    <span className={cn('text-xs font-medium', getStatusTextColor(statusKey))}>
                      {getStatusLabel(statusKey)}
                    </span>
                    <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      in {regions.length} {regions.length === 1 ? 'region' : 'regions'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        ) : (
          // Single status (backward compatible)
          <div className="flex items-center justify-between">
            <span className={cn(
              'px-2.5 py-1 rounded-full text-xs font-semibold',
              getStatusStyles(status)
            )}>
              {getStatusLabel(status)}
            </span>
          </div>
        )}
      </div>
      
      {/* Regional Details (collapsible) */}
      {hasRegionalData && (
        <div className={cn(
          'rounded-lg border overflow-hidden',
          isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between p-3 transition-colors',
              isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
            )}
            onClick={() => setRegionalDetailsExpanded(!regionalDetailsExpanded)}
          >
            <div className="flex items-center gap-2">
              <Globe className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                Regional Details
              </span>
              <Badge variant="secondary" className="text-[10px]">
                {Object.keys(regionalStatus).length} regions
              </Badge>
            </div>
            {regionalDetailsExpanded ? (
              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            ) : (
              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            )}
          </button>
          
          {regionalDetailsExpanded && (
            <div className={cn(
              'px-3 pb-3 border-t space-y-3',
              isLight ? 'border-stone-200' : 'border-white/[0.06]'
            )}>
              {/* Group regions by status */}
              {statusOrder.map(statusKey => {
                const regions = statusSummary?.[statusKey] || []
                if (regions.length === 0) return null
                
                // Get status indicator styles
                const getStatusIndicator = (status) => {
                  switch (status) {
                    case 'ACTIVE': return { icon: null, dotColor: isLight ? 'bg-emerald-500' : 'bg-emerald-400' }
                    case 'LEGACY': return { icon: AlertTriangle, iconColor: isLight ? 'text-amber-600' : 'text-amber-400' }
                    case 'EOL': return { icon: AlertCircle, iconColor: isLight ? 'text-red-600' : 'text-red-400' }
                    default: return { icon: null, dotColor: isLight ? 'bg-stone-400' : 'bg-slate-400' }
                  }
                }
                
                const indicator = getStatusIndicator(statusKey)
                const StatusIcon = indicator.icon
                
                return (
                  <div key={statusKey} className="pt-3">
                    <div className="flex items-center gap-2 mb-2">
                      {StatusIcon ? (
                        <StatusIcon className={cn('h-3.5 w-3.5', indicator.iconColor)} />
                      ) : (
                        <span className={cn('w-2.5 h-2.5 rounded-full', indicator.dotColor)} />
                      )}
                      <span className={cn(
                        'text-xs font-semibold',
                        statusKey === 'ACTIVE' ? (isLight ? 'text-emerald-700' : 'text-emerald-400') :
                        statusKey === 'LEGACY' ? (isLight ? 'text-amber-700' : 'text-amber-400') :
                        statusKey === 'EOL' ? (isLight ? 'text-red-700' : 'text-red-400') :
                        (isLight ? 'text-stone-700' : 'text-slate-300')
                      )}>
                        {getStatusLabel(statusKey)}
                      </span>
                      <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                        ({regions.length})
                      </span>
                    </div>
                    
                    <div className="grid gap-1">
                      {regions.sort().map(region => {
                        const regionData = regionalStatus[region] || {}
                        const regionName = getRegionDisplayName(region)
                        
                        // Collect date info for this region
                        const dateInfo = []
                        const regionStatus = regionData.status || 'ACTIVE'
                        if (regionData.legacy_date && (regionStatus === 'LEGACY' || regionStatus === 'EOL')) {
                          dateInfo.push({ label: 'Legacy', date: formatShortDate(regionData.legacy_date) })
                        }
                        if (regionData.eol_date && (regionStatus === 'LEGACY' || regionStatus === 'EOL')) {
                          dateInfo.push({ label: 'EOL', date: formatShortDate(regionData.eol_date) })
                        }
                        
                        return (
                          <div
                            key={region}
                            className={cn(
                              'flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-md text-xs',
                              isLight ? 'bg-stone-50/80' : 'bg-white/[0.02]'
                            )}
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              <span className={cn('font-medium', isLight ? 'text-stone-700' : 'text-slate-200')}>
                                {regionName}
                              </span>
                              <span className={cn('font-mono text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                                {region}
                              </span>
                            </div>
                            {dateInfo.length > 0 && (
                              <div className="flex items-center gap-2 flex-shrink-0">
                                {dateInfo.map((info, idx) => (
                                  <span
                                    key={idx}
                                    className={cn(
                                      'text-[10px] px-1.5 py-0.5 rounded',
                                      info.label === 'EOL'
                                        ? (isLight ? 'bg-red-100 text-red-700' : 'bg-red-500/15 text-red-400')
                                        : (isLight ? 'bg-amber-100 text-amber-700' : 'bg-amber-500/15 text-amber-400')
                                    )}
                                  >
                                    {info.label}: {info.date}
                                  </span>
                                ))}
                              </div>
                            )}
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
      )}
      
      {/* Release Date */}
      {formattedReleaseDate && (
        <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]')}>
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Release Date</p>
          <p className={cn('text-sm font-medium mt-1', isLight ? 'text-stone-800' : 'text-white')}>
            {formattedReleaseDate}
          </p>
        </div>
      )}
      
      {/* Legacy Date (only show if not mixed - backward compat for non-regional data) */}
      {legacyDate && !hasRegionalData && (
        <div className={cn('rounded-lg p-3 border', isLight ? 'bg-amber-50 border-amber-200' : 'bg-amber-500/10 border border-amber-500/20')}>
          <p className={cn('text-xs', isLight ? 'text-amber-700' : 'text-amber-400')}>Legacy Date</p>
          <p className={cn('text-sm font-medium mt-1', isLight ? 'text-amber-800' : 'text-amber-300')}>
            {legacyDate}
          </p>
        </div>
      )}
      
      {/* Extended Access Date (only show if not mixed - backward compat) */}
      {extendedAccessDate && !hasRegionalData && (
        <div className={cn('rounded-lg p-3 border', isLight ? 'bg-orange-50 border-orange-200' : 'bg-orange-500/10 border border-orange-500/20')}>
          <p className={cn('text-xs', isLight ? 'text-orange-700' : 'text-orange-400')}>Public Extended Access Date</p>
          <p className={cn('text-sm font-medium mt-1', isLight ? 'text-orange-800' : 'text-orange-300')}>
            {extendedAccessDate}
          </p>
        </div>
      )}
      
      {/* EOL Date (only show if not mixed and not active - backward compat) */}
      {eolDate && !hasRegionalData && (status !== 'ACTIVE' && globalStatus !== 'ACTIVE') && (
        <div className={cn('rounded-lg p-3 border', isLight ? 'bg-red-50 border-red-200' : 'bg-red-500/10 border border-red-500/20')}>
          <p className={cn('text-xs', isLight ? 'text-red-700' : 'text-red-400')}>End of Life Date</p>
          <p className={cn('text-sm font-medium mt-1', isLight ? 'text-red-800' : 'text-red-300')}>
            {eolDate}
          </p>
        </div>
      )}
      
      {/* Suggested Replacement */}
      {recommendedReplacement && (
        <div className={cn('rounded-lg p-3 border', isLight ? 'bg-blue-50 border-blue-200' : 'bg-blue-500/10 border border-blue-500/20')}>
          <p className={cn('text-xs', isLight ? 'text-blue-700' : 'text-blue-400')}>Suggested Replacement</p>
          <p className={cn('text-sm font-medium mt-1', isLight ? 'text-blue-800' : 'text-blue-300')}>
            {recommendedReplacement}
          </p>
          {recommendedModelId && (
            <p className={cn('text-[10px] font-mono mt-1', isLight ? 'text-blue-600' : 'text-blue-400/70')}>
              {recommendedModelId}
            </p>
          )}
        </div>
      )}
      
      {/* No lifecycle data message */}
      {!formattedReleaseDate && !legacyDate && !extendedAccessDate && !eolDate && !recommendedReplacement && !hasRegionalData && (
        <p className={cn('text-sm', isLight ? 'text-stone-500' : 'text-slate-400')}>
          No additional lifecycle information available
        </p>
      )}
    </div>
  )
}

function SpecsTab({ model, user, getPricingForModel, preferredRegion }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  
  // Get pricing result to check if pricing data is available
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion)?.summary : null
  
  // Data extraction
  const inputModalities = model.modalities?.input_modalities ?? model.model_modalities?.input_modalities ?? []
  const outputModalities = model.modalities?.output_modalities ?? model.model_modalities?.output_modalities ?? []
  const capabilities = model.capabilities ?? model.model_capabilities ?? []
  const useCasesRaw = model.use_cases ?? model.model_use_cases ?? []
  const useCases = [...new Set(
    useCasesRaw
      .flatMap(uc => uc.includes(',') ? uc.split(',') : [uc])
      .map(s => s.trim())
      .map(s => s.replace(/^and\s+/i, ''))
      .filter(s => s.length > 0 && s.length < 120)
      .map(s => s.charAt(0).toUpperCase() + s.slice(1))
  )]
  const languages = model.languages ?? model.languages_supported ?? []
  const documentationLinks = model.docs ?? model.documentation_links ?? {}
  const customizations = model.customization?.customization_supported || []

  // Category header component
  const CategoryHeader = ({ icon: Icon, title }) => (
    <div className={cn(
      'flex items-center gap-2 mb-3 pb-2 border-b',
      isLight ? 'border-stone-200' : 'border-white/10'
    )}>
      <Icon className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
      <h3 className={cn('text-sm font-semibold tracking-wide uppercase', isLight ? 'text-stone-700' : 'text-slate-200')}>
        {title}
      </h3>
    </div>
  )

  return (
    <ScrollArea className="h-full">
      <div className="p-6">
        {/* No pricing warning banner */}
        {(!pricingResult || (pricingResult.inputPrice == null && pricingResult.outputPrice == null && pricingResult.imagePrice == null && pricingResult.videoPrice == null)) && (
          <div className={cn(
            'mb-4 p-3 rounded-lg border flex items-start gap-2',
            isLight 
              ? 'bg-amber-50 border-amber-200 text-amber-800' 
              : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
          )}>
            <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium">No pricing data available</p>
              <p className={cn('text-xs mt-0.5', isLight ? 'text-amber-700' : 'text-amber-400/80')}>
                This model is not listed in the AWS Pricing API. Please verify model availability before use.
              </p>
            </div>
          </div>
        )}
        {/* Two-column grid layout */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* LEFT COLUMN */}
          <div className="space-y-6">
            {/* MODEL CAPABILITIES */}
            <div>
              <CategoryHeader icon={Cpu} title="Model Capabilities" />
              <div className="space-y-3">
                {/* Modalities - Always expanded */}
                <CollapsibleSection 
                  title="Input & Output Modalities" 
                  icon={Layers} 
                  defaultExpanded={true}
                  dataSource={<>Source: <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">Bedrock ListFoundationModels API</a></>}
                >
                  <div className="space-y-3">
                    <div>
                      <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Input</p>
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
                      <p className={cn('text-xs mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Output</p>
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

                {/* Capabilities - Expanded by default */}
                {capabilities.length > 0 && (
                  <CollapsibleSection 
                    title="Capabilities" 
                    icon={Zap} 
                    defaultExpanded={true}
                    dataSource={<>Source: <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">ListFoundationModels API</a> (includes data from AWS Console)</>}
                  >
                    <ExpandableTagList
                      label=""
                      items={capabilities}
                      maxVisible={8}
                      isLight={isLight}
                    />
                  </CollapsibleSection>
                )}

                {/* Use Cases - Expanded by default */}
                {useCases.length > 0 && (
                  <CollapsibleSection 
                    title="Use Cases" 
                    icon={BookOpen} 
                    defaultExpanded={true}
                    dataSource={<>Source: <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">ListFoundationModels API</a> (includes data from AWS Console)</>}
                  >
                    <ExpandableTagList
                      label=""
                      items={useCases}
                      maxVisible={8}
                      isLight={isLight}
                    />
                  </CollapsibleSection>
                )}
              </div>
            </div>

            {/* FEATURES & INTEGRATIONS */}
            <div>
              <CategoryHeader icon={Wrench} title="Features & Integrations" />
              <div className="space-y-3">
                {/* Bedrock Features - Expanded by default */}
                {(model.features ?? model.feature_support) && (
                  <CollapsibleSection 
                    title="Bedrock Features" 
                    icon={Layers} 
                    defaultExpanded={true}
                    dataSource={<>Source: <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">ListFoundationModels API</a> (includes data from AWS Console)</>}
                  >
                    <BedrockFeaturesSection featureSupport={model.features ?? model.feature_support} />
                  </CollapsibleSection>
                )}

                {/* Languages - Expanded by default */}
                {languages.length > 0 && (
                  <CollapsibleSection 
                    title="Languages" 
                    icon={Languages} 
                    defaultExpanded={true}
                    dataSource={<>Source: <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">ListFoundationModels API</a> (includes data from AWS Console)</>}
                  >
                    <div className="flex flex-wrap gap-1.5">
                      {languages.map(lang => (
                        <Badge key={lang} variant="secondary" className="text-xs">{lang}</Badge>
                      ))}
                    </div>
                  </CollapsibleSection>
                )}

                {/* Customizations - Expanded by default */}
                {customizations.length > 0 && (
                  <CollapsibleSection 
                    title="Customizations" 
                    icon={Wrench} 
                    defaultExpanded={true}
                    dataSource={<>Source: <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">Bedrock ListFoundationModels API</a></>}
                  >
                    <div className="flex flex-wrap gap-1.5">
                      {customizations.map(custom => (
                        <Badge key={custom} variant="outline" className="text-xs">{custom}</Badge>
                      ))}
                    </div>
                  </CollapsibleSection>
                )}

                {/* Show placeholder if no features */}
                {!(model.features ?? model.feature_support) && languages.length === 0 && customizations.length === 0 && (
                  <div className={cn(
                    'rounded-lg p-4 text-center',
                    isLight ? 'bg-stone-50 border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
                  )}>
                    <p className={cn('text-sm', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      No feature data available
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN */}
          <div className="space-y-6">
            {/* AVAILABILITY & DEPLOYMENT */}
            <div>
              <CategoryHeader icon={Globe} title="Availability & Deployment" />
              <div className="space-y-3">
                {/* Regional Availability - Always expanded */}
                <CollapsibleSection 
                  title="Consumption Options" 
                  icon={Globe} 
                  defaultExpanded={true}
                  dataSource={<>Sources: <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">ListFoundationModels API</a>, <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListInferenceProfiles.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">ListInferenceProfiles API</a>, <a href="https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">AWS Pricing API</a>, <a href="https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">Mantle API</a></>}
                >
                  <AvailabilitySummary model={model} getPricingForModel={getPricingForModel} preferredRegion={preferredRegion} />
                </CollapsibleSection>


              </div>
            </div>

            {/* LIFECYCLE & RESOURCES */}
            <div>
              <CategoryHeader icon={Clock} title="Lifecycle & Resources" />
              <div className="space-y-3">
                {/* Lifecycle Details - Expanded by default */}
                <CollapsibleSection 
                  title="Status & Dates" 
                  icon={Clock} 
                  defaultExpanded={true}
                  dataSource={<>Sources: <a href="https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">AWS Model Lifecycle Docs</a>, <a href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">ListFoundationModels API</a></>}
                >
                  <LifecycleDetailsSection model={model} isLight={isLight} />
                </CollapsibleSection>

                {/* Documentation Links - Always visible (not collapsible) */}
                <div className={cn(
                  'rounded-lg overflow-hidden border p-3',
                  isLight
                    ? 'bg-stone-50/80 border-stone-200/80'
                    : 'bg-white/5 border-white/10'
                )}>
                  <div className="flex items-center gap-2 mb-3">
                    <FileText className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                      Documentation
                    </span>
                  </div>
                  <div className="space-y-2">
                    {Object.keys(documentationLinks).length > 0 ? (
                      <div className="flex flex-col gap-2">
                        {documentationLinks.aws_bedrock_guide && (
                          <a href={documentationLinks.aws_bedrock_guide} target="_blank" rel="noopener noreferrer"
                             className={cn(
                               'flex items-center gap-2 text-sm px-2 py-1.5 rounded-md transition-colors',
                               isLight
                                 ? 'text-blue-600 hover:bg-blue-50'
                                 : 'text-blue-400 hover:bg-blue-500/10'
                             )}>
                            <ExternalLink className="h-3.5 w-3.5 flex-shrink-0" />
                            AWS Bedrock Guide
                          </a>
                        )}
                        {documentationLinks.pricing_guide && (
                          <a href={documentationLinks.pricing_guide} target="_blank" rel="noopener noreferrer"
                             className={cn(
                               'flex items-center gap-2 text-sm px-2 py-1.5 rounded-md transition-colors',
                               isLight
                                 ? 'text-blue-600 hover:bg-blue-50'
                                 : 'text-blue-400 hover:bg-blue-500/10'
                             )}>
                            <ExternalLink className="h-3.5 w-3.5 flex-shrink-0" />
                            Pricing Guide
                          </a>
                        )}
                        {documentationLinks.provider_guide && (
                          <a href={documentationLinks.provider_guide} target="_blank" rel="noopener noreferrer"
                             className={cn(
                               'flex items-center gap-2 text-sm px-2 py-1.5 rounded-md transition-colors',
                               isLight
                                 ? 'text-blue-600 hover:bg-blue-50'
                                 : 'text-blue-400 hover:bg-blue-500/10'
                             )}>
                            <ExternalLink className="h-3.5 w-3.5 flex-shrink-0" />
                            Provider Documentation
                          </a>
                        )}
                      </div>
                    ) : (
                      <p className={cn('text-sm px-2', isLight ? 'text-stone-500' : 'text-slate-400')}>
                        No documentation links available
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
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
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{getRegionDisplayName(region)}</span>
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
  on_demand: { name: 'In Region Inference', icon: '🚀', color: 'text-emerald-500' },
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

function QuotasTab({ model, getPricingForModel, preferredRegion }) {
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedGeos, setExpandedGeos] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const quotas = model.quotas ?? model.model_service_quotas ?? {}
  const allRegions = Object.keys(quotas)
  
  // Get pricing result to check if pricing data is available
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion)?.summary : null

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
      // Skip per-day quotas (derived from per-minute, redundant)
      if (/per\s+day/i.test(quota.quota_name || '')) continue
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
        const regionName = (getRegionDisplayName(region) || '').toLowerCase()
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
                          <span className={cn('font-medium text-xs', isLight ? 'text-stone-800' : 'text-white')}>{getRegionDisplayName(region)}</span>
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
        {/* Internal banner */}
        <div className={cn(
          'flex items-start gap-2 px-3 py-2.5 rounded-lg text-xs mb-4',
          isLight
            ? 'bg-violet-50 text-violet-700 border border-violet-200'
            : 'bg-violet-500/10 text-violet-300 border border-violet-500/20'
        )}>
          <Info className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
          <span>These quotas are internal service quotas and may not reflect customer-facing limits.</span>
        </div>

        {/* No pricing warning banner */}
        {(!pricingResult || (pricingResult.inputPrice == null && pricingResult.outputPrice == null && pricingResult.imagePrice == null && pricingResult.videoPrice == null)) && (
          <div className={cn(
            'mb-4 p-3 rounded-lg border flex items-start gap-2',
            isLight 
              ? 'bg-amber-50 border-amber-200 text-amber-800' 
              : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
          )}>
            <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium">No pricing data available</p>
              <p className={cn('text-xs mt-0.5', isLight ? 'text-amber-700' : 'text-amber-400/80')}>
                This model is not listed in the AWS Pricing API. Please verify model availability before use.
              </p>
            </div>
          </div>
        )}

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
            {renderCategorySection('cross_region', 'Cross-Region Inference', Globe)}
            {renderCategorySection('on_demand', 'In Region Inference', Zap)}
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            {renderCategorySection('batch', 'Batch Inference', Layers)}
            {renderCategorySection('provisioned', 'Provisioned Throughput', Server)}
            {renderCategorySection('customization', 'Customization', Cpu)}
            {renderCategorySection('general', 'General', FileText)}
          </div>
        </div>

        {/* Data Source Info */}
        <div className={cn(
          'mt-6 pt-4 border-t flex items-start gap-1.5 text-[10px]',
          isLight 
            ? 'border-stone-200/60 text-stone-400' 
            : 'border-white/[0.04] text-slate-500'
        )}>
          <Info className="h-3 w-3 flex-shrink-0 mt-0.5" />
          <span>
            Source: <a href="https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">AWS Service Quotas API</a> (per-region)
          </span>
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
          unit: 'Per 1M tokens'
        })
      })
    }
    if (od.output_tokens) {
      od.output_tokens.forEach(p => {
        onDemand.push({
          type: 'output',
          description: p.description || 'Output Tokens',
          price: parseFloat(p.price),
          unit: 'Per 1M tokens'
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
      unit: 'Per 1M tokens'
    })
  }
  if (regionPricing.output_per_1k_tokens !== undefined) {
    onDemand.push({
      type: 'output',
      description: 'Output Tokens',
      price: regionPricing.output_per_1k_tokens,
      unit: 'Per 1M tokens'
    })
  }

  // Handle text sub-object
  if (regionPricing.text) {
    if (regionPricing.text.input_per_1k_tokens !== undefined) {
      onDemand.push({
        type: 'input',
        description: 'Input Tokens',
        price: regionPricing.text.input_per_1k_tokens,
        unit: 'Per 1M tokens'
      })
    }
    if (regionPricing.text.output_per_1k_tokens !== undefined) {
      onDemand.push({
        type: 'output',
        description: 'Output Tokens',
        price: regionPricing.text.output_per_1k_tokens,
        unit: 'Per 1M tokens'
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
    /cache[- ]?write[- ]?1h|1[- ]?h(?:our)?[- ]?cache/i.test(combined) ? '1h Cache Write' :
    /cache[- ]?write(?![- ]?1h)/i.test(combined) ? '5m Cache Write' :
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
    const displayPrice = typeof price === 'number' ? price * 1000 : price
    const text = `$${typeof displayPrice === 'number' ? (displayPrice < 0.01 ? displayPrice.toFixed(4) : displayPrice.toFixed(2)) : displayPrice} ${unit}`
    await navigator.clipboard.writeText(text)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(null), 1500)
  }

  return (
    <div>
      {items.map((item, idx) => {
        const rawPrice = typeof item._price === 'number' ? item._price * 1000 : item._price
        const priceStr = typeof rawPrice === 'number' ? (rawPrice < 0.01 ? rawPrice.toFixed(4) : rawPrice.toFixed(2)) : rawPrice || 'N/A'
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
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>{getRegionDisplayName(region)}</span>
          <span className={cn('text-xs font-mono', isLight ? 'text-stone-600' : 'text-slate-300')}>({region})</span>
        </div>
        <div className="flex items-center gap-3">
          {inputItem && outputItem && (
            <span className="text-xs text-emerald-600 dark:text-emerald-400">
              ${(inputItem.price * 1000) < 0.01 ? (inputItem.price * 1000).toFixed(4) : (inputItem.price * 1000).toFixed(2)} / ${(outputItem.price * 1000) < 0.01 ? (outputItem.price * 1000).toFixed(4) : (outputItem.price * 1000).toFixed(2)}
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
                _unit: (item.unit || 'per 1M tokens').replace(/1K tokens/gi, '1M tokens'),
                _raw: item.description,
              })).sort((a, b) => {
                const typeOrder = { input: 0, output: 1, other: 2 }
                return (typeOrder[a._type] ?? 2) - (typeOrder[b._type] ?? 2)
              })} isLight={isLight} />
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
  'On-Demand': { icon: '🚀', label: 'In Region' },
  'On-Demand Global': { icon: '🌐', label: 'In Region (Global/CRIS)' },
  'On-Demand Long Context': { icon: '📚', label: 'In Region Long Context' },
  'On-Demand GovCloud': { icon: '🏛️', label: 'GovCloud On-Demand' },
  'On-Demand GovCloud InRegion': { icon: '🏛️', label: 'GovCloud On-Demand' },
  'On-Demand Long Context GovCloud': { icon: '🏛️', label: 'GovCloud On-Demand (Long Context)' },
  'On-Demand Long Context GovCloud InRegion': { icon: '🏛️', label: 'GovCloud On-Demand (Long Context)' },
  'Batch': { icon: '📦', label: 'Batch' },
  'Batch Global': { icon: '🌍', label: 'Batch Global' },
  'Batch Long Context': { icon: '📖', label: 'Batch Long Context' },
  'Batch Long Context Global': { icon: '🌏', label: 'Batch Long Context Global' },
  'Batch GovCloud': { icon: '🏛️', label: 'GovCloud Batch' },
  'Batch GovCloud InRegion': { icon: '🏛️', label: 'GovCloud Batch' },
  'Batch Long Context GovCloud': { icon: '🏛️', label: 'GovCloud Batch (Long Context)' },
  'Batch Long Context GovCloud InRegion': { icon: '🏛️', label: 'GovCloud Batch (Long Context)' },
  'Provisioned Throughput': { icon: '⚡', label: 'Provisioned Throughput' },
  'Custom Model': { icon: '🎯', label: 'Custom Model' },
}

// ============================================
// PRICING GROUP HIERARCHY FOR "BY TYPE" VIEW
// ============================================
// Maps pricing_group field values to parent groups and sub-groups
// Schema: inference_mode + geographic_scope → pricing_group name
const PRICING_GROUP_HIERARCHY = [
  {
    id: 'cris',
    label: 'Cross-Region (CRIS)',
    icon: Globe,
    colors: {
      light: {
        bg: 'bg-white/60',
        border: 'border-emerald-200/80',
        headerBg: 'bg-white/80',
        headerBorder: 'border-emerald-200',
        icon: 'text-emerald-600',
        text: 'text-emerald-700',
        badge: 'bg-emerald-100 text-emerald-700'
      },
      dark: {
        bg: 'bg-white/[0.02]',
        border: 'border-emerald-500/20',
        headerBg: 'bg-white/[0.03]',
        headerBorder: 'border-emerald-500/20',
        icon: 'text-emerald-400',
        text: 'text-emerald-400',
        badge: 'bg-emerald-500/20 text-emerald-400'
      }
    },
    subGroups: [
      {
        id: 'global',
        label: 'Global',
        description: 'Same price from any source region',
        pricingGroups: ['On-Demand Global', 'On-Demand Long Context Global', 'Batch Global', 'Batch Long Context Global']
      },
      {
        id: 'geo',
        label: 'Geographic',
        description: 'Price varies by source region',
        pricingGroups: ['On-Demand Geo', 'On-Demand Long Context Geo', 'Batch Geo', 'Batch Long Context Geo']
      },
      {
        id: 'govcloud',
        label: 'GovCloud',
        description: 'US GovCloud regions',
        pricingGroups: ['On-Demand GovCloud', 'On-Demand Long Context GovCloud', 'Batch GovCloud', 'Batch Long Context GovCloud']
      }
    ]
  },
  {
    id: 'in_region',
    label: 'In Region',
    icon: Zap,
    colors: {
      light: {
        bg: 'bg-white/60',
        border: 'border-amber-200/80',
        headerBg: 'bg-white/80',
        headerBorder: 'border-amber-200',
        icon: 'text-amber-600',
        text: 'text-amber-700',
        badge: 'bg-amber-100 text-amber-700'
      },
      dark: {
        bg: 'bg-white/[0.02]',
        border: 'border-teal-500/20',
        headerBg: 'bg-white/[0.03]',
        headerBorder: 'border-teal-500/20',
        icon: 'text-[#1A9E7A]',
        text: 'text-teal-400',
        badge: 'bg-teal-500/20 text-teal-400'
      }
    },
    subGroups: [
      {
        id: 'runtime_api',
        label: 'Runtime API',
        description: 'bedrock-runtime',
        pricingGroups: ['On-Demand', 'On-Demand Long Context', 'Batch', 'Batch Long Context']
      },
      {
        id: 'mantle_api',
        label: 'Mantle API',
        description: 'bedrock-mantle',
        pricingGroups: ['Mantle']
      },
      {
        id: 'govcloud',
        label: 'GovCloud',
        description: 'US GovCloud regions',
        pricingGroups: ['On-Demand GovCloud InRegion', 'On-Demand Long Context GovCloud InRegion', 'Batch GovCloud InRegion', 'Batch Long Context GovCloud InRegion']
      }
    ]
  },
  {
    id: 'reserved',
    label: 'Reserved Tiers',
    icon: Clock,
    colors: {
      light: {
        bg: 'bg-white/60',
        border: 'border-indigo-200/80',
        headerBg: 'bg-white/80',
        headerBorder: 'border-indigo-200',
        icon: 'text-indigo-600',
        text: 'text-indigo-700',
        badge: 'bg-indigo-100 text-indigo-700'
      },
      dark: {
        bg: 'bg-white/[0.02]',
        border: 'border-indigo-500/20',
        headerBg: 'bg-white/[0.03]',
        headerBorder: 'border-indigo-500/20',
        icon: 'text-indigo-400',
        text: 'text-indigo-400',
        badge: 'bg-indigo-500/20 text-indigo-400'
      }
    },
    subGroups: [
      {
        id: '1_month_global',
        label: '1 Month (Global)',
        description: '1 month commitment - same price from any source region',
        pricingGroups: ['Reserved 1 Month Global']
      },
      {
        id: '1_month_geo',
        label: '1 Month (Geo)',
        description: '1 month commitment - price varies by source region',
        pricingGroups: ['Reserved 1 Month Geo']
      },
      {
        id: '3_month_global',
        label: '3 Month (Global)',
        description: '3 month commitment - same price from any source region',
        pricingGroups: ['Reserved 3 Month Global']
      },
      {
        id: '3_month_geo',
        label: '3 Month (Geo)',
        description: '3 month commitment - price varies by source region',
        pricingGroups: ['Reserved 3 Month Geo']
      },
      {
        id: '6_month_global',
        label: '6 Month (Global)',
        description: '6 month commitment - same price from any source region',
        pricingGroups: ['Reserved 6 Month Global']
      },
      {
        id: '6_month_geo',
        label: '6 Month (Geo)',
        description: '6 month commitment - price varies by source region',
        pricingGroups: ['Reserved 6 Month Geo']
      },
      {
        id: 'no_commit',
        label: 'No Commit',
        description: 'Reserved capacity without commitment',
        pricingGroups: ['Reserved No Commit']
      }
    ]
  },
  {
    id: 'provisioned',
    label: 'Provisioned',
    icon: Server,
    colors: {
      light: {
        bg: 'bg-white/60',
        border: 'border-purple-200/80',
        headerBg: 'bg-white/80',
        headerBorder: 'border-purple-200',
        icon: 'text-purple-600',
        text: 'text-purple-700',
        badge: 'bg-purple-100 text-purple-700'
      },
      dark: {
        bg: 'bg-white/[0.02]',
        border: 'border-purple-500/20',
        headerBg: 'bg-white/[0.03]',
        headerBorder: 'border-purple-500/20',
        icon: 'text-purple-400',
        text: 'text-purple-400',
        badge: 'bg-purple-500/20 text-purple-400'
      }
    },
    // No sub-groups - direct items
    subGroups: null,
    pricingGroups: ['Provisioned Throughput']
  },
  {
    id: 'custom_model',
    label: 'Custom Model',
    icon: Wrench,
    colors: {
      light: {
        bg: 'bg-white/60',
        border: 'border-slate-200/80',
        headerBg: 'bg-white/80',
        headerBorder: 'border-slate-200',
        icon: 'text-slate-600',
        text: 'text-slate-700',
        badge: 'bg-slate-100 text-slate-700'
      },
      dark: {
        bg: 'bg-white/[0.02]',
        border: 'border-slate-500/20',
        headerBg: 'bg-white/[0.03]',
        headerBorder: 'border-slate-500/20',
        icon: 'text-slate-400',
        text: 'text-slate-400',
        badge: 'bg-slate-500/20 text-slate-400'
      }
    },
    // No sub-groups - direct items
    subGroups: null,
    pricingGroups: ['Custom Model']
  }
]

// Helper component to display regions grouped by geography
function RegionsByGeoDisplay({ regions, getRegionDisplayName, groupRegionsByGeo, geoGroups, isLight, compact = false, geosOnly = false }) {
  const grouped = groupRegionsByGeo(regions)
  const geoOrder = ['US', 'EU', 'APAC', 'CA', 'SA', 'MX', 'ME', 'AF', 'GOV', 'Other']
  const activeGeos = geoOrder.filter(geoKey => grouped[geoKey]?.length > 0)
  
  // For interactive mode (non-geosOnly), track which geo is selected
  // Default to first geo with regions
  const [selectedGeo, setSelectedGeo] = useState(activeGeos[0] || null)
  
  // Geo display names (no emojis)
  const geoDisplayNames = {
    'US': 'United States',
    'EU': 'Europe', 
    'APAC': 'Asia Pacific',
    'CA': 'Canada',
    'SA': 'South America',
    'MX': 'Mexico',
    'ME': 'Middle East',
    'AF': 'Africa',
    'GOV': 'GovCloud',
    'Other': 'Other'
  }
  
  // For geosOnly mode (Reserved Tiers), just show geo names inline without interaction
  if (geosOnly) {
    return (
      <div className="flex flex-wrap gap-1.5">
        {activeGeos.map(geoKey => {
          const regionCount = grouped[geoKey].length
          return (
            <Tooltip key={geoKey} delayDuration={200}>
              <TooltipTrigger asChild>
                <span className={cn(
                  'inline-flex items-center gap-1 px-2 py-1 rounded text-[10px] cursor-default',
                  isLight 
                    ? 'bg-stone-100 text-stone-600 border border-stone-200' 
                    : 'bg-white/[0.06] text-slate-400 border border-white/[0.08]'
                )}>
                  {geoDisplayNames[geoKey] || geoKey}
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={4}>
                <p className="text-xs">{regionCount} region{regionCount !== 1 ? 's' : ''}</p>
              </TooltipContent>
            </Tooltip>
          )
        })}
      </div>
    )
  }
  
  // Interactive mode: clickable geo badges with expandable regions
  return (
    <div className="space-y-2">
      {/* Geo badges row */}
      <div className="flex flex-wrap gap-1.5">
        {activeGeos.map(geoKey => {
          const isSelected = selectedGeo === geoKey
          const regionCount = grouped[geoKey].length
          return (
            <Tooltip key={geoKey} delayDuration={200}>
              <TooltipTrigger asChild>
                <button
                  onClick={() => setSelectedGeo(isSelected ? null : geoKey)}
                  className={cn(
                    'inline-flex items-center gap-1 px-2 py-1 rounded text-[10px] transition-colors cursor-default',
                    isSelected
                      ? isLight
                        ? 'bg-amber-100 text-amber-700 border border-amber-300'
                        : 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                      : isLight 
                        ? 'bg-stone-100 text-stone-600 border border-stone-200 hover:bg-stone-200/80' 
                        : 'bg-white/[0.06] text-slate-400 border border-white/[0.08] hover:bg-white/[0.1]'
                  )}
                >
                  {geoDisplayNames[geoKey] || geoKey}
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={4}>
                <p className="text-xs">{regionCount} region{regionCount !== 1 ? 's' : ''}</p>
              </TooltipContent>
            </Tooltip>
          )
        })}
      </div>
      
      {/* Regions for selected geo */}
      {selectedGeo && grouped[selectedGeo] && (
        <div className="flex flex-wrap gap-1 pt-1">
          {grouped[selectedGeo].sort().map(region => (
            <Tooltip key={region} delayDuration={200}>
              <TooltipTrigger asChild>
                <span
                  className={cn(
                    'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] cursor-default',
                    isLight 
                      ? 'bg-stone-50 text-stone-500 border border-stone-200' 
                      : 'bg-white/[0.03] text-slate-500 border border-white/[0.06]'
                  )}
                >
                  {getRegionDisplayName(region) || region}
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={4}>
                <p className="font-mono text-xs">{region}</p>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      )}
    </div>
  )
}

// Helper component for price breakdown cards with geo-grouped regions
function PriceBreakdownCard({ 
  label, 
  dotColor, 
  priceBreakdown, 
  unit, 
  formatPrice, 
  getRegionDisplayName, 
  groupRegionsByGeo, 
  geoGroups, 
  isLight 
}) {
  return (
    <div className={cn(
      'rounded-lg border overflow-hidden',
      isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
    )}>
      <div className={cn(
        'flex items-center gap-2 px-3 py-2 border-b',
        isLight ? 'bg-stone-50/50 border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
      )}>
        <span className={cn('inline-block w-2 h-2 rounded-full flex-shrink-0', dotColor)} />
        <span className={cn('text-xs font-medium', isLight ? 'text-stone-700' : 'text-slate-300')}>
          {label}
        </span>
      </div>
      <div className="divide-y divide-stone-100 dark:divide-white/[0.04]">
        {priceBreakdown.map((pb, idx) => (
          <div key={idx} className="px-3 py-2.5">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className={cn(
                  'text-sm font-mono font-semibold',
                  isLight ? 'text-stone-900' : 'text-emerald-400'
                )}>
                  ${formatPrice(pb.price)}
                </span>
                <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                  {unit}
                </span>
              </div>
              <span className={cn(
                'text-[10px] px-1.5 py-0.5 rounded',
                isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/[0.06] text-slate-400'
              )}>
                {pb.regions.length} {pb.regions.length === 1 ? 'region' : 'regions'}
              </span>
            </div>
            <RegionsByGeoDisplay
              regions={pb.regions}
              getRegionDisplayName={getRegionDisplayName}
              groupRegionsByGeo={groupRegionsByGeo}
              geoGroups={geoGroups}
              isLight={isLight}
              compact
            />
          </div>
        ))}
      </div>
    </div>
  )
}

// CRIS Pricing Section with clear Global vs Regional distinction
function CRISPricingSection({
  globalItems,
  regionalItems,
  consolidateByTierAndType,
  formatPrice,
  formatPriceRange,
  expandedSections,
  toggleSection,
  getRegionDisplayName,
  groupRegionsByGeo,
  geoGroups,
  isLight
}) {
  const allItems = [...globalItems, ...regionalItems]
  const totalRegions = new Set(allItems.map(i => i.region)).size
  const sectionKey = 'byType_CRIS'
  const isExpanded = expandedSections[sectionKey] !== false

  // Render a pricing tier card
  const renderTierCard = (tier, tierData, keyPrefix, isGlobal) => {
    const tierKey = `${keyPrefix}_${tier}`
    const hasInput = tierData.input
    const hasOutput = tierData.output
    const hasOther = tierData.other
    const tierRegions = new Set([
      ...(tierData.input?.allRegions || []),
      ...(tierData.output?.allRegions || []),
      ...(tierData.other?.allRegions || [])
    ])
    const totalTierRegions = tierRegions.size

    return (
      <div key={tier} className={cn(
        'rounded-lg p-3 border',
        isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
      )}>
        <div className="flex items-center justify-between mb-2">
          <span className={cn('text-xs font-medium', isLight ? 'text-stone-700' : 'text-slate-200')}>
            {tier}
          </span>
          <button
            onClick={() => toggleSection(tierKey)}
            className={cn(
              'text-[10px] px-1.5 py-0.5 rounded transition-colors',
              isLight
                ? 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                : 'bg-white/10 text-slate-300 hover:bg-white/20'
            )}
          >
            {totalTierRegions} {isGlobal ? 'source' : ''} regions {expandedSections[tierKey] ? '▲' : '▼'}
          </button>
        </div>

        {/* Consolidated Input/Output display */}
        {hasInput && hasOutput && (
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500" />
              <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Input:</span>
              <span className={cn('text-xs font-mono font-semibold', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                {formatPriceRange(tierData.input)}
              </span>
            </div>
            <span className={cn('text-xs', isLight ? 'text-stone-300' : 'text-slate-600')}>|</span>
            <div className="flex items-center gap-2">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Output:</span>
              <span className={cn('text-xs font-mono font-semibold', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                {formatPriceRange(tierData.output)}
              </span>
            </div>
          </div>
        )}

        {/* Expanded breakdown */}
        {expandedSections[tierKey] && (
          <div className={cn(
            'mt-3 pt-3 border-t space-y-2',
            isLight ? 'border-stone-200' : 'border-white/[0.06]'
          )}>
            {hasInput && tierData.input.priceBreakdown.length > 1 && (
              <PriceBreakdownCard
                label="Input"
                dotColor="bg-blue-500"
                priceBreakdown={tierData.input.priceBreakdown}
                unit={tierData.input.unit}
                formatPrice={formatPrice}
                getRegionDisplayName={getRegionDisplayName}
                groupRegionsByGeo={groupRegionsByGeo}
                geoGroups={geoGroups}
                isLight={isLight}
              />
            )}
            {hasOutput && tierData.output.priceBreakdown.length > 1 && (
              <PriceBreakdownCard
                label="Output"
                dotColor="bg-emerald-500"
                priceBreakdown={tierData.output.priceBreakdown}
                unit={tierData.output.unit}
                formatPrice={formatPrice}
                getRegionDisplayName={getRegionDisplayName}
                groupRegionsByGeo={groupRegionsByGeo}
                geoGroups={geoGroups}
                isLight={isLight}
              />
            )}
            {/* Single price - show regions */}
            {hasInput && hasOutput && tierData.input.priceBreakdown.length === 1 && tierData.output.priceBreakdown.length === 1 && (
              <div className={cn(
                'rounded-lg p-2',
                isLight ? 'bg-stone-50' : 'bg-white/[0.02]'
              )}>
                <p className={cn('text-[10px] font-medium mb-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>
                  {isGlobal ? 'Source Regions (same price from any region):' : 'Available in:'}
                </p>
                <RegionsByGeoDisplay
                  regions={[...tierRegions]}
                  getRegionDisplayName={getRegionDisplayName}
                  groupRegionsByGeo={groupRegionsByGeo}
                  geoGroups={geoGroups}
                  isLight={isLight}
                  compact
                />
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  // Consolidate items for each section
  const globalConsolidated = globalItems.length > 0 ? consolidateByTierAndType(globalItems) : {}
  const regionalConsolidated = regionalItems.length > 0 ? consolidateByTierAndType(regionalItems) : {}

  const tierOrder = ['Standard', 'Cache Read', '5m Cache Write', '1h Cache Write', 'Flex', 'Priority', 'Standard (Long Context)', 'Cache Read (Long Context)', '5m Cache Write (Long Context)', '1h Cache Write (Long Context)', 'Batch Standard', 'Batch Cache Read', 'Batch 5m Cache Write', 'Batch 1h Cache Write', 'Batch Standard (Long Context)', 'Batch Cache Read (Long Context)', 'Batch 5m Cache Write (Long Context)', 'Batch 1h Cache Write (Long Context)']
  const sortTiers = (tiers) => tiers.sort((a, b) => {
    const idxA = tierOrder.indexOf(a)
    const idxB = tierOrder.indexOf(b)
    if (idxA !== -1 && idxB !== -1) return idxA - idxB
    if (idxA !== -1) return -1
    if (idxB !== -1) return 1
    return a.localeCompare(b)
  })

  return (
    <div className={cn(
      'rounded-lg overflow-hidden border',
      isLight
        ? 'bg-stone-50/80 border-stone-200/80'
        : 'bg-white/5 border-white/10'
    )}>
      <button
        className={cn(
          'w-full flex items-center justify-between p-3 transition-colors',
          isLight ? 'hover:bg-stone-100/80' : 'hover:bg-white/5'
        )}
        onClick={() => toggleSection(sectionKey)}
      >
        <div className="flex items-center gap-2">
          <Globe className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
          <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
            Cross-Region (CRIS)
          </span>
          <Badge variant="secondary" className="text-[10px]">
            {totalRegions} regions
          </Badge>
        </div>
        {isExpanded ? (
          <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
        ) : (
          <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
        )}
      </button>

      {isExpanded && (
        <div className={cn('px-3 pb-3 border-t space-y-4', isLight ? 'border-stone-200' : 'border-white/10')}>
          {/* Global Pricing Section */}
          {globalItems.length > 0 && (
            <div className="pt-3">
              <div className={cn(
                'rounded-lg border overflow-hidden',
                isLight ? 'border-emerald-200 bg-emerald-50/30' : 'border-emerald-500/20 bg-emerald-500/5'
              )}>
                <div className={cn(
                  'flex items-center gap-2 px-3 py-2 border-b',
                  isLight ? 'bg-emerald-50 border-emerald-200' : 'bg-emerald-500/10 border-emerald-500/20'
                )}>
                  <Globe className={cn('h-4 w-4', isLight ? 'text-emerald-600' : 'text-emerald-400')} />
                  <span className={cn('text-xs font-semibold uppercase tracking-wide', isLight ? 'text-emerald-700' : 'text-emerald-400')}>
                    Global Pricing
                  </span>
                  <span className={cn('text-[10px]', isLight ? 'text-emerald-600' : 'text-emerald-500')}>
                    (same price from any source region)
                  </span>
                </div>
                <div className="p-3 space-y-2">
                  {sortTiers(Object.keys(globalConsolidated)).filter(tier => {
                    const tierData = globalConsolidated[tier]
                    // Only show tier if it has actual pricing (input or output with a price)
                    const hasInputPrice = tierData.input?.minPrice != null
                    const hasOutputPrice = tierData.output?.minPrice != null
                    return hasInputPrice || hasOutputPrice
                  }).map(tier => 
                    renderTierCard(tier, globalConsolidated[tier], 'CRIS_global', true)
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Regional Pricing Section */}
          {regionalItems.length > 0 && (
            <div className="pt-1">
              <div className={cn(
                'rounded-lg border overflow-hidden',
                isLight ? 'border-amber-200 bg-amber-50/30' : 'border-amber-500/20 bg-amber-500/5'
              )}>
                <div className={cn(
                  'flex items-center gap-2 px-3 py-2 border-b',
                  isLight ? 'bg-amber-50 border-amber-200' : 'bg-amber-500/10 border-amber-500/20'
                )}>
                  <Route className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-amber-400')} />
                  <span className={cn('text-xs font-semibold uppercase tracking-wide', isLight ? 'text-amber-700' : 'text-amber-400')}>
                    Regional Pricing
                  </span>
                  <span className={cn('text-[10px]', isLight ? 'text-amber-600' : 'text-amber-500')}>
                    (varies by source region)
                  </span>
                </div>
                <div className="p-3 space-y-2">
                  {sortTiers(Object.keys(regionalConsolidated)).filter(tier => {
                    const tierData = regionalConsolidated[tier]
                    const hasInputPrice = tierData.input?.minPrice != null
                    const hasOutputPrice = tierData.output?.minPrice != null
                    return hasInputPrice || hasOutputPrice
                  }).map(tier => 
                    renderTierCard(tier, regionalConsolidated[tier], 'CRIS_regional', false)
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function PricingTab({ model, getPricingForModel, preferredRegion = 'us-east-1', user }) {
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedSections, setExpandedSections] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const geoInfo = {
    'US': { icon: '🇺🇸', name: 'United States' },
    'EU': { icon: '🇪🇺', name: 'Europe' },
    'APAC': { icon: '🌏', name: 'Asia Pacific' },
    'CA': { icon: '🇨🇦', name: 'Canada' },
    'SA': { icon: '🌎', name: 'South America' },
    'ME': { icon: '🏜️', name: 'Middle East' },
    'GOV': { icon: '🏛️', name: 'GovCloud' },
    'Other': { icon: '📍', name: 'Other' }
  }

  const geoOrder = ['US', 'EU', 'APAC', 'CA', 'SA', 'ME', 'GOV', 'Other']

  const getGeoForRegion = (region) => {
    if (region.startsWith('us-gov-')) return 'GOV'
    if (region.startsWith('us-')) return 'US'
    if (region.startsWith('eu-')) return 'EU'
    if (region.startsWith('ap-')) return 'APAC'
    if (region.startsWith('ca-')) return 'CA'
    if (region.startsWith('sa-')) return 'SA'
    if (region.startsWith('me-') || region.startsWith('il-')) return 'ME'
    return 'Other'
  }

  const toggleSection = (key) => setExpandedSections(prev => ({ ...prev, [key]: !prev[key] }))

  // Get pricing from new source
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
  const fullPricing = pricingResult?.fullPricing

  // Fallback to model's embedded pricing
  const legacyPricing = model.pricing ?? model.model_pricing ?? model.comprehensive_pricing ?? {}
  const legacyByRegion = legacyPricing.by_region || {}

  // Process pricing structure: group -> geo -> region -> items
  const pricingByGroupGeoRegion = {}
  let allRegions = []

  // Get GovCloud inference type from model
  const govcloudInferenceType = model.availability?.govcloud?.inference_type

  if (fullPricing?.regions) {
    allRegions = Object.keys(fullPricing.regions)
    for (const [region, regionData] of Object.entries(fullPricing.regions)) {
      if (!regionData?.pricing_groups) continue
      const geo = getGeoForRegion(region)
      const isGovCloud = geo === 'GOV'
      
      for (const [groupName, items] of Object.entries(regionData.pricing_groups)) {
        // For GovCloud regions, route to appropriate virtual group based on inference_type
        let targetGroup = groupName
        if (isGovCloud) {
          if (govcloudInferenceType === 'cris') {
            // Route to CRIS GovCloud group
            if (groupName === 'On-Demand' || groupName === 'On-Demand Geo') targetGroup = 'On-Demand GovCloud'
            else if (groupName === 'On-Demand Long Context' || groupName === 'On-Demand Long Context Geo') targetGroup = 'On-Demand Long Context GovCloud'
            else if (groupName === 'Batch' || groupName === 'Batch Geo') targetGroup = 'Batch GovCloud'
            else if (groupName === 'Batch Long Context' || groupName === 'Batch Long Context Geo') targetGroup = 'Batch Long Context GovCloud'
          } else if (govcloudInferenceType === 'in_region') {
            // Route to In-Region GovCloud group
            if (groupName === 'On-Demand' || groupName === 'On-Demand Geo') targetGroup = 'On-Demand GovCloud InRegion'
            else if (groupName === 'On-Demand Long Context' || groupName === 'On-Demand Long Context Geo') targetGroup = 'On-Demand Long Context GovCloud InRegion'
            else if (groupName === 'Batch' || groupName === 'Batch Geo') targetGroup = 'Batch GovCloud InRegion'
            else if (groupName === 'Batch Long Context' || groupName === 'Batch Long Context Geo') targetGroup = 'Batch Long Context GovCloud InRegion'
          }
        }
        
        if (!pricingByGroupGeoRegion[targetGroup]) pricingByGroupGeoRegion[targetGroup] = {}
        if (!pricingByGroupGeoRegion[targetGroup][geo]) pricingByGroupGeoRegion[targetGroup][geo] = {}
        pricingByGroupGeoRegion[targetGroup][geo][region] = items
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

  const pricingGroupOrder = [
    // CRIS (Cross-Region)
    'On-Demand Global', 'On-Demand Geo', 'On-Demand Long Context Global', 'On-Demand Long Context Geo',
    'Batch Global', 'Batch Geo', 'Batch Long Context Global', 'Batch Long Context Geo',
    'On-Demand GovCloud', 'On-Demand Long Context GovCloud', 'Batch GovCloud', 'Batch Long Context GovCloud',
    // In-Region
    'On-Demand', 'On-Demand Long Context', 'Batch', 'Batch Long Context', 'Mantle',
    'On-Demand GovCloud InRegion', 'On-Demand Long Context GovCloud InRegion', 'Batch GovCloud InRegion', 'Batch Long Context GovCloud InRegion',
    // Reserved
    'Reserved No Commit', 'Reserved No Commit Global', 'Reserved No Commit Geo',
    'Reserved 1 Month', 'Reserved 1 Month Global', 'Reserved 1 Month Geo',
    'Reserved 3 Month', 'Reserved 3 Month Global', 'Reserved 3 Month Geo',
    'Reserved 6 Month', 'Reserved 6 Month Global', 'Reserved 6 Month Geo',
    // Other
    'Provisioned Throughput', 'Custom Model'
  ]
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
      <div className={cn(
        'mb-4 p-4 rounded-lg border flex items-start gap-3',
        isLight 
          ? 'bg-amber-50 border-amber-200 text-amber-800' 
          : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
      )}>
        <AlertTriangle className="h-5 w-5 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-medium">No pricing data available</p>
          <p className={cn('text-sm mt-1', isLight ? 'text-amber-700' : 'text-amber-400/80')}>
            This model is not listed in the AWS Pricing API. Please verify model availability before use.
          </p>
        </div>
      </div>
    )
  }


  // Get tier from pricing label - includes inference mode (Batch) and context type (Long Context)
  const getTierFromLabel = (label, groupName = '', dimension = '') => {
    const groupLower = groupName.toLowerCase()
    const dimLower = dimension.toLowerCase()
    
    // Check modifiers from group name OR dimension
    const isLongContext = groupLower.includes('long context')
    const isBatch = groupLower.includes('batch') || dimLower.includes('-batch')
    
    // Build tier name components
    const prefix = isBatch ? 'Batch ' : ''
    const lcSuffix = isLongContext ? ' (Long Context)' : ''
    
    // Detect tier from the label (pricing variant) or dimension
    if (/cache[- ]?read/i.test(label)) return `${prefix}Cache Read${lcSuffix}`.trim()
    if (/cache[- ]?write[- ]?1h|1h[- ]?cache/i.test(label)) return `${prefix}1h Cache Write${lcSuffix}`.trim()
    if (/cache[- ]?write(?![- ]?1h)|5m[- ]?cache/i.test(label)) return `${prefix}5m Cache Write${lcSuffix}`.trim()
    if (/\bflex\b/i.test(label) || dimLower.includes('-flex')) return `${prefix}Flex${lcSuffix}`.trim()
    if (/\bpriority\b/i.test(label) || dimLower.includes('-priority')) return `${prefix}Priority${lcSuffix}`.trim()
    if (/no[- ]?commit/i.test(label)) return 'No Commit'
    if (/6[- ]?month/i.test(label)) return '6 Month'
    if (/3[- ]?month/i.test(label)) return '3 Month'
    if (/1[- ]?month/i.test(label)) return '1 Month'
    
    // Default tier - distinguish batch and long context
    return `${prefix}Standard${lcSuffix}`.trim()
  }


  // Group items by price to deduplicate across regions
  const groupByPrice = (items) => {
    const priceGroups = {}
    items.forEach(item => {
      const price = item._price ?? 0
      const key = `${price.toFixed(6)}_${item._label}_${item._type}`
      if (!priceGroups[key]) {
        priceGroups[key] = { ...item, regions: [] }
      }
      if (item.region && !priceGroups[key].regions.includes(item.region)) {
        priceGroups[key].regions.push(item.region)
      }
    })
    return Object.values(priceGroups)
  }

  // Flatten all pricing items with region info for processing
  const getAllPricingItems = () => {
    const items = []
    for (const [groupName, geoData] of Object.entries(pricingByGroupGeoRegion)) {
      for (const [geo, regions] of Object.entries(geoData)) {
        for (const [region, regionItems] of Object.entries(regions)) {
          regionItems.forEach(item => {
            const { label } = simplifyPricingDescription(item.description, item.dimension)
            // Use is_input/is_output from data, fallback to parsing if not available
            const type = item.is_output ? 'output' : item.is_input ? 'input' : 'other'
            items.push({
              ...item,
              _label: label,
              _type: type,
              _price: item.price_per_thousand ?? item.price_per_unit,
              _unit: (item.unit_label || `per ${item.unit || 'unit'}`).replace(/1K tokens/gi, '1M tokens'),
              _raw: item.description || item.dimension,
              _groupName: groupName,
              _tier: getTierFromLabel(label, groupName, item.dimension),
              _isLongContext: groupName.toLowerCase().includes('long context'),
              region,
              geo
            })
          })
        }
      }
    }
    return items
  }

  // Filter items by search query
  const filterItems = (items) => {
    if (!searchQuery) return items
    const query = searchQuery.toLowerCase()
    return items.filter(item => {
      const regionName = (getRegionDisplayName(item.region) || '').toLowerCase()
      const geoName = (geoInfo[item.geo]?.name || '').toLowerCase()
      return (
        item.region?.toLowerCase().includes(query) ||
        regionName.includes(query) ||
        item.geo?.toLowerCase().includes(query) ||
        geoName.includes(query) ||
        item._label?.toLowerCase().includes(query) ||
        item._groupName?.toLowerCase().includes(query) ||
        item._tier?.toLowerCase().includes(query)
      )
    })
  }

  // Format price for display
  const formatPrice = (price) => {
    if (price == null) return 'N/A'
    const displayPrice = typeof price === 'number' ? price * 1000 : price
    if (typeof displayPrice === 'number') {
      return displayPrice < 0.01 ? displayPrice.toFixed(4) : displayPrice.toFixed(2)
    }
    return displayPrice
  }

  // ============================================
  // VIEW 1: BY TYPE (Hierarchy-Based View)
  // ============================================
  const renderByTypeView = () => {
    const allItems = filterItems(getAllPricingItems())
    
    // Group items by their pricing_group field (_groupName)
    const itemsByPricingGroup = {}
    allItems.forEach(item => {
      const groupName = item._groupName
      if (!itemsByPricingGroup[groupName]) {
        itemsByPricingGroup[groupName] = []
      }
      itemsByPricingGroup[groupName].push(item)
    })

    // Detect if model has CRIS availability but only In-Region pricing
    // This means the In-Region pricing also applies to CRIS
    const consumptionOptions = model.consumption_options || []
    const hasCrisAvailability = consumptionOptions.includes('cross_region_inference')
    const hasCrisGeoPricing = !!(
      itemsByPricingGroup['On-Demand Geo']?.length > 0 ||
      itemsByPricingGroup['On-Demand Long Context Geo']?.length > 0 ||
      itemsByPricingGroup['Batch Geo']?.length > 0
    )
    const hasCrisPricing = !!(
      itemsByPricingGroup['On-Demand Global']?.length > 0 ||
      hasCrisGeoPricing ||
      itemsByPricingGroup['On-Demand Long Context Global']?.length > 0 ||
      itemsByPricingGroup['Batch Global']?.length > 0
    )
    const hasInRegionPricing = !!(
      itemsByPricingGroup['On-Demand']?.length > 0 ||
      itemsByPricingGroup['On-Demand Long Context']?.length > 0
    )
    // Show indicator when model has CRIS availability but only In-Region pricing (no CRIS-specific pricing)
    const showCrisUsesInRegionPricing = hasCrisAvailability && hasInRegionPricing && !hasCrisPricing

    // Detect if model has In-Region availability but only CRIS Geo pricing (no In-Region pricing)
    // CRIS Geo pricing = In-Region pricing (same price), so we show that CRIS Geo also applies to In-Region
    const hasInRegionAvailability = consumptionOptions.includes('on_demand') || 
                                     (model.in_region && model.in_region.length > 0)
    const showInRegionUsesCrisGeoPricing = hasInRegionAvailability && !hasInRegionPricing && hasCrisGeoPricing

    // Consolidate items by tier and type, computing price ranges
    const consolidateByTierAndType = (items) => {
      const byTier = {}
      items.forEach(item => {
        const tier = item._tier || 'Standard'
        if (!byTier[tier]) byTier[tier] = { input: [], output: [], other: [] }
        const typeKey = item._type === 'input' ? 'input' : item._type === 'output' ? 'output' : 'other'
        byTier[tier][typeKey].push(item)
      })

      // For each tier, compute consolidated data
      const result = {}
      for (const [tier, types] of Object.entries(byTier)) {
        result[tier] = {}
        for (const [typeKey, typeItems] of Object.entries(types)) {
          if (typeItems.length === 0) continue
          
          // Get all unique prices and regions
          const prices = typeItems.map(i => i._price).filter(p => p != null)
          const allRegions = [...new Set(typeItems.map(i => i.region).filter(Boolean))]
          const minPrice = Math.min(...prices)
          const maxPrice = Math.max(...prices)
          const unit = typeItems[0]._unit
          
          // Group by price for detailed breakdown
          const byPrice = {}
          typeItems.forEach(item => {
            const priceKey = (item._price ?? 0).toFixed(6)
            if (!byPrice[priceKey]) {
              byPrice[priceKey] = { price: item._price, regions: [], unit: item._unit }
            }
            if (item.region && !byPrice[priceKey].regions.includes(item.region)) {
              byPrice[priceKey].regions.push(item.region)
            }
          })
          
          result[tier][typeKey] = {
            minPrice,
            maxPrice,
            hasRange: minPrice !== maxPrice,
            unit,
            totalRegions: allRegions.length,
            allRegions,
            priceBreakdown: Object.values(byPrice).sort((a, b) => a.price - b.price),
            rawItems: typeItems // Keep raw items for detailed view
          }
        }
      }
      return result
    }

    // Format price range for display
    const formatPriceRange = (data) => {
      if (!data) return null
      if (data.hasRange) {
        return `$${formatPrice(data.minPrice)} - $${formatPrice(data.maxPrice)}`
      }
      return `$${formatPrice(data.minPrice)}`
    }

    // Tier ordering for consistent display
    const tierOrder = [
      'Standard',
      'Cache Read', '5m Cache Write', '1h Cache Write',
      'Flex', 'Priority',
      'Standard (Long Context)',
      'Cache Read (Long Context)', '5m Cache Write (Long Context)',
      '1h Cache Write (Long Context)', 'Flex (Long Context)', 'Priority (Long Context)',
      'Batch Standard', 'Batch Cache Read', 'Batch 5m Cache Write', 'Batch 1h Cache Write', 'Batch Standard (Long Context)', 'Batch Cache Read (Long Context)', 'Batch 5m Cache Write (Long Context)', 'Batch 1h Cache Write (Long Context)', 'No Commit', '1 Month', '3 Month', '6 Month'
    ]

    const sortTiers = (tiers) => tiers.sort((a, b) => {
      const idxA = tierOrder.indexOf(a)
      const idxB = tierOrder.indexOf(b)
      if (idxA !== -1 && idxB !== -1) return idxA - idxB
      if (idxA !== -1) return -1
      if (idxB !== -1) return 1
      return a.localeCompare(b)
    })

    // Render tier cards for a set of items
    const renderTierCards = (items, keyPrefix, geosOnly = false) => {
      const consolidated = consolidateByTierAndType(items)
      const sortedTiers = sortTiers(Object.keys(consolidated))

      return sortedTiers.filter(tier => {
        const tierData = consolidated[tier]
        return tierData.input?.minPrice != null || tierData.output?.minPrice != null || tierData.other?.minPrice != null
      }).map(tier => {
        const tierData = consolidated[tier]
        const tierKey = `${keyPrefix}_${tier}`
        const hasInput = tierData.input
        const hasOutput = tierData.output
        const hasOther = tierData.other
        const hasInputOutput = hasInput && hasOutput
        
        const tierRegions = new Set([
          ...(tierData.input?.allRegions || []),
          ...(tierData.output?.allRegions || []),
          ...(tierData.other?.allRegions || [])
        ])
        const totalTierRegions = tierRegions.size

        return (
          <div key={tier} className={cn(
            'rounded-lg border overflow-hidden',
            isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
          )}>
            <button
              onClick={() => toggleSection(tierKey)}
              className={cn(
                'w-full flex items-center justify-between px-3 py-2 text-left transition-colors',
                isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
              )}
            >
              <div className="flex items-center gap-3 flex-wrap min-w-0">
                <span className={cn('text-xs font-medium flex-shrink-0', isLight ? 'text-stone-700' : 'text-slate-200')}>
                  {tier}
                </span>
                {hasInputOutput && (
                  <div className="flex items-center gap-2 text-xs">
                    <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-400')}>In:</span>
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500" />
                    <span className={cn('font-mono font-semibold', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                      {formatPriceRange(tierData.input)}
                    </span>
                    <span className={cn(isLight ? 'text-stone-400' : 'text-slate-500')}>/</span>
                    <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-400')}>Out:</span>
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span className={cn('font-mono font-semibold', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                      {formatPriceRange(tierData.output)}
                    </span>
                    <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                      {tierData.input.unit}
                    </span>
                  </div>
                )}
                {hasInput && !hasOutput && (
                  <div className="flex items-center gap-2 text-xs">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500" />
                    <span className={cn(isLight ? 'text-stone-500' : 'text-slate-400')}>Input:</span>
                    <span className={cn('font-mono font-semibold', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                      {formatPriceRange(tierData.input)}
                    </span>
                    <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                      {tierData.input.unit}
                    </span>
                  </div>
                )}
                {hasOutput && !hasInput && (
                  <div className="flex items-center gap-2 text-xs">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span className={cn(isLight ? 'text-stone-500' : 'text-slate-400')}>Output:</span>
                    <span className={cn('font-mono font-semibold', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                      {formatPriceRange(tierData.output)}
                    </span>
                    <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                      {tierData.output.unit}
                    </span>
                  </div>
                )}
                {hasOther && !hasInput && !hasOutput && (
                  <div className="flex items-center gap-2 text-xs">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-400" />
                    <span className={cn('font-mono font-semibold', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                      {formatPriceRange(tierData.other)}
                    </span>
                    <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                      {tierData.other.unit}
                    </span>
                  </div>
                )}
              </div>
              <div className={cn(
                'flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded flex-shrink-0',
                isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/10 text-slate-300'
              )}>
                {totalTierRegions} regions
                {expandedSections[tierKey] ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              </div>
            </button>
            
            {expandedSections[tierKey] && (
              <div className={cn(
                'px-3 pb-3 border-t space-y-3',
                isLight ? 'border-stone-100 bg-stone-50/50' : 'border-white/5 bg-white/[0.01]'
              )}>
                <div className="pt-2">
                  <RegionsByGeoDisplay
                    regions={[...tierRegions]}
                    getRegionDisplayName={getRegionDisplayName}
                    groupRegionsByGeo={groupRegionsByGeo}
                    geoGroups={geoGroups}
                    isLight={isLight}
                    compact
                    geosOnly={geosOnly}
                  />
                </div>
                
                <div className={cn(
                  'rounded border overflow-hidden',
                  isLight ? 'border-stone-200' : 'border-white/10'
                )}>
                  <div className={cn(
                    'px-2 py-1.5 text-[10px] font-medium',
                    isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/5 text-slate-400'
                  )}>
                    Raw Pricing Data
                  </div>
                  <div className="max-h-40 overflow-y-auto">
                    <table className="w-full text-[10px]">
                      <thead className={cn('sticky top-0', isLight ? 'bg-stone-50' : 'bg-[#1a1a1a]')}>
                        <tr className={isLight ? 'text-stone-500' : 'text-slate-500'}>
                          <th className="text-left px-2 py-1 font-medium">Type</th>
                          <th className="text-left px-2 py-1 font-medium">Price</th>
                          <th className="text-left px-2 py-1 font-medium">Region</th>
                          <th className="text-left px-2 py-1 font-medium">Dimension</th>
                        </tr>
                      </thead>
                      <tbody className={cn('divide-y', isLight ? 'divide-stone-100' : 'divide-white/5')}>
                        {(() => {
                          const inputItems = tierData.input?.rawItems || []
                          const outputItems = tierData.output?.rawItems || []
                          const otherItems = tierData.other?.rawItems || []
                          
                          // Interleave input and output items for balanced display
                          const interleaved = []
                          const maxLen = Math.max(inputItems.length, outputItems.length, otherItems.length)
                          for (let i = 0; i < maxLen && interleaved.length < 30; i++) {
                            if (inputItems[i]) interleaved.push(inputItems[i])
                            if (outputItems[i]) interleaved.push(outputItems[i])
                            if (otherItems[i]) interleaved.push(otherItems[i])
                          }
                          return interleaved
                        })().map((item, idx) => (
                          <Popover key={idx}>
                            <PopoverTrigger asChild>
                              <tr 
                                className={cn(
                                  'cursor-pointer transition-colors',
                                  isLight 
                                    ? 'text-stone-600 hover:bg-stone-100/80' 
                                    : 'text-slate-400 hover:bg-white/[0.04]'
                                )}
                              >
                                <td className="px-2 py-1">
                                  <span className={cn(
                                    'inline-block w-1.5 h-1.5 rounded-full mr-1',
                                    item._type === 'input' ? 'bg-blue-500' : item._type === 'output' ? 'bg-emerald-500' : 'bg-gray-400'
                                  )} />
                                  {item._type}
                                </td>
                                <td className={cn('px-2 py-1 font-mono', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                                  ${formatPrice(item._price)}
                                </td>
                                <td className="px-2 py-1">{item.region}</td>
                                <td className="px-2 py-1 truncate max-w-[200px]" title={item.dimension || item._raw}>
                                  {item.dimension || item._raw || '-'}
                                </td>
                              </tr>
                            </PopoverTrigger>
                            <PopoverContent 
                              className={cn(
                                'w-96 max-h-[400px] overflow-y-auto',
                                isLight 
                                  ? 'bg-white border-stone-200' 
                                  : 'bg-[#1a1a1a] border-white/10'
                              )}
                              side="right"
                              align="start"
                            >
                              <div className="p-3 space-y-3">
                                <div className={cn(
                                  'text-xs font-semibold pb-2 border-b flex items-center gap-2',
                                  isLight ? 'text-stone-800 border-stone-200' : 'text-slate-200 border-white/10'
                                )}>
                                  <span className={cn(
                                    'inline-block w-2 h-2 rounded-full',
                                    item._type === 'input' ? 'bg-blue-500' : item._type === 'output' ? 'bg-emerald-500' : 'bg-gray-400'
                                  )} />
                                  Raw Pricing Details
                                </div>
                                
                                {/* Description - prominently displayed */}
                                {item.description && (
                                  <div className={cn(
                                    'p-2 rounded-lg text-xs',
                                    isLight ? 'bg-amber-50 border border-amber-200' : 'bg-amber-900/20 border border-amber-700/30'
                                  )}>
                                    <div className={cn(
                                      'text-[10px] font-medium mb-1',
                                      isLight ? 'text-amber-700' : 'text-amber-400'
                                    )}>
                                      Description
                                    </div>
                                    <div className={isLight ? 'text-amber-900' : 'text-amber-200'}>
                                      {item.description}
                                    </div>
                                  </div>
                                )}
                                
                                {/* Essential fields only */}
                                <div className="space-y-1.5 text-[11px]">
                                  <div className="flex gap-2">
                                    <span className={cn(
                                      'font-medium w-20 shrink-0',
                                      isLight ? 'text-stone-500' : 'text-slate-500'
                                    )}>
                                      Dimension:
                                    </span>
                                    <span className={cn(
                                      'break-all',
                                      isLight ? 'text-stone-800' : 'text-slate-300'
                                    )}>
                                      {item.dimension || 'N/A'}
                                    </span>
                                  </div>
                                  <div className="flex gap-2">
                                    <span className={cn(
                                      'font-medium w-20 shrink-0',
                                      isLight ? 'text-stone-500' : 'text-slate-500'
                                    )}>
                                      Price:
                                    </span>
                                    <span className={cn(
                                      'font-mono',
                                      isLight ? 'text-stone-800' : 'text-slate-300'
                                    )}>
                                      ${formatPrice(item._price ?? item.price_per_unit)}
                                    </span>
                                  </div>
                                  <div className="flex gap-2">
                                    <span className={cn(
                                      'font-medium w-20 shrink-0',
                                      isLight ? 'text-stone-500' : 'text-slate-500'
                                    )}>
                                      Unit:
                                    </span>
                                    <span className={isLight ? 'text-stone-800' : 'text-slate-300'}>
                                      {item.unit || item._unit || 'N/A'}
                                    </span>
                                  </div>
                                  <div className="flex gap-2">
                                    <span className={cn(
                                      'font-medium w-20 shrink-0',
                                      isLight ? 'text-stone-500' : 'text-slate-500'
                                    )}>
                                      Region:
                                    </span>
                                    <span className={isLight ? 'text-stone-800' : 'text-slate-300'}>
                                      {item.region || 'N/A'}
                                    </span>
                                  </div>
                                  <div className="flex gap-2">
                                    <span className={cn(
                                      'font-medium w-20 shrink-0',
                                      isLight ? 'text-stone-500' : 'text-slate-500'
                                    )}>
                                      Group:
                                    </span>
                                    <span className={isLight ? 'text-stone-800' : 'text-slate-300'}>
                                      {item.pricing_group || item._groupName || 'N/A'}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </PopoverContent>
                          </Popover>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )
      })
    }

    // Render a sub-group section (nested inside parent group)
    const renderSubGroup = (subGroup, items, parentId, colors, isFirstSubGroup, showInRegionEquivalence = false, geosOnly = false) => {
      const subGroupKey = `byType_${parentId}_${subGroup.id}`
      const isExpanded = expandedSections[subGroupKey] ?? isFirstSubGroup
      const regionCount = new Set(items.map(i => i.region)).size

      return (
        <div key={subGroup.id} className={cn(
          'rounded-lg border overflow-hidden',
          isLight ? 'bg-white/60 border-stone-200/60' : 'bg-white/[0.02] border-white/[0.06]'
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between px-3 py-2 transition-colors',
              isLight ? 'hover:bg-stone-50/80' : 'hover:bg-white/[0.03]'
            )}
            onClick={() => toggleSection(subGroupKey)}
          >
            <div className="flex items-center gap-2">
              <span className={cn('text-xs font-medium', isLight ? 'text-stone-700' : 'text-slate-200')}>
                {subGroup.label}
              </span>
              {subGroup.description && (
                <span className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
                  ({subGroup.description})
                </span>
              )}
              {/* Show indicator when CRIS Geo pricing also applies to In-Region requests */}
              {subGroup.id === 'geo' && showInRegionEquivalence && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className={cn(
                      'inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border',
                      isLight 
                        ? 'bg-amber-100 text-amber-700 border-amber-200' 
                        : 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                    )}>
                      <Zap className="h-3 w-3" />
                      = In-Region
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs">
                    <p className="text-xs">This model has In-Region availability, and the CRIS Geo pricing shown here also applies to In-Region requests (same price).</p>
                  </TooltipContent>
                </Tooltip>
              )}
              <span className={cn(
                'text-[10px] px-1.5 py-0.5 rounded',
                isLight ? 'bg-stone-100 text-stone-600' : 'bg-white/10 text-slate-400'
              )}>
                {regionCount} regions
              </span>
            </div>
            {isExpanded ? (
              <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
            ) : (
              <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
            )}
          </button>
          {isExpanded && (
            <div className={cn('px-3 pb-3 border-t space-y-2', isLight ? 'border-stone-100' : 'border-white/[0.04]')}>
              <div className="pt-2 space-y-2">
                {renderTierCards(items, subGroupKey, geosOnly)}
              </div>
            </div>
          )}
        </div>
      )
    }

    // Render a parent group section
    const renderParentGroup = (parentGroup, isFirstWithData) => {
      const colors = isLight ? parentGroup.colors.light : parentGroup.colors.dark
      const Icon = parentGroup.icon
      const sectionKey = `byType_${parentGroup.id}`
      const isExpanded = expandedSections[sectionKey] ?? isFirstWithData

      // Collect all items for this parent group
      let allParentItems = []
      let subGroupsWithData = []

      if (parentGroup.subGroups) {
        // Has sub-groups - collect items per sub-group
        const hideInRegion = model.availability?.hide_in_region ?? false
        const mantleHasPricing = model.availability?.mantle?.has_pricing ?? false
        
        parentGroup.subGroups.forEach(subGroup => {
          // Skip Runtime API sub-group when hide_in_region is true
          // BUT only if Mantle has its own pricing - otherwise show In-Region pricing
          if (hideInRegion && mantleHasPricing && parentGroup.id === 'in_region' && subGroup.id === 'runtime_api') {
            return
          }
          
          const subGroupItems = subGroup.pricingGroups.flatMap(pg => itemsByPricingGroup[pg] || [])
          if (subGroupItems.length > 0) {
            subGroupsWithData.push({ subGroup, items: subGroupItems })
            allParentItems.push(...subGroupItems)
          }
        })
      } else if (parentGroup.pricingGroups) {
        // No sub-groups - direct items
        allParentItems = parentGroup.pricingGroups.flatMap(pg => itemsByPricingGroup[pg] || [])
      }

      // Skip if no data
      if (allParentItems.length === 0) return null

      const totalRegions = new Set(allParentItems.map(i => i.region)).size

      return (
        <div key={parentGroup.id} className={cn(
          'rounded-lg overflow-hidden border',
          colors.bg,
          colors.border
        )}>
          <button
            className={cn(
              'w-full flex items-center justify-between p-3 transition-colors',
              isLight ? 'hover:bg-white/40' : 'hover:bg-white/[0.03]'
            )}
            onClick={() => toggleSection(sectionKey)}
          >
            <div className="flex items-center gap-2">
              <Icon className={cn('h-4 w-4', colors.icon)} />
              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                {parentGroup.label}
              </span>
              {/* Show indicator when In-Region pricing also applies to CRIS */}
              {parentGroup.id === 'in_region' && showCrisUsesInRegionPricing && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className={cn(
                      'text-[10px] px-1.5 py-0.5 rounded cursor-help flex items-center gap-1',
                      isLight ? 'bg-emerald-100 text-emerald-700 border border-emerald-200' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    )}>
                      <Globe className="h-3 w-3" />
                      = CRIS
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs">
                    <p className="text-xs">This model supports Cross-Region Inference (CRIS), and the In-Region pricing shown here also applies to CRIS requests.</p>
                  </TooltipContent>
                </Tooltip>
              )}
              {/* Show badge for Provisioned group indicating why user can see it */}
              {parentGroup.id === 'provisioned' && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className={cn(
                      'text-[9px] px-1.5 py-0.5 rounded border font-medium',
                      getSectionBadge(user, 'availability')?.light || 'bg-purple-100 text-purple-700 border-purple-200/60',
                      !isLight && (getSectionBadge(user, 'availability')?.dark || 'bg-purple-500/15 text-purple-400 border-purple-500/20')
                    )}>
                      {getSectionBadge(user, 'availability')?.text || 'BETA'}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="top">
                    <p className="text-xs">Visible to {getSectionBadge(user, 'availability')?.text === 'ADM' ? 'admins' : getSectionBadge(user, 'availability')?.text === 'OP' ? 'operators' : 'beta users'}</p>
                  </TooltipContent>
                </Tooltip>
              )}
              <Badge className={cn('text-[10px]', colors.badge)}>
                {totalRegions} regions
              </Badge>
            </div>
            {isExpanded ? (
              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            ) : (
              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            )}
          </button>

          {isExpanded && (
            <div className={cn('px-3 pb-3 border-t space-y-3', colors.headerBorder)}>
              {parentGroup.subGroups ? (
                // Render sub-groups
                <div className="pt-3 space-y-3">
                  {subGroupsWithData.map(({ subGroup, items }, idx) => 
                    renderSubGroup(
                      subGroup, 
                      items, 
                      parentGroup.id, 
                      colors, 
                      idx === 0,
                      // Pass the In-Region equivalence flag only for CRIS parent group
                      parentGroup.id === 'cris' && showInRegionUsesCrisGeoPricing,
                      // Pass geosOnly for Reserved tiers
                      parentGroup.id === 'reserved'
                    )
                  )}
                </div>
              ) : (
                // Render direct tier cards (no sub-groups)
                <div className="pt-3 space-y-2">
                  {renderTierCards(allParentItems, sectionKey, parentGroup.id === 'reserved')}
                </div>
              )}
            </div>
          )}
        </div>
      )
    }

    // Check if user can view provisioned pricing
    const canViewProvisioned = canViewProvisionedPricing(user)

    // Determine which parent groups have data
    const parentGroupsWithData = PRICING_GROUP_HIERARCHY.filter(parentGroup => {
      // Hide Provisioned group from non-privileged users
      if (parentGroup.id === 'provisioned' && !canViewProvisioned) return false
      
      if (parentGroup.subGroups) {
        return parentGroup.subGroups.some(sg => 
          sg.pricingGroups.some(pg => itemsByPricingGroup[pg]?.length > 0)
        )
      }
      return parentGroup.pricingGroups?.some(pg => itemsByPricingGroup[pg]?.length > 0)
    })

    const firstParentWithData = parentGroupsWithData[0]?.id

    // Split into columns for layout
    const leftColumnIds = ['in_region', 'cris']
    const rightColumnIds = ['reserved', 'provisioned', 'custom_model']

    return (
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="space-y-4">
          {parentGroupsWithData
            .filter(pg => leftColumnIds.includes(pg.id))
            .map(pg => renderParentGroup(pg, pg.id === firstParentWithData))}
        </div>
        <div className="space-y-4">
          {parentGroupsWithData
            .filter(pg => rightColumnIds.includes(pg.id))
            .map(pg => renderParentGroup(pg, pg.id === firstParentWithData))}
        </div>
      </div>
    )
  }

  // ============================================
  // MAIN RENDER
  // ============================================
  return (
    <ScrollArea className="h-full">
      <div className="p-6">
        {/* Header with Search and Source Info - inline layout */}
        <div className="flex items-center gap-4 mb-6">
          {/* Search on the left */}
          <div className="relative w-full max-w-md flex-shrink-0">
            <Search className={cn('absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
            <Input
              placeholder="Search by region, tier, or pricing type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
          
          {/* Source Info centered in remaining space */}
          <div className="flex-1 flex justify-center">
            <div className={cn(
              'flex items-center gap-2 px-3 py-2 rounded-lg text-xs',
              isLight
                ? 'bg-stone-100/80 text-stone-600 border border-stone-200/60'
                : 'bg-white/[0.03] text-slate-400 border border-white/[0.06]'
            )}>
              <Info className={cn('h-3.5 w-3.5 flex-shrink-0', isLight ? 'text-stone-400' : 'text-slate-500')} />
              <span>
                Source: <a href="https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-list-query-api.html" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">AWS Price List API</a>. Verify at <a href="https://aws.amazon.com/bedrock/pricing/" target="_blank" rel="noopener noreferrer" className="underline decoration-current hover:opacity-80">Bedrock Pricing</a> before sharing.
              </span>
            </div>
          </div>
        </div>

        {/* AWS Pricing Calculator Banner */}
        <a
          href="https://calculator.aws/#/createCalculator/bedrock"
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            'flex items-center gap-2.5 px-4 py-2.5 rounded-lg mb-6 text-xs font-medium transition-colors group',
            isLight
              ? 'bg-blue-50/60 border border-blue-200/40 text-blue-700 hover:bg-blue-50'
              : 'bg-blue-500/[0.08] border border-blue-500/20 text-blue-400 hover:bg-blue-500/[0.12]'
          )}
        >
          <Calculator className="h-4 w-4 flex-shrink-0" />
          <span>Ready to estimate costs? <span className="underline decoration-current underline-offset-2">Open AWS Pricing Calculator</span></span>
          <ExternalLink className="h-3 w-3 ml-auto flex-shrink-0 opacity-50 group-hover:opacity-80 transition-opacity" />
        </a>

        {/* View Content */}
        {renderByTypeView()}
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
  const user = useAuthStore(s => s.user)
  const showQuotas = canViewQuotas(user)
  const { theme } = useTheme()
  const isLight = theme === 'light'

  if (!model) return null

  const lifecycleStatus = model.lifecycle?.status ?? model.model_lifecycle?.status ?? model.model_status ?? 'ACTIVE'
  const globalStatus = model.lifecycle?.global_status ?? model.model_lifecycle?.global_status
  const statusSummary = model.lifecycle?.status_summary ?? model.model_lifecycle?.status_summary
  
  // Helper function to get status styles
  const getStatusStyles = (status) => {
    const normalizedStatus = (status || 'ACTIVE').toUpperCase()
    switch (normalizedStatus) {
      case 'ACTIVE':
        return isLight
          ? 'bg-emerald-100 text-emerald-700 border border-emerald-200'
          : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
      case 'LEGACY':
        return isLight
          ? 'bg-amber-100 text-amber-700 border border-amber-200'
          : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
      case 'EOL':
        return isLight
          ? 'bg-red-100 text-red-700 border border-red-200'
          : 'bg-red-500/20 text-red-400 border border-red-500/30'
      case 'MIXED':
        return isLight
          ? 'bg-purple-100 text-purple-700 border border-purple-200'
          : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
      default:
        return isLight
          ? 'bg-stone-100 text-stone-700 border border-stone-200'
          : 'bg-white/10 text-slate-400 border border-white/20'
    }
  }
  
  const getStatusLabel = (status) => {
    const normalizedStatus = (status || 'ACTIVE').toUpperCase()
    switch (normalizedStatus) {
      case 'ACTIVE': return 'Active'
      case 'LEGACY': return 'Legacy'
      case 'EOL': return 'End of Life'
      case 'MIXED': return 'Mixed'
      default: return normalizedStatus
    }
  }
  
  // Helper to render status badges - handles both single and MIXED status
  const renderStatusBadges = () => {
    // If global_status is MIXED and we have status_summary, render multiple badges
    if (globalStatus === 'MIXED' && statusSummary) {
      const statusOrder = ['LEGACY', 'ACTIVE', 'EOL']
      const activeStatuses = statusOrder.filter(s => statusSummary[s]?.length > 0)
      
      if (activeStatuses.length > 0) {
        return (
          <>
            {activeStatuses.map(s => (
              <span
                key={s}
                className={cn(
                  'px-1.5 py-0.5 rounded-full text-[9px] font-semibold uppercase tracking-wide',
                  getStatusStyles(s)
                )}
              >
                {getStatusLabel(s)}
              </span>
            ))}
          </>
        )
      }
    }
    
    // Single status display (backward compatible)
    return (
      <span className={cn(
        'px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide',
        getStatusStyles(lifecycleStatus)
      )}>
        {getStatusLabel(lifecycleStatus)}
      </span>
    )
  }

  const contextWindow = model.specs?.context_window ?? model.converse_data?.context_window
  const extendedContext = model.specs?.extended_context ?? model.converse_data?.extended_context
  const hasExtendedContext = model.specs?.extended_context != null || model.converse_data?.has_extended_context
  const maxOutput = model.specs?.max_output ?? model.specs?.max_output_tokens ?? model.converse_data?.max_output_tokens
  const regions = model.availability?.on_demand?.regions ?? model.in_region ?? []
  const capabilities = model.capabilities ?? model.model_capabilities ?? []
  const streamingSupported = model.streaming ?? model.streaming_supported
  const crisSupported = model.availability?.cross_region?.supported ?? model.cross_region_inference?.supported
  const mantleSupported = model.availability?.mantle?.supported
  const mantleRegions = model.availability?.mantle?.regions ?? []
  const inputModalities = model.modalities?.input_modalities ?? model.model_modalities?.input_modalities ?? []
  const outputModalities = model.modalities?.output_modalities ?? model.model_modalities?.output_modalities ?? []

  // Compute quota stats
  const quotas = model.quotas ?? model.model_service_quotas ?? {}
  const quotaRegions = Object.keys(quotas)
  let totalQuotas = 0
  let adjustableQuotas = 0
  const quotaCategories = new Set()
  for (const region of quotaRegions) {
    const regionQuotas = quotas[region] || []
    for (const quota of regionQuotas) {
      // Skip per-day quotas (derived from per-minute, redundant)
      if (/per\s+day/i.test(quota.quota_name || '')) continue
      totalQuotas++
      if (quota.adjustable) adjustableQuotas++
      quotaCategories.add(categorizeQuota(quota.quota_name || ''))
    }
  }

  // Compute pricing stats
  const pricingResult = getPricingForModel ? getPricingForModel(model, preferredRegion) : null
  
  // Detect available pricing tiers (Flex, Priority) from pricing data
  const availableTiers = new Set()
  const fullPricing = pricingResult?.fullPricing
  if (fullPricing?.regions) {
    for (const regionData of Object.values(fullPricing.regions)) {
      if (regionData?.pricing_groups) {
        for (const items of Object.values(regionData.pricing_groups)) {
          for (const item of items) {
            if (item.dimensions?.tier) {
              availableTiers.add(item.dimensions.tier)
            }
          }
        }
      }
    }
  }
  const hasFlexTier = availableTiers.has('flex')
  const hasPriorityTier = availableTiers.has('priority')
  
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
                {renderStatusBadges()}
                {model.availability?.mantle?.only && (
                  <span className={cn(
                    'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold',
                    isLight
                      ? 'bg-violet-100 text-violet-700 border border-violet-200'
                      : 'bg-violet-500/15 text-violet-400 border border-violet-500/30'
                  )}>
                    Mantle Only
                  </span>
                )}
                {!model.availability?.mantle?.only && model.availability?.mantle?.supported && (
                  <span className={cn(
                    'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold',
                    isLight
                      ? 'bg-violet-100 text-violet-700 border border-violet-200'
                      : 'bg-violet-500/15 text-violet-400 border border-violet-500/30'
                  )}>
                    Mantle
                  </span>
                )}
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
                    {hasExtendedContext && extendedContext ? (
                      <Tooltip delayDuration={200}>
                        <TooltipTrigger asChild>
                          <span className="cursor-default">
                            {formatNumber(contextWindow)}
                            <span className={cn('ml-1', isLight ? 'text-amber-500' : 'text-emerald-400')}>
                              | {formatNumber(extendedContext)}
                            </span>
                          </span>
                        </TooltipTrigger>
                        <TooltipContent side="bottom" className="max-w-[220px] text-xs">
                          <p><strong>{formatNumber(contextWindow)}</strong> base context window</p>
                          <p><strong>{formatNumber(extendedContext)}</strong> extended context (beta)</p>
                        </TooltipContent>
                      </Tooltip>
                    ) : (
                      formatNumber(contextWindow)
                    )}
                  </p>
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
                  {/* Features */}
                  <div className="space-y-3">
                    <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      Features
                    </h3>
                    <div className="flex flex-wrap gap-1.5">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className={cn(
                            'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
                            streamingSupported
                              ? isLight ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              : isLight ? 'bg-stone-100 text-stone-400 border border-stone-200' : 'bg-white/[0.03] text-slate-500 border border-white/[0.06]'
                          )}>
                            <Radio className="h-3.5 w-3.5" />
                            Streaming
                          </span>
                        </TooltipTrigger>
                        <TooltipContent side="top">
                          <p className="text-xs">Supports streaming responses</p>
                        </TooltipContent>
                      </Tooltip>
                      <span className={cn(
                        'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
                        crisSupported
                          ? isLight ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : isLight ? 'bg-stone-100 text-stone-400 border border-stone-200' : 'bg-white/[0.03] text-slate-500 border border-white/[0.06]'
                      )}>
                        <Globe className="h-3.5 w-3.5" />
                        CRIS
                      </span>
                      <span className={cn(
                        'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
                        mantleSupported
                          ? isLight ? 'bg-violet-50 text-violet-700 border border-violet-200' : 'bg-violet-500/10 text-violet-400 border border-violet-500/20'
                          : isLight ? 'bg-stone-100 text-stone-400 border border-stone-200' : 'bg-white/[0.03] text-slate-500 border border-white/[0.06]'
                      )}>
                        <Cpu className="h-3.5 w-3.5" />
                        Mantle
                      </span>
                      {hasFlexTier && (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className={cn(
                              'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
                              isLight 
                                ? 'bg-blue-50 text-blue-700 border border-blue-200' 
                                : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                            )}>
                              <Zap className="h-3.5 w-3.5" />
                              Flex
                            </span>
                          </TooltipTrigger>
                          <TooltipContent side="top">
                            <p className="text-xs">Supports Flex pricing tier (lower cost, variable latency)</p>
                          </TooltipContent>
                        </Tooltip>
                      )}
                      {hasPriorityTier && (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className={cn(
                              'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
                              isLight 
                                ? 'bg-amber-50 text-amber-700 border border-amber-200' 
                                : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                            )}>
                              <Zap className="h-3.5 w-3.5" />
                              Priority
                            </span>
                          </TooltipTrigger>
                          <TooltipContent side="top">
                            <p className="text-xs">Supports Priority pricing tier (guaranteed low latency)</p>
                          </TooltipContent>
                        </Tooltip>
                      )}
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

              {showQuotas && activeTab === 'quotas' && (
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
                  {showQuotas && (
                    <TabsTrigger value="quotas" className="rounded-none border-b-2 border-transparent data-[state=active]:border-current px-6 py-3">
                      <span className="flex items-center gap-2">
                        Service Quotas
                        {getSectionBadge(user, 'quotas') && (
                          <span className={cn(
                            'text-[9px] font-bold leading-none px-1.5 py-0.5 rounded-full border',
                            isLight ? getSectionBadge(user, 'quotas').light : getSectionBadge(user, 'quotas').dark
                          )}>
                            {getSectionBadge(user, 'quotas').text}
                          </span>
                        )}
                      </span>
                    </TabsTrigger>
                  )}
                  <TabsTrigger value="pricing" className="rounded-none border-b-2 border-transparent data-[state=active]:border-current px-6 py-3">
                    Pricing
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="specs" className="flex-1 mt-0 min-h-0 overflow-hidden">
                  <SpecsTab model={model} user={user} getPricingForModel={getPricingForModel} preferredRegion={preferredRegion} />
                </TabsContent>

                {showQuotas && (
                  <TabsContent value="quotas" className="flex-1 mt-0 min-h-0 overflow-hidden">
                    <QuotasTab model={model} getPricingForModel={getPricingForModel} preferredRegion={preferredRegion} />
                  </TabsContent>
                )}

                <TabsContent value="pricing" className="flex-1 mt-0 min-h-0 overflow-hidden">
                  <PricingTab model={model} getPricingForModel={getPricingForModel} preferredRegion={preferredRegion} user={user} />
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  )
}
