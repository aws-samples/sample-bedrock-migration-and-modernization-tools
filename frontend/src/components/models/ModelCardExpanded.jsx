import { useState } from 'react'
import { Star, Globe, Zap, MessageSquare, Image, FileText, Video, Mic, Check, X, ChevronDown, ChevronRight, Search, Database, Languages, Cpu, Layers, Package, Server, ExternalLink, Copy, DollarSign, GitCompareArrows, Radio, Info, Bot, BookOpen, Workflow, Shield, Clock, Route,  Wrench, AlertTriangle, AlertCircle } from 'lucide-react'
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
import { canViewQuotas, getSectionBadge } from '@/config/admin'
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
function CollapsibleSection({ title, icon: Icon, children, defaultExpanded = false, infoLink = null }) {
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
                      {getRegionDisplayName(region)} <span className={cn('font-mono ml-1', isLight ? 'text-stone-500' : 'text-slate-300')}>({region})</span>
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

// On-Demand Availability section with model ID highlight, stats, and geo-grouped regions
function OnDemandAvailabilitySection({ model }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const regions = model.availability?.on_demand?.regions ?? model.in_region ?? []
  const grouped = groupRegionsByGeo(regions)
  const geoCount = Object.keys(grouped).length
  const modelId = model.model_id
  const isMantleOnly = model.availability?.mantle?.only

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

      {/* Model ID highlight bar */}
      {modelId && !isMantleOnly && (
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
            {regions.length > 0 ? (
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
              <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{regions.length}</p>
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
          <p className={cn('text-xs font-medium mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Regions by Geography</p>
          <RegionalAvailabilityGrouped regions={regions} />
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

// Cross-Region Inference Section grouped by geographic scope
function CrossRegionInferenceSection({ crisData }) {
  const [expandedScopes, setExpandedScopes] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const profiles = crisData.profiles || []
  const sourceRegions = crisData.regions || []

  // Helper to get profile fields (handles both field naming conventions)
  const getProfileId = (p) => p.profile_id || p.inference_profile_id
  const getProfileName = (p) => p.profile_name || p.inference_profile_name
  const getSourceRegion = (p) => p.source_region || p.region

  // Extract geographic scope prefix from profile ID (e.g., "us" from "us.anthropic.claude...")
  const getScopePrefix = (profile) => {
    const profileId = getProfileId(profile)
    return profileId?.split('.')[0]?.toLowerCase() || null
  }

  // Parse routing regions from description field
  // e.g., "Routes requests to Claude in us-east-1, us-east-2 and us-west-2." -> ["us-east-1", "us-east-2", "us-west-2"]
  const parseRoutingRegions = (description) => {
    if (!description) return []
    // Match region patterns like us-east-1, eu-west-2, ap-northeast-1, etc.
    const regionPattern = /[a-z]{2,4}-[a-z]+-\d/g
    const matches = description.match(regionPattern)
    return matches ? [...new Set(matches)] : []
  }

  // Group profiles by profile_id, collecting source_region → routing mappings
  // Each source region can route to different destinations!
  const profilesMap = new Map()
  for (const profile of profiles) {
    const profileId = getProfileId(profile)
    if (!profileId) continue

    const existing = profilesMap.get(profileId)
    const sourceRegion = getSourceRegion(profile)
    const routingRegions = parseRoutingRegions(profile.description)

    if (existing) {
      // Add source region with its specific routing destinations
      if (sourceRegion && !existing.routingBySource[sourceRegion]) {
        existing.routingBySource[sourceRegion] = routingRegions
      }
    } else {
      profilesMap.set(profileId, {
        profile,
        scopePrefix: getScopePrefix(profile),
        // Map of source_region → destination_regions
        routingBySource: sourceRegion ? { [sourceRegion]: routingRegions } : {}
      })
    }
  }

  // Normalize scope - group related regions together
  // JP, AU, APAC → APAC; CA, SA, ME, AF returned as their own scopes
  const normalizeScope = (scope) => {
    const upperScope = scope?.toUpperCase()
    if (!upperScope) return 'UNKNOWN'
    if (upperScope === 'JP' || upperScope === 'AU' || upperScope === 'APAC') {
      return 'APAC'
    }
    if (upperScope === 'GLOBAL' || upperScope === 'US' || upperScope === 'EU' ||
        upperScope === 'CA' || upperScope === 'SA' || upperScope === 'ME' || upperScope === 'AF') {
      return upperScope
    }
    // Unknown scopes — return as-is (uppercased)
    return upperScope
  }

  // Group by scope prefix dynamically (with normalization)
  const profilesByScope = {}
  for (const [profileId, data] of profilesMap) {
    const scope = normalizeScope(data.scopePrefix)
    if (!profilesByScope[scope]) {
      profilesByScope[scope] = []
    }
    profilesByScope[scope].push(data)
  }

  // Get available scopes sorted (global first, then alphabetically)
  const availableScopes = Object.keys(profilesByScope).sort((a, b) => {
    if (a === 'GLOBAL') return -1
    if (b === 'GLOBAL') return 1
    return a.localeCompare(b)
  })

  // Dynamic scope display info
  const getScopeDisplay = (scope) => {
    const icons = {
      'GLOBAL': '🌐', 'US': '🇺🇸', 'EU': '🇪🇺', 'APAC': '🌏',
      'CA': '🇨🇦', 'SA': '🌎', 'ME': '🌍', 'AF': '🌍'
    }
    const labels = {
      'GLOBAL': 'Global', 'US': 'United States', 'EU': 'Europe', 'APAC': 'Asia Pacific',
      'CA': 'Canada', 'SA': 'South America', 'ME': 'Middle East', 'AF': 'Africa'
    }
    return {
      icon: icons[scope] || '📍',
      label: labels[scope] || scope
    }
  }

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
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Geographic Scopes</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-amber-700' : 'text-[#1A9E7A]')}>{availableScopes.length}</p>
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

      {/* CRIS Endpoints grouped by geographic scope */}
      {crisData.supported && availableScopes.length > 0 && (
        <div className="space-y-3">
          <p className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>CRIS Endpoints by Geographic Scope</p>

          {availableScopes.map(scopeKey => {
            const scopeDisplay = getScopeDisplay(scopeKey)
            const scopeProfiles = profilesByScope[scopeKey]
            const isExpanded = expandedScopes[scopeKey]

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
                    <span>{scopeDisplay.icon}</span>
                    <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                      {scopeDisplay.label}
                    </span>
                    <Badge variant="info" className="text-xs">{scopeProfiles.length} endpoint{scopeProfiles.length !== 1 ? 's' : ''}</Badge>
                  </div>
                  {isExpanded ? (
                    <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                  ) : (
                    <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                  )}
                </button>
                {isExpanded && (
                  <div className={cn('px-3 pb-3 pt-3 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/[0.06]')}>
                    {scopeProfiles.map(({ profile, routingBySource }, idx) => {
                      const sourceRegionsList = Object.keys(routingBySource).sort()
                      return (
                        <div key={`${getProfileId(profile)}-${idx}`} className={cn(
                          'rounded p-2',
                          isLight ? 'bg-stone-50 border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
                        )}>
                          <p className={cn('text-sm font-medium', isLight ? 'text-stone-900' : 'text-white')}>
                            {getProfileName(profile)}
                          </p>
                          <CopyableText
                            text={getProfileId(profile)}
                            isLight={isLight}
                            className={cn('text-xs font-mono mt-0.5', isLight ? 'text-stone-500 hover:text-stone-700' : 'text-[#c0c1c5] hover:text-white')}
                          />
                          {profile.status && profile.status !== 'ACTIVE' && (
                            <div className="flex items-center gap-2 mt-1">
                              <Badge variant="destructive" className="text-[10px]">{profile.status}</Badge>
                            </div>
                          )}

                          {/* Source regions display */}
                          {sourceRegionsList.length > 0 && (
                            <div className="mt-3">
                              {scopeKey === 'GLOBAL' ? (
                                <>
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
                                  <p className={cn('text-[10px] mt-2 italic flex items-center gap-1', isLight ? 'text-stone-400' : 'text-slate-500')}>
                                    <Globe className="h-3 w-3" />
                                    All source regions route to all supported destination regions
                                  </p>
                                </>
                              ) : (
                                <>
                                  <p className={cn('text-[10px] mb-2 font-medium', isLight ? 'text-stone-600' : 'text-slate-300')}>
                                    Routing by Source Region ({sourceRegionsList.length} sources)
                                  </p>
                                  <div className="space-y-1.5">
                                    {sourceRegionsList.map(sourceRegion => {
                                      const destinations = routingBySource[sourceRegion] || []
                                      return (
                                        <div key={sourceRegion} className={cn(
                                          'flex items-start gap-2 text-[10px] p-1.5 rounded',
                                          isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.04]'
                                        )}>
                                          <Tooltip delayDuration={200}>
                                            <TooltipTrigger asChild>
                                              <Badge variant="secondary" className="text-[10px] cursor-default shrink-0">
                                                {getRegionDisplayName(sourceRegion)}
                                              </Badge>
                                            </TooltipTrigger>
                                            <TooltipContent side="left" sideOffset={4}>
                                              <p className="font-mono text-xs">{sourceRegion}</p>
                                            </TooltipContent>
                                          </Tooltip>
                                          <span className={cn('shrink-0', isLight ? 'text-stone-400' : 'text-slate-500')}>→</span>
                                          <div className="flex flex-wrap gap-1">
                                            {destinations.length > 0 ? destinations.sort().map(dest => (
                                              <Tooltip key={dest} delayDuration={200}>
                                                <TooltipTrigger asChild>
                                                  <Badge variant="info" className="text-[10px] cursor-default">
                                                    {getRegionDisplayName(dest)}
                                                  </Badge>
                                                </TooltipTrigger>
                                                <TooltipContent side="bottom" sideOffset={4}>
                                                  <p className="font-mono text-xs">{dest}</p>
                                                </TooltipContent>
                                              </Tooltip>
                                            )) : (
                                              <span className={cn('italic', isLight ? 'text-stone-400' : 'text-slate-500')}>All supported regions</span>
                                            )}
                                          </div>
                                        </div>
                                      )
                                    })}
                                  </div>
                                </>
                              )}
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
          <p className={cn('text-xs font-medium mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>Regions by Geography</p>
          <RegionalAvailabilityGrouped regions={regions} />
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
  const regions = batchData.regions ?? batchData.supported_regions ?? []
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
                          {getRegionDisplayName(region)}
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

// Mantle Inference Section with grouped regions
function MantleInferenceSection({ mantleData }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const regions = mantleData.regions || []
  const grouped = groupRegionsByGeo(regions)

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
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>Regions</p>
          <p className={cn('text-lg font-bold', isLight ? 'text-violet-700' : 'text-violet-400')}>{regions.length}</p>
        </div>
      </div>

      {/* Regions grouped by geography */}
      {mantleData.supported && regions.length > 0 && (
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
              <Globe className={cn('h-4 w-4', isLight ? 'text-violet-600' : 'text-violet-400')} />
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
                          {getRegionDisplayName(region)}
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

      {/* Endpoint pattern info */}
      {mantleData.supported && (
        <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>
          Endpoint: <code className={cn('font-mono px-1 py-0.5 rounded', isLight ? 'bg-stone-100' : 'bg-white/[0.06]')}>{mantleData.mantle_endpoint_pattern ?? 'bedrock-mantle.{region}.api.aws'}</code>
        </p>
      )}
    </div>
  )
}

// Consumption option explanations for info popover
const consumptionExplanations = {
  'In Region': 'On-demand inference in a specific AWS region. Pay per token/request with no commitment.',
  'Cross-Region (CRIS)': 'Cross-Region Inference Service routes requests to available capacity across regions for higher throughput.',
  'Batch': 'Process large volumes of requests asynchronously at lower cost. Results delivered to S3.',
  'Mantle': 'Managed inference endpoints with dedicated capacity and custom configurations.',
}

// Compact availability summary with expandable detail sections for each inference type
function AvailabilitySummary({ model }) {
  const [expandedTypes, setExpandedTypes] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const isMantleOnly = model.availability?.mantle?.only
  const regions = model.availability?.on_demand?.regions ?? model.in_region ?? []
  const crisData = model.availability?.cross_region ?? model.cross_region_inference ?? {}
  const batchData = model.availability?.batch ?? model.batch_inference_supported ?? {}
  const mantleData = model.availability?.mantle ?? {}

  const toggleType = (label) => {
    setExpandedTypes(prev => ({ ...prev, [label]: !prev[label] }))
  }

  const types = [
    {
      label: 'In Region',
      supported: isMantleOnly ? false : (((model.availability?.on_demand?.regions ?? model.in_region)?.length > 0) || (regions.length > 0 && (model.inference_types_supported || []).includes('ON_DEMAND'))),
      count: isMantleOnly ? 0 : ((model.availability?.on_demand?.regions ?? model.in_region)?.length ?? model.total_in_region ?? regions.length),
      detail: () => <OnDemandAvailabilitySection model={model} />,
    },
    {
      label: 'Cross-Region (CRIS)',
      supported: isMantleOnly ? false : !!crisData.supported,
      count: isMantleOnly ? 0 : (crisData.regions?.length ?? crisData.profiles?.length ?? 0),
      detail: () => <CrossRegionInferenceSection crisData={crisData} />,
    },
    {
      label: 'Batch',
      supported: isMantleOnly ? false : !!batchData.supported,
      count: isMantleOnly ? 0 : (batchData.regions?.length ?? batchData.supported_regions?.length ?? 0),
      detail: () => <BatchInferenceSection batchData={batchData} />,
    },
    {
      label: 'Mantle',
      supported: !!mantleData.supported,
      count: mantleData.regions?.length ?? 0,
      highlight: isMantleOnly,
      detail: () => <MantleInferenceSection mantleData={mantleData} />,
    },
  ]

  return (
    <div className="space-y-1.5">
      {/* Header with info button */}
      <div className="flex items-center justify-between mb-2">
        <span className={cn('text-xs font-medium', isLight ? 'text-stone-600' : 'text-slate-400')}>
          Availability
        </span>
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
      {types.map(({ label, supported, count, highlight, detail }) => {
        const isExpanded = expandedTypes[label]
        return (
          <div key={label}>
            <button
              onClick={() => supported && toggleType(label)}
              disabled={!supported}
              className={cn(
                'w-full flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs transition-all',

                // Supported: colored left accent + tinted bg
                supported && !highlight && (isLight
                  ? 'bg-emerald-50/70 border-l-[3px] border-l-emerald-500 border-y border-r border-emerald-200/60 hover:bg-emerald-100/70 cursor-pointer'
                  : 'bg-emerald-500/[0.08] border-l-[3px] border-l-emerald-500 border-y border-r border-emerald-500/20 hover:bg-emerald-500/[0.12] cursor-pointer'
                ),

                // Mantle highlight: violet accent
                supported && highlight && (isLight
                  ? 'bg-violet-50 border-l-[3px] border-l-violet-500 border-y border-r border-violet-200 hover:bg-violet-100/70 cursor-pointer'
                  : 'bg-violet-500/10 border-l-[3px] border-l-violet-400 border-y border-r border-violet-500/30 hover:bg-violet-500/[0.15] cursor-pointer'
                ),

                // Unsupported: dimmed and muted
                !supported && (isLight
                  ? 'bg-stone-50/40 border border-stone-200/50 opacity-50 cursor-not-allowed'
                  : 'bg-white/[0.01] border border-white/[0.04] opacity-40 cursor-not-allowed'
                ),

                isExpanded && 'rounded-b-none'
              )}
            >
              <div className="flex items-center gap-1.5">
                {supported && (
                  isExpanded
                    ? <ChevronDown className={cn('h-3 w-3', highlight ? (isLight ? 'text-violet-600' : 'text-violet-400') : (isLight ? 'text-emerald-600' : 'text-emerald-400'))} />
                    : <ChevronRight className={cn('h-3 w-3', highlight ? (isLight ? 'text-violet-600' : 'text-violet-400') : (isLight ? 'text-emerald-600' : 'text-emerald-400'))} />
                )}
                <span className={cn(
                  'font-medium',
                  supported && highlight && (isLight ? 'text-violet-700' : 'text-violet-300'),
                  supported && !highlight && (isLight ? 'text-stone-800' : 'text-emerald-200'),
                  !supported && (isLight ? 'text-stone-500' : 'text-slate-500')
                )}>
                  {label}
                </span>
              </div>
              <div className="flex items-center gap-2">
                {supported && count > 0 && (
                  <span className={cn(
                    'text-[10px] font-mono tabular-nums font-semibold',
                    highlight
                      ? (isLight ? 'text-violet-600' : 'text-violet-400')
                      : (isLight ? 'text-emerald-700' : 'text-emerald-400')
                  )}>
                    {count} {count === 1 ? 'region' : 'regions'}
                  </span>
                )}
                <span className={cn(
                  'inline-flex items-center justify-center w-[18px] h-[18px] rounded-full flex-shrink-0',
                  supported && highlight && (isLight ? 'bg-violet-500 text-white' : 'bg-violet-400 text-violet-950'),
                  supported && !highlight && (isLight ? 'bg-emerald-500 text-white' : 'bg-emerald-400 text-emerald-950'),
                  !supported && (isLight ? 'bg-stone-200 text-stone-400' : 'bg-white/[0.06] text-slate-600')
                )}>
                  {supported ? <Check className="h-3 w-3 stroke-[2.5]" /> : <X className="h-2.5 w-2.5" />}
                </span>
              </div>
            </button>
            {isExpanded && supported && (
              <div className={cn(
                'px-3 py-3 rounded-b-md border-l-[3px] border-r border-b border-t-0',
                highlight
                  ? isLight ? 'bg-violet-50/40 border-l-violet-500 border-r-violet-200 border-b-violet-200' : 'bg-violet-500/5 border-l-violet-400 border-r-violet-500/20 border-b-violet-500/20'
                  : isLight ? 'bg-emerald-50/30 border-l-emerald-500 border-r-emerald-200/60 border-b-emerald-200/60' : 'bg-emerald-500/[0.04] border-l-emerald-500 border-r-emerald-500/20 border-b-emerald-500/20'
              )}>
                {detail()}
              </div>
            )}
          </div>
        )
      })}

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
    // If it's already a string, return as-is
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

function SpecsTab({ model }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  
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
  const consumptionOptions = model.consumption_options || []
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
        {/* Two-column grid layout */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* LEFT COLUMN */}
          <div className="space-y-6">
            {/* MODEL CAPABILITIES */}
            <div>
              <CategoryHeader icon={Cpu} title="Model Capabilities" />
              <div className="space-y-3">
                {/* Modalities - Always expanded */}
                <CollapsibleSection title="Input & Output Modalities" icon={Layers} defaultExpanded={true}>
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

                {/* Capabilities - Collapsed by default */}
                {capabilities.length > 0 && (
                  <CollapsibleSection title="Capabilities" icon={Zap} defaultExpanded={false}>
                    <ExpandableTagList
                      label=""
                      items={capabilities}
                      maxVisible={8}
                      isLight={isLight}
                    />
                  </CollapsibleSection>
                )}

                {/* Use Cases - Collapsed by default */}
                {useCases.length > 0 && (
                  <CollapsibleSection title="Use Cases" icon={BookOpen} defaultExpanded={false}>
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
                {/* Bedrock Features - Collapsed by default */}
                {(model.features ?? model.feature_support) && (
                  <CollapsibleSection title="Bedrock Features" icon={Layers} defaultExpanded={false}>
                    <BedrockFeaturesSection featureSupport={model.features ?? model.feature_support} />
                  </CollapsibleSection>
                )}

                {/* Languages - Collapsed by default */}
                {languages.length > 0 && (
                  <CollapsibleSection title="Languages" icon={Languages} defaultExpanded={false}>
                    <div className="flex flex-wrap gap-1.5">
                      {languages.map(lang => (
                        <Badge key={lang} variant="secondary" className="text-xs">{lang}</Badge>
                      ))}
                    </div>
                  </CollapsibleSection>
                )}

                {/* Customizations - Collapsed by default */}
                {customizations.length > 0 && (
                  <CollapsibleSection title="Customizations" icon={Wrench} defaultExpanded={false}>
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
                <CollapsibleSection title="Regional Availability" icon={Globe} defaultExpanded={true}>
                  <AvailabilitySummary model={model} />
                </CollapsibleSection>

                {/* Consumption Options - Collapsed by default */}
                {consumptionOptions.length > 0 && (
                  <CollapsibleSection title="Consumption Options" icon={Server} defaultExpanded={false}>
                    <div className="flex flex-wrap gap-1.5">
                      {consumptionOptions.map(opt => {
                        const labels = {
                          'on_demand': 'In Region',
                          'batch': 'Batch',
                          'provisioned': 'Provisioned',
                          'provisioned_throughput': 'Provisioned Throughput',
                          'cross_region_inference': 'Cross-Region Inference',
                          'mantle': 'Mantle Inference'
                        }
                        return (
                          <Badge key={opt} variant="info" className="text-xs">
                            {labels[opt] || opt}
                          </Badge>
                        )
                      })}
                    </div>
                  </CollapsibleSection>
                )}
              </div>
            </div>

            {/* LIFECYCLE & RESOURCES */}
            <div>
              <CategoryHeader icon={Clock} title="Lifecycle & Resources" />
              <div className="space-y-3">
                {/* Lifecycle Details - Collapsed by default */}
                <CollapsibleSection title="Status & Dates" icon={Clock} defaultExpanded={false}>
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

function QuotasTab({ model }) {
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedGeos, setExpandedGeos] = useState({})
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const quotas = model.quotas ?? model.model_service_quotas ?? {}
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
  'Batch': { icon: '📦', label: 'Batch' },
  'Batch Global': { icon: '🌍', label: 'Batch Global' },
  'Batch Long Context': { icon: '📖', label: 'Batch Long Context' },
  'Batch Long Context Global': { icon: '🌏', label: 'Batch Long Context Global' },
  'Provisioned Throughput': { icon: '⚡', label: 'Provisioned Throughput' },
  'Custom Model': { icon: '🎯', label: 'Custom Model' },
}

function PricingTab({ model, getPricingForModel, preferredRegion = 'us-east-1' }) {
  const [searchQuery, setSearchQuery] = useState('')
  const [pricingView, setPricingView] = useState('byType') // 'byType' | 'byRegion'
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

  const pricingGroupOrder = ['On-Demand Global', 'On-Demand', 'On-Demand Long Context', 'On-Demand Long Context Global', 'Batch Global', 'Batch', 'Batch Long Context', 'Batch Long Context Global', 'Provisioned Throughput', 'Custom Model']
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

  // ============================================
  // CONSUMPTION TYPE MAPPING FOR "BY TYPE" VIEW
  // ============================================
  const consumptionTypeMapping = {
    'In Region': ['On-Demand'],
    'CRIS': ['On-Demand Global'],
    'Long Context': ['On-Demand Long Context', 'On-Demand Long Context Global'],
    'Batch': ['Batch', 'Batch Global', 'Batch Long Context', 'Batch Long Context Global'],
    'Provisioned': ['Provisioned Throughput'],
    'Custom': ['Custom Model']
  }

  const consumptionTypeInfo = {
    'In Region': { icon: Zap, label: 'In Region', description: 'Standard on-demand pricing' },
    'CRIS': { icon: Globe, label: 'Cross-Region (CRIS)', description: 'Cross-region inference' },
    'CRIS-Global': { icon: Globe, label: 'Global', description: 'Global cross-region inference' },
    'CRIS-Regional': { icon: Globe, label: 'Regional', description: 'Regional cross-region inference' },
    'Long Context': { icon: BookOpen, label: 'Long Context', description: 'Extended context window pricing' },
    'Batch': { icon: Package, label: 'Batch', description: 'Batch processing with discounts' },
    'Provisioned': { icon: Server, label: 'Provisioned Throughput', description: 'Reserved capacity pricing' },
    'Custom': { icon: Wrench, label: 'Custom Model', description: 'Fine-tuned model pricing' }
  }

  // Get tier from pricing label
  const getTierFromLabel = (label) => {
    if (/cache\s*read/i.test(label)) return 'Cache Read'
    if (/cache\s*write/i.test(label)) return 'Cache Write'
    if (/1h\s*cache/i.test(label)) return '1h Cache'
    if (/flex/i.test(label)) return 'Flex'
    if (/priority/i.test(label)) return 'Priority'
    if (/no\s*commit/i.test(label)) return 'No Commit'
    if (/6\s*month/i.test(label)) return '6 Month'
    if (/1\s*month/i.test(label)) return '1 Month'
    return 'Standard'
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
            const { label, type } = simplifyPricingDescription(item.description, item.dimension)
            items.push({
              ...item,
              _label: label,
              _type: type,
              _price: item.price_per_thousand ?? item.price_per_unit,
              _unit: (item.unit_label || `per ${item.unit || 'unit'}`).replace(/1K tokens/gi, '1M tokens'),
              _raw: item.description || item.dimension,
              _groupName: groupName,
              _tier: getTierFromLabel(label),
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
  // VIEW TOGGLE COMPONENT
  // ============================================
  const ViewToggle = () => (
    <div className={cn(
      'inline-flex items-center rounded-lg p-0.5',
      isLight
        ? 'bg-stone-100/80 border border-stone-200/50'
        : 'bg-white/5 border border-white/10'
    )}>
      {[
        { key: 'byType', label: 'By Type', icon: Layers },
        { key: 'byRegion', label: 'By Region', icon: Globe }
      ].map(({ key, label, icon: Icon }) => (
        <button
          key={key}
          onClick={() => setPricingView(key)}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all',
            pricingView === key
              ? isLight
                ? 'bg-white text-stone-900 shadow-sm'
                : 'bg-white/10 text-white shadow-sm'
              : isLight
                ? 'text-stone-500 hover:text-stone-700'
                : 'text-slate-400 hover:text-white'
          )}
        >
          <Icon className="h-3.5 w-3.5" />
          {label}
        </button>
      ))}
    </div>
  )

  // ============================================
  // VIEW 1: BY TYPE (Consumption-First Hierarchy)
  // ============================================
  const renderByTypeView = () => {
    const allItems = filterItems(getAllPricingItems())
    
    // Group items by consumption type
    const byConsumptionType = {}
    for (const [consumptionType, groupNames] of Object.entries(consumptionTypeMapping)) {
      const typeItems = allItems.filter(item => groupNames.includes(item._groupName))
      if (typeItems.length > 0) {
        byConsumptionType[consumptionType] = typeItems
      }
    }

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
            priceBreakdown: Object.values(byPrice).sort((a, b) => a.price - b.price)
          }
        }
      }
      return result
    }

    // Render a consumption type section
    const renderConsumptionSection = (consumptionType, items, subLabel = null) => {
      const info = consumptionTypeInfo[subLabel || consumptionType]
      const Icon = info?.icon || Zap
      const sectionKey = `byType_${subLabel || consumptionType}`
      const isExpanded = expandedSections[sectionKey] !== false // Default expanded

      // Consolidate by tier and type
      const consolidated = consolidateByTierAndType(items)

      // Sort tiers: Standard first, then alphabetically
      const tierOrder = ['Standard', 'Cache Read', 'Cache Write', '1h Cache', 'Flex', 'Priority', 'No Commit', '1 Month', '6 Month']
      const sortedTiers = Object.keys(consolidated).sort((a, b) => {
        const idxA = tierOrder.indexOf(a)
        const idxB = tierOrder.indexOf(b)
        if (idxA !== -1 && idxB !== -1) return idxA - idxB
        if (idxA !== -1) return -1
        if (idxB !== -1) return 1
        return a.localeCompare(b)
      })

      // Format price range
      const formatPriceRange = (data) => {
        if (!data) return null
        if (data.hasRange) {
          return `$${formatPrice(data.minPrice)} - $${formatPrice(data.maxPrice)}`
        }
        return `$${formatPrice(data.minPrice)}`
      }

      return (
        <div key={subLabel || consumptionType} className={cn(
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
              <Icon className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                {info?.label || consumptionType}
              </span>
              <Badge variant="secondary" className="text-[10px]">
                {new Set(items.map(i => i.region)).size} regions
              </Badge>
            </div>
            {isExpanded ? (
              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            ) : (
              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            )}
          </button>
          {isExpanded && (
            <div className={cn('px-3 pb-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-white/10')}>
              {sortedTiers.map(tier => {
                const tierData = consolidated[tier]
                const tierKey = `${sectionKey}_${tier}`
                const hasInput = tierData.input
                const hasOutput = tierData.output
                const hasOther = tierData.other
                const hasInputOutput = hasInput && hasOutput
                
                // Calculate total unique regions for this tier
                const tierRegions = new Set([
                  ...(tierData.input?.allRegions || []),
                  ...(tierData.output?.allRegions || []),
                  ...(tierData.other?.allRegions || [])
                ])
                const totalTierRegions = tierRegions.size

                return (
                  <div key={tier} className="pt-2">
                    <div className={cn(
                      'rounded-lg p-2',
                      isLight ? 'bg-white border border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
                    )}>
                      <div className={cn('text-xs font-medium mb-2', isLight ? 'text-stone-700' : 'text-slate-200')}>
                        {tier}
                      </div>
                      
                      {/* Consolidated Input/Output row when both exist */}
                      {hasInputOutput && (
                        <div className="mb-1">
                          <div className={cn(
                            'flex items-center justify-between px-2 py-2 rounded',
                            isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
                          )}>
                            <div className="flex items-center gap-4 flex-wrap">
                              {/* Input */}
                              <div className="flex items-center gap-2">
                                <span className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 bg-blue-500" />
                                <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Input:</span>
                                <span className={cn('text-xs font-mono font-semibold tabular-nums', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                                  {formatPriceRange(tierData.input)}
                                  <span className={cn('font-normal ml-1', isLight ? 'text-stone-400' : 'text-slate-400')}>
                                    {tierData.input.unit}
                                  </span>
                                </span>
                              </div>
                              
                              <span className={cn('text-xs', isLight ? 'text-stone-300' : 'text-slate-600')}>|</span>
                              
                              {/* Output */}
                              <div className="flex items-center gap-2">
                                <span className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 bg-emerald-500" />
                                <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Output:</span>
                                <span className={cn('text-xs font-mono font-semibold tabular-nums', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                                  {formatPriceRange(tierData.output)}
                                  <span className={cn('font-normal ml-1', isLight ? 'text-stone-400' : 'text-slate-400')}>
                                    {tierData.output.unit}
                                  </span>
                                </span>
                              </div>
                            </div>
                            
                            <button
                              onClick={() => toggleSection(tierKey)}
                              className={cn(
                                'text-[10px] px-1.5 py-0.5 rounded transition-colors flex-shrink-0',
                                isLight
                                  ? 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                                  : 'bg-white/10 text-slate-300 hover:bg-white/20'
                              )}
                            >
                              {totalTierRegions} regions {expandedSections[tierKey] ? '▲' : '▼'}
                            </button>
                          </div>
                          
                          {/* Expanded price breakdown */}
                          {expandedSections[tierKey] && (
                            <div className={cn(
                              'ml-4 mt-2 space-y-2 px-2 py-2 rounded',
                              isLight ? 'bg-stone-50' : 'bg-white/[0.02]'
                            )}>
                              {/* Input breakdown */}
                              {tierData.input.priceBreakdown.length > 1 && (
                                <div>
                                  <div className={cn('text-[10px] font-medium mb-1 flex items-center gap-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>
                                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500" />
                                    Input breakdown:
                                  </div>
                                  {tierData.input.priceBreakdown.map((pb, idx) => (
                                    <div key={idx} className={cn('text-[10px] ml-3 flex justify-between', isLight ? 'text-stone-500' : 'text-slate-500')}>
                                      <span className="font-mono">${formatPrice(pb.price)}</span>
                                      <span>{pb.regions.length} regions: {pb.regions.sort().map(r => getRegionDisplayName(r) || r).join(', ')}</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                              
                              {/* Output breakdown */}
                              {tierData.output.priceBreakdown.length > 1 && (
                                <div>
                                  <div className={cn('text-[10px] font-medium mb-1 flex items-center gap-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>
                                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                    Output breakdown:
                                  </div>
                                  {tierData.output.priceBreakdown.map((pb, idx) => (
                                    <div key={idx} className={cn('text-[10px] ml-3 flex justify-between', isLight ? 'text-stone-500' : 'text-slate-500')}>
                                      <span className="font-mono">${formatPrice(pb.price)}</span>
                                      <span>{pb.regions.length} regions: {pb.regions.sort().map(r => getRegionDisplayName(r) || r).join(', ')}</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                              
                              {/* Single price - just show regions */}
                              {tierData.input.priceBreakdown.length === 1 && tierData.output.priceBreakdown.length === 1 && (
                                <div className={cn('text-[10px]', isLight ? 'text-stone-500' : 'text-slate-500')}>
                                  {[...tierRegions].sort().map(r => getRegionDisplayName(r) || r).join(', ')}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Standalone Input (when no Output) */}
                      {hasInput && !hasOutput && (
                        <div className="mb-1">
                          <div className={cn(
                            'flex items-center justify-between px-2 py-1.5 rounded',
                            isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
                          )}>
                            <div className="flex items-center gap-2">
                              <span className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 bg-blue-500" />
                              <span className={cn('text-xs', isLight ? 'text-stone-700' : 'text-[#e4e5e7]')}>Input</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={cn('text-xs font-mono font-semibold tabular-nums', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                                {formatPriceRange(tierData.input)}
                                <span className={cn('font-normal ml-1', isLight ? 'text-stone-400' : 'text-slate-400')}>
                                  {tierData.input.unit}
                                </span>
                              </span>
                              <button
                                onClick={() => toggleSection(`${tierKey}_input`)}
                                className={cn(
                                  'text-[10px] px-1.5 py-0.5 rounded transition-colors',
                                  isLight
                                    ? 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                                    : 'bg-white/10 text-slate-300 hover:bg-white/20'
                                )}
                              >
                                {tierData.input.totalRegions} regions {expandedSections[`${tierKey}_input`] ? '▲' : '▼'}
                              </button>
                            </div>
                          </div>
                          {expandedSections[`${tierKey}_input`] && (
                            <div className={cn(
                              'ml-4 mt-1 px-2 py-1.5 rounded text-[10px]',
                              isLight ? 'bg-stone-50 text-stone-500' : 'bg-white/[0.02] text-slate-400'
                            )}>
                              {tierData.input.priceBreakdown.length > 1 ? (
                                tierData.input.priceBreakdown.map((pb, idx) => (
                                  <div key={idx} className="flex justify-between">
                                    <span className="font-mono">${formatPrice(pb.price)}</span>
                                    <span>{pb.regions.sort().map(r => getRegionDisplayName(r) || r).join(', ')}</span>
                                  </div>
                                ))
                              ) : (
                                tierData.input.allRegions.sort().map(r => getRegionDisplayName(r) || r).join(', ')
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Standalone Output (when no Input) */}
                      {hasOutput && !hasInput && (
                        <div className="mb-1">
                          <div className={cn(
                            'flex items-center justify-between px-2 py-1.5 rounded',
                            isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
                          )}>
                            <div className="flex items-center gap-2">
                              <span className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 bg-emerald-500" />
                              <span className={cn('text-xs', isLight ? 'text-stone-700' : 'text-[#e4e5e7]')}>Output</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={cn('text-xs font-mono font-semibold tabular-nums', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                                {formatPriceRange(tierData.output)}
                                <span className={cn('font-normal ml-1', isLight ? 'text-stone-400' : 'text-slate-400')}>
                                  {tierData.output.unit}
                                </span>
                              </span>
                              <button
                                onClick={() => toggleSection(`${tierKey}_output`)}
                                className={cn(
                                  'text-[10px] px-1.5 py-0.5 rounded transition-colors',
                                  isLight
                                    ? 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                                    : 'bg-white/10 text-slate-300 hover:bg-white/20'
                                )}
                              >
                                {tierData.output.totalRegions} regions {expandedSections[`${tierKey}_output`] ? '▲' : '▼'}
                              </button>
                            </div>
                          </div>
                          {expandedSections[`${tierKey}_output`] && (
                            <div className={cn(
                              'ml-4 mt-1 px-2 py-1.5 rounded text-[10px]',
                              isLight ? 'bg-stone-50 text-stone-500' : 'bg-white/[0.02] text-slate-400'
                            )}>
                              {tierData.output.priceBreakdown.length > 1 ? (
                                tierData.output.priceBreakdown.map((pb, idx) => (
                                  <div key={idx} className="flex justify-between">
                                    <span className="font-mono">${formatPrice(pb.price)}</span>
                                    <span>{pb.regions.sort().map(r => getRegionDisplayName(r) || r).join(', ')}</span>
                                  </div>
                                ))
                              ) : (
                                tierData.output.allRegions.sort().map(r => getRegionDisplayName(r) || r).join(', ')
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Other types (not input/output) */}
                      {hasOther && tierData.other.priceBreakdown.map((pb, idx) => {
                        const itemKey = `${tierKey}_other_${idx}`
                        return (
                          <div key={idx} className="mb-1 last:mb-0">
                            <div className={cn(
                              'flex items-center justify-between px-2 py-1.5 rounded',
                              isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
                            )}>
                              <div className="flex items-center gap-2">
                                <span className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 bg-[#6d6e72]" />
                                <span className={cn('text-xs', isLight ? 'text-stone-700' : 'text-[#e4e5e7]')}>
                                  {tier}
                                </span>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className={cn('text-xs font-mono font-semibold tabular-nums', isLight ? 'text-stone-900' : 'text-emerald-400')}>
                                  ${formatPrice(pb.price)}
                                  <span className={cn('font-normal ml-1', isLight ? 'text-stone-400' : 'text-slate-400')}>
                                    {pb.unit}
                                  </span>
                                </span>
                                {pb.regions.length > 1 ? (
                                  <button
                                    onClick={() => toggleSection(itemKey)}
                                    className={cn(
                                      'text-[10px] px-1.5 py-0.5 rounded transition-colors',
                                      isLight
                                        ? 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                                        : 'bg-white/10 text-slate-300 hover:bg-white/20'
                                    )}
                                  >
                                    {pb.regions.length} regions {expandedSections[itemKey] ? '▲' : '▼'}
                                  </button>
                                ) : (
                                  <span className={cn('text-[10px]', isLight ? 'text-stone-400' : 'text-slate-500')}>
                                    {pb.regions[0]}
                                  </span>
                                )}
                              </div>
                            </div>
                            {expandedSections[itemKey] && pb.regions.length > 1 && (
                              <div className={cn(
                                'ml-4 mt-1 px-2 py-1.5 rounded text-[10px]',
                                isLight ? 'bg-stone-50 text-stone-500' : 'bg-white/[0.02] text-slate-400'
                              )}>
                                {pb.regions.sort().map(r => getRegionDisplayName(r) || r).join(', ')}
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
      )
    }
    
    // Render CRIS section with Global/Regional sub-sections
    const renderCRISSection = (items) => {
      // Split items into Global and Regional
      const globalItems = items.filter(item => item._groupName === 'On-Demand Global')
      const regionalItems = items.filter(item => item._groupName !== 'On-Demand Global')
      
      // If only one type exists, render normally
      if (globalItems.length === 0 || regionalItems.length === 0) {
        return renderConsumptionSection('CRIS', items)
      }
      
      // Both exist - render as parent with sub-sections
      const sectionKey = 'byType_CRIS'
      const isExpanded = expandedSections[sectionKey] !== false
      const info = consumptionTypeInfo['CRIS']
      const Icon = info?.icon || Globe
      
      return (
        <div key="CRIS" className={cn(
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
              <Icon className={cn('h-4 w-4', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
              <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                {info?.label || 'Cross-Region (CRIS)'}
              </span>
              <Badge variant="secondary" className="text-[10px]">
                {new Set(items.map(i => i.region)).size} regions
              </Badge>
            </div>
            {isExpanded ? (
              <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            ) : (
              <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
            )}
          </button>
          {isExpanded && (
            <div className={cn('px-3 pb-3 border-t space-y-3', isLight ? 'border-stone-200' : 'border-white/10')}>
              {/* Global sub-section */}
              {globalItems.length > 0 && (
                <div className="pt-2">
                  {renderConsumptionSection('CRIS', globalItems, 'CRIS-Global')}
                </div>
              )}
              {/* Regional sub-section */}
              {regionalItems.length > 0 && (
                <div className="pt-2">
                  {renderConsumptionSection('CRIS', regionalItems, 'CRIS-Regional')}
                </div>
              )}
            </div>
          )}
        </div>
      )
    }

    // Split into left/right columns
    const leftTypes = ['In Region', 'CRIS', 'Long Context']
    const rightTypes = ['Batch', 'Provisioned', 'Custom']
    
    // Helper to render the appropriate section
    const renderSection = (type, items) => {
      if (type === 'CRIS') {
        return renderCRISSection(items)
      }
      return renderConsumptionSection(type, items)
    }

    return (
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="space-y-4">
          {leftTypes.map(type => byConsumptionType[type] && renderSection(type, byConsumptionType[type]))}
        </div>
        <div className="space-y-4">
          {rightTypes.map(type => byConsumptionType[type] && renderSection(type, byConsumptionType[type]))}
        </div>
      </div>
    )
  }
  // ============================================
  // VIEW 3: BY REGION (Simplified Geo-Based View)
  // ============================================
  const renderByRegionView = () => {
    const allItems = filterItems(getAllPricingItems())

    // Group by geo, then by region
    const byGeoRegion = {}
    allItems.forEach(item => {
      const geo = item.geo || 'Other'
      if (!byGeoRegion[geo]) byGeoRegion[geo] = {}
      if (!byGeoRegion[geo][item.region]) byGeoRegion[geo][item.region] = []
      byGeoRegion[geo][item.region].push(item)
    })

    return (
      <div className="space-y-4">
        {geoOrder.map(geo => {
          const geoData = byGeoRegion[geo]
          if (!geoData || Object.keys(geoData).length === 0) return null

          const geoKey = `byRegion_${geo}`
          const isGeoExpanded = expandedSections[geoKey] !== false // Default expanded for first geo

          return (
            <div key={geo} className={cn(
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
                onClick={() => toggleSection(geoKey)}
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm">{geoInfo[geo]?.icon}</span>
                  <span className={cn('font-medium text-sm', isLight ? 'text-stone-900' : 'text-white')}>
                    {geoInfo[geo]?.name}
                  </span>
                  <Badge variant="secondary" className="text-[10px]">
                    {Object.keys(geoData).length} regions
                  </Badge>
                </div>
                {isGeoExpanded ? (
                  <ChevronDown className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                ) : (
                  <ChevronRight className={cn('h-4 w-4', isLight ? 'text-stone-600' : 'text-slate-300')} />
                )}
              </button>
              {isGeoExpanded && (
                <div className={cn('px-3 pb-3 border-t space-y-2', isLight ? 'border-stone-200' : 'border-white/10')}>
                  {Object.entries(geoData).sort(([a], [b]) => a.localeCompare(b)).map(([region, regionItems]) => {
                    const regionKey = `${geoKey}_${region}`
                    const isRegionExpanded = expandedSections[regionKey]

                    // Sort items by type
                    const sortedItems = [...regionItems].sort((a, b) => {
                      const typeOrder = { input: 0, output: 1, other: 2 }
                      return (typeOrder[a._type] ?? 2) - (typeOrder[b._type] ?? 2)
                    })

                    // Quick summary: find standard input/output prices
                    const inputItem = sortedItems.find(i => i._type === 'input' && i._tier === 'Standard')
                    const outputItem = sortedItems.find(i => i._type === 'output' && i._tier === 'Standard')

                    return (
                      <div key={region} className={cn(
                        'rounded-lg overflow-hidden border mt-2',
                        isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border-white/[0.06]'
                      )}>
                        <button
                          className={cn(
                            'w-full flex items-center justify-between p-2 transition-colors',
                            isLight ? 'hover:bg-stone-50' : 'hover:bg-white/[0.04]'
                          )}
                          onClick={() => toggleSection(regionKey)}
                        >
                          <div className="flex items-center gap-2">
                            <Globe className={cn('h-3.5 w-3.5', isLight ? 'text-amber-600' : 'text-[#1A9E7A]')} />
                            <span className={cn('font-medium text-xs', isLight ? 'text-stone-800' : 'text-white')}>
                              {getRegionDisplayName(region)}
                            </span>
                            <span className={cn('text-[10px] font-mono', isLight ? 'text-stone-500' : 'text-slate-400')}>
                              ({region})
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            {inputItem && outputItem && (
                              <span className={cn('text-xs font-mono', isLight ? 'text-emerald-600' : 'text-emerald-400')}>
                                ${formatPrice(inputItem._price)} / ${formatPrice(outputItem._price)}
                              </span>
                            )}
                            {isRegionExpanded ? (
                              <ChevronDown className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
                            ) : (
                              <ChevronRight className={cn('h-3.5 w-3.5', isLight ? 'text-stone-500' : 'text-slate-400')} />
                            )}
                          </div>
                        </button>
                        {isRegionExpanded && (
                          <div className={cn('px-2 pb-2 border-t', isLight ? 'border-stone-100' : 'border-white/5')}>
                            <PricingItemsList items={sortedItems} isLight={isLight} />
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
    )
  }

  // ============================================
  // MAIN RENDER
  // ============================================
  return (
    <ScrollArea className="h-full">
      <div className="p-6">
        {/* Header with Search and View Toggle */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className={cn('absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4', isLight ? 'text-stone-400' : 'text-slate-500')} />
            <Input
              placeholder="Search by region, tier, or pricing type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
          <ViewToggle />
        </div>

        {/* View Content */}
        {pricingView === 'byType' && renderByTypeView()}
        {pricingView === 'byRegion' && renderByRegionView()}
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
  const fullPricing = pricingResult?.fullPricing
  const legacyPricing = model.pricing ?? model.model_pricing ?? model.comprehensive_pricing ?? {}
  const legacyByRegion = legacyPricing.by_region || {}
  let pricingRegions = []
  let pricingTypes = 0
  // Compute consumption options dynamically to ensure alignment with actual capabilities
  const consumptionOptions = (() => {
    const opts = [...(model.consumption_options || [])]
    // Ensure provisioned_throughput is shown if model supports it
    const provisionedData = model.availability?.provisioned ?? model.provisioned_throughput
    if (provisionedData?.supported && !opts.includes('provisioned_throughput') && !opts.includes('provisioned')) {
      opts.push('provisioned_throughput')
    }
    // Ensure mantle is shown if model supports it
    const mantleData = model.availability?.mantle
    if (mantleData?.supported && !opts.includes('mantle')) {
      opts.push('mantle')
    }
    // Ensure cross_region_inference is shown if model supports it
    const crisData = model.availability?.cross_region ?? model.cross_region_inference
    if (crisData?.supported && !opts.includes('cross_region_inference')) {
      opts.push('cross_region_inference')
    }
    // Ensure batch is shown if model supports it
    const batchData = model.availability?.batch ?? model.batch_inference_supported
    if (batchData?.supported && !opts.includes('batch')) {
      opts.push('batch')
    }
    return opts
  })()

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
                  {/* Availability */}
                  <div className="space-y-3">
                    <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                      Availability
                    </h3>
                    {(() => {
                      const crisRegions = model.availability?.cross_region?.regions ?? model.cross_region_inference?.source_regions ?? []
                      const batchRegions = model.availability?.batch?.regions ?? []
                      const provisionedRegions = model.availability?.provisioned?.regions ?? []
                      const allRegions = new Set([...regions, ...crisRegions, ...batchRegions, ...provisionedRegions, ...mantleRegions])
                      return (
                        <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                          <div className="flex items-center justify-between">
                            <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Regions</p>
                            <p className={cn('text-lg font-bold', isLight ? 'text-stone-900' : 'text-white')}>{allRegions.size}</p>
                          </div>
                        </div>
                      )
                    })()}
                    {mantleRegions.length > 0 && (
                      <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                        <div className="flex items-center justify-between">
                          <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Mantle Regions</p>
                          <p className={cn('text-lg font-bold', isLight ? 'text-violet-700' : 'text-violet-400')}>{mantleRegions.length}</p>
                        </div>
                      </div>
                    )}
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
                      <span className={cn(
                        'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
                        mantleSupported
                          ? isLight ? 'bg-violet-50 text-violet-700 border border-violet-200' : 'bg-violet-500/10 text-violet-400 border border-violet-500/20'
                          : isLight ? 'bg-stone-100 text-stone-400 border border-stone-200' : 'bg-white/[0.03] text-slate-500 border border-white/[0.06]'
                      )}>
                        <Cpu className="h-3.5 w-3.5" />
                        Mantle
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

              {activeTab === 'pricing' && (
                <div className="space-y-3">
                  <h3 className={cn('text-xs font-semibold uppercase tracking-wider', isLight ? 'text-stone-500' : 'text-slate-400')}>
                    Pricing Summary
                  </h3>
                  <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                    <p className={cn('text-xs', isLight ? 'text-stone-500' : 'text-slate-400')}>Regions</p>
                    <p className={cn('text-xl font-bold', isLight ? 'text-emerald-700' : 'text-emerald-400')}>{pricingRegions.length}</p>
                  </div>
                  {consumptionOptions.length > 0 && (
                    <div className={cn('rounded-lg p-3 border', isLight ? 'bg-white border-stone-200' : 'bg-white/[0.03] border-white/[0.06]')}>
                      <p className={cn('text-xs mb-2', isLight ? 'text-stone-500' : 'text-slate-400')}>Consumption</p>
                      <div className="flex flex-wrap gap-1.5">
                        {consumptionOptions.map(opt => {
                          const labels = { 'on_demand': 'In Region', 'batch': 'Batch', 'provisioned': 'Provisioned', 'cross_region_inference': 'Cross-Region', 'mantle': 'Mantle' }
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
                  <SpecsTab model={model} />
                </TabsContent>

                {showQuotas && (
                  <TabsContent value="quotas" className="flex-1 mt-0 min-h-0 overflow-hidden">
                    <QuotasTab model={model} />
                  </TabsContent>
                )}

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
