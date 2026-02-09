import { useState, useRef, useEffect } from 'react'
import { ChevronDown, ChevronUp, Filter, X, Search, Check, Building2, Zap, Target, Cpu, MessageSquare, Image, FileText, Video, Mic } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { RegionSelector } from './RegionSelector'
import { useTheme } from '@/components/layout/ThemeProvider'
import {
  geoRegionOptions,
  modelStatusOptions,
  crisSupportOptions,
  streamingSupportOptions,
  contextFilterOptions,
  modalityOptions,
  initialFilterState,
  countActiveFilters,
} from '@/utils/filters'
import { cn } from '@/lib/utils'

// Quick filter pill component
function QuickFilterPill({ label, isActive, onClick, icon: Icon, isLight }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-150',
        isActive
          ? isLight
            ? 'bg-amber-600 text-white shadow-sm'
            : 'bg-[#1A9E7A] text-white shadow-sm'
          : isLight
            ? 'bg-stone-100 text-stone-600 hover:bg-stone-200'
            : 'bg-[#2c2d32] text-[#a0a1a5] hover:bg-[#373a40]'
      )}
    >
      {Icon && <Icon className="h-3.5 w-3.5" />}
      {label}
    </button>
  )
}

// Active filter chip component
function ActiveFilterChip({ label, onRemove, isLight }) {
  return (
    <span className={cn(
      'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium',
      isLight
        ? 'bg-amber-100 text-amber-800'
        : 'bg-[#1A9E7A]/15 text-[#1A9E7A]'
    )}>
      {label}
      <button
        onClick={onRemove}
        className={cn(
          'ml-0.5 rounded-full p-0.5 transition-colors',
          isLight ? 'hover:bg-amber-200' : 'hover:bg-[#1A9E7A]/25'
        )}
      >
        <X className="h-3 w-3" />
      </button>
    </span>
  )
}

// Multi-select dropdown component
function MultiSelectDropdown({ label, options, selected, onChange, placeholder, isLight, compact = false }) {
  const [isOpen, setIsOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const dropdownRef = useRef(null)

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const toggleOption = (option) => {
    if (selected.includes(option)) {
      onChange(selected.filter(o => o !== option))
    } else {
      onChange([...selected, option])
    }
  }

  const filteredOptions = options.filter(opt =>
    opt.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const displayValue = selected.length === 0
    ? placeholder
    : selected.length === 1
      ? selected[0]
      : `${selected.length} selected`

  return (
    <div className="relative" ref={dropdownRef}>
      {!compact && <p className={cn('text-xs mb-1.5 font-medium', isLight ? 'text-stone-600' : 'text-[#a0a1a5]')}>{label}</p>}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex items-center justify-between rounded-md border transition-colors',
          compact
            ? 'h-8 w-auto min-w-[100px] px-2 py-1 text-xs'
            : 'h-9 w-full px-3 py-2 text-sm',
          isLight
            ? 'bg-white border-stone-300 text-stone-900 hover:border-stone-400 focus:outline-none focus:ring-2 focus:ring-amber-600 focus:ring-offset-2'
            : 'bg-[#1a1b1e] border-[#373a40] text-[#e4e5e7] hover:border-[#4a4d54] focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:ring-offset-2 focus:ring-offset-[#1a1b1e]'
        )}
      >
        {compact && <span className={cn('mr-1', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>{label}:</span>}
        <span className={cn(
          'truncate',
          selected.length === 0 && (isLight ? 'text-stone-500' : 'text-[#6d6e72]')
        )}>
          {displayValue}
        </span>
        <ChevronDown className={cn(
          'h-4 w-4 ml-1 transition-transform flex-shrink-0',
          isLight ? 'text-stone-500' : 'text-[#6d6e72]',
          isOpen && 'rotate-180'
        )} />
      </button>

      {isOpen && (
        <div className={cn(
          'absolute z-50 mt-1 w-full rounded-md border shadow-lg animate-slide-down',
          isLight
            ? 'bg-white border-stone-200 shadow-stone-900/10'
            : 'bg-[#25262b] border-[#373a40] shadow-black/20'
        )}>
          {options.length > 5 && (
            <div className={cn(
              'p-2 border-b',
              isLight ? 'border-stone-200' : 'border-[#373a40]'
            )}>
              <div className="relative">
                <Search className={cn(
                  'absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5',
                  isLight ? 'text-stone-500' : 'text-[#6d6e72]'
                )} />
                <Input
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-8 pl-7 text-sm"
                />
              </div>
            </div>
          )}

          <div className="max-h-48 overflow-y-auto p-1">
            {filteredOptions.length === 0 ? (
              <p className={cn(
                'px-3 py-2 text-sm',
                isLight ? 'text-stone-600' : 'text-[#6d6e72]'
              )}>
                No options found
              </p>
            ) : (
              filteredOptions.map(option => (
                <button
                  key={option}
                  type="button"
                  onClick={() => toggleOption(option)}
                  className={cn(
                    'flex w-full items-center gap-2 rounded px-3 py-1.5 text-sm transition-colors',
                    isLight
                      ? 'hover:bg-stone-100 text-stone-700'
                      : 'hover:bg-[#373a40] text-[#e4e5e7]'
                  )}
                >
                  <div className={cn(
                    'flex h-4 w-4 items-center justify-center rounded border transition-colors',
                    selected.includes(option)
                      ? isLight
                        ? 'bg-amber-700 border-amber-700'
                        : 'bg-[#1A9E7A] border-[#1A9E7A]'
                      : isLight
                        ? 'border-stone-300'
                        : 'border-[#4a4d54]'
                  )}>
                    {selected.includes(option) && (
                      <Check className="h-3 w-3 text-white" />
                    )}
                  </div>
                  <span className="truncate">{option}</span>
                </button>
              ))
            )}
          </div>

          {selected.length > 0 && (
            <div className={cn(
              'p-2 border-t',
              isLight ? 'border-stone-200' : 'border-[#373a40]'
            )}>
              <button
                type="button"
                onClick={() => onChange([])}
                className={cn(
                  'w-full text-xs py-1 rounded transition-colors',
                  isLight
                    ? 'text-stone-600 hover:bg-stone-100'
                    : 'text-[#a0a1a5] hover:bg-[#373a40]'
                )}
              >
                Clear all ({selected.length})
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function FilterSelect({ label, value, onChange, options, className, isLight, compact = false }) {
  const selectedOption = options.find(opt => opt.value === value)
  const displayText = selectedOption?.label || value

  if (compact) {
    return (
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className={cn('h-8 text-xs w-auto min-w-[100px]', className)}>
          <span className={cn('mr-1', isLight ? 'text-stone-500' : 'text-[#9a9b9f]')}>{label}:</span>
          <SelectValue>{displayText}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {options.map(opt => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  return (
    <div className={className}>
      <p className={cn('text-xs mb-1.5 font-medium', isLight ? 'text-stone-600' : 'text-[#a0a1a5]')}>{label}</p>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-9">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map(opt => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

// Filter group component
function FilterGroup({ title, icon: Icon, children, isLight }) {
  return (
    <div className={cn(
      'rounded-lg border p-4',
      isLight
        ? 'bg-white border-stone-200'
        : 'bg-[#25262b] border-[#373a40]'
    )}>
      <div className="flex items-center gap-2 mb-3">
        <Icon className={cn(
          'h-4 w-4',
          isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
        )} />
        <h3 className={cn(
          'text-sm font-semibold',
          isLight ? 'text-stone-900' : 'text-[#e4e5e7]'
        )}>
          {title}
        </h3>
      </div>
      <div className="space-y-3">
        {children}
      </div>
    </div>
  )
}

// Modality icons mapping
const modalityIconMap = {
  TEXT: MessageSquare,
  IMAGE: Image,
  DOCUMENT: FileText,
  VIDEO: Video,
  AUDIO: Mic,
  SPEECH: Mic,
}

// Context size labels
const contextSizeLabels = {
  'Small': 'Small (<32K)',
  'Medium': 'Med (32-128K)',
  'Large': 'Large (128K-500K)',
  'XL': 'XL (>500K)',
}

export function ModelFilters({
  filters,
  onFiltersChange,
  availableProviders = [],
  availableCapabilities = [],
  availableUseCases = [],
  availableCustomizations = [],
  availableLanguages = [],
  availableConsumptionOptions = [],
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const activeCount = countActiveFilters(filters)

  const updateFilter = (key, value) => {
    onFiltersChange({ ...filters, [key]: value })
  }

  const resetFilters = () => {
    onFiltersChange(initialFilterState)
  }

  // Generate active filter chips
  const getActiveFilterChips = () => {
    const chips = []

    if (filters.providers.length > 0) {
      filters.providers.forEach(p => {
        chips.push({
          key: `provider-${p}`,
          label: p,
          onRemove: () => updateFilter('providers', filters.providers.filter(x => x !== p))
        })
      })
    }

    if (filters.modality !== 'All Modalities') {
      chips.push({
        key: 'modality',
        label: filters.modality,
        onRemove: () => updateFilter('modality', 'All Modalities')
      })
    }

    if (filters.modelStatus !== 'All Status') {
      chips.push({
        key: 'status',
        label: filters.modelStatus,
        onRemove: () => updateFilter('modelStatus', 'All Status')
      })
    }

    if (filters.contextFilter !== 'All Models') {
      chips.push({
        key: 'context',
        label: filters.contextFilter,
        onRemove: () => updateFilter('contextFilter', 'All Models')
      })
    }

    if (filters.capabilities.length > 0) {
      filters.capabilities.forEach(c => {
        chips.push({
          key: `cap-${c}`,
          label: c,
          onRemove: () => updateFilter('capabilities', filters.capabilities.filter(x => x !== c))
        })
      })
    }

    if (filters.crisSupport !== 'All Models') {
      chips.push({
        key: 'cris',
        label: `CRIS: ${filters.crisSupport}`,
        onRemove: () => updateFilter('crisSupport', 'All Models')
      })
    }

    if (filters.streamingSupport !== 'All Models') {
      chips.push({
        key: 'streaming',
        label: `Streaming: ${filters.streamingSupport}`,
        onRemove: () => updateFilter('streamingSupport', 'All Models')
      })
    }

    return chips
  }

  const activeChips = getActiveFilterChips()

  // Quick filter toggles
  const toggleModality = (modality) => {
    updateFilter('modality', filters.modality === modality ? 'All Modalities' : modality)
  }

  const toggleStatus = (status) => {
    updateFilter('modelStatus', filters.modelStatus === status ? 'All Status' : status)
  }

  const toggleContextSize = (size) => {
    updateFilter('contextFilter', filters.contextFilter === size ? 'All Models' : size)
  }

  return (
    <div className="space-y-3">
      {/* Main filter row: Search + Region + More - all together */}
      <div className="flex items-center gap-2">
        {/* Search bar - grows to fill available space */}
        <div className="relative flex-1">
          <Search className={cn(
            'absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4',
            isLight ? 'text-stone-400' : 'text-[#6d6e72]'
          )} />
          <Input
            placeholder="Search models..."
            value={filters.searchQuery}
            onChange={(e) => updateFilter('searchQuery', e.target.value)}
            className={cn(
              'h-9 pl-9 pr-8 text-sm',
              isLight
                ? 'bg-white border-stone-200 focus:border-amber-500'
                : 'bg-[#25262b] border-[#373a40] focus:border-[#1A9E7A]'
            )}
          />
          {filters.searchQuery && (
            <button
              onClick={() => updateFilter('searchQuery', '')}
              className={cn(
                'absolute right-2 top-1/2 -translate-y-1/2 p-0.5 rounded-full transition-colors',
                isLight ? 'hover:bg-stone-100' : 'hover:bg-[#373a40]'
              )}
            >
              <X className={cn('h-3.5 w-3.5', isLight ? 'text-stone-400' : 'text-[#6d6e72]')} />
            </button>
          )}
        </div>

        {/* Region selector */}
        <RegionSelector
          value={filters.primaryRegion}
          onChange={(v) => updateFilter('primaryRegion', v)}
          className="h-9 w-[200px] flex-shrink-0"
        />

        {/* More filters toggle */}
        <Button
          variant="outline"
          size="sm"
          className={cn(
            'h-9 flex-shrink-0',
            showAdvanced && (isLight ? 'bg-stone-100' : 'bg-[#2c2d32]')
          )}
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <Filter className="h-4 w-4 mr-1.5" />
          More
          {activeCount > 0 && (
            <Badge className={cn(
              'ml-1.5 text-[10px] px-1.5 border-0',
              isLight ? 'bg-amber-600 text-white' : 'bg-[#1A9E7A] text-white'
            )}>
              {activeCount}
            </Badge>
          )}
          {showAdvanced ? (
            <ChevronUp className="h-4 w-4 ml-1" />
          ) : (
            <ChevronDown className="h-4 w-4 ml-1" />
          )}
        </Button>
      </div>

      {/* Active filter chips */}
      {activeChips.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className={cn('text-xs', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>Filters:</span>
          {activeChips.map(chip => (
            <ActiveFilterChip
              key={chip.key}
              label={chip.label}
              onRemove={chip.onRemove}
              isLight={isLight}
            />
          ))}
          <button
            onClick={resetFilters}
            className={cn(
              'text-xs font-medium transition-colors',
              isLight
                ? 'text-amber-700 hover:text-amber-800'
                : 'text-[#1A9E7A] hover:text-[#22b38d]'
            )}
          >
            Clear all
          </button>
        </div>
      )}

      {/* Advanced filters - compact row */}
      {showAdvanced && (
        <div className={cn(
          'border rounded-lg p-2 animate-slide-down',
          isLight
            ? 'bg-stone-50 border-stone-200'
            : 'bg-[#1a1b1e] border-[#373a40]'
        )}>
          <div className="flex flex-wrap gap-2">
            {/* Provider */}
            {availableProviders.length > 0 && (
              <MultiSelectDropdown
                label="Provider"
                options={availableProviders}
                selected={filters.providers}
                onChange={(v) => updateFilter('providers', v)}
                placeholder="All"
                isLight={isLight}
                compact
              />
            )}

            {/* Modality */}
            <FilterSelect
              label="Modality"
              value={filters.modality}
              onChange={(v) => updateFilter('modality', v)}
              options={modalityOptions}
              isLight={isLight}
              compact
            />

            {/* Status */}
            <FilterSelect
              label="Status"
              value={filters.modelStatus}
              onChange={(v) => updateFilter('modelStatus', v)}
              options={modelStatusOptions}
              isLight={isLight}
              compact
            />

            {/* Context Size */}
            <FilterSelect
              label="Context"
              value={filters.contextFilter}
              onChange={(v) => updateFilter('contextFilter', v)}
              options={contextFilterOptions}
              isLight={isLight}
              compact
            />

            {/* CRIS */}
            <FilterSelect
              label="CRIS"
              value={filters.crisSupport}
              onChange={(v) => updateFilter('crisSupport', v)}
              options={crisSupportOptions}
              isLight={isLight}
              compact
            />

            {/* Streaming */}
            <FilterSelect
              label="Stream"
              value={filters.streamingSupport}
              onChange={(v) => updateFilter('streamingSupport', v)}
              options={streamingSupportOptions}
              isLight={isLight}
              compact
            />

            {/* Consumption Options */}
            {availableConsumptionOptions.length > 0 && (
              <MultiSelectDropdown
                label="Consumption"
                options={availableConsumptionOptions}
                selected={filters.consumptionOptions}
                onChange={(v) => updateFilter('consumptionOptions', v)}
                placeholder="All"
                isLight={isLight}
                compact
              />
            )}

            {/* Capabilities */}
            {availableCapabilities.length > 0 && (
              <MultiSelectDropdown
                label="Capabilities"
                options={availableCapabilities}
                selected={filters.capabilities}
                onChange={(v) => updateFilter('capabilities', v)}
                placeholder="All"
                isLight={isLight}
                compact
              />
            )}

            {/* Use Cases */}
            {availableUseCases.length > 0 && (
              <MultiSelectDropdown
                label="Use Cases"
                options={availableUseCases}
                selected={filters.useCases}
                onChange={(v) => updateFilter('useCases', v)}
                placeholder="All"
                isLight={isLight}
                compact
              />
            )}

            {/* Customizations */}
            {availableCustomizations.length > 0 && (
              <MultiSelectDropdown
                label="Custom"
                options={availableCustomizations}
                selected={filters.customizations}
                onChange={(v) => updateFilter('customizations', v)}
                placeholder="All"
                isLight={isLight}
                compact
              />
            )}

            {/* Languages */}
            {availableLanguages.length > 0 && (
              <MultiSelectDropdown
                label="Lang"
                options={availableLanguages}
                selected={filters.languages}
                onChange={(v) => updateFilter('languages', v)}
                placeholder="All"
                isLight={isLight}
                compact
              />
            )}
          </div>
        </div>
      )}
    </div>
  )
}
