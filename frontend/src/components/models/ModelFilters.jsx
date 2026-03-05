import { useState, useRef, useEffect } from 'react'
import { ChevronDown, ChevronUp, Filter, X, Search, Check } from 'lucide-react'
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
  modelStatusOptions,
  contextFilterOptions,
  modalityOptions,
  initialFilterState,
  countActiveFilters,
  pricingFilterOptions,
} from '@/utils/filters'
import { cn } from '@/lib/utils'

// Active filter chip component
function ActiveFilterChip({ label, onRemove, isLight }) {
  return (
    <span className={cn(
      'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium',
      isLight
        ? 'bg-amber-700 text-white'
        : 'bg-[#1A9E7A]/15 text-[#1A9E7A]'
    )}>
      {label}
      <button
        onClick={onRemove}
        className={cn(
          'ml-0.5 rounded-full p-0.5 transition-colors',
          isLight ? 'hover:bg-amber-800' : 'hover:bg-[#1A9E7A]/25'
        )}
      >
        <X className="h-3 w-3" />
      </button>
    </span>
  )
}

// Multi-select dropdown component
function MultiSelectDropdown({ label, options, selected, onChange, placeholder, isLight }) {
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
      <p className={cn('text-[11px] mb-1 font-medium', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>{label}</p>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex items-center justify-between rounded-md border transition-colors w-full h-9 px-2.5 py-1 text-sm',
          isLight
            ? 'bg-transparent border-stone-300 text-stone-700 hover:border-stone-400'
            : 'bg-[#1a1b1e] border-[#373a40] text-[#e4e5e7] hover:border-[#4a4d54]'
        )}
      >
        <span className={cn(
          'truncate',
          selected.length === 0 && (isLight ? 'text-stone-400' : 'text-[#6d6e72]')
        )}>
          {displayValue}
        </span>
        <ChevronDown className={cn(
          'h-3.5 w-3.5 ml-1.5 transition-transform flex-shrink-0',
          isLight ? 'text-stone-400' : 'text-[#6d6e72]',
          isOpen && 'rotate-180'
        )} />
      </button>

      {isOpen && (
        <div className={cn(
          'absolute z-50 mt-1 min-w-[220px] w-full rounded-md border shadow-lg animate-slide-down',
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
                  className="h-7 pl-7 text-xs"
                />
              </div>
            </div>
          )}

          <div className="max-h-48 overflow-y-auto p-1">
            {filteredOptions.length === 0 ? (
              <p className={cn(
                'px-3 py-2 text-xs',
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
                    'flex w-full items-center gap-2 rounded px-2.5 py-1.5 text-xs transition-colors',
                    isLight
                      ? 'hover:bg-stone-100 text-stone-700'
                      : 'hover:bg-[#373a40] text-[#e4e5e7]'
                  )}
                >
                  <div className={cn(
                    'flex h-3.5 w-3.5 items-center justify-center rounded border transition-colors flex-shrink-0',
                    selected.includes(option)
                      ? isLight
                        ? 'bg-amber-700 border-amber-700'
                        : 'bg-[#1A9E7A] border-[#1A9E7A]'
                      : isLight
                        ? 'border-stone-300'
                        : 'border-[#4a4d54]'
                  )}>
                    {selected.includes(option) && (
                      <Check className="h-2.5 w-2.5 text-white" />
                    )}
                  </div>
                  <span className="truncate">{option}</span>
                </button>
              ))
            )}
          </div>

          {selected.length > 0 && (
            <div className={cn(
              'p-1.5 border-t',
              isLight ? 'border-stone-200' : 'border-[#373a40]'
            )}>
              <button
                type="button"
                onClick={() => onChange([])}
                className={cn(
                  'w-full text-[10px] py-1 rounded transition-colors',
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

function ToggleGroup({ label, options, value, onChange, isLight }) {
  return (
    <div>
      <p className={cn('text-[11px] mb-1.5 font-medium', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>{label}</p>
      <div className={cn(
        'inline-flex rounded-md border overflow-hidden h-8',
        isLight ? 'border-stone-300' : 'border-[#373a40]'
      )}>
        {options.map((opt, i) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={cn(
              'px-2.5 py-1 text-xs font-medium transition-colors',
              i > 0 && (isLight ? 'border-l border-stone-300' : 'border-l border-[#373a40]'),
              value === opt.value
                ? isLight
                  ? 'bg-amber-600 text-white'
                  : 'bg-[#1A9E7A] text-white'
                : isLight
                  ? 'bg-transparent text-stone-500 hover:bg-stone-50'
                  : 'bg-[#1a1b1e] text-[#9a9b9f] hover:bg-[#2c2d32]'
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function FilterSelect({ label, value, onChange, options, isLight }) {
  const selectedOption = options.find(opt => opt.value === value)
  const displayText = selectedOption?.label || value

  return (
    <div>
      <p className={cn('text-[11px] mb-1 font-medium', isLight ? 'text-stone-500' : 'text-[#6d6e72]')}>{label}</p>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-9 text-sm">
          <SelectValue>{displayText}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {options.map(opt => (
            <SelectItem key={opt.value} value={opt.value} className="text-xs">
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}


const crisToggleOptions = [
  { value: 'All Models', label: 'All' },
  { value: 'GLOBAL', label: 'Global' },
  { value: 'US', label: 'US' },
  { value: 'EU', label: 'EU' },
  { value: 'APAC', label: 'APAC' },
  { value: 'JP', label: 'JP' },
  { value: 'AU', label: 'AU' },
  { value: 'CA', label: 'CA' },
  { value: 'SA', label: 'SA' },
  { value: 'ME', label: 'ME' },
  { value: 'AF', label: 'AF' },
  { value: 'CRIS Not Supported', label: 'No' },
]

const streamingToggleOptions = [
  { value: 'All Models', label: 'All' },
  { value: 'Streaming Supported', label: 'Yes' },
  { value: 'Streaming Not Supported', label: 'No' },
]

const mantleToggleOptions = [
  { value: 'All Models', label: 'All' },
  { value: 'Mantle Supported', label: 'Yes' },
  { value: 'Mantle Only', label: 'Only' },
  { value: 'No Mantle', label: 'No' },
]

const pricingToggleOptions = [
  { value: 'All Models', label: 'All' },
  { value: 'Has Pricing', label: 'Yes' },
  { value: 'No Pricing', label: 'No' },
]

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

    if (filters.useCases.length > 0) {
      filters.useCases.forEach(uc => {
        chips.push({
          key: `uc-${uc}`,
          label: uc,
          onRemove: () => updateFilter('useCases', filters.useCases.filter(x => x !== uc))
        })
      })
    }

    if (filters.crisSupport !== 'All Models') {
      const crisLabel = filters.crisSupport === 'CRIS Not Supported' ? 'No' : filters.crisSupport
      chips.push({
        key: 'cris',
        label: `CRIS: ${crisLabel}`,
        onRemove: () => updateFilter('crisSupport', 'All Models')
      })
    }

    if (filters.streamingSupport !== 'All Models') {
      chips.push({
        key: 'streaming',
        label: `Streaming: ${filters.streamingSupport === 'Streaming Supported' ? 'Yes' : 'No'}`,
        onRemove: () => updateFilter('streamingSupport', 'All Models')
      })
    }

    if (filters.mantleSupport !== 'All Models') {
      const mantleLabel = filters.mantleSupport === 'Mantle Supported' ? 'Yes' 
        : filters.mantleSupport === 'Mantle Only' ? 'Only' 
        : 'No'
      chips.push({
        key: 'mantle',
        label: `Mantle: ${mantleLabel}`,
        onRemove: () => updateFilter('mantleSupport', 'All Models')
      })
    }

    if (filters.pricingFilter !== 'All Models') {
      chips.push({
        key: 'pricingFilter',
        label: `Pricing: ${filters.pricingFilter === 'Has Pricing' ? 'Yes' : 'No'}`,
        onRemove: () => updateFilter('pricingFilter', 'All Models')
      })
    }

    if (filters.consumptionOptions.length > 0) {
      filters.consumptionOptions.forEach(co => {
        chips.push({
          key: `cons-${co}`,
          label: co,
          onRemove: () => updateFilter('consumptionOptions', filters.consumptionOptions.filter(x => x !== co))
        })
      })
    }

    if (filters.customizations.length > 0) {
      filters.customizations.forEach(cu => {
        chips.push({
          key: `cust-${cu}`,
          label: cu,
          onRemove: () => updateFilter('customizations', filters.customizations.filter(x => x !== cu))
        })
      })
    }

    if (filters.languages.length > 0) {
      chips.push({
        key: 'languages',
        label: `${filters.languages.length} language${filters.languages.length > 1 ? 's' : ''}`,
        onRemove: () => updateFilter('languages', [])
      })
    }

    return chips
  }

  const activeChips = getActiveFilterChips()

  return (
    <div className="space-y-2">
      {/* Main filter row: Search + Region + More */}
      <div className="flex items-center gap-2">
        {/* Search bar */}
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
          Filters
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
        <div className="flex flex-wrap items-center gap-1.5">
          <span className={cn('text-[11px]', isLight ? 'text-stone-400' : 'text-[#6d6e72]')}>Active:</span>
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
              'text-[11px] font-medium ml-1 transition-colors',
              isLight
                ? 'text-amber-700 hover:text-amber-800'
                : 'text-[#1A9E7A] hover:text-[#22b38d]'
            )}
          >
            Clear all
          </button>
        </div>
      )}

      {/* Advanced filters panel */}
      {showAdvanced && (
        <div className={cn(
          'border rounded-lg p-3 animate-slide-down',
          isLight
            ? 'bg-stone-50/50 border-stone-200'
            : 'bg-[#1a1b1e]/50 border-[#373a40]'
        )}>
          {/* Row 1: Primary filters */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-3 gap-y-2.5">
            {availableProviders.length > 0 && (
              <MultiSelectDropdown
                label="Provider"
                options={availableProviders}
                selected={filters.providers}
                onChange={(v) => updateFilter('providers', v)}
                placeholder="All providers"
                isLight={isLight}
              />
            )}

            <FilterSelect
              label="Modality"
              value={filters.modality}
              onChange={(v) => updateFilter('modality', v)}
              options={modalityOptions}
              isLight={isLight}
            />

            <FilterSelect
              label="Status"
              value={filters.modelStatus}
              onChange={(v) => updateFilter('modelStatus', v)}
              options={modelStatusOptions}
              isLight={isLight}
            />

            <FilterSelect
              label="Context Window"
              value={filters.contextFilter}
              onChange={(v) => updateFilter('contextFilter', v)}
              options={contextFilterOptions}
              isLight={isLight}
            />
          </div>

          {/* Divider */}
          <div className={cn('my-2.5 border-t', isLight ? 'border-stone-200/60' : 'border-[#2c2d32]/60')} />

          {/* Row 2: Content & feature filters */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-3 gap-y-2.5">
            {availableUseCases.length > 0 && (
              <MultiSelectDropdown
                label="Use Cases"
                options={availableUseCases}
                selected={filters.useCases}
                onChange={(v) => updateFilter('useCases', v)}
                placeholder="All use cases"
                isLight={isLight}
              />
            )}

            {availableCapabilities.length > 0 && (
              <MultiSelectDropdown
                label="Capabilities"
                options={availableCapabilities}
                selected={filters.capabilities}
                onChange={(v) => updateFilter('capabilities', v)}
                placeholder="All capabilities"
                isLight={isLight}
              />
            )}

            {availableLanguages.length > 0 && (
              <MultiSelectDropdown
                label="Languages"
                options={availableLanguages}
                selected={filters.languages}
                onChange={(v) => updateFilter('languages', v)}
                placeholder="All languages"
                isLight={isLight}
              />
            )}

            {availableCustomizations.length > 0 && (
              <MultiSelectDropdown
                label="Customization"
                options={availableCustomizations}
                selected={filters.customizations}
                onChange={(v) => updateFilter('customizations', v)}
                placeholder="All types"
                isLight={isLight}
              />
            )}
          </div>

          {/* Divider */}
          <div className={cn('my-2.5 border-t', isLight ? 'border-stone-200/60' : 'border-[#2c2d32]/60')} />

          {/* Row 3: Infrastructure & availability filters */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-3 gap-y-2.5 items-end">
            {availableConsumptionOptions.length > 0 && (
              <MultiSelectDropdown
                label="Consumption"
                options={availableConsumptionOptions}
                selected={filters.consumptionOptions}
                onChange={(v) => updateFilter('consumptionOptions', v)}
                placeholder="All options"
                isLight={isLight}
              />
            )}

            <FilterSelect
              label="Cross-Region (CRIS)"
              value={filters.crisSupport}
              onChange={(v) => updateFilter('crisSupport', v)}
              options={crisToggleOptions}
              isLight={isLight}
            />

            <ToggleGroup
              label="Streaming"
              options={streamingToggleOptions}
              value={filters.streamingSupport}
              onChange={(v) => updateFilter('streamingSupport', v)}
              isLight={isLight}
            />

            <ToggleGroup
              label="Mantle"
              options={mantleToggleOptions}
              value={filters.mantleSupport}
              onChange={(v) => updateFilter('mantleSupport', v)}
              isLight={isLight}
            />

            <ToggleGroup
              label="Pricing"
              options={pricingToggleOptions}
              value={filters.pricingFilter}
              onChange={(v) => updateFilter('pricingFilter', v)}
              isLight={isLight}
            />
          </div>
        </div>
      )}
    </div>
  )
}
