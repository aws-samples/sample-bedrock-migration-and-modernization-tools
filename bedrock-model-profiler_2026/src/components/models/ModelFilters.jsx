import { useState, useRef, useEffect } from 'react'
import { ChevronDown, ChevronUp, Filter, X, Search, Check, Building2, Zap, Target, Cpu } from 'lucide-react'
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
      <p className={cn('text-xs mb-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>{label}</p>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex h-9 w-full items-center justify-between rounded-md border px-3 py-2 text-sm',
          isLight
            ? 'bg-white border-stone-300 text-stone-900 focus:outline-none focus:ring-2 focus:ring-amber-600 focus:ring-offset-2'
            : 'bg-slate-950 border-slate-700 text-slate-100 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:ring-offset-2'
        )}
      >
        <span className={cn(
          'truncate',
          selected.length === 0 && (isLight ? 'text-stone-500' : 'text-slate-500')
        )}>
          {displayValue}
        </span>
        <ChevronDown className={cn(
          'h-4 w-4 transition-transform',
          isLight ? 'text-stone-500' : 'text-slate-400',
          isOpen && 'rotate-180'
        )} />
      </button>

      {isOpen && (
        <div className={cn(
          'absolute z-50 mt-1 w-full rounded-md border shadow-lg',
          isLight
            ? 'bg-white/95 border-stone-200/80 backdrop-blur-xl shadow-stone-900/10'
            : 'bg-slate-900/95 border-slate-700/50 backdrop-blur-xl'
        )}>
          {options.length > 5 && (
            <div className={cn(
              'p-2 border-b',
              isLight ? 'border-stone-200' : 'border-slate-700'
            )}>
              <div className="relative">
                <Search className={cn(
                  'absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5',
                  isLight ? 'text-stone-500' : 'text-slate-400'
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
                isLight ? 'text-stone-600' : 'text-slate-400'
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
                      : 'hover:bg-slate-800 text-slate-200'
                  )}
                >
                  <div className={cn(
                    'flex h-4 w-4 items-center justify-center rounded border',
                    selected.includes(option)
                      ? isLight
                        ? 'bg-amber-700 border-amber-700'
                        : 'bg-[#1A9E7A] border-[#1A9E7A]'
                      : isLight
                        ? 'border-stone-300'
                        : 'border-slate-600'
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
              isLight ? 'border-stone-200' : 'border-slate-700'
            )}>
              <button
                type="button"
                onClick={() => onChange([])}
                className={cn(
                  'w-full text-xs py-1 rounded transition-colors',
                  isLight
                    ? 'text-stone-600 hover:bg-stone-100'
                    : 'text-slate-400 hover:bg-slate-800'
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

function FilterSelect({ label, value, onChange, options, className, isLight }) {
  return (
    <div className={className}>
      <p className={cn('text-xs mb-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>{label}</p>
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
        ? 'bg-white/60 border-stone-200/80 backdrop-blur-sm'
        : 'bg-white/5 border-white/10 backdrop-blur-sm'
    )}>
      <div className="flex items-center gap-2 mb-3">
        <Icon className={cn(
          'h-4 w-4',
          isLight ? 'text-amber-600' : 'text-[#1A9E7A]'
        )} />
        <h3 className={cn(
          'text-sm font-semibold',
          isLight ? 'text-stone-900' : 'text-white'
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

  // Format consumption option labels
  const formatConsumptionOption = (opt) => {
    const labels = {
      'on_demand': 'On-Demand',
      'provisioned': 'Provisioned',
      'batch': 'Batch',
      'cross_region_inference': 'Cross-Region'
    }
    return labels[opt] || opt
  }

  return (
    <div className="space-y-4">
      {/* Primary filters row */}
      <div className="flex flex-col sm:flex-row flex-wrap gap-3 items-stretch sm:items-end">
        {/* Search */}
        <div className="w-full sm:flex-1 sm:min-w-[200px]">
          <p className={cn('text-xs mb-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>Search</p>
          <div className="relative">
            <Search className={cn(
              'absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4',
              isLight ? 'text-stone-500' : 'text-slate-400'
            )} />
            <Input
              placeholder="Search models..."
              value={filters.searchQuery}
              onChange={(e) => updateFilter('searchQuery', e.target.value)}
              className="pl-9 h-9"
            />
          </div>
        </div>

        {/* Region */}
        <div className="w-full sm:w-[220px]">
          <p className={cn('text-xs mb-1.5', isLight ? 'text-stone-600' : 'text-slate-400')}>Primary Region</p>
          <RegionSelector
            value={filters.primaryRegion}
            onChange={(v) => updateFilter('primaryRegion', v)}
            className="h-9"
          />
        </div>

        {/* Status */}
        <FilterSelect
          label="Status"
          value={filters.modelStatus}
          onChange={(v) => updateFilter('modelStatus', v)}
          options={modelStatusOptions}
          className="w-full sm:w-[140px]"
          isLight={isLight}
        />

        {/* Buttons row */}
        <div className="flex gap-2 w-full sm:w-auto">
          {/* Advanced toggle */}
          <Button
            variant="outline"
            size="sm"
            className={cn(
              'h-9 flex-1 sm:flex-initial',
              showAdvanced && (isLight ? 'bg-stone-100' : 'bg-slate-800')
            )}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <Filter className="h-4 w-4 mr-2" />
            <span className="sm:inline">Advanced</span>
            {activeCount > 0 && (
              <Badge className={cn(
                'ml-2 text-xs px-1.5 border-0',
                isLight ? 'bg-amber-700 !text-[#faf9f5]' : 'bg-[#1A9E7A] !text-white'
              )}>
                {activeCount}
              </Badge>
            )}
            {showAdvanced ? (
              <ChevronUp className="h-4 w-4 ml-2" />
            ) : (
              <ChevronDown className="h-4 w-4 ml-2" />
            )}
          </Button>

          {/* Reset */}
          {activeCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-9"
              onClick={resetFilters}
            >
              <X className="h-4 w-4 mr-1" />
              <span className="hidden sm:inline">Clear</span>
            </Button>
          )}
        </div>
      </div>

      {/* Advanced filters - 4 groups */}
      {showAdvanced && (
        <div className={cn(
          'border rounded-lg p-4 animate-slide-down',
          isLight
            ? 'bg-stone-50/80 border-stone-200/80 backdrop-blur-xl'
            : 'bg-slate-900/30 border-slate-700/50 backdrop-blur-xl'
        )}>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Group 1: Provider & Location */}
            <FilterGroup title="Provider & Location" icon={Building2} isLight={isLight}>
              {availableProviders.length > 0 && (
                <MultiSelectDropdown
                  label="Providers"
                  options={availableProviders}
                  selected={filters.providers}
                  onChange={(v) => updateFilter('providers', v)}
                  placeholder="Choose options"
                  isLight={isLight}
                />
              )}
              <FilterSelect
                label="Geographic Regions"
                value={filters.geoRegion}
                onChange={(v) => updateFilter('geoRegion', v)}
                options={geoRegionOptions}
                isLight={isLight}
              />
              <FilterSelect
                label="Model Status"
                value={filters.modelStatus}
                onChange={(v) => updateFilter('modelStatus', v)}
                options={modelStatusOptions}
                isLight={isLight}
              />
            </FilterGroup>

            {/* Group 2: Consumption & Features */}
            <FilterGroup title="Consumption & Features" icon={Zap} isLight={isLight}>
              {availableConsumptionOptions.length > 0 && (
                <MultiSelectDropdown
                  label="Consumption Options"
                  options={availableConsumptionOptions}
                  selected={filters.consumptionOptions}
                  onChange={(v) => updateFilter('consumptionOptions', v)}
                  placeholder="All Models"
                  isLight={isLight}
                />
              )}
              <FilterSelect
                label="Cross-Region Inference"
                value={filters.crisSupport}
                onChange={(v) => updateFilter('crisSupport', v)}
                options={crisSupportOptions}
                isLight={isLight}
              />
              <FilterSelect
                label="Streaming Support"
                value={filters.streamingSupport}
                onChange={(v) => updateFilter('streamingSupport', v)}
                options={streamingSupportOptions}
                isLight={isLight}
              />
            </FilterGroup>

            {/* Group 3: Use Cases & Content */}
            <FilterGroup title="Use Cases & Content" icon={Target} isLight={isLight}>
              {availableUseCases.length > 0 && (
                <MultiSelectDropdown
                  label="Use Cases"
                  options={availableUseCases}
                  selected={filters.useCases}
                  onChange={(v) => updateFilter('useCases', v)}
                  placeholder="Choose options"
                  isLight={isLight}
                />
              )}
              <FilterSelect
                label="Modalities"
                value={filters.modality}
                onChange={(v) => updateFilter('modality', v)}
                options={modalityOptions}
                isLight={isLight}
              />
              {availableCapabilities.length > 0 && (
                <MultiSelectDropdown
                  label="Capabilities"
                  options={availableCapabilities}
                  selected={filters.capabilities}
                  onChange={(v) => updateFilter('capabilities', v)}
                  placeholder="Choose options"
                  isLight={isLight}
                />
              )}
            </FilterGroup>

            {/* Group 4: Model Capabilities */}
            <FilterGroup title="Model Capabilities" icon={Cpu} isLight={isLight}>
              {availableCustomizations.length > 0 && (
                <MultiSelectDropdown
                  label="Customization Options"
                  options={availableCustomizations}
                  selected={filters.customizations}
                  onChange={(v) => updateFilter('customizations', v)}
                  placeholder="All Models"
                  isLight={isLight}
                />
              )}
              {availableLanguages.length > 0 && (
                <MultiSelectDropdown
                  label="Languages"
                  options={availableLanguages}
                  selected={filters.languages}
                  onChange={(v) => updateFilter('languages', v)}
                  placeholder="Choose options"
                  isLight={isLight}
                />
              )}
              <FilterSelect
                label="Context Window"
                value={filters.contextFilter}
                onChange={(v) => updateFilter('contextFilter', v)}
                options={contextFilterOptions}
                isLight={isLight}
              />
            </FilterGroup>
          </div>
        </div>
      )}
    </div>
  )
}
