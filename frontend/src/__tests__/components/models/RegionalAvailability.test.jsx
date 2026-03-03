import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { RegionalAvailability } from '@/components/models/RegionalAvailability'

// Mock useTheme
vi.mock('@/components/layout/ThemeProvider', () => ({
  useTheme: () => ({ theme: 'light' })
}))

// Mock useModels hook
vi.mock('@/hooks/useModels', () => ({
  useModels: () => ({
    models: [
      {
        model_id: 'anthropic.claude-3-sonnet',
        model_name: 'Claude 3 Sonnet',
        model_provider: 'Anthropic',
        availability: {
          on_demand: {
            regions: ['us-east-1', 'us-west-2']
          },
          cross_region: {
            supported: true,
            source_regions: ['us-east-1'],
            profiles: []
          },
          mantle: {
            supported: false,
            mantle_regions: []
          }
        }
      },
      {
        model_id: 'amazon.titan-text-express',
        model_name: 'Titan Text Express',
        model_provider: 'Amazon',
        availability: {
          on_demand: {
            regions: ['us-east-1', 'eu-west-1']
          },
          cross_region: {
            supported: false,
            source_regions: [],
            profiles: []
          },
          mantle: {
            supported: false,
            mantle_regions: []
          }
        }
      }
    ],
    loading: false,
    error: null
  })
}))

describe('RegionalAvailability', () => {
  beforeEach(() => {
    // Reset any mocks before each test
    vi.clearAllMocks()
  })

  it('renders table without max-height constraint', () => {
    // Arrange & Act
    render(<RegionalAvailability />)
    
    // Assert - The table container should not have max-h-* classes
    // The component uses overflow-auto for both horizontal and vertical scrolling
    const tableContainer = document.querySelector('.overflow-auto')
    expect(tableContainer).toBeInTheDocument()
    
    // Check that no max-h-* class is present on the container
    const containerClasses = tableContainer?.className || ''
    expect(containerClasses).not.toMatch(/max-h-/)
  })

  it('renders header cells with sticky classes', () => {
    // Arrange & Act
    render(<RegionalAvailability />)
    
    // Assert - Header cells should have sticky positioning
    const stickyHeaders = document.querySelectorAll('th.sticky')
    expect(stickyHeaders.length).toBeGreaterThan(0)
    
    // Check for top-0 class on header cells (sticky to top)
    const topStickyHeaders = document.querySelectorAll('th.sticky.top-0')
    expect(topStickyHeaders.length).toBeGreaterThan(0)
  })

  it('renders model column with sticky left positioning', () => {
    // Arrange & Act
    render(<RegionalAvailability />)
    
    // Assert - Model column header should have sticky left-0
    const modelColumnHeader = document.querySelector('th.sticky.left-0')
    expect(modelColumnHeader).toBeInTheDocument()
    
    // Model column cells in tbody should also be sticky left
    const modelColumnCells = document.querySelectorAll('td.sticky.left-0')
    expect(modelColumnCells.length).toBeGreaterThan(0)
  })

  it('has correct z-index layering for sticky elements', () => {
    // Arrange & Act
    render(<RegionalAvailability />)
    
    // Assert - Header should have higher z-index than body cells
    // The model column header (corner cell) should have z-30
    const cornerCell = document.querySelector('th.sticky.left-0.top-0')
    expect(cornerCell).toBeInTheDocument()
    expect(cornerCell?.className).toMatch(/z-30/)
    
    // Regular header cells should have z-20
    const regularHeaders = document.querySelectorAll('th.sticky.top-0:not(.left-0)')
    if (regularHeaders.length > 0) {
      expect(regularHeaders[0]?.className).toMatch(/z-20/)
    }
    
    // Body sticky cells should have z-10
    const bodyStickyCell = document.querySelector('td.sticky.left-0')
    if (bodyStickyCell) {
      expect(bodyStickyCell.className).toMatch(/z-10/)
    }
  })
})
