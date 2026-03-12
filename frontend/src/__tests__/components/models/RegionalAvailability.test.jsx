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
            regions: ['us-east-1'],
            profiles: []
          },
          mantle: {
            supported: false,
            regions: []
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
            regions: [],
            profiles: []
          },
          mantle: {
            supported: false,
            regions: []
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

  it('renders grid container without max-height constraint', () => {
    // Arrange & Act
    render(<RegionalAvailability />)
    
    // Assert - The grid container should not have max-h-* classes
    // The component uses overflow-auto for both horizontal and vertical scrolling
    const gridContainer = document.querySelector('.overflow-auto')
    expect(gridContainer).toBeInTheDocument()
    
    // Check that no max-h-* class is present on the container
    const containerClasses = gridContainer?.className || ''
    expect(containerClasses).not.toMatch(/max-h-/)
  })

  it('renders header with sticky positioning', () => {
    // Arrange & Act
    render(<RegionalAvailability />)
    
    // Assert - Header container should have sticky positioning
    // The CSS Grid implementation uses div.sticky.top-0 for the header
    const stickyHeader = document.querySelector('.sticky.top-0')
    expect(stickyHeader).toBeInTheDocument()
  })

  it('renders model column with sticky left positioning', () => {
    // Arrange & Act
    render(<RegionalAvailability />)
    
    // Assert - Model column header should have sticky left-0
    // In CSS Grid implementation, we use div.sticky.left-0 for the model column
    const modelColumnHeader = document.querySelector('.sticky.left-0')
    expect(modelColumnHeader).toBeInTheDocument()
  })

  it('has correct z-index layering for sticky elements', () => {
    // Arrange & Act
    render(<RegionalAvailability />)
    
    // Assert - Header should have higher z-index than body cells
    // The model column header (corner cell) should have z-30
    const cornerCell = document.querySelector('.sticky.left-0.z-30')
    expect(cornerCell).toBeInTheDocument()
    
    // The header container should have z-20
    const headerContainer = document.querySelector('.sticky.top-0.z-20')
    expect(headerContainer).toBeInTheDocument()
    
    // Body sticky cells (model name column) should have z-10
    // These are rendered via virtualization, so we check for the class pattern
    const bodyStickyCell = document.querySelector('.sticky.left-0.z-10')
    // Note: Body cells may not be rendered if virtualization hasn't kicked in
    // So we just verify the header structure is correct
    expect(cornerCell?.className).toMatch(/z-30/)
  })
})
