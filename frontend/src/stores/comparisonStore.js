import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const MAX_COMPARISON_MODELS = 5

export const useComparisonStore = create(
  persist(
    (set, get) => ({
      // Selected models for comparison: Array of { model, region }
      selectedModels: [],

      // Add a model to comparison
      addModel: (model, region = 'us-east-1') => {
        const { selectedModels } = get()

        // Check if already selected
        if (selectedModels.some(m => m.model.model_id === model.model_id)) {
          return false
        }

        // Check max limit
        if (selectedModels.length >= MAX_COMPARISON_MODELS) {
          return false
        }

        set({
          selectedModels: [...selectedModels, { model, region }]
        })
        return true
      },

      // Remove a model from comparison
      removeModel: (modelId) => {
        set(state => ({
          selectedModels: state.selectedModels.filter(m => m.model.model_id !== modelId)
        }))
      },

      // Toggle model selection (add if not selected, remove if selected)
      toggleModel: (model, region = 'us-east-1') => {
        const { selectedModels, addModel, removeModel } = get()
        const isSelected = selectedModels.some(m => m.model.model_id === model.model_id)

        if (isSelected) {
          removeModel(model.model_id)
          return false
        } else {
          return addModel(model, region)
        }
      },

      // Update region for a specific model
      updateRegion: (modelId, region) => {
        set(state => ({
          selectedModels: state.selectedModels.map(m =>
            m.model.model_id === modelId ? { ...m, region } : m
          )
        }))
      },

      // Check if a model is selected
      isModelSelected: (modelId) => {
        const { selectedModels } = get()
        return selectedModels.some(m => m.model.model_id === modelId)
      },

      // Clear all selected models
      clearAll: () => {
        set({ selectedModels: [] })
      },

      // Get count of selected models
      getCount: () => {
        return get().selectedModels.length
      },

      // Check if can add more models
      canAddMore: () => {
        return get().selectedModels.length < MAX_COMPARISON_MODELS
      },

      // Get max allowed models
      maxModels: MAX_COMPARISON_MODELS,
    }),
    {
      name: 'bedrock-comparison-storage',
      // Only persist the selectedModels array
      partialize: (state) => ({ selectedModels: state.selectedModels }),
    }
  )
)
