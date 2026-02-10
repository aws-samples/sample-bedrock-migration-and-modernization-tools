import { create } from 'zustand'
import { persist } from 'zustand/middleware'

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
    }),
    {
      name: 'bedrock-comparison-storage',
      // Only persist the selectedModels array
      partialize: (state) => ({ selectedModels: state.selectedModels }),
    }
  )
)
