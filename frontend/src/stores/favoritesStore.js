import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useFavoritesStore = create(
  persist(
    (set, get) => ({
      favoriteIds: [],

      toggleFavorite: (modelId) => {
        const { favoriteIds } = get()
        set({
          favoriteIds: favoriteIds.includes(modelId)
            ? favoriteIds.filter(id => id !== modelId)
            : [...favoriteIds, modelId],
        })
      },

      isFavorite: (modelId) => get().favoriteIds.includes(modelId),

      clearAll: () => set({ favoriteIds: [] }),

      getCount: () => get().favoriteIds.length,
    }),
    {
      name: 'bedrock-favorites-storage',
      partialize: (state) => ({ favoriteIds: state.favoriteIds }),
    }
  )
)
