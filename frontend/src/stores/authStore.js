import { create } from 'zustand'

/**
 * Authentication state store using Zustand
 * Stores user profile information extracted from JWT token
 */
export const useAuthStore = create((set, get) => ({
  // User profile data
  user: null,

  // JWT token for API calls (if needed)
  accessToken: null,

  // Authentication status
  isAuthenticated: false,

  // Set user data from OIDC authentication
  setUser: (authUser) => {
    if (!authUser) {
      set({
        user: null,
        accessToken: null,
        isAuthenticated: false,
      })
      return
    }

    set({
      user: {
        givenName: authUser.profile?.given_name || null,
        email: authUser.profile?.email || null,
        alias: authUser.profile?.identities?.[0]?.userId || authUser.profile?.preferred_username || null,
        sub: authUser.profile?.sub || null,
      },
      accessToken: authUser.access_token,
      isAuthenticated: true,
    })
  },

  // Clear authentication state
  clearUser: () => {
    set({
      user: null,
      accessToken: null,
      isAuthenticated: false,
    })
  },

  // Get display name for user
  getDisplayName: () => {
    const { user } = get()
    if (!user) return null
    return user.givenName || user.alias || user.email || 'User'
  },
}))
