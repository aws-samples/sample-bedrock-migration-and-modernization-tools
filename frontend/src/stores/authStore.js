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

    const profile = authUser.profile || {}

    // Extract Cognito groups from token claims
    const groups = profile['cognito:groups'] || []

    set({
      user: {
        givenName: profile.given_name || null,
        email: profile.email || null,
        alias: profile.identities?.[0]?.userId || profile.preferred_username || null,
        sub: profile.sub || null,
        // Cognito groups (e.g., ['admins'])
        groups: Array.isArray(groups) ? groups : [],
        // OIDC provider geo attributes (non-PII, used for aggregated analytics only)
        country: profile['custom:country'] || null,
        region: profile['custom:region'] || null,
        geoLocation: profile['custom:geo_location'] || null,
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
