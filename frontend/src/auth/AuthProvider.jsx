import { AuthProvider as OidcAuthProvider } from 'react-oidc-context'
import { getOidcConfig, isAuthConfigured } from '@/config/auth'

/**
 * Authentication Provider wrapper
 * Wraps the app with OIDC authentication if configured,
 * otherwise renders children without authentication
 */
export function AuthProvider({ children }) {
  const oidcConfig = getOidcConfig()

  // If auth is not configured, render children without auth wrapper
  if (!isAuthConfigured() || !oidcConfig) {
    return children
  }

  return (
    <OidcAuthProvider {...oidcConfig}>
      {children}
    </OidcAuthProvider>
  )
}
