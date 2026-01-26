import { useEffect, useState } from 'react'
import { useAuth, hasAuthParams } from 'react-oidc-context'
import { useAuthStore } from '@/stores/authStore'
import { isAuthConfigured } from '@/config/auth'
import { Loader2 } from 'lucide-react'

/**
 * Authentication Gate component
 * Handles automatic sign-in redirect and loading states
 * Renders children only when authenticated (if auth is configured)
 */
export function AuthGate({ children }) {
  const [hasTriedSignin, setHasTriedSignin] = useState(false)
  const setUser = useAuthStore((state) => state.setUser)

  // If auth is not configured, render children immediately
  if (!isAuthConfigured()) {
    return children
  }

  return <AuthGateInner hasTriedSignin={hasTriedSignin} setHasTriedSignin={setHasTriedSignin} setUser={setUser}>{children}</AuthGateInner>
}

function AuthGateInner({ children, hasTriedSignin, setHasTriedSignin, setUser }) {
  const auth = useAuth()

  useEffect(() => {
    // Automatic sign-in redirect if not authenticated
    if (
      !hasAuthParams() &&
      !auth.isAuthenticated &&
      !auth.activeNavigator &&
      !auth.isLoading &&
      !hasTriedSignin
    ) {
      auth.signinRedirect()
      setHasTriedSignin(true)
    }

    // Update auth store when authenticated
    if (auth.isAuthenticated && auth.user) {
      setUser(auth.user)
    }
  }, [auth, hasTriedSignin, setHasTriedSignin, setUser])

  // Show loading state during authentication
  if (auth.isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-slate-950">
        <Loader2 className="h-8 w-8 animate-spin text-[#1A9E7A]" />
        <p className="mt-4 text-slate-400">Authenticating...</p>
      </div>
    )
  }

  // Show loading while redirect is in progress
  if (!auth.isAuthenticated && !hasAuthParams()) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-slate-950">
        <Loader2 className="h-8 w-8 animate-spin text-[#1A9E7A]" />
        <p className="mt-4 text-slate-400">Redirecting to sign in...</p>
      </div>
    )
  }

  return children
}
