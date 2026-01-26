import { useAuth } from 'react-oidc-context'
import { useAuthStore } from '@/stores/authStore'
import { isAuthConfigured } from '@/config/auth'
import { LogOut, User } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { useTheme } from './ThemeProvider'

/**
 * User profile display component for the sidebar
 * Shows user info and sign-out button when authenticated
 */
export function UserProfile({ collapsed, mobileMenuOpen }) {
  const { theme } = useTheme()
  const isLight = theme === 'light'
  const getDisplayName = useAuthStore((state) => state.getDisplayName)
  const user = useAuthStore((state) => state.user)

  // Don't render anything if auth is not configured
  if (!isAuthConfigured()) {
    return null
  }

  return <UserProfileInner collapsed={collapsed} mobileMenuOpen={mobileMenuOpen} isLight={isLight} getDisplayName={getDisplayName} user={user} />
}

function UserProfileInner({ collapsed, mobileMenuOpen, isLight, getDisplayName, user }) {
  const auth = useAuth()
  const displayName = getDisplayName()

  const handleSignOut = () => {
    auth.signoutRedirect()
  }

  if (!auth.isAuthenticated || !user) {
    return null
  }

  const showExpanded = !collapsed || mobileMenuOpen

  return (
    <div className={cn(
      'p-3 border-t',
      isLight ? 'border-stone-200' : 'border-slate-800'
    )}>
      <div className={cn(
        'flex items-center gap-3',
        !showExpanded && 'justify-center'
      )}>
        {/* User Avatar */}
        <div className={cn(
          'flex items-center justify-center rounded-full flex-shrink-0',
          'h-8 w-8',
          isLight ? 'bg-amber-100 text-amber-700' : 'bg-[#1A9E7A]/20 text-[#1A9E7A]'
        )}>
          <User className="h-4 w-4" />
        </div>

        {showExpanded && (
          <div className="flex-1 min-w-0">
            <p className={cn(
              'text-sm font-medium truncate',
              isLight ? 'text-stone-900' : 'text-white'
            )}>
              {displayName}
            </p>
            {user.email && (
              <p className={cn(
                'text-xs truncate',
                isLight ? 'text-stone-500' : 'text-slate-400'
              )}>
                {user.email}
              </p>
            )}
          </div>
        )}

        {showExpanded && (
          <Button
            variant="ghost"
            size="icon"
            onClick={handleSignOut}
            className={cn(
              'h-8 w-8 flex-shrink-0',
              isLight
                ? 'text-stone-500 hover:text-stone-900 hover:bg-stone-100'
                : 'text-slate-400 hover:text-white hover:bg-slate-800'
            )}
            title="Sign out"
          >
            <LogOut className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Sign out button when collapsed */}
      {!showExpanded && (
        <Button
          variant="ghost"
          size="icon"
          onClick={handleSignOut}
          className={cn(
            'h-8 w-8 mt-2 mx-auto',
            isLight
              ? 'text-stone-500 hover:text-stone-900 hover:bg-stone-100'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          )}
          title="Sign out"
        >
          <LogOut className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}
