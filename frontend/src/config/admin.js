// Cognito user-group constants
const GROUPS = {
  BETA: 'beta-access-users',
  OPERATORS: 'region-roadmap-operators',
  ADMINS: 'admins',
}

const hasGroup = (user, group) =>
  Array.isArray(user?.groups) && user.groups.includes(group)

// --- Permission functions (groups are additive) ---

export const isAdmin = (user) => hasGroup(user, GROUPS.ADMINS)

export const canViewRoadmap = (user) =>
  [GROUPS.BETA, GROUPS.OPERATORS, GROUPS.ADMINS].some(g => hasGroup(user, g))

export const canViewRegionalAvailability = (user) =>
  [GROUPS.BETA, GROUPS.OPERATORS, GROUPS.ADMINS].some(g => hasGroup(user, g))

export const canViewQuotas = (user) =>
  [GROUPS.BETA, GROUPS.OPERATORS, GROUPS.ADMINS].some(g => hasGroup(user, g))

export const canViewProvisionedPricing = (user) =>
  [GROUPS.BETA, GROUPS.OPERATORS, GROUPS.ADMINS].some(g => hasGroup(user, g))

export const canEditRoadmap = (user) => hasGroup(user, GROUPS.OPERATORS)

export const canViewAnalytics = (user) => hasGroup(user, GROUPS.ADMINS)

export const canViewChangelog = (user) => hasGroup(user, GROUPS.ADMINS)

// --- Sidebar badge definitions ---

const BADGES = {
  BETA: {
    text: 'BETA',
    light: 'bg-amber-100 text-amber-700 border-amber-200/60',
    dark: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
  },
  OP: {
    text: 'OP',
    light: 'bg-sky-100 text-sky-700 border-sky-200/60',
    dark: 'bg-sky-500/15 text-sky-400 border-sky-500/20',
  },
  ADM: {
    text: 'ADM',
    light: 'bg-purple-100 text-purple-700 border-purple-200/60',
    dark: 'bg-purple-500/15 text-purple-400 border-purple-500/20',
  },
}

/**
 * Returns a badge object { text, light, dark } for a sidebar section,
 * based on the user's highest-priority group for that section.
 * Returns null for sections that don't need a badge.
 */
export function getSectionBadge(user, sectionId) {
  if (sectionId === 'admin') return BADGES.ADM
  if (sectionId === 'changelog') return BADGES.ADM

  if (sectionId === 'roadmap') {
    if (hasGroup(user, GROUPS.ADMINS)) return BADGES.ADM
    if (hasGroup(user, GROUPS.OPERATORS)) return BADGES.OP
    if (hasGroup(user, GROUPS.BETA)) return BADGES.BETA
  }

  if (sectionId === 'availability') {
    if (hasGroup(user, GROUPS.ADMINS)) return BADGES.ADM
    if (hasGroup(user, GROUPS.OPERATORS)) return BADGES.OP
    if (hasGroup(user, GROUPS.BETA)) return BADGES.BETA
  }

  if (sectionId === 'quotas') {
    if (hasGroup(user, GROUPS.ADMINS)) return BADGES.ADM
    if (hasGroup(user, GROUPS.OPERATORS)) return BADGES.OP
    if (hasGroup(user, GROUPS.BETA)) return BADGES.BETA
  }

  return null
}
