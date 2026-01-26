/**
 * Authentication configuration for AWS Cognito OIDC
 */

const authConfig = {
  // Cognito User Pool authority URL
  authority: import.meta.env.VITE_COGNITO_AUTHORITY_URL,

  // Cognito App Client ID
  clientId: import.meta.env.VITE_COGNITO_CLIENT_ID,

  // Redirect URI after authentication
  redirectUri: window.location.origin,

  // OAuth response type
  responseType: 'code',

  // OAuth scopes
  scope: 'aws.cognito.signin.user.admin email openid profile',
}

/**
 * Check if authentication is configured
 * Returns true if both authority and clientId are set
 */
export const isAuthConfigured = () => {
  return Boolean(authConfig.authority && authConfig.clientId)
}

/**
 * Get OIDC configuration for react-oidc-context
 */
export const getOidcConfig = () => {
  if (!isAuthConfigured()) {
    return null
  }

  return {
    authority: authConfig.authority,
    client_id: authConfig.clientId,
    redirect_uri: authConfig.redirectUri,
    response_type: authConfig.responseType,
    scope: authConfig.scope,
  }
}

export default authConfig
