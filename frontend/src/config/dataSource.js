/**
 * Data source configuration
 *
 * In production: Fetches data from CloudFront /latest/* path (served from S3 data bucket)
 * In development: 
 *   - First tries S3 proxy /s3-data/* (uses local AWS credentials)
 *   - Falls back to local files in /latest/* (from public directory)
 */

// S3 configuration from environment variables
const S3_BUCKET = import.meta.env.VITE_S3_BUCKET || 'your-data-bucket-name'
const S3_REGION = import.meta.env.VITE_S3_REGION || 'us-east-1'
const S3_PREFIX = import.meta.env.VITE_S3_PREFIX || 'latest'

// In production, data is served from CloudFront /latest/* path
// In development, data is proxied via Vite /s3-data/* middleware (or local fallback)
const isDevelopment = import.meta.env.DEV

// Use local files in development by default (from public/latest/)
// Set VITE_USE_S3_PROXY=true to use S3 proxy instead
const useLocalData = isDevelopment && import.meta.env.VITE_USE_S3_PROXY !== 'true'

// Export URLs based on environment
// In production, /latest/* is served directly from CloudFront data origin
// In development with local data, /latest/* is served from public directory
export const DATA_URLS = {
  models: isDevelopment
    ? (useLocalData ? `/${S3_PREFIX}/bedrock_models.json` : `/s3-data/${S3_PREFIX}/bedrock_models.json`)
    : `/${S3_PREFIX}/bedrock_models.json`,
  pricing: isDevelopment
    ? (useLocalData ? `/${S3_PREFIX}/bedrock_pricing.json` : `/s3-data/${S3_PREFIX}/bedrock_pricing.json`)
    : `/${S3_PREFIX}/bedrock_pricing.json`,
  // New: Frontend config from backend
  frontendConfig: isDevelopment
    ? (useLocalData ? `/config/frontend-config.json` : `/s3-data/config/frontend-config.json`)
    : `/config/frontend-config.json`,
}

// Export config for debugging
export const DATA_SOURCE_CONFIG = {
  isDevelopment,
  useLocalData,
  bucket: S3_BUCKET,
  region: S3_REGION,
  prefix: S3_PREFIX,
}

// Log the data source on startup (only in development)
if (isDevelopment) {
  if (useLocalData) {
    console.log(`[Data Source] Development mode - using LOCAL files from public/latest/`)
  } else {
    console.log(`[Data Source] Development mode - using S3 proxy`)
    console.log(`[Data Source] Bucket: ${S3_BUCKET}`)
  }
  console.log(`[Data Source] Models URL: ${DATA_URLS.models}`)
}
