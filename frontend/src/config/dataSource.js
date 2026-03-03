/**
 * Data source configuration
 *
 * In production: Fetches data from CloudFront /data/* path (served from S3 data bucket)
 * In development: Fetches data via Vite dev server proxy /s3-data/* (uses local AWS credentials)
 */

// S3 configuration from environment variables
const S3_BUCKET = import.meta.env.VITE_S3_BUCKET || 'bedrock-profiler-data-169497827606-dev'
const S3_REGION = import.meta.env.VITE_S3_REGION || 'us-east-1'
const S3_PREFIX = import.meta.env.VITE_S3_PREFIX || 'latest'

// In production, data is served from CloudFront /data/* path
// In development, data is proxied via Vite /s3-data/* middleware
const isDevelopment = import.meta.env.DEV

// Export URLs based on environment
// In production, /latest/* is served directly from CloudFront data origin
export const DATA_URLS = {
  models: isDevelopment
    ? `/s3-data/${S3_PREFIX}/bedrock_models.json`
    : `/${S3_PREFIX}/bedrock_models.json`,
  pricing: isDevelopment
    ? `/s3-data/${S3_PREFIX}/bedrock_pricing.json`
    : `/${S3_PREFIX}/bedrock_pricing.json`,
  // New: Frontend config from backend
  frontendConfig: isDevelopment
    ? `/s3-data/config/frontend-config.json`
    : `/config/frontend-config.json`,
}

// Export config for debugging
export const DATA_SOURCE_CONFIG = {
  isDevelopment,
  bucket: S3_BUCKET,
  region: S3_REGION,
  prefix: S3_PREFIX,
}

// Log the data source on startup (only in development)
if (isDevelopment) {
  console.log(`[Data Source] Development mode - using S3 proxy`)
  console.log(`[Data Source] Bucket: ${S3_BUCKET}`)
  console.log(`[Data Source] Region: ${S3_REGION}`)
  console.log(`[Data Source] Models URL: ${DATA_URLS.models}`)
}
