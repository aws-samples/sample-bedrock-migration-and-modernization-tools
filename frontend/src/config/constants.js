/**
 * Centralized constants for Bedrock Model Profiler frontend.
 *
 * This file consolidates all provider colors, region mappings, and other
 * configuration that was previously duplicated across multiple components.
 *
 * Keeping these in one place makes it easier to:
 * - Add new providers/regions
 * - Maintain consistency across the app
 * - Eventually sync with backend config
 */

// =============================================================================
// Provider Colors - Brand colors for each model provider
// =============================================================================

export const providerColors = {
  Amazon: '#FF9900',
  Anthropic: '#D4A27F',
  Meta: '#0082FB',
  'Mistral AI': '#F54E42',
  Mistral: '#F54E42',
  Cohere: '#39594D',
  'AI21 Labs': '#6C5CE7',
  AI21: '#6C5CE7',
  'Stability AI': '#7C5CFF',
  Stability: '#7C5CFF',
  'Luma AI': '#6366F1',
  Luma: '#6366F1',
  Writer: '#4A90D9',
  NVIDIA: '#76B900',
  DeepSeek: '#4A90D9',
  Qwen: '#6366F1',
  Google: '#4285F4',
  OpenAI: '#10A37F',
  TwelveLabs: '#6366F1',
  MiniMax: '#6366F1',
  'Moonshot AI': '#6366F1',
  default: '#64748b',
}

// Tailwind class version for components that need bg classes
export const providerColorClasses = {
  Amazon: 'bg-[#FF9900]',
  Anthropic: 'bg-[#D4A27F]',
  Meta: 'bg-[#0082FB]',
  'Mistral AI': 'bg-[#F54E42]',
  Mistral: 'bg-[#F54E42]',
  Cohere: 'bg-[#39594D]',
  'AI21 Labs': 'bg-[#6C5CE7]',
  AI21: 'bg-[#6C5CE7]',
  'Stability AI': 'bg-[#7C5CFF]',
  Stability: 'bg-[#7C5CFF]',
  'Luma AI': '#6366F1',
  Luma: 'bg-[#6366F1]',
  default: 'bg-slate-500',
}

// =============================================================================
// AWS Regions - Complete list with coordinates for map display
// =============================================================================

export const regionCoordinates = {
  // US
  'us-east-1': { lat: 38.9519, lng: -77.4480, name: 'N. Virginia', geo: 'US' },
  'us-east-2': { lat: 39.9612, lng: -82.9988, name: 'Ohio', geo: 'US' },
  'us-west-1': { lat: 37.3541, lng: -121.9552, name: 'N. California', geo: 'US' },
  'us-west-2': { lat: 45.8696, lng: -119.6880, name: 'Oregon', geo: 'US' },
  // EU
  'eu-west-1': { lat: 53.3498, lng: -6.2603, name: 'Ireland', geo: 'EU' },
  'eu-west-2': { lat: 51.5074, lng: -0.1278, name: 'London', geo: 'EU' },
  'eu-west-3': { lat: 48.8566, lng: 2.3522, name: 'Paris', geo: 'EU' },
  'eu-central-1': { lat: 50.1109, lng: 8.6821, name: 'Frankfurt', geo: 'EU' },
  'eu-central-2': { lat: 47.3769, lng: 8.5417, name: 'Zurich', geo: 'EU' },
  'eu-north-1': { lat: 59.3293, lng: 18.0686, name: 'Stockholm', geo: 'EU' },
  'eu-south-1': { lat: 45.4642, lng: 9.1900, name: 'Milan', geo: 'EU' },
  'eu-south-2': { lat: 40.4168, lng: -3.7038, name: 'Spain', geo: 'EU' },
  // Asia Pacific
  'ap-south-1': { lat: 19.0760, lng: 72.8777, name: 'Mumbai', geo: 'AP' },
  'ap-south-2': { lat: 17.3850, lng: 78.4867, name: 'Hyderabad', geo: 'AP' },
  'ap-northeast-1': { lat: 35.6762, lng: 139.6503, name: 'Tokyo', geo: 'AP' },
  'ap-northeast-2': { lat: 37.5665, lng: 126.9780, name: 'Seoul', geo: 'AP' },
  'ap-northeast-3': { lat: 34.6937, lng: 135.5023, name: 'Osaka', geo: 'AP' },
  'ap-southeast-1': { lat: 1.3521, lng: 103.8198, name: 'Singapore', geo: 'AP' },
  'ap-southeast-2': { lat: -33.8688, lng: 151.2093, name: 'Sydney', geo: 'AP' },
  'ap-southeast-3': { lat: -6.2088, lng: 106.8456, name: 'Jakarta', geo: 'AP' },
  'ap-southeast-4': { lat: -37.8136, lng: 144.9631, name: 'Melbourne', geo: 'AP' },
  'ap-southeast-5': { lat: 3.1390, lng: 101.6869, name: 'Malaysia', geo: 'AP' },
  'ap-east-1': { lat: 22.3193, lng: 114.1694, name: 'Hong Kong', geo: 'AP' },
  // Canada
  'ca-central-1': { lat: 45.5017, lng: -73.5673, name: 'Montreal', geo: 'CA' },
  'ca-west-1': { lat: 51.0447, lng: -114.0719, name: 'Calgary', geo: 'CA' },
  // South America
  'sa-east-1': { lat: -23.5505, lng: -46.6333, name: 'São Paulo', geo: 'SA' },
  // Middle East
  'me-south-1': { lat: 26.0667, lng: 50.5577, name: 'Bahrain', geo: 'ME' },
  'me-central-1': { lat: 24.4539, lng: 54.3773, name: 'UAE', geo: 'ME' },
  'il-central-1': { lat: 32.0853, lng: 34.7818, name: 'Tel Aviv', geo: 'ME' },
  // Africa
  'af-south-1': { lat: -33.9249, lng: 18.4241, name: 'Cape Town', geo: 'AF' },
}

// AWS regions for dropdown selectors (subset commonly used)
export const awsRegions = [
  { value: 'us-east-1', label: 'N. Virginia (us-east-1)', geo: 'US' },
  { value: 'us-east-2', label: 'Ohio (us-east-2)', geo: 'US' },
  { value: 'us-west-1', label: 'N. California (us-west-1)', geo: 'US' },
  { value: 'us-west-2', label: 'Oregon (us-west-2)', geo: 'US' },
  { value: 'eu-west-1', label: 'Ireland (eu-west-1)', geo: 'EU' },
  { value: 'eu-west-2', label: 'London (eu-west-2)', geo: 'EU' },
  { value: 'eu-west-3', label: 'Paris (eu-west-3)', geo: 'EU' },
  { value: 'eu-central-1', label: 'Frankfurt (eu-central-1)', geo: 'EU' },
  { value: 'eu-north-1', label: 'Stockholm (eu-north-1)', geo: 'EU' },
  { value: 'ap-northeast-1', label: 'Tokyo (ap-northeast-1)', geo: 'AP' },
  { value: 'ap-northeast-2', label: 'Seoul (ap-northeast-2)', geo: 'AP' },
  { value: 'ap-southeast-1', label: 'Singapore (ap-southeast-1)', geo: 'AP' },
  { value: 'ap-southeast-2', label: 'Sydney (ap-southeast-2)', geo: 'AP' },
  { value: 'ap-south-1', label: 'Mumbai (ap-south-1)', geo: 'AP' },
  { value: 'ca-central-1', label: 'Montreal (ca-central-1)', geo: 'CA' },
  { value: 'sa-east-1', label: 'Sao Paulo (sa-east-1)', geo: 'SA' },
]

// =============================================================================
// Geographic Region Groups
// =============================================================================

export const geoRegionOptions = [
  { value: 'All Regions', label: 'All Regions' },
  { value: 'US', label: 'US Regions' },
  { value: 'EU', label: 'EU Regions' },
  { value: 'AP', label: 'Asia Pacific' },
  { value: 'CA', label: 'Canada' },
  { value: 'SA', label: 'South America' },
  { value: 'ME', label: 'Middle East' },
  { value: 'AF', label: 'Africa' },
]

export const geoGroups = {
  US: { name: 'Americas (US)', color: '#3B82F6' },
  EU: { name: 'Europe', color: '#10B981' },
  AP: { name: 'Asia Pacific', color: '#F59E0B' },
  CA: { name: 'Canada', color: '#EF4444' },
  SA: { name: 'South America', color: '#8B5CF6' },
  ME: { name: 'Middle East', color: '#EC4899' },
  AF: { name: 'Africa', color: '#14B8A6' },
}

export const geoPrefixMap = {
  'US': 'us-',
  'EU': 'eu-',
  'AP': 'ap-',
  'CA': 'ca-',
  'SA': 'sa-',
  'ME': 'me-',
  'AF': 'af-',
}

// =============================================================================
// Model Configuration
// =============================================================================

// Context window size categories
export const contextWindowThresholds = {
  small: 32000,
  medium: 128000,
  large: 500000,
}

export function getContextSizeCategory(contextWindow) {
  if (!contextWindow || typeof contextWindow !== 'number') {
    return { label: 'Unknown', color: 'bg-slate-500', tier: 0 }
  }
  if (contextWindow < contextWindowThresholds.small) {
    return { label: 'Small', color: 'bg-slate-500', tier: 1 }
  }
  if (contextWindow < contextWindowThresholds.medium) {
    return { label: 'Medium', color: 'bg-blue-500', tier: 2 }
  }
  if (contextWindow < contextWindowThresholds.large) {
    return { label: 'Large', color: 'bg-emerald-500', tier: 3 }
  }
  return { label: 'XL', color: 'bg-purple-500', tier: 4 }
}

// Consumption option labels
export const consumptionLabels = {
  on_demand: 'In Region',
  provisioned: 'Provisioned',
  provisioned_throughput: 'Provisioned',
  batch: 'Batch',
  cross_region_inference: 'CRIS',
  mantle: 'Mantle',
}

// Capability labels (prettify snake_case raw values)
export const capabilityLabels = {
  chat: 'Chat',
  function_calling: 'Functions',
  image_understanding: 'Vision',
  image_generation: 'Image Gen',
  multimodal: 'Multimodal',
  text_generation: 'Text Gen',
  text_completion: 'Completion',
  embedding: 'Embedding',
  code_generation: 'Code',
  summarization: 'Summarize',
  classification: 'Classify',
  reranking: 'Rerank',
  video_generation: 'Video Gen',
  audio_generation: 'Audio Gen',
  speech_generation: 'Speech',
  document_understanding: 'Document',
}

// =============================================================================
// Modality Configuration
// =============================================================================

export const modalityOptions = [
  { value: 'All Modalities', label: 'All Modalities' },
  { value: 'TEXT', label: 'Text' },
  { value: 'IMAGE', label: 'Image' },
  { value: 'DOCUMENT', label: 'Document' },
  { value: 'VIDEO', label: 'Video' },
  { value: 'AUDIO', label: 'Audio' },
  { value: 'SPEECH', label: 'Speech' },
]

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get provider color (hex) with fallback to default
 */
export function getProviderColor(provider) {
  return providerColors[provider] || providerColors.default
}

/**
 * Get provider color class (Tailwind) with fallback to default
 */
export function getProviderColorClass(provider) {
  return providerColorClasses[provider] || providerColorClasses.default
}

/**
 * Get region display info
 */
export function getRegionInfo(regionCode) {
  return regionCoordinates[regionCode] || { name: regionCode, geo: 'Unknown' }
}
