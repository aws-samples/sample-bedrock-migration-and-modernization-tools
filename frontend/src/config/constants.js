/**
 * Centralized constants for Bedrock Model Profiler frontend.
 *
 * This file re-exports generated constants and adds any manual overrides.
 * The generated constants come from backend profiler-config.json via sync-config.js
 */

// Import generated constants (run 'npm run sync-config' to regenerate)
export {
  providerColors,
  regionCoordinates,
  awsRegions,
  geoRegionOptions,
  geoPrefixMap,
  contextWindowThresholds,
  configMetadata,
  getProviderColor,
  getRegionInfo,
  getContextSizeCategory,
} from './generated-constants.js'

// =============================================================================
// Manual Overrides and Extensions
// =============================================================================

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
  'Luma AI': 'bg-[#6366F1]',
  Luma: 'bg-[#6366F1]',
  default: 'bg-slate-500',
}

// Geographic groups with colors for visualization
export const geoGroups = {
  US: { name: 'Americas (US)', color: '#3B82F6' },
  EU: { name: 'Europe', color: '#10B981' },
  AP: { name: 'Asia Pacific', color: '#F59E0B' },
  CA: { name: 'Canada', color: '#EF4444' },
  SA: { name: 'South America', color: '#8B5CF6' },
  ME: { name: 'Middle East', color: '#EC4899' },
  AF: { name: 'Africa', color: '#14B8A6' },
  GOV: { name: 'GovCloud', color: '#6366F1' },
}

// Consumption option labels
export const consumptionLabels = {
  on_demand: 'In Region',
  provisioned: 'Provisioned',
  provisioned_throughput: 'Provisioned Throughput',
  batch: 'Batch',
  cross_region_inference: 'Cross-Region (CRIS)',
  mantle: 'Mantle',
  reserved: 'Reserved',
}

// Highspot internal documentation links by provider
export const providerHighspotLinks = {
  'Anthropic': 'https://aws.highspot.com/items/65d536d21f1dc338d678c98a?lfrm=shp.0',
  'Cohere': 'https://aws.highspot.com/items/66c2768ceaafe7792af41af3?lfrm=shp.1',
  'DeepSeek': 'https://aws.highspot.com/items/67b7bba26242ce84bc9a96e7',
  'Luma AI': 'https://aws.highspot.com/items/6789432dd115da44a75f18d7',
  'Meta': 'https://aws.highspot.com/items/6621d0bf53770d82756eb910?lfrm=shp.3',
  'Mistral AI': 'https://aws.highspot.com/items/66a0363be5729fa6f48e238f',
  'OpenAI': 'https://aws.highspot.com/items/6890f60545135d9ec3bccc25?lfrm=shp.0',
  'poolside': 'https://aws.highspot.com/items/670431c587fd5d910501bfbf?lfrm=shp.0',
  'Stability AI': 'https://aws.highspot.com/items/66c923fb0581b40e1de3a91f?lfrm=shp.0',
  'TwelveLabs': 'https://aws.highspot.com/items/6871b11d08f05626ccf9d5c8?lfrm=shp.0',
  'Writer': 'https://aws.highspot.com/items/68017cda152ae2f3216fd5d9',
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

// Modality options
export const modalityOptions = [
  { value: 'All Modalities', label: 'All Modalities' },
  { value: 'TEXT', label: 'Text' },
  { value: 'IMAGE', label: 'Image' },
  { value: 'DOCUMENT', label: 'Document' },
  { value: 'VIDEO', label: 'Video' },
  { value: 'AUDIO', label: 'Audio' },
  { value: 'SPEECH', label: 'Speech' },
]

/**
 * Get provider color class (Tailwind) with fallback to default
 */
export function getProviderColorClass(provider) {
  return providerColorClasses[provider] || providerColorClasses.default
}
