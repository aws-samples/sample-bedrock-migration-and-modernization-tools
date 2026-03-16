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
