/**
 * Filter Options Configuration
 * 
 * This file defines curated lists of valid filter options.
 * Raw data values are mapped to these normalized labels.
 * Unknown values are categorized as "Other" or excluded.
 */

// =============================================================================
// CUSTOMIZATION OPTIONS
// =============================================================================
export const CUSTOMIZATION_OPTIONS = {
  // Map raw values to display labels
  'FINE_TUNING': 'Fine-Tuning',
  'DISTILLATION': 'Distillation',
  'CONTINUED_PRE_TRAINING': 'Continued Pre-Training',
}

// =============================================================================
// LANGUAGE OPTIONS
// =============================================================================
export const LANGUAGE_OPTIONS = {
  // Primary languages (shown first, sorted)
  'English': 'English',
  'Chinese': 'Chinese',
  'Spanish': 'Spanish',
  'French': 'French',
  'German': 'German',
  'Japanese': 'Japanese',
  'Korean': 'Korean',
  'Portuguese': 'Portuguese',
  'Italian': 'Italian',
  'Dutch': 'Dutch',
  'Russian': 'Russian',
  'Arabic': 'Arabic',
  'Hindi': 'Hindi',
  'Turkish': 'Turkish',
  'Polish': 'Polish',
  'Vietnamese': 'Vietnamese',
  'Thai': 'Thai',
  'Indonesian': 'Indonesian',
  'Czech': 'Czech',
  'Danish': 'Danish',
  'Swedish': 'Swedish',
  'Norwegian': 'Norwegian',
  'Finnish': 'Finnish',
  'Greek': 'Greek',
  'Hebrew': 'Hebrew',
  'Romanian': 'Romanian',
  'Hungarian': 'Hungarian',
  'Ukrainian': 'Ukrainian',
  'Bengali': 'Bengali',
  'Tamil': 'Tamil',
  'Malay': 'Malay',
  'Croatian': 'Croatian',
  'Catalan': 'Catalan',
  'Bulgarian': 'Bulgarian',
  'Slovak': 'Slovak',
  'Slovenian': 'Slovenian',
  'Serbian': 'Serbian',
  'Lithuanian': 'Lithuanian',
  'Latvian': 'Latvian',
  'Estonian': 'Estonian',
  // Mapping variants
  'Multilingual': 'Multilingual',
  'Multi-lingual': 'Multilingual',
}

// Priority order for languages (most common first)
export const LANGUAGE_PRIORITY = [
  'English', 'Chinese', 'Spanish', 'French', 'German', 'Japanese', 
  'Korean', 'Portuguese', 'Italian', 'Arabic', 'Russian', 'Hindi',
  'Multilingual'
]

// =============================================================================
// CAPABILITY OPTIONS
// =============================================================================
export const CAPABILITY_OPTIONS = {
  // Text capabilities
  'Text generation': 'Text Generation',
  'Text Generation': 'Text Generation',
  'text generation': 'Text Generation',
  'Chat': 'Chat',
  'chat': 'Chat',
  'Conversation': 'Chat',
  'Question answering': 'Question Answering',
  'Summarization': 'Summarization',
  'Translation': 'Translation',
  
  // Code capabilities
  'Code generation': 'Code Generation',
  'Code Generation': 'Code Generation',
  'code generation': 'Code Generation',
  'Coding': 'Code Generation',
  'Code completion': 'Code Completion',
  'Code review': 'Code Review',
  
  // Reasoning
  'Reasoning': 'Reasoning',
  'reasoning': 'Reasoning',
  'Advanced reasoning': 'Advanced Reasoning',
  'Complex reasoning': 'Advanced Reasoning',
  'Mathematical reasoning': 'Math & Logic',
  'Math': 'Math & Logic',
  'Logic': 'Math & Logic',
  
  // Analysis
  'Analysis': 'Analysis',
  'Data analysis': 'Data Analysis',
  'Document analysis': 'Document Analysis',
  'Image analysis': 'Image Analysis',
  
  // Multimodal
  'Vision': 'Vision',
  'vision': 'Vision',
  'Image understanding': 'Vision',
  'Image to text': 'Vision',
  'Text to image': 'Image Generation',
  'Image generation': 'Image Generation',
  'Image Generation': 'Image Generation',
  'Video generation': 'Video Generation',
  'Video understanding': 'Video Understanding',
  'Audio': 'Audio',
  'Speech': 'Speech',
  'Speech to text': 'Speech-to-Text',
  'Text to speech': 'Text-to-Speech',
  
  // Agentic
  'Agentic': 'Agentic',
  'Agentic workflows': 'Agentic',
  'Agentic behavior': 'Agentic',
  'Tool use': 'Tool Use',
  'Function calling': 'Function Calling',
  
  // Embeddings
  'Embeddings': 'Embeddings',
  'Text embeddings': 'Embeddings',
  'embedding': 'Embeddings',
  
  // Context
  'Long context': 'Long Context',
  'Extended context': 'Long Context',
}

// =============================================================================
// USE CASE OPTIONS
// =============================================================================
export const USE_CASE_OPTIONS = {
  // Customer service
  'Customer service': 'Customer Service',
  'Customer support': 'Customer Service',
  'Chatbots': 'Chatbots & Assistants',
  'Virtual assistants': 'Chatbots & Assistants',
  'AI assistants': 'Chatbots & Assistants',
  'Ai assistants': 'Chatbots & Assistants',
  
  // Content
  'Content creation': 'Content Creation',
  'Content generation': 'Content Creation',
  'Copywriting': 'Content Creation',
  'Marketing': 'Marketing',
  
  // Code & Development
  'Software development': 'Software Development',
  'Code generation': 'Code Generation',
  'Code review': 'Code Review',
  'Debugging': 'Debugging',
  'API integration': 'API Integration',
  'API integration development': 'API Integration',
  
  // Analysis & Research
  'Research': 'Research',
  'Data analysis': 'Data Analysis',
  'Document processing': 'Document Processing',
  'Information extraction': 'Information Extraction',
  
  // Business
  'Business intelligence': 'Business Intelligence',
  'Financial analysis': 'Financial Analysis',
  'Legal': 'Legal',
  'Healthcare': 'Healthcare',
  'Education': 'Education',
  
  // Creative
  'Creative writing': 'Creative Writing',
  'Storytelling': 'Storytelling',
  'Image generation': 'Image Generation',
  'Video creation': 'Video Creation',
  
  // Agents
  'Agentic workflows': 'Agentic Workflows',
  'Agentic Workflows': 'Agentic Workflows',
  'Automation': 'Automation',
  'AI agents': 'AI Agents',
  'Ai agents': 'AI Agents',
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Normalize a raw value using the provided options map
 * @param {string} rawValue - Raw value from data
 * @param {Object} optionsMap - Map of raw values to normalized labels
 * @param {boolean} includeUnknown - Whether to include unknown values (default: false)
 * @returns {string|null} Normalized label or null if not found and includeUnknown is false
 */
export function normalizeValue(rawValue, optionsMap, includeUnknown = false) {
  if (!rawValue || typeof rawValue !== 'string') return null
  
  const trimmed = rawValue.trim()
  
  // Direct match
  if (optionsMap[trimmed]) {
    return optionsMap[trimmed]
  }
  
  // Case-insensitive match
  const lowerKey = Object.keys(optionsMap).find(
    key => key.toLowerCase() === trimmed.toLowerCase()
  )
  if (lowerKey) {
    return optionsMap[lowerKey]
  }
  
  // Partial match (if rawValue contains a known key)
  const partialKey = Object.keys(optionsMap).find(
    key => trimmed.toLowerCase().includes(key.toLowerCase()) && key.length > 3
  )
  if (partialKey) {
    return optionsMap[partialKey]
  }
  
  // Return raw value if includeUnknown, otherwise null
  return includeUnknown ? trimmed : null
}

/**
 * Extract and normalize values from models
 * @param {Array} models - Array of model objects
 * @param {string} field - Field name to extract (e.g., 'capabilities', 'languages')
 * @param {Object} optionsMap - Map of raw values to normalized labels
 * @param {Array} priorityOrder - Optional priority order for sorting
 * @returns {Array} Sorted array of unique normalized values
 */
export function extractNormalizedValues(models, field, optionsMap, priorityOrder = []) {
  const normalizedSet = new Set()
  
  models.forEach(m => {
    const values = m[field]
    if (Array.isArray(values)) {
      values.forEach(v => {
        const normalized = normalizeValue(v, optionsMap, false)
        if (normalized) {
          normalizedSet.add(normalized)
        }
      })
    }
  })
  
  // Sort by priority order, then alphabetically
  const result = Array.from(normalizedSet).sort((a, b) => {
    const aIndex = priorityOrder.indexOf(a)
    const bIndex = priorityOrder.indexOf(b)
    
    if (aIndex !== -1 && bIndex !== -1) return aIndex - bIndex
    if (aIndex !== -1) return -1
    if (bIndex !== -1) return 1
    return a.localeCompare(b)
  })
  
  return result
}

/**
 * Get the normalized label for a raw value
 * Used for display in dropdowns and chips
 */
export function getDisplayLabel(rawValue, type) {
  const optionsMaps = {
    customization: CUSTOMIZATION_OPTIONS,
    language: LANGUAGE_OPTIONS,
    capability: CAPABILITY_OPTIONS,
    useCase: USE_CASE_OPTIONS,
  }
  
  const map = optionsMaps[type]
  if (!map) return rawValue
  
  return normalizeValue(rawValue, map, true) || rawValue
}

/**
 * Create a reverse mapping from normalized values to raw values
 * Used for filtering - maps a normalized label back to all raw values that map to it
 * @param {Object} optionsMap - Map of raw values to normalized labels
 * @returns {Object} Map of normalized labels to arrays of raw values
 */
export function createReverseMapping(optionsMap) {
  const reverseMap = {}
  
  for (const [raw, normalized] of Object.entries(optionsMap)) {
    if (!reverseMap[normalized]) {
      reverseMap[normalized] = []
    }
    reverseMap[normalized].push(raw)
  }
  
  return reverseMap
}

// Pre-computed reverse mappings for filter matching
export const CAPABILITY_REVERSE_MAP = createReverseMapping(CAPABILITY_OPTIONS)
export const USE_CASE_REVERSE_MAP = createReverseMapping(USE_CASE_OPTIONS)
export const LANGUAGE_REVERSE_MAP = createReverseMapping(LANGUAGE_OPTIONS)
export const CUSTOMIZATION_REVERSE_MAP = createReverseMapping(CUSTOMIZATION_OPTIONS)
