/**
 * Presentation helpers for the backend response.
 *
 * The backend sends `confidence` already scaled to 0-100 (see
 * `PredictionResponse.confidence`, `ge=0.0, le=100.0`) and `confidence_level`
 * as a free-text string, so the level is derived from the value rather than
 * hard-coded thresholds - those are env-tunable on the backend.
 */

/**
 * @typedef {'high' | 'moderate' | 'low' | 'unknown'} ConfidenceTone
 */

/**
 * @param {number} value
 * @param {number} [decimals]
 * @returns {number}
 */
export function clampPercent(value, decimals = 2) {
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, value))
}

/**
 * @param {number} value
 * @param {number} [decimals]
 * @returns {string} e.g. "93.11%"
 */
export function formatPercent(value, decimals = 2) {
  if (!Number.isFinite(value)) return '-'
  return `${clampPercent(value).toFixed(decimals)}%`
}

/**
 * @param {number} value
 * @returns {string} e.g. "1,502" - `image.width` / `image.height` arrive as floats
 */
export function formatDimension(value) {
  if (!Number.isFinite(value)) return '-'
  return Math.round(value).toLocaleString('en-GB')
}

/**
 * @param {string | undefined | null} level backend `confidence_level`
 * @param {number} confidence percentage 0-100
 * @returns {ConfidenceTone}
 */
export function confidenceTone(level, confidence) {
  switch ((level || '').toLowerCase()) {
    case 'high':
      return 'high'
    case 'moderate':
      return 'moderate'
    case 'low':
      return 'low'
    default:
      break
  }
  // Fall back to the numeric value if the label is ever missing or new.
  if (!Number.isFinite(confidence)) return 'unknown'
  if (confidence >= 80) return 'high'
  if (confidence >= 60) return 'moderate'
  return 'low'
}

/**
 * @param {ConfidenceTone} tone
 * @returns {string}
 */
export function confidenceLabel(tone) {
  switch (tone) {
    case 'high':
      return 'High confidence'
    case 'moderate':
      return 'Moderate confidence'
    case 'low':
      return 'Low confidence'
    default:
      return 'Confidence unknown'
  }
}

/**
 * `disposal.category` is free text ("Recyclable (check locally)",
 * "Hazardous / Special", ...), so it is matched by substring rather than by
 * an exact enum comparison.
 *
 * @param {string | undefined | null} category
 * @returns {'recycle' | 'compost' | 'special' | 'general' | 'unknown'}
 */
export function disposalTone(category) {
  const text = (category || '').toLowerCase()
  if (text.includes('hazard') || text.includes('special')) return 'special'
  if (text.includes('compost')) return 'compost'
  if (text.includes('donate')) return 'special'
  if (text.includes('recycl')) return 'recycle'
  if (text.includes('general')) return 'general'
  return 'unknown'
}

/**
 * @param {number} ms
 * @returns {string} e.g. "34.7 ms"
 */
export function formatMilliseconds(ms) {
  if (!Number.isFinite(ms)) return '-'
  return `${ms.toFixed(ms >= 100 ? 0 : 1)} ms`
}