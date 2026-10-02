/**
 * Client-side image rules, mirrored from the backend so the user gets an
 * instant answer instead of a round trip.
 *
 * Source of truth: `backend/app/config.py` (limits) and
 * `backend/app/utils/image_utils.py` (order of checks).
 */

/** Matches `Settings.allowed_mime_types`. */
export const ACCEPTED_MIME_TYPES = Object.freeze([
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/webp',
])

/** Matches `Settings.allowed_extensions`. */
export const ACCEPTED_EXTENSIONS = Object.freeze([
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
])

/** `Settings.max_upload_bytes` - 5 MiB. */
export const MAX_FILE_BYTES = 5 * 1024 * 1024

/** `Settings.min_image_dimension` - shortest side, in pixels. */
export const MIN_IMAGE_DIMENSION = 16

/** `accept` attribute for the hidden file input. */
export const FILE_INPUT_ACCEPT = ACCEPTED_MIME_TYPES.join(',')

/** Human-readable format list, e.g. "JPG, PNG, WEBP". */
export const ACCEPTED_LABEL = 'JPG, PNG or WEBP'

/** Human-readable size limit. */
export const MAX_FILE_LABEL = '5 MB'

/**
 * @param {string} name
 * @returns {string} lowercase extension including the dot, or '' if none
 */
export function fileExtension(name) {
  const index = String(name ?? '').lastIndexOf('.')
  return index === -1 ? '' : String(name).slice(index).toLowerCase()
}

/**
 * @param {number} bytes
 * @returns {string}
 */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  )
  const value = bytes / 1024 ** exponent
  return `${value.toFixed(exponent === 0 ? 0 : 1)} ${units[exponent]}`
}

/**
 * Validates the file's name and MIME type. Dimensions are checked separately
 * by `readImageDimensions`, because that needs the file to be decoded.
 *
 * @param {File} file
 * @returns {string | null} an error message, or null when the file is fine
 */
export function validateImageFile(file) {
  if (!file) return 'No file was selected.'

  const extension = fileExtension(file.name)
  // The backend requires a filename with a supported extension.
  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    const found = extension ? ` '${extension}'` : ''
    return `Unsupported file type${found}. Please choose a ${ACCEPTED_LABEL} image.`
  }

  // Some browsers leave `type` empty for uncommon-but-valid files, so only
  // reject it when it is present and wrong. The backend requires both.
  if (file.type && !ACCEPTED_MIME_TYPES.includes(file.type.toLowerCase())) {
    return `Unsupported content type '${file.type}'. Allowed types: ${ACCEPTED_MIME_TYPES.join(', ')}.`
  }

  if (file.size === 0) return 'That file is empty.'

  if (file.size > MAX_FILE_BYTES) {
    return `That image is ${formatBytes(file.size)}, which is over the ${MAX_FILE_LABEL} limit.`
  }

  return null
}

/**
 * Decodes the file just far enough to read its pixel dimensions.
 *
 * @param {File} file
 * @returns {Promise<{ width: number, height: number }>}
 */
export function readImageDimensions(file) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file)
    const image = new Image()

    image.onload = () => {
      const size = { width: image.naturalWidth, height: image.naturalHeight }
      URL.revokeObjectURL(objectUrl)
      if (!size.width || !size.height) {
        reject(new Error('Could not read the image dimensions.'))
        return
      }
      resolve(size)
    }

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('That file could not be read as an image. It may be corrupt.'))
    }

    image.src = objectUrl
  })
}

/**
 * Full client-side validation, including the backend's minimum-size rule.
 *
 * @param {File} file
 * @returns {Promise<{ width: number, height: number } | null>} null when invalid
 * @throws {Error} with a user-facing message when the file is rejected
 */
export async function inspectImageFile(file) {
  const basicError = validateImageFile(file)
  if (basicError) throw new Error(basicError)

  const { width, height } = await readImageDimensions(file)
  if (Math.min(width, height) < MIN_IMAGE_DIMENSION) {
    throw new Error(
      `That image is too small. It must be at least ${MIN_IMAGE_DIMENSION}x${MIN_IMAGE_DIMENSION} pixels.`,
    )
  }

  return { width, height }
}