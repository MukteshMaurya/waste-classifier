import { ApiError, TimeoutError } from './errors.js'

/**
 * Base URL of the FastAPI backend.
 *
 * Reads VITE_API_URL (see `.env.example`). No production host is hard-coded
 * here on purpose - the deployed build gets its value from the Vercel
 * environment variable of the same name.
 */
export const API_BASE_URL = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

/**
 * Render's free tier parks an idle service, so the first request of a cold
 * start can take the best part of a minute before uvicorn answers. Both
 * timeouts are deliberately generous.
 */
export const HEALTH_TIMEOUT_MS = 30_000
export const PREDICT_TIMEOUT_MS = 120_000
export const CATEGORIES_TIMEOUT_MS = 30_000

/**
 * @param {number} ms
 * @param {AbortSignal | null} [externalSignal]
 * @returns {{ signal: AbortSignal, cleanup: () => void }}
 */
function withTimeout(ms, externalSignal) {
  const controller = new AbortController()

  const timer = setTimeout(() => controller.abort(), ms)

  const onExternalAbort = () => controller.abort()
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort()
    } else {
      externalSignal.addEventListener('abort', onExternalAbort)
    }
  }

  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer)
      externalSignal?.removeEventListener('abort', onExternalAbort)
    },
  }
}

/**
 * Parses a response body without ever throwing - a proxy or an unhandled
 * backend exception can return HTML or plain text instead of JSON.
 *
 * @param {Response} response
 * @returns {Promise<unknown>}
 */
async function readBody(response) {
  const text = await response.text().catch(() => '')
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

/**
 * Turns FastAPI's `{"detail": [{loc, msg, type}]}` validation envelope into a
 * single readable sentence.
 *
 * @param {unknown} body
 * @returns {string | null}
 */
function readValidationDetail(body) {
  if (!body || typeof body !== 'object' || !Array.isArray(body.detail)) return null
  const messages = body.detail
    .map((item) => {
      if (!item || typeof item !== 'object') return null
      const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null
      const msg = typeof item.msg === 'string' ? item.msg : null
      if (!msg) return null
      return field ? `${String(field)}: ${msg}` : msg
    })
    .filter(Boolean)
  return messages.length ? messages.join(' ') : null
}

/**
 * Builds the user-facing error for a non-2xx response.
 *
 * @param {Response} response
 * @param {unknown} body
 * @param {string} context
 * @returns {ApiError}
 */
function toHttpError(response, body, context) {
  const status = response.status
  const unreachableHint =
    'The API host did not answer. If you are using the free Render tier the service may be asleep - retry in a few seconds.'

  const payload =
    body && typeof body === 'object' && !Array.isArray(body) ? body : null
  const backendMessage =
    payload && typeof payload.message === 'string' ? payload.message : null
  const backendCode = payload && typeof payload.error === 'string' ? payload.error : null

  if (status === 503) {
    return new ApiError(
      backendMessage || 'The classification model is warming up or unavailable.',
      {
        status,
        code: backendCode || 'model_unavailable',
        hint: unreachableHint,
      },
    )
  }

  if (status === 413) {
    return new ApiError(backendMessage || 'That image is too large to upload.', {
      status,
      code: backendCode || 'file_too_large',
      hint: 'Compress or resize the image and try again.',
    })
  }

  if (status === 422) {
    return new ApiError(
      readValidationDetail(body) || 'The backend rejected the request.',
      {
        status,
        code: 'validation_error',
        hint: 'This is a client bug - please report it.',
      },
    )
  }

  if (status === 404) {
    return new ApiError(
      backendMessage || `The API has no ${context} endpoint at this address.`,
      {
        status,
        code: backendCode || 'not_found',
        hint: 'Check that VITE_API_URL points at the backend root URL.',
      },
    )
  }

  if (status >= 500) {
    return new ApiError(
      backendMessage || 'The backend failed while handling the request.',
      {
        status,
        code: backendCode || 'server_error',
        hint:
          'The prediction could not be completed. Retry, and check the backend logs if it keeps failing.',
      },
    )
  }

  return new ApiError(
    backendMessage ||
      (typeof body === 'string' && body.trim()
        ? body.trim().slice(0, 300)
        : `Request failed with status ${status}.`),
    { status, code: backendCode || `http_${status}` },
  )
}

/**
 * @param {unknown} error
 * @param {{ timeoutMs: number }} options
 * @returns {never}
 */
function rethrow(error, { timeoutMs }) {
  if (error instanceof ApiError) throw error

  if (error?.name === 'AbortError') {
    throw new TimeoutError(
      `The backend did not respond within ${Math.round(timeoutMs / 1000)} seconds.`,
      {
        code: 'timeout',
        hint: 'A cold Render service can take a minute to start. Try again.',
        cause: error,
      },
    )
  }

  throw new ApiError('Could not reach the classification API.', {
    code: 'network_error',
    hint:
      'Check that VITE_API_URL is correct, that the backend is running, and that its CORS allow-list (FRONTEND_URL) contains this origin.',
    cause: error,
  })
}

/**
 * @param {string} path
 * @param {RequestInit} init
 * @param {{ timeoutMs: number, context: string }} options
 * @returns {Promise<unknown>}
 */
async function request(path, init, { timeoutMs, context }) {
  const { signal, cleanup } = withTimeout(timeoutMs, init.signal ?? null)

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal,
      headers: { Accept: 'application/json', ...(init.headers ?? {}) },
    })
    const body = await readBody(response)

    if (!response.ok) throw toHttpError(response, body, context)

    return body
  } catch (error) {
    rethrow(error, { timeoutMs })
  } finally {
    cleanup()
  }
}

/**
 * `GET /health`
 *
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<import('./types.js').HealthResponse>}
 */
export async function fetchHealth({ signal } = {}) {
  return request('/health', { method: 'GET', signal }, {
    timeoutMs: HEALTH_TIMEOUT_MS,
    context: 'health',
  })
}

/**
 * `GET /categories`
 *
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<import('./types.js').CategoriesResponse>}
 */
export async function fetchCategories({ signal } = {}) {
  return request('/categories', { method: 'GET', signal }, {
    timeoutMs: CATEGORIES_TIMEOUT_MS,
    context: 'categories',
  })
}

/**
 * `POST /predict` as multipart/form-data with the field name `file`.
 *
 * Content-Type is intentionally not set: the browser must add the multipart
 * boundary itself. Only this one still frame is ever uploaded - the camera
 * preview stays local and is never streamed.
 *
 * @param {File | Blob} file
 * @param {{ filename?: string, signal?: AbortSignal }} [options]
 * @returns {Promise<import('./types.js').PredictionResponse>}
 */
export async function predictImage(file, { filename, signal } = {}) {
  const body = new FormData()
  body.append('file', file, filename || 'upload.jpg')

  return request('/predict', { method: 'POST', body, signal }, {
    timeoutMs: PREDICT_TIMEOUT_MS,
    context: 'predict',
  })
}