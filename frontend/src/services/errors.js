/**
 * Normalised error raised by every request in this module.
 *
 * The UI only ever renders `message` and `hint`, so all transport-level
 * detail (aborts, HTML error pages, plain-text 500s) is funnelled into a
 * single predictable shape.
 */
export class ApiError extends Error {
  /**
   * @param {string} message
   * @param {object} [options]
   * @param {string} [options.hint] Actionable follow-up shown under the message.
   * @param {number} [options.status] HTTP status, when a response was received.
   * @param {string} [options.code] Backend machine-readable error code.
   * @param {unknown} [options.cause]
   */
  constructor(message, { hint, status, code, cause } = {}) {
    super(message)
    this.name = 'ApiError'
    this.hint = hint
    this.status = status
    this.code = code
    this.cause = cause
  }
}

/** Raised when the caller aborted the request (timeout or navigation). */
export class TimeoutError extends ApiError {
  constructor(message, options = {}) {
    super(message, options)
    this.name = 'TimeoutError'
  }
}

/**
 * True when the failure looks like "the host could not be reached at all"
 * rather than "the host answered with an error".
 *
 * @param {unknown} error
 * @returns {boolean}
 */
export function isConnectionError(error) {
  return (
    error instanceof ApiError &&
    (error.code === 'network_error' || error.code === 'timeout')
  )
}