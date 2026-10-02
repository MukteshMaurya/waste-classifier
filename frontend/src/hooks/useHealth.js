import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchHealth } from '../services/api.js'
import { ApiError } from '../services/errors.js'

/**
 * @typedef {'idle' | 'checking' | 'healthy' | 'degraded' | 'unreachable'} HealthState
 */

/**
 * Polls `GET /health` so the header can tell the user, before they upload
 * anything, whether the backend is up. A Render service that is asleep looks
 * identical to an outage from the browser, so the distinction is surfaced as
 * `unreachable` with a retry affordance rather than being silently swallowed.
 *
 * @returns {{
 *   state: HealthState,
 *   message: string | null,
 *   version: string | null,
 *   classCount: number | null,
 *   check: () => void,
 * }}
 */
export function useHealth() {
  const [state, setState] = useState(/** @type {HealthState} */('idle'))
  const [message, setMessage] = useState(/** @type {string | null} */(null))
  const [version, setVersion] = useState(/** @type {string | null} */(null))
  const [classCount, setClassCount] = useState(/** @type {number | null} */(null))
  const [nonce, setNonce] = useState(0)
  const activeController = useRef(/** @type {AbortController | null} */(null))

  useEffect(() => {
    const controller = new AbortController()
    activeController.current = controller
    let cancelled = false

    setState('checking')
    setMessage(null)

    fetchHealth({ signal: controller.signal })
      .then((health) => {
        if (cancelled || !health) return
        setVersion(health.version ?? null)
        setClassCount(Number.isFinite(health.classes) ? health.classes : null)

        // `/health` answers 503 when the model failed to load, which the
        // service layer surfaces as a thrown ApiError - handled below.
        if (health.model_loaded) {
          setState('healthy')
        } else {
          setState('degraded')
          setMessage(health.error || 'The model is not loaded on the server.')
        }
      })
      .catch((error) => {
        if (cancelled || controller.signal.aborted) return
        setState('unreachable')
        setVersion(null)
        setClassCount(null)
        setMessage(
          error instanceof ApiError ? error.message : 'The API is not responding.',
        )
      })

    return () => {
      cancelled = true
      controller.abort()
      if (activeController.current === controller) activeController.current = null
    }
  }, [nonce])

  const check = useCallback(() => setNonce((value) => value + 1), [])

  return { state, message, version, classCount, check }
}