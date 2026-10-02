import { useEffect, useRef, useState } from 'react'
import { fetchCategories } from '../services/api.js'

/**
 * Loads `GET /categories` for the reference grid.
 *
 * Supporting information only: if the call fails the grid is simply hidden,
 * because the prediction flow does not depend on it.
 *
 * @returns {{
 *   classes: import('../services/types.js').ClassInfo[]
 *   isLoading: boolean
 *   failed: boolean
 * }}
 */
export function useCategories() {
  const [classes, setClasses] = useState(
    /** @type {import('../services/types.js').ClassInfo[]} */([]),
  )
  const [isLoading, setIsLoading] = useState(true)
  const [failed, setFailed] = useState(false)
  const controllerRef = useRef(/** @type {AbortController | null} */(null))

  useEffect(() => {
    const controller = new AbortController()
    controllerRef.current = controller
    let cancelled = false

    fetchCategories({ signal: controller.signal })
      .then((response) => {
        if (cancelled) return
        const items = Array.isArray(response?.classes) ? response.classes : []
        setClasses([...items].sort((a, b) => (a.index ?? 0) - (b.index ?? 0)))
        setFailed(false)
      })
      .catch(() => {
        if (cancelled || controller.signal.aborted) return
        setFailed(true)
      })
      .finally(() => {
        if (cancelled) return
        setIsLoading(false)
        if (controllerRef.current === controller) controllerRef.current = null
      })

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [])

  return { classes, isLoading, failed }
}