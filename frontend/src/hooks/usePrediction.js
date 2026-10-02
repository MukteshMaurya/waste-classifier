import { useCallback, useEffect, useRef, useState } from 'react'
import { predictImage } from '../services/api.js'
import { ApiError } from '../services/errors.js'
import { inspectImageFile } from '../utils/files.js'

/**
 * @typedef {'upload' | 'camera'} ImageSource
 * @typedef {'idle' | 'validating' | 'loading' | 'done' | 'error'} PredictionStatus
 */

/**
 * Owns the single uploaded/captured image and the one in-flight `/predict`
 * request. Nothing here fabricates data: `result` is only ever set from the
 * backend response.
 *
 * @returns {{
 *   file: File | null,
 *   previewUrl: string | null,
 *   source: ImageSource | null,
 *   result: import('../services/types.js').PredictionResponse | null,
 *   error: ApiError | null,
 *   status: PredictionStatus,
 *   isBusy: boolean,
 *   selectFile: (file: File, source: ImageSource) => Promise<boolean>,
 *   analyze: () => Promise<void>,
 *   reset: () => void,
 *   clearError: () => void,
 * }}
 */
export function usePrediction() {
  const [file, setFile] = useState(/** @type {File | null} */(null))
  const [previewUrl, setPreviewUrl] = useState(/** @type {string | null} */(null))
  const [source, setSource] = useState(/** @type {ImageSource | null} */(null))
  const [result, setResult] = useState(
    /** @type {import('../services/types.js').PredictionResponse | null} */(null),
  )
  const [error, setError] = useState(/** @type {ApiError | null} */(null))
  const [status, setStatus] = useState(/** @type {PredictionStatus} */('idle'))

  const previewRef = useRef(/** @type {string | null} */(null))
  const controllerRef = useRef(/** @type {AbortController | null} */(null))

  const releasePreview = useCallback(() => {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current)
      previewRef.current = null
    }
  }, [])

  const abortInFlight = useCallback(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
  }, [])

  // Release the blob URL and cancel any request on unmount.
  useEffect(
    () => () => {
      abortInFlight()
      releasePreview()
    },
    [abortInFlight, releasePreview],
  )

  /**
   * @param {File} nextFile
   * @param {ImageSource} nextSource
   * @returns {Promise<boolean>} whether the file was accepted
   */
  const selectFile = useCallback(
    async (nextFile, nextSource) => {
      abortInFlight()
      setResult(null)
      setError(null)
      setStatus('validating')

      try {
        await inspectImageFile(nextFile)
      } catch (validationError) {
        releasePreview()
        setFile(null)
        setPreviewUrl(null)
        setSource(null)
        setStatus('error')
        setError(
          new ApiError(
            validationError instanceof Error
              ? validationError.message
              : 'That image could not be used.',
            { code: 'invalid_file' },
          ),
        )
        return false
      }

      releasePreview()
      const objectUrl = URL.createObjectURL(nextFile)
      previewRef.current = objectUrl

      setFile(nextFile)
      setPreviewUrl(objectUrl)
      setSource(nextSource)
      setStatus('idle')
      return true
    },
    [abortInFlight, releasePreview],
  )

  const analyze = useCallback(async () => {
    if (!file || controllerRef.current) return

    const controller = new AbortController()
    controllerRef.current = controller
    setStatus('loading')
    setError(null)

    try {
      const prediction = await predictImage(file, {
        filename: file.name,
        signal: controller.signal,
      })
      if (controller.signal.aborted) return
      setResult(prediction)
      setStatus('done')
    } catch (requestError) {
      if (controller.signal.aborted) return
      setResult(null)
      setStatus('error')
      setError(
        requestError instanceof ApiError
          ? requestError
          : new ApiError('The prediction failed for an unknown reason.', {
              cause: requestError,
            }),
      )
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
    }
  }, [file])

  const reset = useCallback(() => {
    abortInFlight()
    releasePreview()
    setFile(null)
    setPreviewUrl(null)
    setSource(null)
    setResult(null)
    setError(null)
    setStatus('idle')
  }, [abortInFlight, releasePreview])

  const clearError = useCallback(() => setError(null), [])

  return {
    file,
    previewUrl,
    source,
    result,
    error,
    status,
    isBusy: status === 'loading' || status === 'validating',
    selectFile,
    analyze,
    reset,
    clearError,
  }
}