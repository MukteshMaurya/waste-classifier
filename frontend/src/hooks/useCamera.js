import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../services/errors.js'

/**
 * Longest edge kept in a capture. The classifier resizes to 224x224 anyway,
 * and capping here keeps the JPEG comfortably under the backend's 5 MB limit
 * on high-resolution phone cameras.
 */
const CAPTURE_MAX_EDGE = 1600
const CAPTURE_QUALITY = 0.92

/**
 * @typedef {'idle' | 'starting' | 'active' | 'error'} CameraState
 */

/**
 * Wraps `navigator.mediaDevices.getUserMedia`.
 *
 * The live preview is a purely local `<video>` element - no frame is ever
 * uploaded while the camera is open. A single still JPEG is produced only
 * when `capture()` is called, and that is the only thing the caller sends to
 * `POST /predict`.
 *
 * @returns {{
 *   videoRef: import('react').RefObject<HTMLVideoElement | null>,
 *   state: CameraState,
 *   error: ApiError | null,
 *   isMirrored: boolean,
 *   isSupported: boolean,
 *   start: (facingMode?: 'environment' | 'user') => Promise<void>,
 *   capture: () => Promise<File | null>,
 *   switchCamera: () => Promise<void>,
 *   stop: () => void,
 *   dismissError: () => void,
 * }}
 */
export function useCamera() {
  const videoRef = useRef(/** @type {HTMLVideoElement | null} */(null))
  const streamRef = useRef(/** @type {MediaStream | null} */(null))
  const facingRef = useRef(/** @type {'environment' | 'user'} */('environment'))

  const [state, setState] = useState(/** @type {CameraState} */('idle'))
  const [error, setError] = useState(/** @type {ApiError | null} */(null))
  const [isMirrored, setIsMirrored] = useState(false)

  const isSupported =
    typeof navigator !== 'undefined' &&
    Boolean(navigator.mediaDevices?.getUserMedia) &&
    typeof window !== 'undefined' &&
    Boolean(window.isSecureContext)

  /** Stops every track and detaches the stream from the video element. */
  const stop = useCallback(() => {
    const stream = streamRef.current
    if (stream) {
      for (const track of stream.getTracks()) {
        try {
          track.stop()
        } catch {
          // A track that already ended is not an error worth surfacing.
        }
      }
    }
    streamRef.current = null

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsMirrored(false)
    setState('idle')
  }, [])

  // Belt and braces: releasing the camera when the component unmounts keeps
  // the browser's recording indicator from staying on.
  useEffect(() => stop, [stop])

  /**
   * @param {MediaStream} stream
   */
  const attach = useCallback((stream) => {
    streamRef.current = stream
    const video = videoRef.current
    if (video) {
      video.srcObject = stream
      const play = video.play()
      if (play?.catch) play.catch(() => undefined)
    }

    // Only mirror the preview when the user-facing camera is active.
    const track = stream.getVideoTracks()[0]
    const settings = track?.getSettings?.() ?? {}
    const facing = settings.facingMode
    setIsMirrored(facing === 'user')
  }, [])

  /**
   * @param {DOMException | Error} failure
   * @returns {ApiError}
   */
  const describeFailure = (failure) => {
    switch (failure?.name) {
      case 'NotAllowedError':
      case 'SecurityError':
        return new ApiError('Camera permission was denied.', {
          code: 'camera_denied',
          hint: 'Allow camera access in your browser settings, then try again.',
          cause: failure,
        })
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return new ApiError('No camera was found on this device.', {
          code: 'camera_missing',
          hint: 'Upload a photo from your gallery instead.',
          cause: failure,
        })
      case 'NotReadableError':
      case 'TrackStartError':
        return new ApiError('The camera is already in use by another application.', {
          code: 'camera_busy',
          hint: 'Close other apps or tabs using the camera and try again.',
          cause: failure,
        })
      case 'OverconstrainedError':
        return new ApiError('This camera cannot provide a usable video stream.', {
          code: 'camera_constraints',
          hint: 'Try the other camera, or upload a photo instead.',
          cause: failure,
        })
      default:
        return new ApiError('The camera could not be started.', {
          code: 'camera_unavailable',
          hint: 'Upload a photo from your gallery instead.',
          cause: failure,
        })
    }
  }

  /**
   * @param {{ facingMode?: 'environment' | 'user' }} [options]
   */
  const start = useCallback(
    async ({ facingMode = 'environment' } = {}) => {
      if (!isSupported) {
        setError(
          new ApiError('This browser cannot access the camera on this page.', {
            code: 'camera_unsupported',
            hint: window.isSecureContext
              ? 'Try a current version of Chrome, Edge, Firefox or Safari.'
              : 'The camera API requires HTTPS (or localhost).',
          }),
        )
        setState('error')
        return
      }

      setState('starting')
      setError(null)

      try {
        // Stop any previous session first, otherwise the old track keeps the
        // camera light on.
        const previous = streamRef.current
        if (previous) {
          for (const track of previous.getTracks()) track.stop()
          streamRef.current = null
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        })

        facingRef.current = facingMode
        attach(stream)
        setState('active')
      } catch (failure) {
        setState('error')
        setError(describeFailure(failure))
      }
    },
    [attach, isSupported],
  )

  /**
   * Grabs the current frame as a JPEG File.
   *
   * @returns {Promise<File | null>}
   */
  const capture = useCallback(async () => {
    const video = videoRef.current
    if (!video || !streamRef.current) {
      setError(
        new ApiError('The camera is not running, so no photo was taken.', {
          code: 'camera_inactive',
          hint: 'Start the camera and try again.',
        }),
      )
      return null
    }

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) {
      setError(
        new ApiError('The camera preview is not ready yet.', {
          code: 'camera_not_ready',
          hint: 'Wait a moment and capture again.',
        }),
      )
      return null
    }

    const scale = Math.min(
      1,
      CAPTURE_MAX_EDGE / Math.max(video.videoWidth, video.videoHeight),
    )
    const width = Math.round(video.videoWidth * scale)
    const height = Math.round(video.videoHeight * scale)

    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height

    const context = canvas.getContext('2d')
    if (!context) {
      setError(
        new ApiError('This browser cannot read the camera frame.', {
          code: 'canvas_unavailable',
        }),
      )
      return null
    }

    // Mirror the pixels when previewing the user-facing camera so the photo
    // matches what the user saw.
    if (isMirrored) {
      context.translate(width, 0)
      context.scale(-1, 1)
    }
    context.drawImage(video, 0, 0, width, height)

    const blob = await new Promise((resolve) => {
      canvas.toBlob(resolve, 'image/jpeg', CAPTURE_QUALITY)
    })

    if (!blob) {
      setError(
        new ApiError('The photo could not be encoded.', {
          code: 'capture_failed',
          hint: 'Capture again, or upload an image instead.',
        }),
      )
      return null
    }

    const stamp = new Date().toISOString().replace(/[:.]/g, '-')
    return new File([blob], `capture-${stamp}.jpg`, {
      type: 'image/jpeg',
      lastModified: Date.now(),
    })
  }, [isMirrored])

  const switchCamera = useCallback(
    async () => {
      await start({ facingMode: facingRef.current === 'environment' ? 'user' : 'environment' })
    },
    [start],
  )

  const dismissError = useCallback(() => setError(null), [])

  return {
    videoRef,
    state,
    error,
    isMirrored,
    isSupported,
    start,
    capture,
    switchCamera,
    stop,
    dismissError,
  }
}