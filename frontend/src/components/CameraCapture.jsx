import { useEffect, useRef } from 'react'
import ErrorBanner from './ErrorBanner.jsx'

/**
 * Live camera modal.
 *
 * Guarantees, in order of importance:
 *  1. The stream is stopped on unmount, on cancel, and on capture.
 *  2. Nothing is uploaded while the preview is open. `onCapture` receives one
 *     still JPEG and that is the only network payload.
 *  3. Escape cancels, and focus moves into the dialog while it is open.
 *
 * @param {{
 *   isOpen: boolean,
 *   isSupported: boolean,
 *   state: import('../hooks/useCamera.js').CameraState,
 *   error: import('../services/errors.js').ApiError | null,
 *   isMirrored: boolean,
 *   videoRef: import('react').RefObject<HTMLVideoElement | null>,
 *   start: (options?: { facingMode?: 'environment' | 'user' }) => Promise<void>,
 *   switchCamera: () => Promise<void>,
 *   capture: () => Promise<File | null>,
 *   stop: () => void,
 *   onCapture: (file: File) => void | Promise<void>,
 *   onCancel: () => void,
 *   onDismissError: () => void,
 * }}
 */
export default function CameraCapture({
  isOpen,
  isSupported,
  state,
  error,
  isMirrored,
  videoRef,
  start,
  switchCamera,
  capture,
  stop,
  onCapture,
  onCancel,
  onDismissError,
}) {
  const dialogRef = useRef(/** @type {HTMLDivElement | null} */(null))
  const closeButtonRef = useRef(/** @type {HTMLButtonElement | null} */(null))

  // Opening the dialog is what actually acquires the stream.
  useEffect(() => {
    if (!isOpen) return
    void start()
    return () => stop()
  }, [isOpen, start, stop])

  useEffect(() => {
    if (!isOpen) return
    closeButtonRef.current?.focus()
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onCancel()
        return
      }
      if (event.key !== 'Tab') return

      // Minimal focus trap so tabbing cannot land behind the overlay.
      const focusable = dialogRef.current?.querySelectorAll(
        'button:not([disabled]), [href], input, video[controls]',
      )
      if (!focusable || focusable.length === 0) return

      const first = focusable[0]
      const last = focusable[focusable.length - 1]

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onCancel])

  if (!isOpen) return null

  const handleCapture = async () => {
    const file = await capture()
    if (!file) return
    await onCapture(file)
  }

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onCancel()
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="camera-title"
        ref={dialogRef}
      >
        <header className="modal__header">
          <h2 id="camera-title" className="modal__title">
            Capture a photo
          </h2>
          <button
            ref={closeButtonRef}
            type="button"
            className="icon-button"
            onClick={onCancel}
            aria-label="Cancel and close the camera"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </header>

        <div className="modal__body">
          <div className="camera">
            <video
              ref={videoRef}
              className={`camera__video${isMirrored ? ' camera__video--mirrored' : ''}`}
              playsInline
              muted
              autoPlay
              aria-label="Live camera preview"
            />

            {state !== 'active' && (
              <div className="camera__overlay">
                {state === 'starting' && (
                  <p className="camera__message">
                    <span className="spinner" aria-hidden="true" />
                    Starting the camera&hellip;
                  </p>
                )}

                {state === 'idle' && (
                  <div className="camera__message">
                    <button
                      type="button"
                      className="button button--ghost"
                      onClick={() => void start()}
                    >
                      Start camera
                    </button>
                  </div>
                )}

                {state === 'error' && (
                  <div className="camera__message">
                    <p>The camera is not available.</p>
                    <button
                      type="button"
                      className="button button--ghost"
                      onClick={() => void start()}
                    >
                      Try again
                    </button>
                  </div>
                )}
              </div>
            )}

            <div className="camera__guide" aria-hidden="true" />
          </div>

          {!isSupported && (
            <ErrorBanner
              tone="warning"
              title="Camera not available"
              message="This browser will not let this page use the camera."
              hint={
                window.isSecureContext
                  ? 'Upload an image instead, or open the site in a current browser.'
                  : 'The camera API requires HTTPS (or localhost). Upload an image instead.'
              }
            />
          )}

          {error && (
            <ErrorBanner
              tone="warning"
              title="Camera problem"
              message={error.message}
              hint={error.hint}
              action={{ label: 'Dismiss', onClick: onDismissError }}
            />
          )}

          <p className="camera__note">
            Nothing is uploaded while the camera is open. Only the single photo
            you capture is sent to the classifier.
          </p>
        </div>

        <footer className="modal__footer">
          <button type="button" className="button button--ghost" onClick={onCancel}>
            Cancel
          </button>

          {state === 'active' && (
            <button
              type="button"
              className="button button--ghost"
              onClick={() => void switchCamera()}
            >
              Switch camera
            </button>
          )}

          <button
            type="button"
            className="button button--primary"
            onClick={() => void handleCapture()}
            disabled={state !== 'active'}
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M4 8h3l1.5-2h7L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z" />
              <circle cx="12" cy="13.5" r="3.2" />
            </svg>
            {state === 'active' ? 'Capture photo' : 'Camera starting'}
          </button>
        </footer>
      </div>
    </div>
  )
}