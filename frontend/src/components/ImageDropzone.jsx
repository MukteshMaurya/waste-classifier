import { useCallback, useId, useRef, useState } from 'react'
import {
  ACCEPTED_LABEL,
  FILE_INPUT_ACCEPT,
  MAX_FILE_LABEL,
} from '../utils/files.js'

/**
 * Drag-and-drop + file-picker zone.
 *
 * Client-side validation is a convenience only - `inspectImageFile` inside
 * `usePrediction.selectFile` is what ultimately gates the upload, so a file
 * dropped from outside the browser cannot bypass the checks.
 *
 * @param {{
 *   onFile: (file: File) => void,
 *   disabled?: boolean,
 * }}
 */
export default function ImageDropzone({ onFile, disabled = false }) {
  const inputRef = useRef(/** @type {HTMLInputElement | null} */(null))
  const [isDragging, setIsDragging] = useState(false)
  const inputId = useId()
  const hintId = `${inputId}-hint`
  // Guards against a stray dragleave from a child element flipping the state.
  const dragDepth = useRef(0)

  const openPicker = useCallback(() => {
    if (!disabled) inputRef.current?.click()
  }, [disabled])

  const handleDragEnter = useCallback(
    (event) => {
      event.preventDefault()
      if (disabled) return
      dragDepth.current += 1
      setIsDragging(true)
    },
    [disabled],
  )

  const handleDragOver = useCallback(
    (event) => {
      event.preventDefault()
      if (disabled) return
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'
    },
    [disabled],
  )

  const handleDragLeave = useCallback(
    (event) => {
      event.preventDefault()
      dragDepth.current = Math.max(0, dragDepth.current - 1)
      if (dragDepth.current === 0) setIsDragging(false)
    },
    [],
  )

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault()
      dragDepth.current = 0
      setIsDragging(false)
      if (disabled) return

      const file = event.dataTransfer?.files?.[0]
      if (file) onFile(file)
    },
    [disabled, onFile],
  )

  const handleInputChange = useCallback(
    (event) => {
      const file = event.target.files?.[0]
      // Reset so re-picking the same file still fires a change event.
      event.target.value = ''
      if (file && !disabled) onFile(file)
    },
    [disabled, onFile],
  )

  return (
    <div
      className={`dropzone${isDragging ? ' dropzone--active' : ''}`}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        id={inputId}
        className="visually-hidden"
        type="file"
        accept={FILE_INPUT_ACCEPT}
        aria-describedby={hintId}
        disabled={disabled}
        onChange={handleInputChange}
      />

      <div className="dropzone__inner">
        <span className="dropzone__icon" aria-hidden="true">
          <svg
            viewBox="0 0 24 24"
            width="34"
            height="34"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 16V4" />
            <path d="m7 9 5-5 5 5" />
            <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
          </svg>
        </span>

        <p className="dropzone__title">
          {isDragging ? 'Drop your image here' : 'Drag an image here'}
        </p>
        <p className="dropzone__subtitle">
          {ACCEPTED_LABEL} &middot; up to {MAX_FILE_LABEL}
        </p>

        <button
          type="button"
          className="button button--ghost"
          onClick={openPicker}
          disabled={disabled}
          aria-describedby={hintId}
        >
          Choose a file
        </button>

        <p id={hintId} className="dropzone__hint">
          Take a clear photo of a single item on a plain background for the
          best result.
        </p>
      </div>
    </div>
  )
}