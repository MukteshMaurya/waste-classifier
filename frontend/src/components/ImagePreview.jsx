import { formatBytes } from '../utils/files.js'

/**
 * Preview of the currently selected image, with the controls that replace or
 * clear it. The preview is a local blob URL - nothing is uploaded until the
 * user presses Analyze.
 *
 * @param {{
 *   file: File
 *   previewUrl: string | null
 *   source: 'upload' | 'camera' | null
 *   onRemove: () => void
 *   onReplace: () => void
 *   disabled?: boolean
 * }}
 */
export default function ImagePreview({
  file,
  previewUrl,
  source,
  onRemove,
  onReplace,
  disabled = false,
}) {
  if (!file) return null

  return (
    <div className="preview">
      <div className="preview__frame">
        {previewUrl ? (
          <img
            className="preview__image"
            src={previewUrl}
            alt="Selected waste item, ready to analyse"
          />
        ) : (
          <div className="preview__placeholder">No preview available</div>
        )}

        <span className="preview__badge">
          {source === 'camera' ? 'Camera capture' : 'Uploaded file'}
        </span>
      </div>

      <div className="preview__meta">
        <p className="preview__name" title={file.name}>
          {file.name}
        </p>
        <p className="preview__facts">
          {formatBytes(file.size)}
          {file.type ? ` · ${file.type.replace('image/', '').toUpperCase()}` : ''}
        </p>

        <div className="preview__actions">
          <button
            type="button"
            className="button button--small button--ghost"
            onClick={onReplace}
            disabled={disabled}
          >
            Replace
          </button>
          <button
            type="button"
            className="button button--small button--danger"
            onClick={onRemove}
            disabled={disabled}
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M5 7h14M10 7V5.5A1.5 1.5 0 0 1 11.5 4h1A1.5 1.5 0 0 1 14 5.5V7M6.5 7l.8 12a1.5 1.5 0 0 0 1.5 1.4h6.4a1.5 1.5 0 0 0 1.5-1.4l.8-12" />
            </svg>
            Remove
          </button>
        </div>
      </div>
    </div>
  )
}