/**
 * Inline, dismissible error surface used for API, camera and validation
 * problems. The message is always the backend's own wording when the backend
 * supplied one - nothing is invented here.
 *
 * @param {{
 *   tone?: 'error' | 'warning' | 'info'
 *   title?: string
 *   message: string
 *   hint?: string | null
 *   action?: { label: string, onClick: () => void } | null
 *   onDismiss?: () => void
 * }}
 */
export default function ErrorBanner({
  tone = 'error',
  title,
  message,
  hint,
  action,
  onDismiss,
}) {
  if (!message) return null

  return (
    <div className={`banner banner--${tone}`} role="alert" aria-live="assertive">
      <span className="banner__icon" aria-hidden="true">
        {tone === 'error' ? (
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="9" />
            <path d="M12 7.5v5M12 16.2v.1" />
          </svg>
        ) : tone === 'warning' ? (
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10.3 4.3 2.6 17.6A1.5 1.5 0 0 0 3.9 20h16.2a1.5 1.5 0 0 0 1.3-2.4L13.7 4.3a1.5 1.5 0 0 0-2.6 0Z" />
            <path d="M12 9.5v4M12 17v.1" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="9" />
            <path d="M12 11v5.5M12 7.8v.1" />
          </svg>
        )}
      </span>

      <div className="banner__content">
        {title && <p className="banner__title">{title}</p>}
        <p className="banner__message">{message}</p>
        {hint && <p className="banner__hint">{hint}</p>}

        {(action || onDismiss) && (
          <div className="banner__actions">
            {action && (
              <button type="button" className="button button--small" onClick={action.onClick}>
                {action.label}
              </button>
            )}
            {onDismiss && (
              <button type="button" className="button button--small button--quiet" onClick={onDismiss}>
                Dismiss
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}