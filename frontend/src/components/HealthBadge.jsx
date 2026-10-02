/**
 * Live `GET /health` indicator for the header.
 *
 * The five states map to: not checked yet, in flight, model loaded, model
 * failed to load, and "no answer from the API" - which on Render's free tier
 * usually means the service is asleep rather than broken.
 *
 * @param {{
 *   state: 'idle' | 'checking' | 'healthy' | 'degraded' | 'unreachable'
 *   message?: string | null
 *   version?: string | null
 *   classCount?: number | null
 *   onRetry: () => void
 * }}
 */
export default function HealthBadge({ state, message, version, classCount, onRetry }) {
  const view = {
    idle: { tone: 'pending', dot: 'pending', text: 'Checking API' },
    checking: { tone: 'pending', dot: 'pending', text: 'Checking API' },
    healthy: {
      tone: 'ok',
      dot: 'ok',
      text: classCount ? `API online · ${classCount} classes` : 'API online',
    },
    degraded: { tone: 'warn', dot: 'warn', text: 'Model not loaded' },
    unreachable: { tone: 'down', dot: 'down', text: 'API unreachable' },
  }[state] || { tone: 'pending', dot: 'pending', text: 'Checking API' }

  const title = [
    view.text,
    message,
    version ? `v${version}` : null,
    state === 'unreachable'
      ? 'The backend may be asleep on the free Render tier. Wait a moment and retry.'
      : null,
  ]
    .filter(Boolean)
    .join(' · ')

  return (
    <div className={`health health--${view.tone}`} title={title}>
      <span className={`health__dot health__dot--${view.dot}`} aria-hidden="true" />
      <span className="health__text">{view.text}</span>

      {state === 'unreachable' && (
        <button
          type="button"
          className="health__retry"
          onClick={onRetry}
          aria-label="Retry the API health check"
        >
          Retry
        </button>
      )}
    </div>
  )
}