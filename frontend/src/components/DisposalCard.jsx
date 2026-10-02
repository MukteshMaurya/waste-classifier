import { disposalTone } from '../utils/format.js'

/**
 * Recycling / disposal guidance, rendered only from
 * `PredictionResponse.disposal`. Nothing is hard-coded here: if the backend
 * omits a key, the corresponding row is simply not shown.
 *
 * @param {{ disposal: import('../services/types.js').DisposalInfo }}
 */
export default function DisposalCard({ disposal }) {
  if (!disposal) return null

  const tone = disposalTone(disposal.category)
  const tips = Array.isArray(disposal.tips) ? disposal.tips : []

  return (
    <section className={`disposal disposal--${tone}`}>
      <header className="disposal__header">
        <span className="disposal__icon" aria-hidden="true">
          {disposal.icon || '♻️'}
        </span>
        <div>
          <h3 className="disposal__title">How to dispose of it</h3>
          {disposal.category && (
            <p className="disposal__category">{disposal.category}</p>
          )}
        </div>
      </header>

      {disposal.instructions && (
        <p className="disposal__instructions">{disposal.instructions}</p>
      )}

      {disposal.bin_colour && (
        <p className="disposal__bin">
          <span className="disposal__bin-label">Bin / drop-off</span>
          <span className="disposal__bin-value">{disposal.bin_colour}</span>
        </p>
      )}

      {tips.length > 0 && (
        <>
          <h4 className="disposal__tips-title">Good to know</h4>
          <ul className="disposal__tips">
            {tips.map((tip) => (
              <li key={tip}>{tip}</li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}