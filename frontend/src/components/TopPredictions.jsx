import { confidenceTone, formatPercent } from '../utils/format.js'

/**
 * Ranked alternatives from `PredictionResponse.top_predictions`.
 *
 * Each entry is `{ class, confidence }` - note the wire key is `class`, not
 * `name`, because the backend schema declares an alias.
 *
 * @param {{ items: import('../services/types.js').ClassPrediction[] }}
 */
export default function TopPredictions({ items }) {
  if (!items?.length) return null

  return (
    <section className="panel">
      <h3 className="panel__title">Top {items.length} predictions</h3>
      <p className="panel__subtitle">
        The classifier's full ranked output. Only the first entry is the
        reported result.
      </p>

      <ol className="ranking">
        {items.map((item, index) => {
          const tone = confidenceTone(null, item.confidence)
          return (
            <li className="ranking__row" key={item.class}>
              <span className="ranking__rank">{index + 1}</span>
              <span className="ranking__label">{item.class}</span>
              <span className="ranking__bar" aria-hidden="true">
                <span
                  className={`ranking__fill ranking__fill--${tone}`}
                  style={{ width: `${Math.min(100, Math.max(0, item.confidence))}%` }}
                />
              </span>
              <span className="ranking__value">{formatPercent(item.confidence)}</span>
            </li>
          )
        })}
      </ol>
    </section>
  )
}