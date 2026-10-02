import DisposalCard from './DisposalCard.jsx'
import TopPredictions from './TopPredictions.jsx'
import {
  confidenceLabel,
  confidenceTone,
  formatDimension,
  formatMilliseconds,
  formatPercent,
} from '../utils/format.js'

/**
 * Full result view for one `POST /predict` response.
 *
 * Every value comes from the backend payload:
 *   prediction, confidence (already a percentage), confidence_level,
 *   is_confident, message, top_predictions[], disposal{}, image{}, model{},
 *   processing_time_ms.
 *
 * @param {{
 *   result: import('../services/types.js').PredictionResponse
 *   previewUrl: string | null
 * }}
 */
export default function PredictionResult({ result, previewUrl }) {
  if (!result) return null

  const tone = confidenceTone(result.confidence_level, result.confidence)
  const percent = Math.min(100, Math.max(0, result.confidence || 0))
  const image = result.image || {}
  const model = result.model || {}

  return (
    <section className="result" aria-live="polite">
      <header className="result__header">
        <div>
          <p className="result__eyebrow">Result</p>
          <h2 className="result__category">
            <span aria-hidden="true">{result.disposal?.icon || '♻️'}</span>
            {result.prediction}
          </h2>
          <p className={`result__level result__level--${tone}`}>
            {confidenceLabel(tone)}
          </p>
        </div>

        <div className="result__confidence">
          <span className="result__confidence-value">
            {formatPercent(result.confidence)}
          </span>
          <span className="result__confidence-label">confidence</span>
        </div>
      </header>

      <div className={`meter meter--${tone}`}>
        <div
          className="meter__fill"
          style={{ width: `${percent}%` }}
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(percent)}
          aria-label={`Prediction confidence ${formatPercent(result.confidence)}`}
        />
      </div>

      {result.message && (
        <p className={`result__message${result.is_confident ? '' : ' result__message--warn'}`}>
          {result.message}
        </p>
      )}

      <div className="result__grid">
        {previewUrl && (
          <figure className="result__figure">
            <img src={previewUrl} alt={`Analysed waste item, classified as ${result.prediction}`} />
            <figcaption>Analysed image</figcaption>
          </figure>
        )}

        <div className="result__detail">
          <TopPredictions items={result.top_predictions} />
        </div>
      </div>

      <DisposalCard disposal={result.disposal} />

      <dl className="meta">
        <div className="meta__item">
          <dt>Image</dt>
          <dd>
            {formatDimension(image.width)} &times; {formatDimension(image.height)}
            {Number.isFinite(image.size_kb) ? ` · ${image.size_kb} KB` : ''}
          </dd>
        </div>
        <div className="meta__item">
          <dt>Inference time</dt>
          <dd>{formatMilliseconds(result.processing_time_ms)}</dd>
        </div>
        <div className="meta__item">
          <dt>Model</dt>
          <dd>
            {model.name || 'waste_classifier.onnx'}
            {Number.isFinite(model.num_classes)
              ? ` · ${model.num_classes} classes`
              : ''}
          </dd>
        </div>
      </dl>
    </section>
  )
}