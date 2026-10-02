/**
 * Typed description of the backend contract.
 *
 * Every field below was read from `backend/app/schemas.py` and
 * `backend/app/api/routes.py`. The wire format is produced by FastAPI with
 * `response_model_by_alias=True`, so `ClassPrediction.name` is serialised
 * under its alias `class`.
 *
 * @typedef {'High' | 'Moderate' | 'Low'} ConfidenceLevel
 *
 * @typedef {object} ClassPrediction
 * @property {string} class          Waste category name (alias of `name`).
 * @property {number} confidence     Percentage 0-100, rounded to 2 decimals.
 *
 * @typedef {object} DisposalInfo
 * @property {string} category      e.g. "Recyclable", "Compostable".
 * @property {string} instructions  Plain-language disposal guidance.
 * @property {string} icon          Emoji.
 * @property {string} bin_colour    Typical bin / drop-off point.
 * @property {string[]} tips        Extra practical notes.
 *
 * @typedef {object} PredictionImage
 * @property {number} width         JSON float.
 * @property {number} height        JSON float.
 * @property {number} size_kb       Bytes / 1024, 1 decimal.
 *
 * @typedef {object} PredictionModel
 * @property {string} name
 * @property {number} num_classes
 * @property {string | null} architecture
 *
 * @typedef {object} PredictionResponse
 * @property {boolean} success
 * @property {string} prediction
 * @property {number} confidence           Percentage 0-100 (already scaled).
 * @property {ConfidenceLevel} confidence_level
 * @property {boolean} is_confident
 * @property {string | null} message       Advisory note, not an error.
 * @property {ClassPrediction[]} top_predictions
 * @property {DisposalInfo} disposal
 * @property {PredictionImage} image
 * @property {PredictionModel} model
 * @property {number} processing_time_ms
 *
 * @typedef {object} HealthResponse
 * @property {'healthy' | 'degraded'} status
 * @property {boolean} model_loaded
 * @property {number} classes
 * @property {string | null} error
 * @property {string} version
 *
 * @typedef {object} ClassInfo
 * @property {string} name
 * @property {number} index
 * @property {string} category
 * @property {string} icon
 * @property {string} bin_colour
 * @property {string} instructions
 *
 * @typedef {object} CategoriesResponse
 * @property {number} count
 * @property {ClassInfo[]} classes
 *
 * @typedef {object} BackendErrorBody
 * @property {false} success
 * @property {string} error
 * @property {string} message
 */

export {}