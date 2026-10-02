import { useCallback, useEffect, useRef, useState } from 'react'
import CameraCapture from '../components/CameraCapture.jsx'
import CategoryGrid from '../components/CategoryGrid.jsx'
import ErrorBanner from '../components/ErrorBanner.jsx'
import HealthBadge from '../components/HealthBadge.jsx'
import ImageDropzone from '../components/ImageDropzone.jsx'
import ImagePreview from '../components/ImagePreview.jsx'
import PredictionResult from '../components/PredictionResult.jsx'
import { useCamera } from '../hooks/useCamera.js'
import { useCategories } from '../hooks/useCategories.js'
import { useHealth } from '../hooks/useHealth.js'
import { usePrediction } from '../hooks/usePrediction.js'
import { API_BASE_URL } from '../services/api.js'

/**
 * The single page of the application.
 *
 * Flow: choose or capture an image -> press Analyze -> POST /predict ->
 * render the response. There is no client-side router because the app has one
 * route; `vercel.json` still rewrites unknown paths to `index.html` so a
 * bookmarked or reloaded deep link still works.
 */
export default function HomePage() {
  const health = useHealth()
  const { classes, isLoading: categoriesLoading } = useCategories()
  const prediction = usePrediction()
  const camera = useCamera()

  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const resultRef = useRef(/** @type {HTMLDivElement | null} */(null))
  const shouldFocusResult = useRef(false)

  const {
    file,
    previewUrl,
    source,
    result,
    error,
    status,
    isBusy,
    selectFile,
    analyze,
    reset,
    clearError,
  } = prediction

  // Move focus to the result so keyboard and screen-reader users are told
  // that the analysis finished.
  useEffect(() => {
    if (!result || !shouldFocusResult.current) return
    shouldFocusResult.current = false
    resultRef.current?.focus()
  }, [result])

  const handleFile = useCallback(
    /** @param {File} nextFile */
    async (nextFile) => {
      await selectFile(nextFile, 'upload')
    },
    [selectFile],
  )

  const handleCapture = useCallback(
    /** @param {File} captured */
    async (captured) => {
      setIsCameraOpen(false)
      camera.stop()
      await selectFile(captured, 'camera')
    },
    [camera, selectFile],
  )

  const closeCamera = useCallback(() => {
    setIsCameraOpen(false)
    camera.stop()
  }, [camera])

  const handleAnalyze = useCallback(async () => {
    if (!file || isBusy) return
    shouldFocusResult.current = true
    await analyze()
  }, [analyze, file, isBusy])

  const handleReset = useCallback(() => {
    reset()
  }, [reset])

  return (
    <div className="page">
      <header className="topbar">
        <div className="topbar__brand">
          <span className="topbar__logo" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M7 19H4.8A1.8 1.8 0 0 1 3 17.2V8.8A1.8 1.8 0 0 1 4.8 7H7" />
              <path d="M10 4.5 12 2.5l2 2" />
              <path d="M12 2.5V14" />
              <path d="M7 12h10l-1.4 7H8.4L7 12Z" />
            </svg>
          </span>
          <div>
            <h1 className="topbar__title">AI Waste Classifier</h1>
            <p className="topbar__subtitle">
              Identify a waste item and get disposal guidance
            </p>
          </div>
        </div>

        <HealthBadge
          state={health.state}
          message={health.message}
          version={health.version}
          classCount={health.classCount}
          onRetry={health.check}
        />
      </header>

      <main className="content">
        {health.state === 'unreachable' && (
          <ErrorBanner
            tone="warning"
            title="The classification API is not responding"
            message={
              health.message ??
              `No answer from ${API_BASE_URL}.`
            }
            hint="On the free Render tier the service sleeps when idle, so the first request after a pause can take up to a minute. Wait a moment and retry."
            action={{ label: 'Retry now', onClick: health.check }}
          />
        )}

        {health.state === 'degraded' && (
          <ErrorBanner
            tone="warning"
            title="The model is not loaded on the server"
            message={health.message ?? 'The backend reported a degraded state.'}
            hint="Predictions will fail until the ONNX model loads. Check the Render service logs."
            action={{ label: 'Re-check', onClick: health.check }}
          />
        )}

        <div className="layout">
          <section className="panel panel--input">
            <h2 className="panel__title">Add a photo</h2>
            <p className="panel__subtitle">
              Upload an existing image, or take one with your camera.
            </p>

            {!file ? (
              <>
                <ImageDropzone onFile={handleFile} disabled={isBusy} />

                <button
                  type="button"
                  className="button button--ghost button--block"
                  onClick={() => setIsCameraOpen(true)}
                  disabled={isBusy || !camera.isSupported}
                  title={
                    camera.isSupported
                      ? undefined
                      : 'The camera API needs HTTPS (or localhost) and a supported browser.'
                  }
                >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M3 8.5A1.5 1.5 0 0 1 4.5 7h2.2l1.4-2h8l1.4 2h2.2A1.5 1.5 0 0 1 21 8.5v9A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5v-9Z" />
                    <circle cx="12" cy="13" r="3.4" />
                  </svg>
                  Use camera
                </button>

                {!camera.isSupported && (
                  <p className="panel__note">
                    The camera needs HTTPS (or localhost) and browser
                    permission. Uploading works everywhere.
                  </p>
                )}
              </>
            ) : (
              <>
                <ImagePreview
                  file={file}
                  previewUrl={previewUrl}
                  source={source}
                  onRemove={handleReset}
                  onReplace={handleReset}
                  disabled={isBusy}
                />

                <div className="actions">
                  <button
                    type="button"
                    className="button button--primary button--block"
                    onClick={() => void handleAnalyze()}
                    disabled={isBusy}
                  >
                    {status === 'loading' ? (
                      <>
                        <span className="spinner" aria-hidden="true" />
                        Analyzing&hellip;
                      </>
                    ) : result ? (
                      'Analyze again'
                    ) : (
                      <>
                        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                          <circle cx="11" cy="11" r="6.5" />
                          <path d="m16 16 4 4" />
                        </svg>
                        Analyze
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    className="button button--ghost button--block"
                    onClick={() => setIsCameraOpen(true)}
                    disabled={isBusy}
                  >
                    Retake with camera
                  </button>
                </div>
              </>
            )}

            {error && (
              <ErrorBanner
                title={
                  error.code === 'invalid_file'
                    ? 'That image cannot be used'
                    : 'Analysis failed'
                }
                message={error.message}
                hint={error.hint}
                onDismiss={clearError}
              />
            )}

            {status === 'loading' && (
              <p className="panel__note">
                Uploading one still image to the classifier. This usually takes
                a couple of seconds.
              </p>
            )}
          </section>

          <div
            className="results"
            ref={resultRef}
            tabIndex={-1}
            aria-live="polite"
          >
            {result ? (
              <PredictionResult result={result} previewUrl={previewUrl} />
            ) : (
              <section className="panel panel--placeholder">
                <h2 className="panel__title">No result yet</h2>
                <p className="panel__subtitle">
                  Add a photo and press Analyze. The prediction, confidence
                  score, the top three candidates and disposal guidance all
                  come straight from the backend.
                </p>
                <ol className="steps">
                  <li>Photograph or upload one waste item.</li>
                  <li>Prefer a plain background and good light.</li>
                  <li>Press Analyze and read the disposal advice.</li>
                </ol>
              </section>
            )}
          </div>
        </div>

        {!categoriesLoading && categories.length > 0 && (
          <CategoryGrid classes={classes} />
        )}
      </main>

      <footer className="footer">
        <p>
          API: <code>{API_BASE_URL}</code>
        </p>
        <p>
          Results are a suggestion from a 10-class image model. Always confirm
          with your local recycling rules.
        </p>
      </footer>

      <CameraCapture
        isOpen={isCameraOpen}
        isSupported={camera.isSupported}
        state={camera.state}
        error={camera.error}
        isMirrored={camera.isMirrored}
        videoRef={camera.videoRef}
        start={camera.start}
        switchCamera={camera.switchCamera}
        capture={camera.capture}
        stop={camera.stop}
        onCapture={handleCapture}
        onCancel={closeCamera}
        onDismissError={camera.dismissError}
      />
    </div>
  )
}