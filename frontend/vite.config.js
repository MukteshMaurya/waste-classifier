import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * The backend runs with CORS restricted to an explicit FRONTEND_URL allow-list
 * (default http://localhost:5173), so the dev server must keep using port 5173.
 * `strictPort: true` makes Vite fail loudly instead of silently hopping to 5174,
 * which would be rejected by the backend's CORS allow-list.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendOrigin = (env.VITE_API_URL || 'http://127.0.0.1:8000').replace(
    /\/+$/,
    '',
  )

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      strictPort: true,
      proxy: {
        // Dev-only convenience so the browser can call same-origin /predict
        // even when FRONTEND_URL has not been updated yet. In production the
        // app talks to VITE_API_URL directly, so this proxy is never used.
        '/api': {
          target: backendOrigin,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
    preview: {
      host: true,
      port: 4173,
    },
    build: {
      outDir: 'dist',
      assetsDir: 'assets',
      sourcemap: false,
      // Keep the deployed bundle small and cache-friendly.
      chunkSizeWarningLimit: 900,
    },
  }
})