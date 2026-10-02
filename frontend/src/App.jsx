import HomePage from './pages/HomePage.jsx'
import './App.css'

/**
 * Root of the application. A single page today - no router is mounted, so a
 * direct visit to any path still renders the app (see `vercel.json`).
 */
export default function App() {
  return <HomePage />
}