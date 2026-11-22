import { Component, lazy, Suspense } from 'react'
import { Route, Routes, Link } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import Layout from './components/Layout'
import ConnectScreen from './components/ConnectScreen'
import { useConnection } from './lib/connection'
import Overview from './pages/Overview'

const Projects = lazy(() => import('./pages/Projects'))
const NewProject = lazy(() => import('./pages/NewProject'))
const ProjectDetail = lazy(() => import('./pages/ProjectDetail'))
const ClipEditor = lazy(() => import('./pages/ClipEditor'))
const SettingsPage = lazy(() => import('./pages/Settings'))
const SystemPage = lazy(() => import('./pages/System'))

function PageFallback() {
  return <div className="page-loading" role="status" aria-label="Loading"><Loader2 className="spin" size={20} /></div>
}

class ErrorBoundary extends Component {
  state = { error: null }
  static getDerivedStateFromError(error) { return { error } }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="empty" role="alert">
        <h3>Something went wrong</h3>
        <p>{String(this.state.error?.message || this.state.error)}</p>
        <button className="btn btn-secondary btn-md" onClick={() => window.location.reload()}>Reload</button>
      </div>
    )
  }
}

function NotFound() {
  return (
    <div className="empty">
      <p className="eyebrow">404</p>
      <h3>Page not found</h3>
      <p>The page you’re looking for doesn’t exist.</p>
      <Link to="/" className="btn btn-primary btn-md">Back to overview</Link>
    </div>
  )
}

export default function App() {
  const { status } = useConnection()

  if (status === 'checking') {
    return <div className="boot" role="status"><Loader2 className="spin" size={22} /><span>Connecting to Studio…</span></div>
  }
  if (status !== 'ready') return <ConnectScreen />

  const wrap = (el) => <ErrorBoundary><Suspense fallback={<PageFallback />}>{el}</Suspense></ErrorBoundary>

  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={wrap(<Overview />)} />
        <Route path="projects" element={wrap(<Projects />)} />
        <Route path="new" element={wrap(<NewProject />)} />
        <Route path="projects/:jobId" element={wrap(<ProjectDetail />)} />
        <Route path="projects/:jobId/clips/:rank" element={wrap(<ClipEditor />)} />
        <Route path="settings" element={wrap(<SettingsPage />)} />
        <Route path="system" element={wrap(<SystemPage />)} />
        {/* Legacy dashboard links */}
        <Route path="job/:jobId" element={wrap(<ProjectDetail />)} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
