import { useEffect, useState } from 'react'
import { NavLink, useLocation, useNavigate, useOutlet } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import {
  LayoutDashboard, FolderKanban, Plus, Settings, Activity, Moon, Sun, Menu, X, CodeXml, Wifi, WifiOff,
} from 'lucide-react'
import { useConnection } from '../lib/connection'
import { storage } from '../lib/storage'
import { useHotkeys } from '../lib/hooks'
import { Button, IconButton } from './ui'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/projects', label: 'Projects', icon: FolderKanban },
  { to: '/system', label: 'System', icon: Activity },
  { to: '/settings', label: 'Settings', icon: Settings },
]

function useTheme() {
  const [theme, setTheme] = useState(() => storage.get('osc.theme', 'system'))
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    storage.set('osc.theme', theme)
  }, [theme])
  const resolved = theme === 'system'
    ? (window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
    : theme
  return [resolved, () => setTheme(resolved === 'dark' ? 'light' : 'dark')]
}

export function Logo({ compact = false }) {
  return (
    <span className="logo">
      <span className="logo-mark" aria-hidden>
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none">
          <path d="M8 5.5v13l10-6.5-10-6.5Z" fill="currentColor" />
          <path d="M4 4v16" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" opacity=".55" />
        </svg>
      </span>
      {!compact && <span className="logo-text">Clipping<span>Studio</span></span>}
    </span>
  )
}

export default function Layout() {
  const [theme, toggleTheme] = useTheme()
  const [mobileOpen, setMobileOpen] = useState(false)
  const { status, base } = useConnection()
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => { setMobileOpen(false) }, [location.pathname])
  useHotkeys({ n: () => navigate('/new') })

  const host = (() => {
    if (!base) return 'This server'
    try { return new URL(base).host } catch { return base }
  })()

  const sidebar = (
    <>
      <div className="sidebar-top">
        <NavLink to="/" className="sidebar-brand" aria-label="Clipping Studio home"><Logo /></NavLink>
        <Button variant="primary" icon={Plus} className="sidebar-cta" onClick={() => navigate('/new')}>
          New project
          <kbd className="kbd kbd-on-accent" aria-hidden>N</kbd>
        </Button>
        <nav className="sidebar-nav" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? 'is-active' : ''}`}>
              {({ isActive }) => (
                <>
                  {isActive && <motion.span layoutId="nav-active" className="nav-active" transition={{ type: 'spring', stiffness: 500, damping: 40 }} />}
                  <Icon size={17} aria-hidden />
                  <span>{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="sidebar-bottom">
        <NavLink to="/settings?tab=connection" className="conn-pill" title={base || 'Same-origin backend'}>
          {status === 'ready' ? <Wifi size={14} aria-hidden /> : <WifiOff size={14} aria-hidden />}
          <span className={`conn-dot conn-${status}`} />
          <span className="conn-host">{host}</span>
        </NavLink>
        <div className="sidebar-row">
          <IconButton icon={theme === 'dark' ? Sun : Moon} label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} onClick={toggleTheme} />
          <a className="icon-btn icon-btn-md" href="https://github.com/NaufalRizqullah/opensource-clipping" target="_blank" rel="noopener noreferrer" aria-label="Source code on GitHub" title="Source code">
            <CodeXml size={18} aria-hidden />
          </a>
        </div>
      </div>
    </>
  )

  return (
    <div className="shell">
      <a href="#main" className="skip-link">Skip to content</a>
      <aside className="sidebar" aria-label="Sidebar">{sidebar}</aside>

      <header className="topbar">
        <NavLink to="/" aria-label="Home"><Logo /></NavLink>
        <div className="topbar-actions">
          <IconButton icon={Plus} label="New project" onClick={() => navigate('/new')} />
          <IconButton icon={mobileOpen ? X : Menu} label={mobileOpen ? 'Close menu' : 'Open menu'} onClick={() => setMobileOpen((v) => !v)} aria-expanded={mobileOpen} />
        </div>
      </header>

      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div className="drawer-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setMobileOpen(false)} />
            <motion.aside
              className="sidebar sidebar-drawer"
              initial={{ x: '-100%' }} animate={{ x: 0 }} exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 380, damping: 38 }}
              aria-label="Mobile menu"
            >
              {sidebar}
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <main id="main" className="main" tabIndex={-1}>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={location.pathname}
            className="page"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6, transition: { duration: 0.12 } }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            <FrozenOutlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}

/** Keeps the exiting page's content stable during the route transition. */
function FrozenOutlet() {
  const outlet = useOutlet()
  const [frozen] = useState(outlet)
  return frozen
}

export function PageHeader({ eyebrow, title, description, actions, back }) {
  return (
    <header className="page-header">
      <div className="page-header-text">
        {back}
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="page-desc">{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  )
}
