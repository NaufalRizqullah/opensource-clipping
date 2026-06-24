import { useState, useEffect } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Menu, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/lib/use-theme'
import Dashboard from './pages/Dashboard'
import NewJob from './pages/NewJob'
import JobDetail from './pages/JobDetail'
import Settings from './pages/Settings'
import Analytics from './pages/Analytics'

const NAV = [
  { to: '/', end: true, label: 'Dashboard' },
  { to: '/new', end: false, label: 'New Job' },
  { to: '/analytics', end: false, label: 'Analytics' },
  { to: '/settings', end: false, label: 'Settings' },
]

function Brand() {
  return (
    <div className="leading-tight">
      <div className="font-display text-base font-bold tracking-tight">TrendSplice</div>
      <div className="text-[11px] text-muted-foreground">Turn long content into timely moments.</div>
    </div>
  )
}

function NavItems({ onNavigate }) {
  return NAV.map(({ to, end, label }) => (
    <NavLink
      key={to}
      to={to}
      end={end}
      onClick={onNavigate}
      className={({ isActive }) =>
        cn(
          'rounded-lg px-3 py-2.5 text-sm font-semibold transition-colors',
          isActive
            ? 'bg-accent text-accent-foreground'
            : 'text-muted-foreground hover:bg-secondary/70 hover:text-foreground'
        )
      }
    >
      {label}
    </NavLink>
  ))
}

function ThemeToggle({ theme, onToggle, className }) {
  const isDark = theme === 'dark'
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className={cn(
        'rounded-lg px-2.5 py-1.5 text-xs font-semibold text-muted-foreground transition-colors hover:bg-secondary/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background',
        className
      )}
    >
      {isDark ? 'Light' : 'Dark'}
    </button>
  )
}

function App() {
  const { theme, toggle } = useTheme()
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()

  // Close drawer on route change.
  useEffect(() => { setMenuOpen(false) }, [location.pathname])

  // Escape to close + lock body scroll while the drawer is open.
  useEffect(() => {
    if (!menuOpen) return
    const onKey = (e) => { if (e.key === 'Escape') setMenuOpen(false) }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [menuOpen])

  return (
    <div className="min-h-screen md:grid md:grid-cols-[264px_1fr]">
      {/* Sidebar (desktop) */}
      <aside className="sticky top-0 hidden h-screen flex-col border-r bg-card/70 backdrop-blur md:flex">
        <div className="px-6 py-6">
          <Brand />
        </div>
        <nav className="flex flex-1 flex-col gap-1 px-3 py-2">
          <NavItems />
        </nav>
        <div className="flex items-center justify-between gap-2 border-t px-6 py-5">
          <span className="text-[11px] text-muted-foreground">TrendSplice</span>
          <ThemeToggle theme={theme} onToggle={toggle} className="-mr-1" />
        </div>
      </aside>

      {/* Top bar (mobile) */}
      <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b bg-card/80 px-4 py-3 backdrop-blur md:hidden">
        <Brand />
        <div className="flex items-center gap-1">
          <ThemeToggle theme={theme} onToggle={toggle} />
          <button
            type="button"
            onClick={() => setMenuOpen(true)}
            aria-label="Open menu"
            aria-expanded={menuOpen}
            className="inline-grid size-9 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            <Menu className="size-5" />
          </button>
        </div>
      </header>

      {/* Mobile drawer */}
      <div className={cn('fixed inset-0 z-50 md:hidden', !menuOpen && 'pointer-events-none')}>
        <div
          onClick={() => setMenuOpen(false)}
          className={cn(
            'absolute inset-0 bg-foreground/40 backdrop-blur-sm transition-opacity duration-300',
            menuOpen ? 'opacity-100' : 'opacity-0'
          )}
        />
        <aside
          className={cn(
            'absolute left-0 top-0 flex h-full w-72 max-w-[82%] flex-col border-r bg-card shadow-2xl transition-transform duration-300 ease-out',
            menuOpen ? 'translate-x-0' : '-translate-x-full'
          )}
          role="dialog"
          aria-modal="true"
          aria-label="Navigation"
        >
          <div className="flex items-center justify-between px-5 py-5">
            <Brand />
            <button
              type="button"
              onClick={() => setMenuOpen(false)}
              aria-label="Close menu"
              className="inline-grid size-9 shrink-0 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            >
              <X className="size-5" />
            </button>
          </div>
          <nav className="flex flex-1 flex-col gap-1 px-3 py-2">
            <NavItems onNavigate={() => setMenuOpen(false)} />
          </nav>
          <div className="flex items-center justify-between gap-2 border-t px-5 py-4">
            <span className="text-[11px] text-muted-foreground">TrendSplice</span>
            <ThemeToggle theme={theme} onToggle={toggle} className="-mr-1" />
          </div>
        </aside>
      </div>

      {/* Main content */}
      <main className="min-w-0">
        <div className="mx-auto w-full max-w-6xl px-5 py-8 sm:px-8 md:px-10 md:py-12">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/new" element={<NewJob />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/job/:jobId" element={<JobDetail />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App
