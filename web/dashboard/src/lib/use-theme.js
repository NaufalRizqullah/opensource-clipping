import { useState, useEffect, useCallback } from 'react'

const STORAGE_KEY = 'osc-theme'

// Initial value reflects what index.html already applied before paint.
function getInitialTheme() {
  if (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) {
    return 'dark'
  }
  return 'light'
}

/**
 * Theme state lifted to the app shell. Call once and pass `theme` + `toggle`
 * down so the desktop sidebar and mobile drawer stay in sync.
 */
export function useTheme() {
  const [theme, setTheme] = useState(getInitialTheme)

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    try {
      localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* storage unavailable (private mode), class still applied */
    }
  }, [theme])

  const toggle = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }, [])

  return { theme, setTheme, toggle }
}
