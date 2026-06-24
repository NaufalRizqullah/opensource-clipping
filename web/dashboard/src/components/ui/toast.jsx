import { createContext, useContext, useState, useCallback, useRef } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

const ToastContext = createContext(null)

const BORDER = {
  success: 'border-l-success',
  error: 'border-l-destructive',
  info: 'border-l-primary',
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])
  const idRef = useRef(0)

  const dismiss = useCallback((id) => {
    setToasts((list) => list.filter((t) => t.id !== id))
  }, [])

  const toast = useCallback((opts) => {
    const id = ++idRef.current
    const t = {
      id,
      title: typeof opts === 'string' ? opts : opts.title,
      description: opts.description,
      variant: opts.variant || 'info',
      duration: opts.duration ?? 4000,
    }
    setToasts((list) => [...list, t])
    if (t.duration > 0) setTimeout(() => dismiss(id), t.duration)
    return id
  }, [dismiss])

  return (
    <ToastContext.Provider value={{ toast, dismiss }}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[100] flex w-[calc(100vw-2rem)] max-w-sm flex-col gap-2.5">
        {toasts.map((t) => (
          <div
            key={t.id}
            role="status"
            className={cn(
              'animate-in-up pointer-events-auto flex items-start gap-3 rounded-xl border border-l-4 bg-card p-3.5 shadow-[0_10px_40px_rgba(20,19,26,0.18)]',
              BORDER[t.variant] || BORDER.info
            )}
          >
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold leading-snug">{t.title}</div>
              {t.description && (
                <div className="mt-0.5 text-xs text-muted-foreground">{t.description}</div>
              )}
            </div>
            <button
              type="button"
              onClick={() => dismiss(t.id)}
              aria-label="Dismiss"
              className="-m-1 rounded-md p-1 text-muted-foreground transition-colors hover:text-foreground"
            >
              <X className="size-4" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within a ToastProvider')
  return ctx
}
