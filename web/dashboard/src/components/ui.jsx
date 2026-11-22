import { forwardRef, useEffect, useId, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Check, Copy, Loader2, X } from 'lucide-react'
import { STATUS_META } from '../lib/format'

export const Button = forwardRef(function Button(
  { variant = 'secondary', size = 'md', loading = false, icon: Icon, iconRight: IconRight, children, className = '', disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      className={`btn btn-${variant} btn-${size} ${className}`}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...props}
    >
      {loading ? <Loader2 size={size === 'sm' ? 14 : 16} className="spin" aria-hidden /> : Icon ? <Icon size={size === 'sm' ? 14 : 16} aria-hidden /> : null}
      {children && <span>{children}</span>}
      {IconRight && !loading && <IconRight size={size === 'sm' ? 14 : 16} aria-hidden />}
    </button>
  )
})

export function IconButton({ icon: Icon, label, size = 'md', className = '', ...props }) {
  return (
    <button className={`icon-btn icon-btn-${size} ${className}`} aria-label={label} title={label} {...props}>
      <Icon size={size === 'sm' ? 14 : 18} aria-hidden />
    </button>
  )
}

export function Card({ as: Tag = 'section', className = '', children, ...props }) {
  return <Tag className={`card ${className}`} {...props}>{children}</Tag>
}

export function CardHeader({ title, description, icon: Icon, actions }) {
  return (
    <header className="card-header">
      <div className="card-header-text">
        <h3 className="card-title">{Icon && <Icon size={16} aria-hidden />}{title}</h3>
        {description && <p className="card-desc">{description}</p>}
      </div>
      {actions && <div className="card-actions">{actions}</div>}
    </header>
  )
}

export function Badge({ tone = 'neutral', children, dot = false, className = '' }) {
  return <span className={`badge badge-${tone} ${className}`}>{dot && <span className="badge-dot" />}{children}</span>
}

export function StatusBadge({ status }) {
  const meta = STATUS_META[status] || { label: status, tone: 'neutral' }
  const live = meta.tone === 'info' || status === 'queued'
  return <Badge tone={meta.tone} dot className={live ? 'badge-live' : ''}>{meta.label}</Badge>
}

export function ProgressBar({ value = 0, tone = 'accent', indeterminate = false, label }) {
  return (
    <div
      className={`progress progress-${tone} ${indeterminate ? 'progress-indeterminate' : ''}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(value)}
      aria-label={label}
    >
      <motion.div
        className="progress-fill"
        initial={false}
        animate={{ width: indeterminate ? '35%' : `${Math.max(0, Math.min(100, value))}%` }}
        transition={{ type: 'spring', stiffness: 90, damping: 22 }}
      />
    </div>
  )
}

export function Toggle({ checked, onChange, label, description, disabled, id }) {
  const autoId = useId()
  const inputId = id || autoId
  return (
    <label className={`toggle-row ${disabled ? 'is-disabled' : ''}`} htmlFor={inputId}>
      <span className="toggle-text">
        <span className="toggle-label">{label}</span>
        {description && <span className="toggle-desc">{description}</span>}
      </span>
      <span className="switch">
        <input id={inputId} type="checkbox" role="switch" checked={!!checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
        <span className="switch-track"><span className="switch-thumb" /></span>
      </span>
    </label>
  )
}

export function Segmented({ value, onChange, options, label, size = 'md', layoutId }) {
  const fallbackId = useId()
  return (
    <div className={`segmented segmented-${size}`} role="radiogroup" aria-label={label}>
      {options.map((opt) => {
        const active = opt.value === value
        const Icon = opt.icon
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            className={`segmented-item ${active ? 'is-active' : ''}`}
            onClick={() => onChange(opt.value)}
          >
            {active && (
              <motion.span layoutId={layoutId || fallbackId} className="segmented-pill" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />
            )}
            <span className="segmented-label">{Icon && <Icon size={14} aria-hidden />}{opt.label}</span>
          </button>
        )
      })}
    </div>
  )
}

export function Field({ label, hint, error, children, htmlFor, trailing }) {
  return (
    <div className={`field ${error ? 'has-error' : ''}`}>
      {(label || trailing) && (
        <div className="field-top">
          {label && <label className="field-label" htmlFor={htmlFor}>{label}</label>}
          {trailing}
        </div>
      )}
      {children}
      {error ? <p className="field-error" role="alert">{error}</p> : hint ? <p className="field-hint">{hint}</p> : null}
    </div>
  )
}

export const Input = forwardRef(function Input({ className = '', icon: Icon, ...props }, ref) {
  if (!Icon) return <input ref={ref} className={`input ${className}`} {...props} />
  return (
    <div className="input-wrap">
      <Icon size={16} className="input-icon" aria-hidden />
      <input ref={ref} className={`input has-icon ${className}`} {...props} />
    </div>
  )
})

export function Select({ className = '', children, ...props }) {
  return <select className={`input select ${className}`} {...props}>{children}</select>
}

export function Textarea({ className = '', ...props }) {
  return <textarea className={`input textarea ${className}`} {...props} />
}

export function Skeleton({ className = '', style }) {
  return <div className={`skeleton ${className}`} style={style} aria-hidden />
}

export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <motion.div className="empty" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      {Icon && <div className="empty-icon"><Icon size={22} aria-hidden /></div>}
      <h3>{title}</h3>
      {description && <p>{description}</p>}
      {action}
    </motion.div>
  )
}

export function CopyButton({ value, label = 'Copy', size = 'sm', variant = 'ghost', onCopied }) {
  const [copied, setCopied] = useState(false)
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(String(value ?? ''))
    } catch {
      const ta = document.createElement('textarea')
      ta.value = String(value ?? '')
      ta.setAttribute('readonly', '')
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      ta.remove()
    }
    setCopied(true)
    onCopied?.()
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setCopied(false), 1600)
  }
  return (
    <Button type="button" size={size} variant={variant} icon={copied ? Check : Copy} onClick={copy} disabled={!value} aria-live="polite">
      {label && (copied ? 'Copied' : label)}
    </Button>
  )
}

export function Modal({ open, onClose, title, description, children, footer, size = 'md' }) {
  const panelRef = useRef(null)
  useEffect(() => {
    if (!open) return
    const prev = document.activeElement
    const onKey = (e) => { if (e.key === 'Escape') onClose?.() }
    window.addEventListener('keydown', onKey)
    const t = setTimeout(() => panelRef.current?.querySelector('[data-autofocus], button, input, textarea, select')?.focus(), 30)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      clearTimeout(t)
      document.body.style.overflow = ''
      prev?.focus?.()
    }
  }, [open, onClose])

  return (
    <AnimatePresence>
      {open && (
        <motion.div className="modal-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={(e) => e.target === e.currentTarget && onClose?.()}>
          <motion.div
            ref={panelRef}
            className={`modal modal-${size}`}
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
            initial={{ opacity: 0, y: 24, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98, transition: { duration: 0.15 } }}
            transition={{ type: 'spring', stiffness: 380, damping: 32 }}
          >
            <header className="modal-header">
              <div>
                <h2 id="modal-title">{title}</h2>
                {description && <p>{description}</p>}
              </div>
              <IconButton icon={X} label="Close" size="sm" onClick={onClose} />
            </header>
            {children && <div className="modal-body">{children}</div>}
            {footer && <footer className="modal-footer">{footer}</footer>}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export function ConfirmModal({ open, onClose, onConfirm, title, description, confirmLabel = 'Confirm', tone = 'danger', loading }) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      description={description}
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant={tone === 'danger' ? 'danger' : 'primary'} onClick={onConfirm} loading={loading} data-autofocus>{confirmLabel}</Button>
        </>
      }
    />
  )
}

export function ScoreRing({ score, size = 40 }) {
  if (score == null) return null
  const value = Math.max(0, Math.min(100, Number(score) > 10 ? Number(score) : Number(score) * 10))
  const r = (size - 6) / 2
  const c = 2 * Math.PI * r
  const tone = value >= 80 ? 'var(--success)' : value >= 60 ? 'var(--accent)' : 'var(--warning)'
  return (
    <div className="score-ring" style={{ width: size, height: size }} title={`Viral score ${score}`} aria-label={`Viral score ${score}`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--ring-track)" strokeWidth="3" fill="none" />
        <motion.circle
          cx={size / 2} cy={size / 2} r={r} stroke={tone} strokeWidth="3" fill="none" strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - value / 100) }}
          transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <span>{Math.round(Number(score))}</span>
    </div>
  )
}

export function Kbd({ children }) {
  return <kbd className="kbd">{children}</kbd>
}
