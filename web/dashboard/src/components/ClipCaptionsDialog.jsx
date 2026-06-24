import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/components/ui/toast'

function CopyRow({ label, value, multiline }) {
  const { toast } = useToast()
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      toast({ variant: 'success', title: `${label} copied to clipboard` })
    } catch {
      toast({ variant: 'error', title: 'Copy failed', description: 'Clipboard is unavailable in this context.' })
    }
  }
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</span>
        <button
          type="button"
          onClick={copy}
          className="text-xs font-semibold text-primary transition-colors hover:text-primary/80"
        >
          Copy
        </button>
      </div>
      <div className={cn('rounded-lg border bg-secondary/40 px-3 py-2 text-sm', multiline && 'whitespace-pre-wrap')}>
        {value}
      </div>
    </div>
  )
}

function Section({ title, children }) {
  return (
    <section className="space-y-3">
      <div className="text-sm font-bold">{title}</div>
      {children}
    </section>
  )
}

/** True when a clip carries any ready-to-post social copy. */
export function hasCaptions(clip) {
  const m = clip?.metadata || {}
  return Boolean(
    m.youtube_title_final || m.youtube_description_final ||
    (Array.isArray(m.youtube_tags_final) && m.youtube_tags_final.length) ||
    m.tiktok_caption_final || m.hastag
  )
}

export function ClipCaptionsDialog({ clip, open, onClose }) {
  useEffect(() => {
    if (!open) return
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [open, onClose])

  if (!open || !clip) return null

  const m = clip.metadata || {}
  const tags = Array.isArray(m.youtube_tags_final) ? m.youtube_tags_final : []
  const ytTitle = m.youtube_title_final || clip.title || clip.title_en
  const ytDesc = m.youtube_description_final
  const tiktok = m.tiktok_caption_final
  const hashtags = m.hastag
  const reason = m.alasan

  // Portal to <body> so a transformed ancestor (e.g. the page's animate-in-up
  // wrapper) can't become the containing block for our fixed overlay.
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-end justify-center sm:items-center sm:p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Clip post kit"
    >
      <div className="absolute inset-0 bg-foreground/50 backdrop-blur-sm" onClick={onClose} />
      <div className="animate-sheet-up relative flex max-h-[88vh] w-full max-w-lg flex-col rounded-t-2xl border bg-card shadow-2xl sm:rounded-2xl">
        <div className="flex items-start justify-between gap-3 border-b px-5 py-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="font-display text-base font-bold">Post kit</h2>
              <Badge variant="secondary">#{clip.rank}</Badge>
            </div>
            <p className="mt-0.5 line-clamp-1 text-sm text-muted-foreground">{ytTitle}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-mr-1 inline-grid size-9 shrink-0 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary/70 hover:text-foreground"
          >
            <X className="size-5" />
          </button>
        </div>

        <div className="flex-1 space-y-6 overflow-auto px-5 py-5">
          {reason && (
            <div className="rounded-lg bg-accent/60 p-3 text-sm leading-relaxed text-accent-foreground">
              {reason}
            </div>
          )}

          <Section title="YouTube">
            {ytTitle && <CopyRow label="Title" value={ytTitle} />}
            {ytDesc && <CopyRow label="Description" value={ytDesc} multiline />}
            {tags.length > 0 && <CopyRow label="Tags" value={tags.join(', ')} />}
          </Section>

          {(tiktok || hashtags) && (
            <Section title="TikTok / Shorts">
              {tiktok && <CopyRow label="Caption" value={tiktok} multiline />}
              {hashtags && <CopyRow label="Hashtags" value={hashtags} />}
            </Section>
          )}
        </div>
      </div>
    </div>,
    document.body
  )
}
