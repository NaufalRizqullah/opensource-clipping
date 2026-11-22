export const ACTIVE_STATES = new Set(['queued', 'downloading', 'transcribing', 'analyzing', 'rendering'])
export const isActive = (job) => ACTIVE_STATES.has(job?.status)

export const STATUS_META = {
  queued: { label: 'Queued', tone: 'neutral' },
  downloading: { label: 'Downloading', tone: 'info' },
  transcribing: { label: 'Transcribing', tone: 'info' },
  analyzing: { label: 'Analyzing', tone: 'info' },
  rendering: { label: 'Rendering', tone: 'info' },
  completed: { label: 'Completed', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
  cancelled: { label: 'Cancelled', tone: 'warning' },
}

export const PIPELINE_STEPS = [
  { key: 'download', label: 'Download' },
  { key: 'transcribe', label: 'Transcribe' },
  { key: 'analyze', label: 'AI analysis' },
  { key: 'metadata', label: 'Metadata' },
  { key: 'diarization', label: 'Speakers' },
  { key: 'render', label: 'Render' },
  { key: 'done', label: 'Done' },
]

export function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return '—'
  const s = Math.max(0, Math.round(Number(seconds)))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}` : `${m}:${String(sec).padStart(2, '0')}`
}

export function formatTimecode(seconds) {
  if (!Number.isFinite(seconds)) return '0:00'
  return formatDuration(seconds)
}

const rtf = typeof Intl !== 'undefined' ? new Intl.RelativeTimeFormat('en', { numeric: 'auto' }) : null

export function timeAgo(value) {
  if (!value) return ''
  const date = new Date(value)
  const diff = (date.getTime() - Date.now()) / 1000
  if (!Number.isFinite(diff)) return ''
  const units = [['year', 31536000], ['month', 2592000], ['week', 604800], ['day', 86400], ['hour', 3600], ['minute', 60]]
  for (const [unit, secs] of units) {
    if (Math.abs(diff) >= secs) return rtf ? rtf.format(Math.round(diff / secs), unit) : date.toLocaleString()
  }
  return 'just now'
}

export function formatDate(value) {
  if (!value) return ''
  return new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let n = bytes
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++ }
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`
}

export function youtubeId(url) {
  if (!url) return null
  try {
    const u = new URL(url)
    if (u.hostname === 'youtu.be') return u.pathname.slice(1).split('/')[0] || null
    if (u.hostname.endsWith('youtube.com')) {
      if (u.searchParams.get('v')) return u.searchParams.get('v')
      const m = u.pathname.match(/\/(shorts|live|embed)\/([\w-]{6,})/)
      return m ? m[2] : null
    }
  } catch { /* invalid */ }
  return null
}

export function detectPlatform(url) {
  try {
    const host = new URL(url).hostname.replace(/^www\./, '')
    if (/(^|\.)youtube\.com$|^youtu\.be$/.test(host)) return 'youtube'
    if (/(^|\.)tiktok\.com$/.test(host)) return 'tiktok'
    if (/(^|\.)instagram\.com$/.test(host)) return 'instagram'
    if (/(^|\.)drive\.google\.com$|(^|\.)docs\.google\.com$/.test(host)) return 'gdrive'
  } catch { /* invalid */ }
  return null
}

export function isHttpUrl(value) {
  try {
    const u = new URL(String(value).trim())
    return (u.protocol === 'http:' || u.protocol === 'https:') && !!u.hostname
  } catch { return false }
}

export function jobTitle(job) {
  const first = job?.clips?.[0]
  const t = first?.title_en || first?.title
  if (t) return t
  if (job?.upload_filename) return job.upload_filename.replace(/^[a-f0-9]{12}_/, '').replace(/\.[^.]+$/, '').replace(/_/g, ' ')
  if (job?.url) {
    try {
      const u = new URL(job.url)
      return `${u.hostname.replace(/^www\./, '')}${u.pathname.length > 1 ? u.pathname : ''}`.slice(0, 64)
    } catch { return job.url }
  }
  return `Project ${job?.id || ''}`
}

export function jobThumbnail(job, mediaUrl) {
  const clip = job?.clips?.find((c) => c.thumbnail_url)
  if (clip) return mediaUrl(clip.thumbnail_url)
  const yt = youtubeId(job?.url)
  return yt ? `https://i.ytimg.com/vi/${encodeURIComponent(yt)}/hqdefault.jpg` : null
}

export function jobPercent(job) {
  if (job?.status === 'completed') return 100
  const p = job?.progress
  if (!p) return 0
  return Math.max(0, Math.min(100, Number(p.percent) || 0))
}

export function clipTitle(clip) {
  return clip?.metadata?.youtube_title_final || clip?.title_en || clip?.title || `Clip #${clip?.rank ?? ''}`
}

export function plural(n, word) {
  return `${n} ${word}${n === 1 ? '' : 's'}`
}

export const PLATFORM_LABEL = { youtube: 'YouTube', tiktok: 'TikTok', instagram: 'Instagram', gdrive: 'Google Drive' }
export const sourceLabel = (job) => (job?.upload_filename ? 'Upload' : PLATFORM_LABEL[job?.source] || job?.source || 'Link')
