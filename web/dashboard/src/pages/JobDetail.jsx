import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { fetchJob, createSSEConnection } from '../api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { ClipCaptionsDialog, hasCaptions } from '@/components/ClipCaptionsDialog'

const STEPS = [
  { key: 'download', label: 'Download' },
  { key: 'transcribe', label: 'Transcribe' },
  { key: 'analyze', label: 'AI Analysis' },
  { key: 'metadata', label: 'Metadata' },
  { key: 'render', label: 'Render' },
  { key: 'done', label: 'Done' },
]
const TERMINAL = ['completed', 'failed', 'cancelled']

/** Friendly download filename built from the clip title (falls back to rank). */
function downloadName(clip) {
  const base = (clip.title || clip.title_en || `clip-${clip.rank}`)
    .replace(/[<>:"/\\|?*\n\r]+/g, '')
    .trim()
    .slice(0, 80)
  const ext = (clip.download_url?.split('.').pop() || 'mp4').split('?')[0]
  return `${base || `clip-${clip.rank}`}.${ext}`
}
const STATUS_VARIANT = {
  completed: 'success', failed: 'destructive', cancelled: 'secondary',
}

function JobDetail() {
  const { jobId } = useParams()
  const [job, setJob] = useState(null)
  const [loading, setLoading] = useState(true)
  const [captionsClip, setCaptionsClip] = useState(null)

  useEffect(() => {
    let sse = null
    const load = async () => {
      try {
        const data = await fetchJob(jobId)
        setJob(data)
        if (!TERMINAL.includes(data.status)) {
          sse = createSSEConnection(jobId, (event) => {
            if (event.type === 'completed') {
              fetchJob(jobId).then(setJob)
            } else if (event.type === 'progress') {
              setJob((prev) => prev ? { ...prev, status: event.status, progress: event.progress, error: event.error } : prev)
            }
          })
        }
      } catch (err) {
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    load()
    return () => { if (sse) sse.close() }
  }, [jobId])

  useEffect(() => {
    if (!job) return
    if (TERMINAL.includes(job.status)) return
    const interval = setInterval(async () => {
      try { setJob(await fetchJob(jobId)) } catch {}
    }, 3000)
    return () => clearInterval(interval)
  }, [jobId, job?.status])

  if (loading) {
    return <div className="grid place-items-center py-32 text-muted-foreground"><Loader2 className="size-6 animate-spin" /></div>
  }
  if (!job) {
    return (
      <div className="grid place-items-center py-32 text-center">
        <h3 className="font-display text-xl font-bold">Job not found</h3>
        <Button asChild variant="secondary" className="mt-4"><Link to="/">Back to all jobs</Link></Button>
      </div>
    )
  }

  const currentStep = job.progress?.step || ''
  const percent = job.progress?.percent || 0
  const currentIdx = STEPS.findIndex((x) => x.key === currentStep)
  const running = !TERMINAL.includes(job.status)

  return (
    <div className="animate-in-up space-y-6">
      {/* Header */}
      <div>
        <Link to="/" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
          All jobs
        </Link>
        <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-3">
              <h1 className="font-display text-3xl font-bold tracking-tight">Job</h1>
              <span className="font-mono text-sm text-muted-foreground">#{job.id}</span>
              <Badge variant={STATUS_VARIANT[job.status] || 'default'}>{job.status}</Badge>
            </div>
            <p className="mt-1 truncate text-sm text-muted-foreground">{job.url || job.upload_filename || 'Unknown source'}</p>
          </div>
          <Button asChild variant="outline" size="sm">
            <Link to="/new" state={{ reuseJob: job }}>Clone &amp; Rerun</Link>
          </Button>
        </div>
      </div>

      {/* Progress */}
      {running && (
        <Card className="p-6">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-semibold">Progress</span>
            <span className="font-display text-sm font-bold tabular-nums text-primary">{Math.round(percent)}%</span>
          </div>
          <Progress value={percent} />
          {job.progress?.message && (
            <p className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" /> {job.progress.message}
            </p>
          )}
          <div className="mt-6 flex flex-wrap gap-2">
            {STEPS.map((s, i) => {
              const done = i < currentIdx
              const active = i === currentIdx
              return (
                <div
                  key={s.key}
                  className={cn(
                    'rounded-full border px-2.5 py-1 text-xs font-medium',
                    done && 'border-transparent bg-success/12 text-success',
                    active && 'border-primary/30 bg-primary/10 text-primary',
                    !done && !active && 'text-muted-foreground'
                  )}
                >
                  {s.label}
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {/* Error */}
      {job.error && (
        <Card className="border-destructive/30 bg-destructive/5 p-5">
          <div className="font-semibold text-destructive">Error</div>
          <p className="mt-2 break-words text-sm text-muted-foreground">{job.error}</p>
        </Card>
      )}

      {/* Clips */}
      {job.clips && job.clips.length > 0 && (
        <div className="space-y-3">
          <h2 className="font-display text-lg font-bold tracking-tight">
            Generated clips <span className="text-muted-foreground">({job.clips.length})</span>
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {job.clips.map((clip, i) => (
              <Card key={i} className="group/clip overflow-hidden transition-all hover:border-primary/30 hover:shadow-[0_8px_30px_rgba(20,19,26,0.10)]">
                <div className="relative bg-black">
                  <video
                    className="max-h-[460px] w-full"
                    controls
                    preload="metadata"
                    poster={clip.thumbnail_url || undefined}
                    src={clip.download_url}
                  />
                  <Badge
                    variant={clip.rank === 1 ? 'lime' : 'secondary'}
                    className="absolute left-2 top-2 shadow-sm"
                  >
                    #{clip.rank}
                  </Badge>
                </div>
                <div className="space-y-3 p-4">
                  <div className="line-clamp-2 text-sm font-semibold leading-snug">
                    {clip.title || clip.title_en || `Clip ${clip.rank}`}
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    {clip.viral_score != null && (
                      <Badge variant="lime">Score {clip.viral_score}</Badge>
                    )}
                    {clip.duration != null && <span>{Math.round(clip.duration)}s</span>}
                  </div>
                  <div className="flex gap-2">
                    <Button asChild variant="secondary" size="sm" className="flex-1">
                      <a href={clip.download_url} download={downloadName(clip)}>Download</a>
                    </Button>
                    {hasCaptions(clip) && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => setCaptionsClip(clip)}
                        title="View AI-generated captions & hashtags"
                      >
                        Post kit
                      </Button>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Empty (done but no clips) */}
      {!running && (!job.clips || job.clips.length === 0) && !job.error && (
        <Card className="grid place-items-center px-6 py-14 text-center text-muted-foreground">
          <p className="text-sm">No clips were produced for this job.</p>
        </Card>
      )}

      {/* Log */}
      {job.log && job.log.length > 0 && (
        <div className="space-y-2">
          <h2 className="font-display text-sm font-bold uppercase tracking-wider text-muted-foreground">Activity log</h2>
          <div className="max-h-72 overflow-auto rounded-xl border bg-[#15141b] p-4 font-mono text-xs leading-relaxed text-zinc-300">
            {job.log.map((line, i) => <div key={i}>{line}</div>)}
          </div>
        </div>
      )}

      <ClipCaptionsDialog
        clip={captionsClip}
        open={!!captionsClip}
        onClose={() => setCaptionsClip(null)}
      />
    </div>
  )
}

export default JobDetail
