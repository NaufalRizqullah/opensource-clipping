import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { fetchJobs, fetchHealth, deleteJob } from '../api'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/components/ui/toast'

const STATUS = {
  queued: { label: 'Queued', variant: 'secondary' },
  downloading: { label: 'Downloading', variant: 'default' },
  transcribing: { label: 'Transcribing', variant: 'default' },
  analyzing: { label: 'Analyzing', variant: 'default' },
  rendering: { label: 'Rendering', variant: 'default' },
  completed: { label: 'Completed', variant: 'success' },
  failed: { label: 'Failed', variant: 'destructive' },
  cancelled: { label: 'Cancelled', variant: 'secondary' },
}
const RUNNING = ['queued', 'downloading', 'transcribing', 'analyzing', 'rendering']

function formatDate(s) {
  if (!s) return '-'
  return new Date(s).toLocaleString('en-US', {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
  })
}

function StatCard({ label, value, loading }) {
  return (
    <Card className="p-5">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</span>
      {loading
        ? <Skeleton className="mt-3 h-8 w-14" />
        : <div className="mt-3 font-display text-[32px] font-bold leading-none tabular-nums">{value}</div>}
    </Card>
  )
}

function JobRowSkeleton() {
  return (
    <div className="flex items-center gap-4 rounded-xl border bg-card p-4">
      <div className="min-w-0 flex-1 space-y-2">
        <Skeleton className="h-4 w-2/3" />
        <Skeleton className="h-3 w-1/3" />
      </div>
      <Skeleton className="h-6 w-20 rounded-full" />
    </div>
  )
}

function StatusBadge({ status }) {
  const s = STATUS[status] || { label: status, variant: 'secondary' }
  return <Badge variant={s.variant} className="shrink-0">{s.label}</Badge>
}

function Dashboard() {
  const { toast } = useToast()
  const [jobs, setJobs] = useState([])
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState(null)

  const handleDelete = async (job) => {
    if (!window.confirm(`Delete job #${job.id}? This removes its output files and cannot be undone.`)) return
    setDeletingId(job.id)
    try {
      await deleteJob(job.id)
      setJobs((prev) => prev.filter((j) => j.id !== job.id))
      toast({ variant: 'success', title: 'Job deleted', description: `#${job.id} was removed.` })
    } catch (err) {
      toast({ variant: 'error', title: 'Delete failed', description: err.message })
    } finally {
      setDeletingId(null)
    }
  }

  const loadData = async () => {
    try {
      const [jobsData, healthData] = await Promise.all([fetchJobs(), fetchHealth()])
      setJobs(jobsData.jobs || [])
      setHealth(healthData)
    } catch (err) {
      console.error('Failed to load dashboard:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 5000)
    return () => clearInterval(interval)
  }, [])

  const totalCompleted = jobs.filter((j) => j.status === 'completed').length
  const totalClips = jobs.reduce((acc, j) => acc + (j.clips?.length || 0), 0)

  return (
    <div className="animate-in-up space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-4xl font-bold tracking-tight">Dashboard</h1>
          <p className="mt-1.5 text-muted-foreground">Overview of all your clipping jobs</p>
        </div>
        <Button asChild size="lg">
          <Link to="/new">New Job</Link>
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Total Jobs" value={jobs.length} loading={loading} />
        <StatCard label="Completed" value={totalCompleted} loading={loading} />
        <StatCard label="Total Clips" value={totalClips} loading={loading} />
        <Card className="p-5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">System</span>
          {loading ? (
            <div className="mt-3 space-y-2">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-4 w-20" />
            </div>
          ) : (
            <div className="mt-3 space-y-1.5 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">GPU</span>
                <span className={health?.gpu_available ? 'font-medium text-success' : 'text-muted-foreground'}>
                  {health?.gpu_available ? 'Ready' : 'CPU only'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">FFmpeg</span>
                <span className={health?.ffmpeg_available ? 'font-medium text-success' : 'font-medium text-destructive'}>
                  {health?.ffmpeg_available ? 'OK' : 'Missing'}
                </span>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* Job list */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="font-display text-lg font-bold tracking-tight">Recent jobs</h2>
          {jobs.length > 0 && <span className="text-sm text-muted-foreground">{jobs.length} total</span>}
        </div>

        {loading ? (
          <div className="space-y-2.5">
            {Array.from({ length: 4 }).map((_, i) => <JobRowSkeleton key={i} />)}
          </div>
        ) : jobs.length === 0 ? (
          <Card className="bg-dot-grid grid place-items-center px-6 py-16 text-center">
            <h3 className="font-display text-xl font-bold">No jobs yet</h3>
            <p className="mt-1.5 max-w-sm text-sm text-muted-foreground">
              Create your first job and turn a long video into viral short-form clips automatically.
            </p>
            <Button asChild className="mt-6">
              <Link to="/new">Create first job</Link>
            </Button>
          </Card>
        ) : (
          <div className="space-y-2.5">
            {jobs.map((job) => {
              const title = job.url
                ? (() => { try { return new URL(job.url).hostname.replace('www.', '') } catch { return job.url } })()
                : job.upload_filename || 'Upload'
              const running = RUNNING.includes(job.status)
              return (
                <Link
                  key={job.id}
                  to={`/job/${job.id}`}
                  className="group flex items-center gap-4 rounded-xl border bg-card p-4 transition-all hover:border-primary/40 hover:shadow-[0_6px_24px_rgba(20,19,26,0.07)]"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate font-semibold">{title}</span>
                      <span className="shrink-0 font-mono text-[11px] text-muted-foreground">#{job.id}</span>
                    </div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
                      <span>{formatDate(job.created_at)}</span>
                      {job.clips?.length > 0 && <span>{job.clips.length} clips</span>}
                      {running && job.progress?.message && (
                        <span className="truncate font-medium text-primary">{job.progress.message}</span>
                      )}
                      {job.error && <span className="text-destructive">Error</span>}
                    </div>
                  </div>
                  <StatusBadge status={job.status} />
                  <button
                    type="button"
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleDelete(job) }}
                    disabled={deletingId === job.id}
                    aria-label={`Delete job ${job.id}`}
                    className="shrink-0 rounded-md px-2 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:opacity-50"
                  >
                    {deletingId === job.id ? 'Deleting…' : 'Delete'}
                  </button>
                </Link>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
