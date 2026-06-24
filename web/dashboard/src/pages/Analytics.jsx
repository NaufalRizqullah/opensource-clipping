import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { fetchJobs } from '../api'
import { cn } from '@/lib/utils'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'

const STATUS_LABEL = {
  queued: 'Queued', downloading: 'Downloading', transcribing: 'Transcribing',
  analyzing: 'Analyzing', rendering: 'Rendering', completed: 'Completed',
  failed: 'Failed', cancelled: 'Cancelled',
}
const STATUS_ORDER = ['completed', 'rendering', 'analyzing', 'transcribing', 'downloading', 'queued', 'failed', 'cancelled']
const BAR_COLOR = {
  completed: 'bg-success', failed: 'bg-destructive', cancelled: 'bg-muted-foreground/50',
}

function formatDuration(totalSeconds) {
  if (!totalSeconds) return '0s'
  const s = Math.round(totalSeconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${sec}s`
  return `${sec}s`
}

function StatCard({ label, value, sub, loading }) {
  return (
    <Card className="p-5">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</span>
      {loading ? (
        <Skeleton className="mt-3 h-8 w-16" />
      ) : (
        <>
          <div className="mt-3 font-display text-[32px] font-bold leading-none tabular-nums">{value}</div>
          {sub && <div className="mt-1.5 text-xs text-muted-foreground">{sub}</div>}
        </>
      )}
    </Card>
  )
}

function Analytics() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchJobs()
      .then((data) => setJobs(data.jobs || []))
      .catch((err) => console.error('Failed to load analytics:', err))
      .finally(() => setLoading(false))
  }, [])

  const total = jobs.length
  const completed = jobs.filter((j) => j.status === 'completed').length
  const failed = jobs.filter((j) => j.status === 'failed').length
  const finished = completed + failed
  const successRate = finished > 0 ? Math.round((completed / finished) * 100) : null

  const allClips = jobs.flatMap((j) => j.clips || [])
  const totalClips = allClips.length
  const durations = allClips.map((c) => c.duration).filter((d) => typeof d === 'number')
  const totalDuration = durations.reduce((a, b) => a + b, 0)
  const avgDuration = durations.length ? totalDuration / durations.length : 0
  const longest = durations.length ? Math.max(...durations) : 0
  const jobsWithClips = jobs.filter((j) => (j.clips?.length || 0) > 0).length
  const avgClipsPerJob = jobsWithClips ? (totalClips / jobsWithClips) : 0

  const statusCounts = jobs.reduce((acc, j) => {
    acc[j.status] = (acc[j.status] || 0) + 1
    return acc
  }, {})
  const presentStatuses = STATUS_ORDER.filter((s) => statusCounts[s])
  const maxCount = Math.max(1, ...Object.values(statusCounts))

  return (
    <div className="animate-in-up space-y-8">
      <div>
        <h1 className="font-display text-4xl font-bold tracking-tight">Analytics</h1>
        <p className="mt-1.5 text-muted-foreground">Output and reliability across all your clipping jobs</p>
      </div>

      {!loading && total === 0 ? (
        <Card className="bg-dot-grid grid place-items-center px-6 py-16 text-center">
          <h3 className="font-display text-xl font-bold">Nothing to analyze yet</h3>
          <p className="mt-1.5 max-w-sm text-sm text-muted-foreground">
            Run a few clipping jobs and your output and reliability stats will show up here.
          </p>
          <Button asChild className="mt-6">
            <Link to="/new">Create a job</Link>
          </Button>
        </Card>
      ) : (
        <>
          {/* Headline stats */}
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatCard label="Total Jobs" value={total} loading={loading} />
            <StatCard
              label="Success Rate"
              value={successRate == null ? '-' : `${successRate}%`}
              sub={loading ? null : `${completed} done · ${failed} failed`}
              loading={loading}
            />
            <StatCard label="Total Clips" value={totalClips} sub={loading ? null : `${avgClipsPerJob.toFixed(1)} avg / job`} loading={loading} />
            <StatCard label="Total Runtime" value={formatDuration(totalDuration)} sub={loading ? null : 'across all clips'} loading={loading} />
          </div>

          <div className="grid items-start gap-5 lg:grid-cols-2">
            {/* Status breakdown */}
            <Card className="p-6">
              <h2 className="mb-5 font-display text-base font-bold">Status breakdown</h2>
              {loading ? (
                <div className="space-y-3">
                  {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-5 w-full" />)}
                </div>
              ) : (
                <div className="space-y-3">
                  {presentStatuses.map((s) => (
                    <div key={s}>
                      <div className="mb-1 flex items-center justify-between text-sm">
                        <span className="font-medium">{STATUS_LABEL[s] || s}</span>
                        <span className="tabular-nums text-muted-foreground">{statusCounts[s]}</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-secondary">
                        <div
                          className={cn('h-full rounded-full', BAR_COLOR[s] || 'bg-primary')}
                          style={{ width: `${(statusCounts[s] / maxCount) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Card>

            {/* Clip output */}
            <Card className="p-6">
              <h2 className="mb-5 font-display text-base font-bold">Clip output</h2>
              {loading ? (
                <div className="space-y-3">
                  {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-5 w-full" />)}
                </div>
              ) : (
                <div className="divide-y text-sm">
                  <div className="flex items-center justify-between py-2.5">
                    <span className="text-muted-foreground">Average clip length</span>
                    <span className="font-medium tabular-nums">{durations.length ? formatDuration(avgDuration) : '-'}</span>
                  </div>
                  <div className="flex items-center justify-between py-2.5">
                    <span className="text-muted-foreground">Longest clip</span>
                    <span className="font-medium tabular-nums">{durations.length ? formatDuration(longest) : '-'}</span>
                  </div>
                  <div className="flex items-center justify-between py-2.5">
                    <span className="text-muted-foreground">Avg clips per job</span>
                    <span className="font-medium tabular-nums">{jobsWithClips ? avgClipsPerJob.toFixed(1) : '-'}</span>
                  </div>
                  <div className="flex items-center justify-between py-2.5">
                    <span className="text-muted-foreground">Jobs producing clips</span>
                    <span className="font-medium tabular-nums">{jobsWithClips} / {total}</span>
                  </div>
                </div>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  )
}

export default Analytics
