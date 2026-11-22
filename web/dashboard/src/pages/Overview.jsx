import { useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'motion/react'
import { ArrowRight, Clapperboard, Cpu, Film, Flame, Gauge, Plus, Sparkles, Zap, CheckCircle2, XCircle } from 'lucide-react'
import { api } from '../lib/api'
import { useDocumentTitle, useLocalState, useResource } from '../lib/hooks'
import { STATUS_META, isActive, jobTitle, plural } from '../lib/format'
import { PageHeader } from '../components/Layout'
import { Button, Card, CardHeader, EmptyState, Skeleton } from '../components/ui'
import { ClipCard, ProjectCard, ProjectCardSkeleton } from '../components/project'

function greeting() {
  const h = new Date().getHours()
  return h < 5 ? 'Working late' : h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening'
}

function Stat({ icon: Icon, label, value, sub, index }) {
  return (
    <motion.div
      className="stat"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="stat-icon"><Icon size={16} aria-hidden /></div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </motion.div>
  )
}

export default function Overview() {
  useDocumentTitle('Overview')
  const navigate = useNavigate()
  const [profile] = useLocalState('osc.profile', { name: '' })
  const jobsRes = useResource((signal) => api.jobs({ signal }), [], {
    interval: (d) => (d?.jobs?.some(isActive) ? 4000 : 20000),
  })
  const health = useResource((signal) => api.health({ signal }), [], { interval: 30000 })

  const jobs = useMemo(() => jobsRes.data?.jobs || [], [jobsRes.data])
  const stats = useMemo(() => {
    const done = jobs.filter((j) => j.status === 'completed')
    const failed = jobs.filter((j) => j.status === 'failed')
    const clips = jobs.flatMap((j) => (j.clips || []).map((c) => ({ ...c, jobId: j.id })))
    const finished = done.length + failed.length
    return {
      total: jobs.length,
      active: jobs.filter(isActive),
      clips,
      rate: finished ? Math.round((done.length / finished) * 100) : null,
      top: [...clips].sort((a, b) => (b.viral_score ?? -1) - (a.viral_score ?? -1)).slice(0, 8),
    }
  }, [jobs])

  const h = health.data
  const name = profile?.name?.trim()

  return (
    <>
      <PageHeader
        eyebrow={new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' })}
        title={`${greeting()}${name ? `, ${name}` : ''}`}
        description="Turn long-form videos into short, high-retention clips."
        actions={<Button variant="primary" icon={Plus} onClick={() => navigate('/new')}>New project</Button>}
      />

      <div className="stats-grid">
        {jobsRes.loading && !jobsRes.data ? (
          [0, 1, 2, 3].map((i) => <Skeleton key={i} className="stat" style={{ height: 112 }} />)
        ) : (
          <>
            <Stat index={0} icon={Clapperboard} label="Projects" value={stats.total} sub={stats.active.length ? `${stats.active.length} processing` : 'All idle'} />
            <Stat index={1} icon={Film} label="Clips rendered" value={stats.clips.length} sub="Ready to publish" />
            <Stat index={2} icon={Gauge} label="Success rate" value={stats.rate == null ? '—' : `${stats.rate}%`} sub="Finished projects" />
            <Stat index={3} icon={Cpu} label="Renderer" value={h ? (h.gpu_available ? 'GPU' : 'CPU') : '—'} sub={h ? (h.ffmpeg_available ? 'FFmpeg ready' : 'FFmpeg missing') : 'Checking…'} />
          </>
        )}
      </div>

      {jobsRes.error && <p className="alert alert-danger">{jobsRes.error.message}</p>}

      {stats.active.length > 0 && (
        <section className="section">
          <div className="section-head">
            <h2><Zap size={16} aria-hidden />Processing now</h2>
          </div>
          <div className="project-grid">
            {stats.active.map((job, i) => <ProjectCard key={job.id} job={job} index={i} />)}
          </div>
        </section>
      )}

      {!jobsRes.loading && jobs.length === 0 ? (
        <Card className="hero-empty">
          <div className="hero-empty-art" aria-hidden><Sparkles size={26} /></div>
          <h2>Create your first clip project</h2>
          <p>Paste a YouTube, TikTok, Instagram or Google Drive link — or upload a file. The AI finds the strongest moments, frames faces, adds karaoke captions and renders ready-to-post shorts.</p>
          <div className="row gap-sm wrap center">
            <Button variant="primary" icon={Plus} onClick={() => navigate('/new')}>Start a project</Button>
            <Button variant="ghost" onClick={() => navigate('/settings?tab=keys')}>Add API keys</Button>
          </div>
          <ul className="feature-list">
            <li><CheckCircle2 size={14} aria-hidden />Faster-Whisper word-level transcripts</li>
            <li><CheckCircle2 size={14} aria-hidden />Gemini / NVIDIA / vLLM highlight curation</li>
            <li><CheckCircle2 size={14} aria-hidden />Face tracking, B-roll, BGM ducking</li>
          </ul>
        </Card>
      ) : (
        <div className="overview-grid">
          <section className="section">
            <div className="section-head">
              <h2><Flame size={16} aria-hidden />Top clips</h2>
              <Link to="/projects" className="link-subtle">All projects <ArrowRight size={14} aria-hidden /></Link>
            </div>
            {jobsRes.loading && !jobsRes.data ? (
              <div className="clip-grid">{[0, 1, 2, 3].map((i) => <Skeleton key={i} style={{ aspectRatio: '9/14', borderRadius: 14 }} />)}</div>
            ) : stats.top.length ? (
              <div className="clip-grid">
                {stats.top.map((clip, i) => <ClipCard key={`${clip.jobId}-${clip.rank}`} clip={clip} jobId={clip.jobId} index={i} />)}
              </div>
            ) : (
              <EmptyState icon={Film} title="No clips yet" description="Rendered clips will appear here, ranked by viral score." />
            )}
          </section>

          <aside className="stack-md">
            <Card>
              <CardHeader title="Recent projects" actions={<Link to="/projects" className="link-subtle">View all</Link>} />
              <ul className="recent-list">
                {jobsRes.loading && !jobsRes.data
                  ? [0, 1, 2].map((i) => <li key={i}><ProjectCardSkeleton /></li>)
                  : jobs.slice(0, 5).map((job) => (
                    <li key={job.id}>
                      <Link to={`/projects/${encodeURIComponent(job.id)}`} className="recent-item">
                        <span className={`recent-dot tone-${job.status}`} aria-hidden />
                        <span className="recent-text">
                          <span className="truncate">{jobTitle(job)}</span>
                          <span className="recent-sub">{STATUS_META[job.status]?.label || job.status} · {plural(job.clips?.length || 0, 'clip')}</span>
                        </span>
                        <ArrowRight size={14} aria-hidden className="recent-arrow" />
                      </Link>
                    </li>
                  ))}
              </ul>
            </Card>
            <Card>
              <CardHeader title="System" actions={<Link to="/system" className="link-subtle">Details</Link>} />
              <ul className="check-list">
                <li>{h?.gpu_available ? <CheckCircle2 size={15} className="ok" /> : <XCircle size={15} className="warn" />}CUDA GPU {h?.gpu_available ? 'available' : 'not detected'}</li>
                <li>{h?.ffmpeg_available ? <CheckCircle2 size={15} className="ok" /> : <XCircle size={15} className="bad" />}FFmpeg {h?.ffmpeg_available ? 'installed' : 'missing'}</li>
                <li><Zap size={15} className="muted" />{h ? `${h.jobs_running} running · ${h.jobs_queued} queued` : 'Loading…'}</li>
              </ul>
            </Card>
          </aside>
        </div>
      )}
    </>
  )
}
