import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import {
  ArrowLeft, Ban, Download, ExternalLink, FileText, Film, Image as ImageIcon, Repeat2, Terminal, Trash2, AlertOctagon, FolderOpen, Settings2,
} from 'lucide-react'
import { api, mediaUrl } from '../lib/api'
import { useDocumentTitle, useLiveJob, useResource } from '../lib/hooks'
import { sourceLabel, formatBytes, formatDate, isActive, jobPercent, jobTitle, plural, timeAgo } from '../lib/format'
import { useToast } from '../lib/toast'
import { PageHeader } from '../components/Layout'
import { Badge, Button, Card, CardHeader, ConfirmModal, CopyButton, EmptyState, ProgressBar, Skeleton, StatusBadge } from '../components/ui'
import { ClipCard, PipelineStepper, SourceIcon } from '../components/project'

function fileIcon(name) {
  if (/\.(mp4|mkv|webm|mov)$/i.test(name)) return Film
  if (/\.(jpe?g|png)$/i.test(name)) return ImageIcon
  return FileText
}

export default function ProjectDetail() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const toast = useToast()
  const { data: job, error, loading, reload } = useLiveJob(jobId)
  const active = isActive(job)
  const outputs = useResource(
    (signal) => (job && !active ? api.outputs(jobId, { signal }).catch((e) => (e.status === 404 ? { files: [] } : Promise.reject(e))) : Promise.resolve(null)),
    [jobId, job?.status],
  )
  const [confirm, setConfirm] = useState(null) // 'cancel' | 'delete'
  const [busy, setBusy] = useState(false)
  const [showLog, setShowLog] = useState(false)
  const logRef = useRef(null)

  useDocumentTitle(job ? jobTitle(job) : 'Project')

  useEffect(() => {
    if (showLog && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [showLog, job?.log?.length])

  const sortedClips = useMemo(() => [...(job?.clips || [])].sort((a, b) => a.rank - b.rank), [job?.clips])

  const doCancelOrDelete = async () => {
    setBusy(true)
    try {
      const res = await api.deleteJob(jobId)
      if (res.cancel_requested) {
        toast.info('Cancellation requested — finishing the current step.')
        reload()
      } else {
        toast.success('Project deleted')
        navigate('/projects')
      }
    } catch (err) {
      toast.error(err.message)
    } finally {
      setBusy(false)
      setConfirm(null)
    }
  }

  if (loading && !job) {
    return (
      <div className="stack-lg">
        <Skeleton style={{ height: 34, width: '45%' }} />
        <Skeleton style={{ height: 120 }} />
        <div className="clip-grid">{[0, 1, 2, 3].map((i) => <Skeleton key={i} style={{ aspectRatio: '9/14' }} />)}</div>
      </div>
    )
  }
  if (error && !job) {
    return (
      <EmptyState
        icon={AlertOctagon}
        title={error.status === 404 ? 'Project not found' : 'Could not load project'}
        description={error.status === 404 ? 'It may have been deleted, or the server was restarted with a different store.' : error.message}
        action={<Button onClick={() => navigate('/projects')} icon={ArrowLeft}>Back to projects</Button>}
      />
    )
  }
  if (!job) return null

  const percent = jobPercent(job)
  const configEntries = Object.entries(job.config || {}).filter(([k, v]) => v !== null && v !== '' && !['url', 'upload_filename', 'source'].includes(k))

  return (
    <>
      <PageHeader
        back={<Link to="/projects" className="back-link"><ArrowLeft size={14} aria-hidden />Projects</Link>}
        title={jobTitle(job)}
        description={
          <span className="row gap-sm wrap align-center">
            <StatusBadge status={job.status} />
            <span className="meta-item"><SourceIcon job={job} />{job.upload_filename ? 'Uploaded file' : sourceLabel(job)}</span>
            <span className="meta-item" title={formatDate(job.created_at)}>Created {timeAgo(job.created_at)}</span>
            <span className="meta-item mono">ID {job.id}<CopyButton value={job.id} label="" /></span>
          </span>
        }
        actions={
          <>
            {job.url && (
              <a className="btn btn-ghost btn-md" href={job.url} target="_blank" rel="noopener noreferrer"><ExternalLink size={16} aria-hidden /><span>Source</span></a>
            )}
            {!active && <Button icon={Repeat2} onClick={() => navigate('/new', { state: { reuseJob: job } })}>Rerun</Button>}
            {active ? (
              <Button variant="danger-ghost" icon={Ban} onClick={() => setConfirm('cancel')} disabled={job.cancel_requested}>{job.cancel_requested ? 'Cancelling…' : 'Cancel'}</Button>
            ) : (
              <Button variant="danger-ghost" icon={Trash2} onClick={() => setConfirm('delete')}>Delete</Button>
            )}
          </>
        }
      />

      <Card className="pipeline-card">
        <PipelineStepper job={job} />
        <AnimatePresence initial={false}>
          {active && (
            <motion.div className="pipeline-progress" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
              <div className="row between gap-md">
                <span className="pipeline-msg">{job.cancel_requested ? 'Cancelling after the current operation…' : job.progress?.message || 'Waiting for a free worker…'}</span>
                <span className="mono pipeline-pct">{Math.round(percent)}%</span>
              </div>
              <ProgressBar value={percent} indeterminate={job.status === 'queued'} label="Overall progress" />
            </motion.div>
          )}
        </AnimatePresence>
        {job.status === 'failed' && job.error && (
          <div className="alert alert-danger" role="alert"><AlertOctagon size={16} aria-hidden /><div><strong>Processing failed</strong><p className="pre-wrap">{job.error}</p></div></div>
        )}
        {job.status === 'cancelled' && <div className="alert alert-warning">This project was cancelled.{job.clips?.length ? ` ${plural(job.clips.length, 'clip')} finished before stopping.` : ''}</div>}
      </Card>

      <section className="section">
        <div className="section-head">
          <h2><Film size={16} aria-hidden />Clips {sortedClips.length > 0 && <Badge>{sortedClips.length}</Badge>}</h2>
        </div>
        {sortedClips.length ? (
          <div className="clip-grid">
            {sortedClips.map((clip, i) => <ClipCard key={clip.rank} clip={clip} jobId={job.id} index={i} />)}
          </div>
        ) : active ? (
          <div className="clip-grid">
            {Array.from({ length: Math.min(job.config?.clips || 4, 8) }, (_, i) => (
              <div key={i} className="clip-placeholder"><Skeleton className="fill" /><span>Clip {i + 1}</span></div>
            ))}
          </div>
        ) : (
          <EmptyState icon={Film} title="No clips produced" description={job.status === 'failed' ? 'Check the error and log, adjust settings, then rerun.' : 'This project finished without rendered clips.'} />
        )}
      </section>

      <div className="detail-grid">
        <Card>
          <button type="button" className="collapse-btn" onClick={() => setShowLog((v) => !v)} aria-expanded={showLog}>
            <span className="card-title"><Terminal size={16} aria-hidden />Activity log <Badge>{job.log?.length || 0}</Badge></span>
            <span className="link-subtle">{showLog ? 'Hide' : 'Show'}</span>
          </button>
          <AnimatePresence initial={false}>
            {showLog && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} style={{ overflow: 'hidden' }}>
                <pre ref={logRef} className="log" tabIndex={0} aria-label="Activity log">
                  {job.log?.length ? job.log.join('\n') : 'No log entries yet.'}
                </pre>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>

        <Card>
          <CardHeader title="Files" icon={FolderOpen} description={outputs.data?.files?.length ? plural(outputs.data.files.length, 'file') : undefined} />
          {active ? (
            <p className="muted small">Files appear when processing ends.</p>
          ) : outputs.loading && !outputs.data ? (
            <Skeleton style={{ height: 80 }} />
          ) : outputs.data?.files?.length ? (
            <ul className="file-list">
              {outputs.data.files.filter((f) => !/^(jobs|video_asli)\b/.test(f.filename)).map((f) => {
                const Icon = fileIcon(f.filename)
                return (
                  <li key={f.filename}>
                    <Icon size={15} aria-hidden className="muted" />
                    <span className="truncate" title={f.filename}>{f.filename}</span>
                    <span className="mono muted small">{formatBytes(f.size_bytes)}</span>
                    <a className="icon-btn icon-btn-sm" href={mediaUrl(f.download_url)} download={f.filename} aria-label={`Download ${f.filename}`} title="Download"><Download size={14} /></a>
                  </li>
                )
              })}
            </ul>
          ) : (
            <p className="muted small">No output files.</p>
          )}
        </Card>

        <Card>
          <CardHeader title="Configuration" icon={Settings2} />
          <dl className="config-list">
            {configEntries.map(([k, v]) => (
              <div key={k}><dt>{k.replace(/_/g, ' ')}</dt><dd className="mono">{typeof v === 'boolean' ? (v ? 'on' : 'off') : String(v)}</dd></div>
            ))}
          </dl>
        </Card>
      </div>

      <ConfirmModal
        open={confirm === 'cancel'}
        onClose={() => setConfirm(null)}
        onConfirm={doCancelOrDelete}
        loading={busy}
        title="Cancel this project?"
        description="The worker stops after the current operation. Clips already rendered are kept."
        confirmLabel="Cancel project"
      />
      <ConfirmModal
        open={confirm === 'delete'}
        onClose={() => setConfirm(null)}
        onConfirm={doCancelOrDelete}
        loading={busy}
        title="Delete this project?"
        description="This removes the project from the Studio list and cannot be undone. Rendered files stay in the server’s outputs folder."
        confirmLabel="Delete project"
      />
    </>
  )
}
