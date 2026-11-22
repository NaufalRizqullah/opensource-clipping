import { useState } from 'react'
import { motion } from 'motion/react'
import { Activity, CheckCircle2, Cpu, Power, RefreshCw, XCircle, Server, PackageCheck } from 'lucide-react'
import { api, fetchReadiness } from '../lib/api'
import { useConnection } from '../lib/connection'
import { useDocumentTitle, useResource } from '../lib/hooks'
import { useToast } from '../lib/toast'
import { PageHeader } from '../components/Layout'
import { Badge, Button, Card, CardHeader, ConfirmModal, Skeleton } from '../components/ui'

const CHECK_LABELS = {
  numpy: 'NumPy runtime',
  renderer: 'Renderer (OpenCV, MediaPipe)',
  transcription: 'Transcription (Faster-Whisper)',
  ffmpeg: 'FFmpeg',
  runtime: 'Python runtime',
}

export default function SystemPage() {
  useDocumentTitle('System')
  const toast = useToast()
  const { check } = useConnection()
  const health = useResource((signal) => api.health({ signal }), [], { interval: 10000 })
  const ready = useResource((signal) => fetchReadiness(signal), [])
  const [confirm, setConfirm] = useState(false)
  const [busy, setBusy] = useState(false)
  const h = health.data
  const r = ready.data

  const shutdown = async () => {
    setBusy(true)
    try {
      await api.shutdown()
      toast.info('Server is shutting down…')
      setTimeout(check, 2500)
    } catch (err) { toast.error(err.message) } finally { setBusy(false); setConfirm(false) }
  }

  const tiles = [
    { icon: Server, label: 'API', value: h ? 'Online' : '—', ok: !!h },
    { icon: Cpu, label: 'GPU', value: h ? (h.gpu_available ? 'CUDA available' : 'CPU only') : '—', ok: h?.gpu_available },
    { icon: PackageCheck, label: 'FFmpeg', value: h ? (h.ffmpeg_available ? 'Installed' : 'Missing') : '—', ok: h?.ffmpeg_available },
    { icon: Activity, label: 'Queue', value: h ? `${h.jobs_running} running · ${h.jobs_queued} queued` : '—', ok: true },
  ]

  return (
    <>
      <PageHeader title="System" description={h ? `Backend v${h.version}` : 'Server health and pipeline readiness.'} actions={<Button icon={RefreshCw} onClick={() => { health.reload(); ready.reload() }}>Refresh</Button>} />

      <div className="stats-grid">
        {tiles.map((t, i) => (
          <motion.div key={t.label} className="stat" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
            <div className={`stat-icon ${t.ok === false ? 'is-bad' : ''}`}><t.icon size={16} aria-hidden /></div>
            <div className="stat-label">{t.label}</div>
            <div className="stat-value stat-value-sm">{health.loading && !h ? <Skeleton style={{ height: 20, width: 90 }} /> : t.value}</div>
          </motion.div>
        ))}
      </div>

      <div className="detail-grid two">
        <Card>
          <CardHeader
            title="Pipeline readiness"
            description="Imports the native video and ML stack in an isolated process — the same check run before every job."
            actions={r && <Badge tone={r.ready ? 'success' : 'danger'} dot>{r.ready ? 'Ready' : 'Not ready'}</Badge>}
          />
          {ready.loading && !r ? (
            <div className="stack-sm"><Skeleton style={{ height: 40 }} /><Skeleton style={{ height: 40 }} /><Skeleton style={{ height: 40 }} /><p className="muted small">Probing dependencies — this can take up to a minute on first run.</p></div>
          ) : ready.error ? (
            <p className="alert alert-danger">{ready.error.message}</p>
          ) : (
            <ul className="readiness">
              {Object.entries(r.checks || {}).map(([name, c]) => (
                <li key={name} className={c.ok ? 'is-ok' : 'is-bad'}>
                  {c.ok ? <CheckCircle2 size={17} aria-hidden /> : <XCircle size={17} aria-hidden />}
                  <div>
                    <strong>{CHECK_LABELS[name] || name}</strong>
                    {!c.ok && <p>{c.message || 'Unavailable'}{c.module ? ` (${c.module})` : ''}</p>}
                  </div>
                </li>
              ))}
            </ul>
          )}
          {r?.versions && Object.keys(r.versions).length > 0 && (
            <dl className="config-list" style={{ marginTop: 16 }}>
              {Object.entries(r.versions).map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{String(v)}</dd></div>)}
            </dl>
          )}
        </Card>

        <Card className="danger-zone">
          <CardHeader title="Danger zone" />
          <p className="muted">Stop the backend process. Running jobs are cancelled. You’ll need to start the server again from your terminal or notebook.</p>
          <Button variant="danger" icon={Power} onClick={() => setConfirm(true)} style={{ marginTop: 12 }}>Shut down server</Button>
        </Card>
      </div>

      <ConfirmModal open={confirm} onClose={() => setConfirm(false)} onConfirm={shutdown} loading={busy} title="Shut down the server?" description="All running jobs will be cancelled and the Studio will disconnect." confirmLabel="Shut down" />
    </>
  )
}
