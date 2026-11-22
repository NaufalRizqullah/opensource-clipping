import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import { Check, Clapperboard, Clock, Film, Link2, Upload, X, SquarePlay, Music2, Camera, HardDrive } from 'lucide-react'
import { mediaUrl } from '../lib/api'
import {
  PIPELINE_STEPS, clipTitle, sourceLabel, formatDuration, isActive, jobPercent, jobThumbnail, jobTitle, plural, timeAgo,
} from '../lib/format'
import { ProgressBar, ScoreRing, StatusBadge } from './ui'

const SOURCE_ICON = { youtube: SquarePlay, tiktok: Music2, instagram: Camera, gdrive: HardDrive }

export function SourceIcon({ job, size = 14 }) {
  const Icon = job?.upload_filename ? Upload : SOURCE_ICON[job?.source] || Link2
  return <Icon size={size} aria-hidden />
}

export function ProjectCard({ job, index = 0 }) {
  const thumb = jobThumbnail(job, mediaUrl)
  const active = isActive(job)
  const percent = jobPercent(job)
  const [imgOk, setImgOk] = useState(true)

  return (
    <motion.article
      className="project-card"
      layout
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.32, delay: Math.min(index, 10) * 0.035, ease: [0.22, 1, 0.36, 1] }}
    >
      <Link to={`/projects/${encodeURIComponent(job.id)}`} className="project-link" aria-label={`Open ${jobTitle(job)}`}>
        <div className="project-thumb">
          {thumb && imgOk ? (
            <img src={thumb} alt="" loading="lazy" decoding="async" onError={() => setImgOk(false)} />
          ) : (
            <div className="thumb-placeholder"><Clapperboard size={26} aria-hidden /></div>
          )}
          <div className="project-thumb-overlay" />
          <div className="project-thumb-top"><StatusBadge status={job.status} /></div>
          {job.clips?.length > 0 && (
            <div className="project-thumb-bottom"><span className="chip-glass"><Film size={12} aria-hidden />{plural(job.clips.length, 'clip')}</span></div>
          )}
        </div>
        <div className="project-body">
          <h3 className="project-title" title={jobTitle(job)}>{jobTitle(job)}</h3>
          <div className="project-meta">
            <span className="meta-item"><SourceIcon job={job} />{sourceLabel(job)}</span>
            <span className="meta-sep" aria-hidden>•</span>
            <span className="meta-item"><Clock size={12} aria-hidden />{timeAgo(job.created_at)}</span>
            <span className="meta-sep" aria-hidden>•</span>
            <span className="meta-item mono">{job.config?.ratio || '9:16'}</span>
          </div>
          {active && (
            <div className="project-progress">
              <ProgressBar value={percent} indeterminate={job.status === 'queued'} label="Progress" />
              <div className="project-progress-text">
                <span className="truncate">{job.cancel_requested ? 'Cancelling…' : job.progress?.message || 'Waiting in queue…'}</span>
                <span className="mono">{Math.round(percent)}%</span>
              </div>
            </div>
          )}
          {job.status === 'failed' && job.error && <p className="project-error truncate" title={job.error}>{job.error}</p>}
        </div>
      </Link>
    </motion.article>
  )
}

export function ProjectCardSkeleton() {
  return (
    <div className="project-card is-skeleton" aria-hidden>
      <div className="project-thumb skeleton" />
      <div className="project-body">
        <div className="skeleton" style={{ height: 16, width: '80%' }} />
        <div className="skeleton" style={{ height: 12, width: '55%', marginTop: 10 }} />
      </div>
    </div>
  )
}

export function ClipCard({ clip, jobId, index = 0 }) {
  const videoRef = useRef(null)
  const [hover, setHover] = useState(false)
  const [thumbOk, setThumbOk] = useState(true)
  const thumb = clip.thumbnail_url ? mediaUrl(clip.thumbnail_url) : null
  const src = mediaUrl(clip.download_url)

  const start = () => {
    setHover(true)
    const v = videoRef.current
    if (v) { v.currentTime = 0; v.play().catch(() => {}) }
  }
  const stop = () => {
    setHover(false)
    videoRef.current?.pause()
  }

  return (
    <motion.article
      className="clip-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.36, delay: Math.min(index, 12) * 0.04, ease: [0.22, 1, 0.36, 1] }}
    >
      <Link
        to={`/projects/${encodeURIComponent(jobId)}/clips/${clip.rank}`}
        className="clip-link"
        onMouseEnter={start}
        onMouseLeave={stop}
        onFocus={start}
        onBlur={stop}
        aria-label={`Open clip ${clip.rank}: ${clipTitle(clip)}`}
      >
        <div className="clip-media">
          {thumb && thumbOk && <img src={thumb} alt="" loading="lazy" decoding="async" className={hover ? 'is-hidden' : ''} onError={() => setThumbOk(false)} />}
          <video ref={videoRef} src={src} muted playsInline loop preload={thumb && thumbOk ? 'none' : 'metadata'} className={hover || !thumb || !thumbOk ? '' : 'is-hidden'} />
          <div className="clip-media-overlay" />
          <span className="clip-rank">#{clip.rank}</span>
          {clip.duration != null && <span className="clip-duration mono">{formatDuration(clip.duration)}</span>}
        </div>
        <div className="clip-body">
          <h4 className="clip-title">{clipTitle(clip)}</h4>
          <ScoreRing score={clip.viral_score} size={34} />
        </div>
      </Link>
    </motion.article>
  )
}

export function PipelineStepper({ job }) {
  const status = job?.status
  const current = job?.progress?.step_number || (status === 'completed' ? 7 : status === 'queued' ? 0 : 1)
  const failed = status === 'failed' || status === 'cancelled'

  return (
    <ol className="stepper" aria-label="Pipeline progress">
      {PIPELINE_STEPS.map((step, i) => {
        const n = i + 1
        const done = status === 'completed' || n < current
        const now = !done && n === current && !failed && status !== 'queued'
        const broken = failed && n === current
        const state = done ? 'done' : now ? 'active' : broken ? 'failed' : 'pending'
        return (
          <li key={step.key} className={`step step-${state}`} aria-current={now ? 'step' : undefined}>
            <span className="step-dot">
              {done ? <Check size={12} strokeWidth={3} aria-hidden /> : broken ? <X size={12} strokeWidth={3} aria-hidden /> : <span className="step-n">{n}</span>}
              {now && <motion.span className="step-pulse" animate={{ scale: [1, 1.8], opacity: [0.5, 0] }} transition={{ duration: 1.6, repeat: Infinity, ease: 'easeOut' }} />}
            </span>
            <span className="step-label">{step.label}</span>
            {i < PIPELINE_STEPS.length - 1 && (
              <span className="step-line" aria-hidden>
                <motion.span className="step-line-fill" initial={false} animate={{ scaleX: done ? 1 : 0 }} transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }} />
              </span>
            )}
          </li>
        )
      })}
    </ol>
  )
}
