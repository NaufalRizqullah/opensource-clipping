import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import {
  ArrowLeft, ChevronLeft, ChevronRight, Download, FileJson, Image as ImageIcon, RotateCcw, SquarePlay, Music2, Info, X, Plus, ExternalLink, Keyboard,
} from 'lucide-react'
import { mediaUrl } from '../lib/api'
import { useDocumentTitle, useLiveJob, useHotkeys } from '../lib/hooks'
import { clipTitle, formatDuration, youtubeId } from '../lib/format'
import { storage } from '../lib/storage'
import { useToast } from '../lib/toast'
import VideoPlayer from '../components/VideoPlayer'
import { Badge, Button, Card, CopyButton, EmptyState, Field, IconButton, Input, Kbd, ScoreRing, Segmented, Skeleton, Textarea } from '../components/ui'

const LIMITS = { title: 100, description: 5000, tagsTotal: 500, tiktok: 2200 }

function aiDraft(clip) {
  const m = clip?.metadata || {}
  const tags = Array.isArray(m.youtube_tags_final) ? m.youtube_tags_final : typeof m.youtube_tags_final === 'string' ? m.youtube_tags_final.split(',') : []
  return {
    title: String(m.youtube_title_final || clip?.title_en || clip?.title || ''),
    description: String(m.youtube_description_final || ''),
    tags: tags.map((t) => String(t).trim()).filter(Boolean),
    tiktok: String(m.tiktok_caption_final || ''),
  }
}

function Counter({ value, max }) {
  const n = value.length
  return <span className={`counter mono ${n > max ? 'is-over' : n > max * 0.9 ? 'is-near' : ''}`}>{n}/{max}</span>
}

export default function ClipEditor() {
  const { jobId, rank } = useParams()
  const navigate = useNavigate()
  const toast = useToast()
  const { data: job, loading, error } = useLiveJob(jobId)
  const clips = useMemo(() => [...(job?.clips || [])].sort((a, b) => a.rank - b.rank), [job?.clips])
  const index = clips.findIndex((c) => String(c.rank) === String(rank))
  const clip = clips[index]
  const prev = clips[index - 1]
  const next = clips[index + 1]

  const draftKey = `osc.draft.${jobId}.${rank}`
  const [draft, setDraft] = useState(null)
  const [tab, setTab] = useState('youtube')
  const [tagInput, setTagInput] = useState('')
  const [showKeys, setShowKeys] = useState(false)

  useDocumentTitle(clip ? clipTitle(clip) : 'Clip')

  useEffect(() => {
    if (!clip) return
    setDraft(storage.getJSON(draftKey, null) || aiDraft(clip))
    // Re-init only when switching clips, not on every background refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftKey, !!clip])

  useEffect(() => {
    if (!draft || !clip) return
    const t = setTimeout(() => storage.setJSON(draftKey, draft), 300)
    return () => clearTimeout(t)
  }, [draft, draftKey, clip])

  const go = (c) => c && navigate(`/projects/${encodeURIComponent(jobId)}/clips/${c.rank}`, { replace: true })
  useHotkeys({ '[': () => go(prev), ']': () => go(next), '?': () => setShowKeys((v) => !v) })

  if (loading && !job) {
    return <div className="editor-grid"><Skeleton style={{ aspectRatio: '9/16', maxHeight: '78vh' }} /><Skeleton style={{ height: 480 }} /></div>
  }
  if (error && !job) return <EmptyState title="Could not load clip" description={error.message} action={<Button onClick={() => navigate('/projects')}>Back</Button>} />
  if (!clip) {
    return <EmptyState title="Clip not found" description="This clip doesn’t exist in the project." action={<Button icon={ArrowLeft} onClick={() => navigate(`/projects/${encodeURIComponent(jobId)}`)}>Back to project</Button>} />
  }

  const ai = aiDraft(clip)
  const dirty = draft && JSON.stringify(draft) !== JSON.stringify(ai)
  const update = (patch) => setDraft((d) => ({ ...d, ...patch }))
  const tagsTotal = draft ? draft.tags.join(',').length : 0

  const addTags = (raw) => {
    const incoming = raw.split(',').map((t) => t.trim().replace(/^#/, '')).filter(Boolean)
    if (!incoming.length) return
    setDraft((d) => ({ ...d, tags: Array.from(new Set([...d.tags, ...incoming])) }))
    setTagInput('')
  }

  const reset = () => { setDraft(ai); storage.remove(draftKey); toast.info('Restored the AI-generated metadata') }

  const exportJson = () => {
    const data = {
      project: jobId, rank: clip.rank, file: clip.filename, duration: clip.duration, viral_score: clip.viral_score,
      youtube: { title: draft.title, description: draft.description, tags: draft.tags },
      tiktok: { caption: draft.tiktok },
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `clip-${clip.rank}-metadata.json`
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }

  const ytId = youtubeId(job.url)
  const sourceLink = ytId && clip.start_time != null ? `https://www.youtube.com/watch?v=${encodeURIComponent(ytId)}&t=${Math.floor(clip.start_time)}s` : null
  const fullDescription = draft ? [draft.description, draft.tags.length ? draft.tags.map((t) => `#${t.replace(/\s+/g, '')}`).join(' ') : ''].filter(Boolean).join('\n\n') : ''

  return (
    <>
      <div className="editor-top">
        <Link to={`/projects/${encodeURIComponent(jobId)}`} className="back-link"><ArrowLeft size={14} aria-hidden />Project</Link>
        <div className="editor-nav">
          <IconButton icon={ChevronLeft} label="Previous clip ([)" onClick={() => go(prev)} disabled={!prev} />
          <span className="mono muted">{index + 1} / {clips.length}</span>
          <IconButton icon={ChevronRight} label="Next clip (])" onClick={() => go(next)} disabled={!next} />
          <IconButton icon={Keyboard} label="Keyboard shortcuts (?)" onClick={() => setShowKeys((v) => !v)} />
        </div>
      </div>

      <div className="editor-grid">
        <div className="editor-stage">
          <motion.div key={clip.rank} initial={{ opacity: 0, scale: 0.985 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}>
            <VideoPlayer src={mediaUrl(clip.download_url)} poster={clip.thumbnail_url ? mediaUrl(clip.thumbnail_url) : undefined} ratio={job.config?.ratio || '9:16'} />
          </motion.div>
          <AnimatePresence>
            {showKeys && (
              <motion.div className="keys-card" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
                <div className="row between"><strong>Shortcuts</strong><IconButton icon={X} size="sm" label="Close" onClick={() => setShowKeys(false)} /></div>
                <ul>
                  <li><Kbd>Space</Kbd>/<Kbd>K</Kbd> Play / pause</li>
                  <li><Kbd>J</Kbd>/<Kbd>L</Kbd> Back / forward 5s</li>
                  <li><Kbd>←</Kbd>/<Kbd>→</Kbd> Step 1s</li>
                  <li><Kbd>M</Kbd> Mute · <Kbd>F</Kbd> Fullscreen</li>
                  <li><Kbd>[</Kbd>/<Kbd>]</Kbd> Previous / next clip</li>
                </ul>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="editor-panel stack-md">
          <Card className="editor-head">
            <div className="row gap-md align-start">
              <div className="grow min-0">
                <p className="eyebrow">Clip #{clip.rank}</p>
                <h1 className="editor-title">{clipTitle(clip)}</h1>
                <div className="row gap-sm wrap" style={{ marginTop: 8 }}>
                  <Badge>{formatDuration(clip.duration)}</Badge>
                  {clip.start_time != null && <Badge>Source {formatDuration(clip.start_time)}–{formatDuration(clip.end_time)}</Badge>}
                  {dirty && <Badge tone="warning">Edited</Badge>}
                </div>
              </div>
              <ScoreRing score={clip.viral_score} size={52} />
            </div>
            <div className="row gap-sm wrap" style={{ marginTop: 16 }}>
              <a className="btn btn-primary btn-md" href={mediaUrl(clip.download_url)} download={clip.filename}><Download size={16} aria-hidden /><span>Download MP4</span></a>
              {clip.thumbnail_url && <a className="btn btn-secondary btn-md" href={mediaUrl(clip.thumbnail_url)} download><ImageIcon size={16} aria-hidden /><span>Thumbnail</span></a>}
              <Button icon={FileJson} variant="ghost" onClick={exportJson} disabled={!draft}>Export</Button>
              {sourceLink && <a className="btn btn-ghost btn-md" href={sourceLink} target="_blank" rel="noopener noreferrer"><ExternalLink size={16} aria-hidden /><span>In source</span></a>}
            </div>
          </Card>

          {draft && (
            <Card>
              <div className="row between gap-sm wrap" style={{ marginBottom: 16 }}>
                <Segmented value={tab} onChange={setTab} label="Platform" size="sm" layoutId="editor-tab" options={[
                  { value: 'youtube', label: 'YouTube', icon: SquarePlay },
                  { value: 'tiktok', label: 'TikTok / Reels', icon: Music2 },
                ]} />
                <Button size="sm" variant="ghost" icon={RotateCcw} onClick={reset} disabled={!dirty}>Reset to AI</Button>
              </div>

              <AnimatePresence mode="wait" initial={false}>
                {tab === 'youtube' ? (
                  <motion.div key="yt" className="stack-md" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 8 }} transition={{ duration: 0.16 }}>
                    <Field label="Title" htmlFor="m-title" trailing={<span className="row gap-xs"><Counter value={draft.title} max={LIMITS.title} /><CopyButton value={draft.title} label="" /></span>}>
                      <Input id="m-title" value={draft.title} onChange={(e) => update({ title: e.target.value })} />
                    </Field>
                    <Field label="Description" htmlFor="m-desc" trailing={<span className="row gap-xs"><Counter value={draft.description} max={LIMITS.description} /><CopyButton value={fullDescription} label="" /></span>}>
                      <Textarea id="m-desc" rows={6} value={draft.description} onChange={(e) => update({ description: e.target.value })} />
                    </Field>
                    <Field label="Tags" htmlFor="m-tags" trailing={<span className="row gap-xs"><Counter value={'x'.repeat(tagsTotal)} max={LIMITS.tagsTotal} /><CopyButton value={draft.tags.join(', ')} label="" /></span>} hint="Press Enter or comma to add. Paste a comma-separated list to add many.">
                      <div className="tag-input">
                        <AnimatePresence initial={false}>
                          {draft.tags.map((t) => (
                            <motion.span key={t} className="tag" layout initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }}>
                              {t}
                              <button type="button" aria-label={`Remove tag ${t}`} onClick={() => update({ tags: draft.tags.filter((x) => x !== t) })}><X size={11} /></button>
                            </motion.span>
                          ))}
                        </AnimatePresence>
                        <input
                          id="m-tags"
                          value={tagInput}
                          placeholder={draft.tags.length ? '' : 'Add tags…'}
                          onChange={(e) => (e.target.value.includes(',') ? addTags(e.target.value) : setTagInput(e.target.value))}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') { e.preventDefault(); addTags(tagInput) }
                            if (e.key === 'Backspace' && !tagInput && draft.tags.length) update({ tags: draft.tags.slice(0, -1) })
                          }}
                          onBlur={() => addTags(tagInput)}
                        />
                        {tagInput && <button type="button" className="icon-btn icon-btn-sm" onClick={() => addTags(tagInput)} aria-label="Add tag"><Plus size={13} /></button>}
                      </div>
                    </Field>
                  </motion.div>
                ) : (
                  <motion.div key="tt" className="stack-md" initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -8 }} transition={{ duration: 0.16 }}>
                    <Field label="Caption" htmlFor="m-tt" trailing={<span className="row gap-xs"><Counter value={draft.tiktok} max={LIMITS.tiktok} /><CopyButton value={draft.tiktok} label="" /></span>} hint="Works for TikTok, Instagram Reels and Facebook Reels.">
                      <Textarea id="m-tt" rows={8} value={draft.tiktok} onChange={(e) => update({ tiktok: e.target.value })} />
                    </Field>
                    <div className="caption-preview" aria-label="Caption preview">
                      <span className="caption-handle">@yourchannel</span>
                      <p>{draft.tiktok || <span className="muted">Your caption preview appears here.</span>}</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <p className="field-hint row gap-xs" style={{ marginTop: 14 }}><Info size={12} aria-hidden />Edits are saved automatically in this browser.</p>
            </Card>
          )}
        </div>
      </div>
    </>
  )
}
