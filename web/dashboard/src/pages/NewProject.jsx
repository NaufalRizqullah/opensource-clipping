import { useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import {
  ArrowRight, Bot, Captions, ChevronDown, CloudUpload, FileVideo, Link2, Mic2, Music, Repeat2, Rocket,
  ScissorsLineDashed, Sparkles, Trash2, Wand2, Clapperboard, Layers, Users, Video,
} from 'lucide-react'
import { api, uploadVideo } from '../lib/api'
import { useDocumentTitle } from '../lib/hooks'
import { PLATFORM_LABEL, detectPlatform, formatBytes, isHttpUrl, youtubeId } from '../lib/format'
import { useToast } from '../lib/toast'
import { PageHeader } from '../components/Layout'
import { Button, Card, CardHeader, Field, Input, ProgressBar, Segmented, Select, Toggle } from '../components/ui'

const RATIOS = [
  { value: '9:16', label: 'Vertical', hint: 'Shorts · Reels · TikTok', w: 9, h: 16 },
  { value: '4:5', label: 'Portrait', hint: 'Feed posts', w: 4, h: 5 },
  { value: '1:1', label: 'Square', hint: 'Universal', w: 1, h: 1 },
  { value: '3:4', label: 'Classic', hint: 'Pinterest', w: 3, h: 4 },
  { value: '16:9', label: 'Landscape', hint: 'YouTube', w: 16, h: 9 },
]

const FONTS = [
  { value: 'HORMOZI', label: 'Hormozi', sample: 'STOP SCROLLING', cls: 'font-hormozi' },
  { value: 'DEFAULT', label: 'Montserrat', sample: 'Stop scrolling', cls: 'font-default' },
  { value: 'STORYTELLER', label: 'Storyteller', sample: 'Stop scrolling', cls: 'font-story' },
  { value: 'CINEMATIC', label: 'Cinematic', sample: 'STOP SCROLLING', cls: 'font-cine' },
]

const PRESETS = [
  { id: 'viral', label: 'Viral', icon: Sparkles, desc: 'Hook glitch, karaoke, B-roll, BGM',
    set: { use_broll: true, use_hook_glitch: true, use_auto_bgm: true, use_karaoke_effect: true, hook_v2: false, silence_trim: true, font_style: 'HORMOZI', use_split_screen: false, use_camera_switch: false, no_subs: false } },
  { id: 'podcast', label: 'Podcast', icon: Mic2, desc: 'Active-speaker camera switching',
    set: { use_broll: false, use_hook_glitch: true, use_auto_bgm: false, use_karaoke_effect: true, hook_v2: false, silence_trim: true, font_style: 'DEFAULT', use_split_screen: false, use_camera_switch: true, no_subs: false } },
  { id: 'retention', label: 'Retention', icon: Layers, desc: 'Multi-hook intro, trimmed pacing',
    set: { use_broll: true, use_hook_glitch: true, use_auto_bgm: true, use_karaoke_effect: true, hook_v2: true, silence_trim: true, font_style: 'HORMOZI', use_split_screen: false, use_camera_switch: false, no_subs: false } },
  { id: 'clean', label: 'Clean', icon: Clapperboard, desc: 'Cinematic captions, no extras',
    set: { use_broll: false, use_hook_glitch: false, use_auto_bgm: false, use_karaoke_effect: false, hook_v2: false, silence_trim: false, font_style: 'CINEMATIC', use_split_screen: false, use_camera_switch: false, no_subs: false } },
]

const DEFAULTS = {
  clips: 7, ratio: '9:16', font_style: 'HORMOZI', words_per_sub: 5, hook_duration: 3, render_height: '1080', source_height: 'max',
  use_broll: true, use_hook_glitch: true, use_auto_bgm: true, use_karaoke_effect: true, use_split_screen: false, use_camera_switch: false,
  no_subs: false, hook_v2: false, hook_v2_items: 3, no_segment_trim: false, silence_trim: false,
  whisper_model: 'large-v3', whisper_device: 'cuda', use_dlp_subs: false,
  ai_provider: 'gemini', gemini_model: 'gemini-3-flash-preview', vllm_model: '', local_max_tokens: '', face_detector: 'mediapipe', load_gemini_json: false,
}

const ALLOWED_EXT = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.ts']
const MAX_UPLOAD = 2 * 1024 ** 3

function computeType(device) {
  return device === 'cpu' ? 'int8' : device === 'auto' ? 'default' : 'float16'
}

export default function NewProject() {
  useDocumentTitle('New project')
  const navigate = useNavigate()
  const location = useLocation()
  const toast = useToast()
  const reuseJob = location.state?.reuseJob

  const [mode, setMode] = useState(reuseJob ? 'reuse' : 'url')
  const [url, setUrl] = useState('')
  const [source, setSource] = useState('youtube')
  const [reuseId, setReuseId] = useState(reuseJob?.id || '')
  const [cfg, setCfg] = useState(() => {
    const merged = { ...DEFAULTS }
    for (const [k, v] of Object.entries(reuseJob?.config || {})) {
      if (k in DEFAULTS && v !== null && v !== undefined) merged[k] = typeof DEFAULTS[k] === 'string' ? String(v) : v
    }
    return { ...merged, load_gemini_json: !!reuseJob }
  })
  const [preset, setPreset] = useState(reuseJob ? null : 'viral')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [settings, setSettings] = useState(null)
  const [errors, setErrors] = useState({})
  const [submitting, setSubmitting] = useState(false)

  // Upload state
  const fileInput = useRef(null)
  const uploadAbort = useRef(null)
  const [file, setFile] = useState(null)
  const [uploadPct, setUploadPct] = useState(0)
  const [uploaded, setUploaded] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const set = (patch) => { setCfg((c) => ({ ...c, ...patch })); setPreset(null) }

  useEffect(() => {
    const controller = new AbortController()
    api.settings({ signal: controller.signal }).then((s) => {
      setSettings(s)
      if (reuseJob) return
      setCfg((c) => ({
        ...c,
        clips: s.default_clips ?? c.clips,
        ratio: s.default_ratio || c.ratio,
        font_style: s.default_font_style || c.font_style,
        whisper_model: s.default_whisper_model || c.whisper_model,
        whisper_device: s.gpu_available ? (s.default_whisper_device || c.whisper_device) : 'cpu',
        ai_provider: s.default_ai_provider || c.ai_provider,
      }))
    }).catch(() => {})
    return () => controller.abort()
  }, [reuseJob])

  useEffect(() => () => uploadAbort.current?.abort(), [])

  const detected = useMemo(() => (isHttpUrl(url) ? detectPlatform(url) : null), [url])
  useEffect(() => { if (detected) setSource(detected) }, [detected])
  const ytId = youtubeId(url)

  const applyPreset = (p) => { setCfg((c) => ({ ...c, ...p.set })); setPreset(p.id) }

  const pickFile = async (f) => {
    if (!f) return
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!ALLOWED_EXT.includes(ext)) { setErrors({ file: `Unsupported file type ${ext || ''}. Use ${ALLOWED_EXT.join(', ')}` }); return }
    if (f.size > MAX_UPLOAD) { setErrors({ file: 'File is larger than 2 GB.' }); return }
    if (!f.size) { setErrors({ file: 'File is empty.' }); return }
    setErrors({})
    setFile(f)
    setUploaded(null)
    setUploadPct(0)
    setUploading(true)
    const controller = new AbortController()
    uploadAbort.current = controller
    try {
      const res = await uploadVideo(f, { onProgress: setUploadPct, signal: controller.signal })
      setUploaded(res)
      toast.success('Upload complete')
    } catch (err) {
      if (err.name !== 'AbortError') setErrors({ file: err.message })
      setFile(null)
    } finally {
      setUploading(false)
    }
  }

  const clearFile = () => {
    uploadAbort.current?.abort()
    setFile(null); setUploaded(null); setUploadPct(0)
  }

  const keyMissing = (() => {
    if (!settings) return null
    if (cfg.load_gemini_json && mode === 'reuse') return null
    if (cfg.ai_provider === 'gemini' && !settings.google_api_key_set) return 'Gemini API key is not configured.'
    if (cfg.ai_provider === 'nvidia' && !settings.nvidia_api_key_set) return 'NVIDIA API key is not configured.'
    return null
  })()

  const validate = () => {
    const e = {}
    if (mode === 'url' && !isHttpUrl(url)) e.url = 'Enter a valid video link (https://…)'
    if (mode === 'upload' && !uploaded) e.file = uploading ? 'Wait for the upload to finish.' : 'Choose a video file to upload.'
    if (mode === 'reuse' && !/^[A-Za-z0-9_-]{1,64}$/.test(reuseId.trim())) e.reuse = 'Enter a valid project ID.'
    if (cfg.ai_provider === 'vllm' && cfg.vllm_model && !/^[A-Za-z0-9._:/@+-]{1,256}$/.test(cfg.vllm_model)) e.vllm_model = 'Invalid model name.'
    if (cfg.ai_provider === 'vllm' && cfg.local_max_tokens !== '' && cfg.local_max_tokens != null) {
      const n = Number(cfg.local_max_tokens)
      if (!Number.isInteger(n) || n < 1 || n > 131072) e.local_max_tokens = 'Whole number between 1 and 131072.'
    }
    setErrors(e)
    return Object.keys(e).length === 0
  }

  const submit = async (ev) => {
    ev?.preventDefault()
    if (!validate()) return
    setSubmitting(true)
    const payload = {
      ...(mode === 'url' ? { url: url.trim(), source } : {}),
      ...(mode === 'upload' ? { upload_filename: uploaded.filename } : {}),
      ...(mode === 'reuse' ? { reuse_job_id: reuseId.trim(), load_gemini_json: !!cfg.load_gemini_json } : { load_gemini_json: false }),
      clips: Number(cfg.clips),
      ratio: cfg.ratio,
      font_style: cfg.font_style,
      words_per_sub: Number(cfg.words_per_sub),
      hook_duration: Number(cfg.hook_duration),
      render_height: String(cfg.render_height),
      source_height: String(cfg.source_height),
      use_broll: !!cfg.use_broll,
      use_hook_glitch: !!cfg.use_hook_glitch,
      use_auto_bgm: !!cfg.use_auto_bgm,
      use_karaoke_effect: !!cfg.use_karaoke_effect,
      use_split_screen: !!cfg.use_split_screen,
      use_camera_switch: !!cfg.use_camera_switch,
      no_subs: !!cfg.no_subs,
      hook_v2: !!cfg.hook_v2,
      hook_v2_items: Number(cfg.hook_v2_items),
      no_segment_trim: !!cfg.no_segment_trim,
      silence_trim: !!cfg.silence_trim,
      whisper_model: cfg.whisper_model,
      whisper_device: cfg.whisper_device,
      whisper_compute_type: computeType(cfg.whisper_device),
      use_dlp_subs: !!cfg.use_dlp_subs,
      ai_provider: cfg.ai_provider,
      gemini_model: cfg.gemini_model || DEFAULTS.gemini_model,
      face_detector: cfg.face_detector,
      ...(cfg.ai_provider === 'vllm' && cfg.vllm_model ? { vllm_model: cfg.vllm_model.trim() } : {}),
      ...(cfg.ai_provider === 'vllm' && cfg.local_max_tokens !== '' && cfg.local_max_tokens != null ? { local_max_tokens: Number(cfg.local_max_tokens) } : {}),
    }
    try {
      const job = await api.createJob(payload)
      toast.success('Project started')
      navigate(`/projects/${encodeURIComponent(job.id)}`)
    } catch (err) {
      toast.error(err.message)
      setErrors({ submit: err.message })
    } finally {
      setSubmitting(false)
    }
  }

  const ratio = RATIOS.find((r) => r.value === cfg.ratio) || RATIOS[0]
  const featureCount = ['use_broll', 'use_hook_glitch', 'use_auto_bgm', 'use_karaoke_effect', 'hook_v2', 'silence_trim', 'use_camera_switch', 'use_split_screen'].filter((k) => cfg[k]).length

  return (
    <form onSubmit={submit} noValidate>
      <PageHeader
        eyebrow="New project"
        title="What are we clipping today?"
        description="Drop in a long video. Choose a look. The pipeline does the rest."
      />

      <div className="new-grid">
        <div className="stack-lg">
          {/* Source */}
          <Card>
            <CardHeader title="Source" icon={Video} actions={
              <Segmented value={mode} onChange={(m) => { setMode(m); setErrors({}) }} label="Source type" size="sm" layoutId="source-mode" options={[
                { value: 'url', label: 'Link', icon: Link2 },
                { value: 'upload', label: 'Upload', icon: CloudUpload },
                { value: 'reuse', label: 'Reuse', icon: Repeat2 },
              ]} />
            } />
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={mode} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.18 }}>
                {mode === 'url' && (
                  <div className="source-url">
                    <Field label="Video link" htmlFor="src-url" error={errors.url} hint="YouTube, TikTok, Instagram or Google Drive. Platform is detected automatically.">
                      <Input id="src-url" icon={Link2} type="url" inputMode="url" placeholder="https://www.youtube.com/watch?v=…" value={url} onChange={(e) => setUrl(e.target.value)} autoFocus spellCheck={false} />
                    </Field>
                    <div className="row gap-md wrap align-end">
                      <Field label="Platform" htmlFor="src-platform">
                        <Select id="src-platform" value={source} onChange={(e) => setSource(e.target.value)}>
                          <option value="youtube">YouTube</option>
                          <option value="tiktok">TikTok</option>
                          <option value="instagram">Instagram</option>
                          <option value="gdrive">Google Drive</option>
                        </Select>
                      </Field>
                      {detected && <span className="badge badge-success">Detected: {PLATFORM_LABEL[detected]}</span>}
                    </div>
                    <AnimatePresence>
                      {ytId && (
                        <motion.div className="yt-preview" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
                          <img src={`https://i.ytimg.com/vi/${encodeURIComponent(ytId)}/hqdefault.jpg`} alt="Video thumbnail preview" loading="lazy" />
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}

                {mode === 'upload' && (
                  <div>
                    <input ref={fileInput} type="file" accept={ALLOWED_EXT.join(',') + ',video/*'} hidden onChange={(e) => { pickFile(e.target.files?.[0]); e.target.value = '' }} />
                    {!file ? (
                      <button
                        type="button"
                        className={`dropzone ${dragOver ? 'is-over' : ''} ${errors.file ? 'has-error' : ''}`}
                        onClick={() => fileInput.current?.click()}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={(e) => { e.preventDefault(); setDragOver(false); pickFile(e.dataTransfer.files?.[0]) }}
                      >
                        <span className="dropzone-icon"><CloudUpload size={22} aria-hidden /></span>
                        <strong>Drop a video here or click to browse</strong>
                        <span>MP4, MKV, MOV, WebM, AVI · up to 2 GB</span>
                      </button>
                    ) : (
                      <div className="file-row">
                        <span className="file-icon"><FileVideo size={18} aria-hidden /></span>
                        <div className="file-info">
                          <div className="row between gap-sm"><strong className="truncate">{file.name}</strong><span className="mono muted">{formatBytes(file.size)}</span></div>
                          <ProgressBar value={uploadPct * 100} tone={uploaded ? 'success' : 'accent'} label="Upload progress" />
                          <span className="field-hint">{uploaded ? 'Uploaded — ready to process' : `Uploading… ${Math.round(uploadPct * 100)}%`}</span>
                        </div>
                        <Button type="button" variant="ghost" size="sm" icon={Trash2} onClick={clearFile} aria-label={uploading ? 'Cancel upload' : 'Remove file'}>{uploading ? 'Cancel' : 'Remove'}</Button>
                      </div>
                    )}
                    {errors.file && <p className="field-error" role="alert">{errors.file}</p>}
                  </div>
                )}

                {mode === 'reuse' && (
                  <div className="stack-md">
                    <Field label="Project ID to reuse" htmlFor="src-reuse" error={errors.reuse} hint="Reuses the downloaded source of a finished project — skips the download step.">
                      <Input id="src-reuse" icon={Repeat2} value={reuseId} onChange={(e) => setReuseId(e.target.value)} placeholder="e.g. d20b47341e08" spellCheck={false} className="mono" />
                    </Field>
                    <Toggle label="Bypass AI analysis" description="Reuse the cached AI highlight selection to save credits." checked={cfg.load_gemini_json} onChange={(v) => setCfg((c) => ({ ...c, load_gemini_json: v }))} />
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </Card>

          {/* Presets */}
          <Card>
            <CardHeader title="Style preset" icon={Wand2} description="A starting point — fine-tune everything below." />
            <div className="preset-grid">
              {PRESETS.map((p) => (
                <button type="button" key={p.id} className={`preset ${preset === p.id ? 'is-active' : ''}`} onClick={() => applyPreset(p)} aria-pressed={preset === p.id}>
                  <span className="preset-icon"><p.icon size={18} aria-hidden /></span>
                  <strong>{p.label}</strong>
                  <span>{p.desc}</span>
                </button>
              ))}
            </div>
          </Card>

          {/* Output */}
          <Card>
            <CardHeader title="Output" icon={Clapperboard} />
            <div className="stack-lg">
              <Field label="Number of clips" htmlFor="clips" trailing={<span className="value-pill mono">{cfg.clips}</span>}>
                <input id="clips" type="range" min="1" max="30" value={cfg.clips} onChange={(e) => set({ clips: Number(e.target.value) })} className="range" style={{ '--pct': `${((cfg.clips - 1) / 29) * 100}%` }} />
              </Field>

              <div className="field">
                <span className="field-label">Aspect ratio</span>
                <div className="ratio-grid" role="radiogroup" aria-label="Aspect ratio">
                  {RATIOS.map((r) => {
                    const scale = 34 / Math.max(r.w, r.h)
                    return (
                      <button type="button" key={r.value} role="radio" aria-checked={cfg.ratio === r.value} className={`ratio ${cfg.ratio === r.value ? 'is-active' : ''}`} onClick={() => set({ ratio: r.value })}>
                        <span className="ratio-box-wrap"><span className="ratio-box" style={{ width: r.w * scale, height: r.h * scale }} /></span>
                        <strong>{r.value}</strong>
                        <span>{r.hint}</span>
                      </button>
                    )
                  })}
                </div>
              </div>

              <div className="field">
                <span className="field-label">Caption style</span>
                <div className="font-grid" role="radiogroup" aria-label="Caption style">
                  {FONTS.map((f) => (
                    <button type="button" key={f.value} role="radio" aria-checked={cfg.font_style === f.value} className={`font-card ${cfg.font_style === f.value ? 'is-active' : ''}`} onClick={() => set({ font_style: f.value })}>
                      <span className={`font-sample ${f.cls}`}>{f.sample.split(' ').map((w, i) => <span key={i} className={i === 1 ? 'hl' : ''}>{w} </span>)}</span>
                      <span className="font-name">{f.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          {/* Features */}
          <Card>
            <CardHeader title="Enhancements" icon={Sparkles} description={`${featureCount} enabled`} />
            <div className="toggle-grid">
              <Toggle label="Karaoke captions" description="Word-by-word highlight" checked={cfg.use_karaoke_effect} onChange={(v) => set({ use_karaoke_effect: v })} disabled={cfg.no_subs} />
              <Toggle label="Hook glitch intro" description="3s teaser with TV-glitch transition" checked={cfg.use_hook_glitch} onChange={(v) => set({ use_hook_glitch: v })} />
              <Toggle label="Multi-hook intro (V2)" description="3–6 micro-hooks before the clip" checked={cfg.hook_v2} onChange={(v) => set({ hook_v2: v })} />
              <Toggle label="B-roll footage" description={settings && !settings.pexels_api_key_set ? 'Needs a Pexels API key' : 'Contextual stock clips from Pexels'} checked={cfg.use_broll} onChange={(v) => set({ use_broll: v })} />
              <Toggle label="Background music" description="Auto BGM with voice ducking" checked={cfg.use_auto_bgm} onChange={(v) => set({ use_auto_bgm: v })} />
              <Toggle label="Silence trim" description="Cut dead air between phrases" checked={cfg.silence_trim} onChange={(v) => set({ silence_trim: v })} />
              <Toggle label="Camera switch" description={settings && !settings.hf_token_set ? 'Podcast mode · needs HF token' : 'Podcast · follow the active speaker'} checked={cfg.use_camera_switch} onChange={(v) => set({ use_camera_switch: v, ...(v ? { use_split_screen: false } : {}) })} />
              <Toggle label="Split screen" description={settings && !settings.hf_token_set ? 'Podcast mode · needs HF token' : 'Podcast · stacked speakers'} checked={cfg.use_split_screen} onChange={(v) => set({ use_split_screen: v, ...(v ? { use_camera_switch: false } : {}) })} />
              <Toggle label="No subtitles" description="Render clean, without captions" checked={cfg.no_subs} onChange={(v) => set({ no_subs: v })} />
            </div>
          </Card>

          {/* AI */}
          <Card>
            <CardHeader title="AI & transcription" icon={Bot} />
            <div className="form-grid">
              <Field label="AI provider" htmlFor="ai-provider">
                <Select id="ai-provider" value={cfg.ai_provider} onChange={(e) => set({ ai_provider: e.target.value })}>
                  <option value="gemini">Google Gemini</option>
                  <option value="nvidia">NVIDIA NIM</option>
                  <option value="vllm">vLLM (self-hosted)</option>
                </Select>
              </Field>
              {cfg.ai_provider === 'gemini' && (
                <Field label="Gemini model" htmlFor="gemini-model">
                  <Input id="gemini-model" value={cfg.gemini_model} onChange={(e) => set({ gemini_model: e.target.value })} spellCheck={false} className="mono" />
                </Field>
              )}
              {cfg.ai_provider === 'vllm' && (
                <>
                  <Field label="vLLM model" htmlFor="vllm-model" error={errors.vllm_model} hint="Empty uses the server default.">
                    <Input id="vllm-model" value={cfg.vllm_model} onChange={(e) => set({ vllm_model: e.target.value })} placeholder={settings?.vllm_model || 'Server default'} className="mono" maxLength={256} />
                  </Field>
                  <Field label="Max output tokens" htmlFor="vllm-tokens" error={errors.local_max_tokens}>
                    <Input id="vllm-tokens" type="number" min="1" max="131072" value={cfg.local_max_tokens} onChange={(e) => set({ local_max_tokens: e.target.value })} placeholder={String(settings?.local_max_tokens || 4096)} />
                  </Field>
                </>
              )}
              <Field label="Whisper model" htmlFor="whisper-model">
                <Select id="whisper-model" value={cfg.whisper_model} onChange={(e) => set({ whisper_model: e.target.value })}>
                  <option value="large-v3">large-v3 · best quality</option>
                  <option value="medium">medium · balanced</option>
                  <option value="small">small · fast</option>
                  <option value="base">base · fastest</option>
                </Select>
              </Field>
              <Field label="Device" htmlFor="whisper-device" hint={settings && !settings.gpu_available ? 'No GPU detected on the server.' : undefined}>
                <Select id="whisper-device" value={cfg.whisper_device} onChange={(e) => set({ whisper_device: e.target.value })}>
                  <option value="cuda">CUDA GPU</option>
                  <option value="cpu">CPU</option>
                  <option value="auto">Auto</option>
                </Select>
              </Field>
            </div>
            <div className="stack-sm" style={{ marginTop: 14 }}>
              <Toggle label="Use platform subtitles" description="Skip Whisper when the video already has subtitles" checked={cfg.use_dlp_subs} onChange={(v) => set({ use_dlp_subs: v })} />
            </div>
          </Card>

          {/* Advanced */}
          <Card>
            <button type="button" className="collapse-btn" onClick={() => setShowAdvanced((v) => !v)} aria-expanded={showAdvanced}>
              <span className="card-title"><ScissorsLineDashed size={16} aria-hidden />Advanced</span>
              <motion.span animate={{ rotate: showAdvanced ? 180 : 0 }}><ChevronDown size={18} aria-hidden /></motion.span>
            </button>
            <AnimatePresence initial={false}>
              {showAdvanced && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }} style={{ overflow: 'hidden' }}>
                  <div className="form-grid" style={{ paddingTop: 16 }}>
                    <Field label="Words per caption" htmlFor="wps" trailing={<span className="value-pill mono">{cfg.words_per_sub}</span>}>
                      <input id="wps" type="range" min="1" max="15" value={cfg.words_per_sub} onChange={(e) => set({ words_per_sub: Number(e.target.value) })} className="range" style={{ '--pct': `${((cfg.words_per_sub - 1) / 14) * 100}%` }} />
                    </Field>
                    <Field label="Hook duration (s)" htmlFor="hookdur" trailing={<span className="value-pill mono">{cfg.hook_duration}s</span>}>
                      <input id="hookdur" type="range" min="1" max="10" value={cfg.hook_duration} onChange={(e) => set({ hook_duration: Number(e.target.value) })} className="range" style={{ '--pct': `${((cfg.hook_duration - 1) / 9) * 100}%` }} />
                    </Field>
                    {cfg.hook_v2 && (
                      <Field label="Micro-hooks" htmlFor="hookitems" trailing={<span className="value-pill mono">{cfg.hook_v2_items}</span>}>
                        <input id="hookitems" type="range" min="2" max="6" value={cfg.hook_v2_items} onChange={(e) => set({ hook_v2_items: Number(e.target.value) })} className="range" style={{ '--pct': `${((cfg.hook_v2_items - 2) / 4) * 100}%` }} />
                      </Field>
                    )}
                    <Field label="Render height" htmlFor="rh">
                      <Select id="rh" value={cfg.render_height} onChange={(e) => set({ render_height: e.target.value })}>
                        <option value="720">720p</option>
                        <option value="1080">1080p</option>
                        <option value="1440">1440p</option>
                        <option value="2160">2160p (4K)</option>
                      </Select>
                    </Field>
                    <Field label="Source download quality" htmlFor="sh">
                      <Select id="sh" value={cfg.source_height} onChange={(e) => set({ source_height: e.target.value })}>
                        <option value="max">Best available</option>
                        <option value="2160">Up to 2160p</option>
                        <option value="1440">Up to 1440p</option>
                        <option value="1080">Up to 1080p</option>
                        <option value="720">Up to 720p</option>
                      </Select>
                    </Field>
                    <Field label="Face detector" htmlFor="fd">
                      <Select id="fd" value={cfg.face_detector} onChange={(e) => set({ face_detector: e.target.value })}>
                        <option value="mediapipe">MediaPipe BlazeFace</option>
                        <option value="yolo">YOLO</option>
                      </Select>
                    </Field>
                  </div>
                  <div style={{ marginTop: 14 }}>
                    <Toggle label="Disable smart segment trimming" description="Keep each highlight as one continuous cut" checked={cfg.no_segment_trim} onChange={(v) => set({ no_segment_trim: v })} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>
        </div>

        {/* Summary */}
        <aside className="new-summary">
          <Card className="summary-card">
            <div className="summary-preview">
              <div className="summary-frame" style={{ aspectRatio: `${ratio.w} / ${ratio.h}` }}>
                <motion.div key={cfg.font_style} className={`summary-caption ${FONTS.find((f) => f.value === cfg.font_style)?.cls}`} initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
                  {cfg.no_subs ? <span className="muted">No captions</span> : <>this changes <span className="hl">everything</span></>}
                </motion.div>
              </div>
            </div>
            <dl className="summary-list">
              <div><dt>Clips</dt><dd>{cfg.clips}</dd></div>
              <div><dt>Format</dt><dd>{ratio.value} · {cfg.render_height}p</dd></div>
              <div><dt>Captions</dt><dd>{cfg.no_subs ? 'Off' : FONTS.find((f) => f.value === cfg.font_style)?.label}</dd></div>
              <div><dt>AI</dt><dd>{cfg.ai_provider === 'gemini' ? 'Gemini' : cfg.ai_provider === 'nvidia' ? 'NVIDIA NIM' : 'vLLM'}</dd></div>
              <div><dt>Whisper</dt><dd>{cfg.use_dlp_subs ? 'Platform subs' : `${cfg.whisper_model} · ${cfg.whisper_device}`}</dd></div>
              <div><dt>Extras</dt><dd className="summary-tags">
                {cfg.use_hook_glitch && <span>Hook</span>}{cfg.hook_v2 && <span>Multi-hook</span>}{cfg.use_broll && <span>B-roll</span>}
                {cfg.use_auto_bgm && <span><Music size={10} aria-hidden />BGM</span>}{cfg.use_camera_switch && <span><Users size={10} aria-hidden />Cam switch</span>}
                {cfg.use_split_screen && <span>Split</span>}{cfg.silence_trim && <span>Trim</span>}{cfg.use_karaoke_effect && !cfg.no_subs && <span><Captions size={10} aria-hidden />Karaoke</span>}
                {!featureCount && <span className="muted">None</span>}
              </dd></div>
            </dl>
            {keyMissing && <p className="alert alert-warning">{keyMissing} <a href="/settings?tab=keys" onClick={(e) => { e.preventDefault(); navigate('/settings?tab=keys') }}>Add it in Settings</a>.</p>}
            {errors.submit && <p className="alert alert-danger" role="alert">{errors.submit}</p>}
            <Button type="submit" variant="primary" size="lg" icon={Rocket} iconRight={ArrowRight} loading={submitting} disabled={uploading} className="w-full">
              {submitting ? 'Starting…' : 'Generate clips'}
            </Button>
            <p className="field-hint center">Processing runs on the connected server. You can leave this page.</p>
          </Card>
        </aside>
      </div>
    </form>
  )
}
