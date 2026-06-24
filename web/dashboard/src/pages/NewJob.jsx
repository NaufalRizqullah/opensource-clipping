import { useState, useRef, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { createJob, uploadVideo } from '../api'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
} from '@/components/ui/select'

function Field({ label, children, hint }) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

function ToggleRow({ label, desc, checked, onChange }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4 rounded-lg px-2 py-2 transition-colors hover:bg-secondary/50">
      <div className="min-w-0">
        <div className="text-sm font-semibold">{label}</div>
        {desc && <div className="text-xs text-muted-foreground">{desc}</div>}
      </div>
      <Switch checked={checked} onCheckedChange={onChange} />
    </label>
  )
}

function SectionTitle({ children }) {
  return <h3 className="mb-4 font-display text-sm font-bold uppercase tracking-wider text-muted-foreground">{children}</h3>
}

function NewJob() {
  const navigate = useNavigate()
  const location = useLocation()
  const fileRef = useRef(null)

  const [mode, setMode] = useState('url')
  const [url, setUrl] = useState('')
  const [uploadFilename, setUploadFilename] = useState('')
  const [uploading, setUploading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [reuseJobId, setReuseJobId] = useState('')

  // Config
  const [clips, setClips] = useState(7)
  const [ratio, setRatio] = useState('9:16')
  const [source, setSource] = useState('youtube')
  const [fontStyle, setFontStyle] = useState('HORMOZI')
  const [whisperModel, setWhisperModel] = useState('base')
  const [whisperDevice, setWhisperDevice] = useState('cpu')
  const [aiProvider, setAiProvider] = useState('gemini')

  // Toggles
  const [useBroll, setUseBroll] = useState(true)
  const [useHookGlitch, setUseHookGlitch] = useState(true)
  const [useBgm, setUseBgm] = useState(false)
  const [useKaraoke, setUseKaraoke] = useState(true)
  const [noSubs, setNoSubs] = useState(false)
  const [hookV2, setHookV2] = useState(false)
  const [silenceTrim, setSilenceTrim] = useState(false)
  const [useDlpSubs, setUseDlpSubs] = useState(false)
  const [loadGeminiJson, setLoadGeminiJson] = useState(false)
  const [useFitBlur, setUseFitBlur] = useState(true)

  useEffect(() => {
    const reuseJob = location.state?.reuseJob
    if (reuseJob) {
      setReuseJobId(reuseJob.id)
      setUrl(reuseJob.url || '')
      setUploadFilename(reuseJob.upload_filename || '')
      setMode('reuse')
      setSource(reuseJob.source || 'youtube')

      const config = reuseJob.config || {}
      if (config.clips !== undefined) setClips(config.clips)
      if (config.ratio !== undefined) setRatio(config.ratio)
      if (config.font_style !== undefined) setFontStyle(config.font_style)
      if (config.whisper_model !== undefined) setWhisperModel(config.whisper_model)
      if (config.whisper_device !== undefined) setWhisperDevice(config.whisper_device)
      if (config.ai_provider !== undefined) setAiProvider(config.ai_provider)
      if (config.use_broll !== undefined) setUseBroll(config.use_broll)
      if (config.use_hook_glitch !== undefined) setUseHookGlitch(config.use_hook_glitch)
      if (config.use_auto_bgm !== undefined) setUseBgm(config.use_auto_bgm)
      if (config.use_karaoke_effect !== undefined) setUseKaraoke(config.use_karaoke_effect)
      if (config.hook_v2 !== undefined) setHookV2(config.hook_v2)
      if (config.silence_trim !== undefined) setSilenceTrim(config.silence_trim)
      if (config.use_dlp_subs !== undefined) setUseDlpSubs(config.use_dlp_subs)
      if (config.no_subs !== undefined) setNoSubs(config.no_subs)
      if (config.use_fit_blur !== undefined) setUseFitBlur(config.use_fit_blur)
      setLoadGeminiJson(true)
    }
  }, [location.state])

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError('')
    try {
      const result = await uploadVideo(file)
      setUploadFilename(result.filename)
      setMode('upload')
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (mode === 'url' && !url.trim()) return setError('URL cannot be empty')
    if (mode === 'upload' && !uploadFilename) return setError('Please upload a video first')
    if (mode === 'reuse' && !reuseJobId.trim()) return setError('Job ID cannot be empty')

    setSubmitting(true)
    try {
      const payload = {
        ...(mode === 'url' ? { url: url.trim() } : { upload_filename: uploadFilename }),
        source,
        clips,
        ratio,
        font_style: fontStyle,
        whisper_model: whisperModel,
        whisper_device: whisperDevice,
        ai_provider: aiProvider,
        use_broll: useBroll,
        use_hook_glitch: useHookGlitch,
        use_auto_bgm: useBgm,
        use_karaoke_effect: useKaraoke,
        no_subs: noSubs,
        hook_v2: hookV2,
        silence_trim: silenceTrim,
        use_dlp_subs: useDlpSubs,
        load_gemini_json: loadGeminiJson,
        use_fit_blur: useFitBlur,
        ...(reuseJobId.trim() ? { reuse_job_id: reuseJobId.trim() } : {}),
      }
      const job = await createJob(payload)
      navigate(`/job/${job.id}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const featureToggles = [
    { label: 'Fit (no crop)', desc: 'Whole frame, blurred bars, nothing cut', checked: useFitBlur, onChange: setUseFitBlur },
    { label: 'B-Roll Footage', desc: 'Insert contextual stock footage', checked: useBroll, onChange: setUseBroll },
    { label: 'Hook Glitch', desc: 'Glitch transition intro', checked: useHookGlitch, onChange: setUseHookGlitch },
    { label: 'Background Music', desc: 'Auto BGM matching the mood', checked: useBgm, onChange: setUseBgm },
    { label: 'Karaoke Effect', desc: 'Word-by-word highlight', checked: useKaraoke, onChange: setUseKaraoke },
    { label: 'Hook V2', desc: 'Multi-hook intro clips', checked: hookV2, onChange: setHookV2 },
    { label: 'Silence Trim', desc: 'Remove dead air', checked: silenceTrim, onChange: setSilenceTrim },
    { label: 'YouTube Subs', desc: 'Skip Whisper if subs exist', checked: useDlpSubs, onChange: setUseDlpSubs },
    { label: 'No Subtitles', desc: 'Render without text', checked: noSubs, onChange: setNoSubs },
    { label: 'Bypass AI', desc: 'Reuse existing Gemini JSON', checked: loadGeminiJson, onChange: setLoadGeminiJson },
  ]

  return (
    <div className="animate-in-up space-y-8">
      <div>
        <h1 className="font-display text-4xl font-bold tracking-tight">New Job</h1>
        <p className="mt-1.5 text-muted-foreground">Generate viral short clips from a long video</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Source */}
        <Card className="p-6">
          <SectionTitle>Video Source</SectionTitle>
          <Tabs value={mode} onValueChange={setMode}>
            <TabsList>
              <TabsTrigger value="url">URL</TabsTrigger>
              <TabsTrigger value="upload">Upload</TabsTrigger>
              <TabsTrigger value="reuse">Reuse</TabsTrigger>
            </TabsList>

            <TabsContent value="url">
              <div className="grid gap-4 sm:grid-cols-[1fr_200px]">
                <Field label="Video URL" hint="Supports YouTube, TikTok, Instagram, Google Drive">
                  <Input
                    type="url"
                    placeholder="https://www.youtube.com/watch?v=..."
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                  />
                </Field>
                <Field label="Platform">
                  <Select value={source} onValueChange={setSource}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="youtube">YouTube</SelectItem>
                      <SelectItem value="tiktok">TikTok</SelectItem>
                      <SelectItem value="instagram">Instagram</SelectItem>
                      <SelectItem value="gdrive">Google Drive</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
              </div>
            </TabsContent>

            <TabsContent value="upload">
              <Label className="mb-1.5 block">Upload Video</Label>
              {uploadFilename ? (
                <div className="flex items-center gap-3 rounded-lg border bg-secondary/40 p-3">
                  <span className="min-w-0 flex-1 truncate text-sm font-medium">{uploadFilename}</span>
                  <Button type="button" variant="ghost" size="sm" onClick={() => { setUploadFilename(''); fileRef.current?.click() }}>
                    Change
                  </Button>
                </div>
              ) : (
                <div className="rounded-xl border border-dashed p-8 text-center">
                  <input ref={fileRef} type="file" accept="video/*" onChange={handleFileUpload} className="hidden" />
                  <Button type="button" variant="secondary" onClick={() => fileRef.current?.click()} disabled={uploading}>
                    {uploading ? <><Loader2 className="size-4 animate-spin" /> Uploading…</> : 'Select video file'}
                  </Button>
                  <p className="mt-3 text-xs text-muted-foreground">MP4, MKV, AVI, MOV, WebM (max 2GB)</p>
                </div>
              )}
            </TabsContent>

            <TabsContent value="reuse">
              <Field label="Reuse Job ID" hint="Bypass download using an existing job ID. (Clone & Rerun fills this automatically.)">
                <Input
                  placeholder="Example: d20b47341e08"
                  value={reuseJobId}
                  onChange={(e) => setReuseJobId(e.target.value)}
                />
              </Field>
            </TabsContent>
          </Tabs>
        </Card>

        {/* Config */}
        <div className="grid items-start gap-5 lg:grid-cols-3">
          <Card className="p-6">
            <SectionTitle>Basics</SectionTitle>
            <div className="space-y-4">
              <Field label="Number of Clips">
                <Input type="number" min="1" max="30" value={clips} onChange={(e) => setClips(parseInt(e.target.value) || 7)} />
              </Field>
              <Field label="Aspect Ratio">
                <Select value={ratio} onValueChange={setRatio}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="9:16">9:16 · TikTok / Reels</SelectItem>
                    <SelectItem value="16:9">16:9 · Horizontal</SelectItem>
                    <SelectItem value="1:1">1:1 · Square</SelectItem>
                    <SelectItem value="3:4">3:4</SelectItem>
                    <SelectItem value="4:5">4:5</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Font Style">
                <Select value={fontStyle} onValueChange={setFontStyle}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="HORMOZI">Hormozi (Bold)</SelectItem>
                    <SelectItem value="DEFAULT">Default (Montserrat)</SelectItem>
                    <SelectItem value="STORYTELLER">Storyteller (Inter)</SelectItem>
                    <SelectItem value="CINEMATIC">Cinematic (Bebas Neue)</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
            </div>
          </Card>

          <Card className="p-6">
            <SectionTitle>AI &amp; Whisper</SectionTitle>
            <div className="space-y-4">
              <Field label="AI Provider">
                <Select value={aiProvider} onValueChange={setAiProvider}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gemini">Google Gemini</SelectItem>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="anthropic">Anthropic Claude</SelectItem>
                    <SelectItem value="nvidia">NVIDIA NIM</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Whisper Model" hint="On CPU, smaller is much faster">
                <Select value={whisperModel} onValueChange={setWhisperModel}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="large-v3">large-v3 (Best quality)</SelectItem>
                    <SelectItem value="medium">medium (Balanced)</SelectItem>
                    <SelectItem value="small">small (Fast)</SelectItem>
                    <SelectItem value="base">base (Fastest)</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Device">
                <Select value={whisperDevice} onValueChange={setWhisperDevice}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cpu">CPU</SelectItem>
                    <SelectItem value="cuda">CUDA (GPU)</SelectItem>
                    <SelectItem value="auto">Auto</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
            </div>
          </Card>

          <Card className="p-6">
            <SectionTitle>Features</SectionTitle>
            <div className="-mx-2 space-y-0.5">
              {featureToggles.map((t) => <ToggleRow key={t.label} {...t} />)}
            </div>
          </Card>
        </div>

        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/8 px-4 py-3 text-sm font-medium text-destructive">
            {error}
          </div>
        )}

        <div className="flex items-center gap-3">
          <Button type="submit" size="lg" disabled={submitting}>
            {submitting ? <><Loader2 className="size-4 animate-spin" /> Starting…</> : 'Start Clipping'}
          </Button>
          {useFitBlur && (
            <span className="text-xs text-muted-foreground">Fit mode on, whole frame kept</span>
          )}
        </div>
      </form>
    </div>
  )
}

export default NewJob
