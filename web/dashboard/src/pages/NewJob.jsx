import { useState, useRef, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { createJob, uploadVideo, fetchSettings } from '../api'

function NewJob() {
  const navigate = useNavigate()
  const location = useLocation()
  const fileRef = useRef(null)
  const providerTouched = useRef(false)

  const [mode, setMode] = useState('url') // 'url' or 'upload'
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
  const [whisperModel, setWhisperModel] = useState('large-v3')
  const [whisperDevice, setWhisperDevice] = useState('cuda')
  const [aiProvider, setAiProvider] = useState('gemini')
  const [vllmModel, setVllmModel] = useState('')
  const [localMaxTokens, setLocalMaxTokens] = useState('')
  const [vllmDefaults, setVllmDefaults] = useState({ model: '', tokens: 4096 })

  useEffect(() => {
    let cancelled = false
    fetchSettings().then(data => {
      if (cancelled) return
      setVllmDefaults({ model: data.vllm_model || '', tokens: data.local_max_tokens || 4096 })
      if (!providerTouched.current && !location.state?.reuseJob && ['gemini', 'nvidia', 'vllm'].includes(data.default_ai_provider)) {
        setAiProvider(data.default_ai_provider)
      }
    }).catch(() => {})
    return () => { cancelled = true }
  }, [location.state])

  // Toggles
  const [useBroll, setUseBroll] = useState(true)
  const [useHookGlitch, setUseHookGlitch] = useState(true)
  const [useBgm, setUseBgm] = useState(true)
  const [useKaraoke, setUseKaraoke] = useState(true)
  const [noSubs, setNoSubs] = useState(false)
  const [hookV2, setHookV2] = useState(false)
  const [silenceTrim, setSilenceTrim] = useState(false)
  const [useDlpSubs, setUseDlpSubs] = useState(false)
  const [loadGeminiJson, setLoadGeminiJson] = useState(false)

  // Load from location state if user clicked "Clone / Rerun"
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
      if (config.vllm_model) setVllmModel(config.vllm_model)
      if (config.local_max_tokens != null) setLocalMaxTokens(String(config.local_max_tokens))
      
      if (config.use_broll !== undefined) setUseBroll(config.use_broll)
      if (config.use_hook_glitch !== undefined) setUseHookGlitch(config.use_hook_glitch)
      if (config.use_auto_bgm !== undefined) setUseBgm(config.use_auto_bgm)
      if (config.use_karaoke_effect !== undefined) setUseKaraoke(config.use_karaoke_effect)
      if (config.hook_v2 !== undefined) setHookV2(config.hook_v2)
      if (config.silence_trim !== undefined) setSilenceTrim(config.silence_trim)
      if (config.use_dlp_subs !== undefined) setUseDlpSubs(config.use_dlp_subs)
      if (config.no_subs !== undefined) setNoSubs(config.no_subs)
      
      // Default to true when cloning to save AI tokens, user can untoggle
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

    if (mode === 'url' && !url.trim()) {
      setError('URL tidak boleh kosong')
      return
    }
    if (mode === 'upload' && !uploadFilename) {
      setError('Silakan upload video terlebih dahulu')
      return
    }
    if (mode === 'reuse' && !reuseJobId.trim()) {
      setError('Job ID tidak boleh kosong')
      return
    }
    if (aiProvider === 'vllm' && vllmModel.trim() && !/^[A-Za-z0-9._:/@+-]{1,256}$/.test(vllmModel.trim())) {
      setError('Nama model vLLM tidak valid')
      return
    }
    if (aiProvider === 'vllm' && localMaxTokens !== '' && (!Number.isInteger(Number(localMaxTokens)) || Number(localMaxTokens) < 1 || Number(localMaxTokens) > 131072)) {
      setError('Batas token output harus berupa angka bulat antara 1 dan 131072')
      return
    }

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
        ...(aiProvider === 'vllm' && whisperDevice === 'cpu' ? { whisper_compute_type: 'int8' } : {}),
        ...(aiProvider === 'vllm' && vllmModel.trim() ? { vllm_model: vllmModel.trim() } : {}),
        ...(aiProvider === 'vllm' && localMaxTokens !== '' ? { local_max_tokens: Number(localMaxTokens) } : {}),
        use_broll: useBroll,
        use_hook_glitch: useHookGlitch,
        use_auto_bgm: useBgm,
        use_karaoke_effect: useKaraoke,
        no_subs: noSubs,
        hook_v2: hookV2,
        silence_trim: silenceTrim,
        use_dlp_subs: useDlpSubs,
        load_gemini_json: loadGeminiJson,
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

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h2>New Clipping Job</h2>
          <p>Generate viral short clips dari video panjang</p>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Source Selection */}
        <div className="card" style={{ marginBottom: '16px' }}>
          <h3 className="card-title" style={{ marginBottom: '16px' }}>📥 Video Source</h3>

          {/* Mode toggle */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
            <button
              type="button"
              className={`btn ${mode === 'url' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setMode('url')}
            >
              🔗 From URL
            </button>
            <button
              type="button"
              className={`btn ${mode === 'upload' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setMode('upload')}
            >
              📁 Upload File
            </button>
            <button
              type="button"
              className={`btn ${mode === 'reuse' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setMode('reuse')}
            >
              🔁 Reuse Job
            </button>
          </div>

          {mode === 'url' && (
            <>
              <div className="form-group">
                <label className="form-label">Video URL</label>
                <input
                  className="form-input"
                  type="url"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
                <p className="form-hint">Mendukung YouTube, TikTok, Instagram, Google Drive</p>
              </div>
              <div className="form-group" style={{ maxWidth: '200px' }}>
                <label className="form-label">Platform</label>
                <select className="form-select" value={source} onChange={(e) => setSource(e.target.value)}>
                  <option value="youtube">YouTube</option>
                  <option value="tiktok">TikTok</option>
                  <option value="instagram">Instagram</option>
                  <option value="gdrive">Google Drive</option>
                </select>
              </div>
            </>
          )}

          {mode === 'upload' && (
            <div className="form-group">
              <label className="form-label">Upload Video</label>
              {uploadFilename ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ color: 'var(--success)' }}>✅ {uploadFilename}</span>
                  <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setUploadFilename(''); fileRef.current?.click() }}>
                    Change
                  </button>
                </div>
              ) : (
                <div>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="video/*"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                  />
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => fileRef.current?.click()}
                    disabled={uploading}
                  >
                    {uploading ? <><span className="spinner"></span> Uploading...</> : '📁 Select Video File'}
                  </button>
                  <p className="form-hint">MP4, MKV, AVI, MOV, WebM (max 2GB)</p>
                </div>
              )}
            </div>
          )}

          {mode === 'reuse' && (
            <div className="form-group">
              <label className="form-label">Reuse Job ID</label>
              <input
                className="form-input"
                type="text"
                placeholder="Contoh: d20b47341e08"
                value={reuseJobId}
                onChange={(e) => setReuseJobId(e.target.value)}
              />
              <p className="form-hint" style={{ marginTop: '4px' }}>Bypass download dengan job ID lama. (Jika menggunakan Clone & Rerun, biarkan form ini terisi).</p>
            </div>
          )}
        </div>

        {/* Main Config */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px', marginBottom: '16px' }}>
          {/* Basic Settings */}
          <div className="config-section">
            <h4>🎯 Basic Settings</h4>
            <div className="form-group">
              <label className="form-label">Number of Clips</label>
              <input className="form-input" type="number" min="1" max="30" value={clips} onChange={(e) => setClips(parseInt(e.target.value) || 7)} />
            </div>
            <div className="form-group">
              <label className="form-label">Aspect Ratio</label>
              <select className="form-select" value={ratio} onChange={(e) => setRatio(e.target.value)}>
                <option value="9:16">9:16 (Vertical — TikTok/Reels)</option>
                <option value="16:9">16:9 (Horizontal)</option>
                <option value="1:1">1:1 (Square)</option>
                <option value="3:4">3:4</option>
                <option value="4:5">4:5</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Font Style</label>
              <select className="form-select" value={fontStyle} onChange={(e) => setFontStyle(e.target.value)}>
                <option value="HORMOZI">Hormozi (Bold)</option>
                <option value="DEFAULT">Default (Montserrat)</option>
                <option value="STORYTELLER">Storyteller (Inter)</option>
                <option value="CINEMATIC">Cinematic (Bebas Neue)</option>
              </select>
            </div>
          </div>

          {/* AI Settings */}
          <div className="config-section">
            <h4>🤖 AI & Whisper</h4>
            <div className="form-group">
              <label className="form-label">AI Provider</label>
              <select className="form-select" value={aiProvider} onChange={(e) => { providerTouched.current = true; setAiProvider(e.target.value) }}>
                <option value="gemini">Google Gemini</option>
                <option value="nvidia">NVIDIA NIM</option>
                <option value="vllm">vLLM (Self-hosted)</option>
              </select>
            </div>
            {aiProvider === 'vllm' && <>
              <div className="form-group">
                <label className="form-label" htmlFor="vllm-model">Model vLLM</label>
                <input id="vllm-model" className="form-input" maxLength="256" value={vllmModel} onChange={(e) => setVllmModel(e.target.value)} placeholder={vllmDefaults.model || 'Gunakan model default server'} />
                <p className="form-hint">Kosongkan untuk default server. Gunakan nama model atau alias yang sudah dijalankan di Kaggle.</p>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="vllm-tokens">Batas token output</label>
                <input id="vllm-tokens" className="form-input" type="number" min="1" max="131072" step="1" value={localMaxTokens} onChange={(e) => setLocalMaxTokens(e.target.value)} placeholder={String(vllmDefaults.tokens)} />
                <p className="form-hint">Prompt dan jatah token output harus muat dalam kapasitas konteks model. Anda dapat mengubah kapasitas konteks saat menjalankan server vLLM.</p>
                <p className="form-hint">Mulai dengan 1–2 klip. Pada satu GPU 16 GB, pilih Whisper CPU atau YouTube Subs agar memori cukup. vLLM tidak otomatis beralih ke cloud.</p>
              </div>
            </>}
            <div className="form-group">
              <label className="form-label">Whisper Model</label>
              <select className="form-select" value={whisperModel} onChange={(e) => setWhisperModel(e.target.value)}>
                <option value="large-v3">large-v3 (Best quality)</option>
                <option value="medium">medium (Balanced)</option>
                <option value="small">small (Fast)</option>
                <option value="base">base (Fastest)</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Device</label>
              <select className="form-select" value={whisperDevice} onChange={(e) => setWhisperDevice(e.target.value)}>
                <option value="cuda">CUDA (GPU)</option>
                <option value="cpu">CPU</option>
                <option value="auto">Auto</option>
              </select>
            </div>
          </div>

          {/* Feature Toggles */}
          <div className="config-section">
            <h4>✨ Features</h4>
            <ToggleRow label="B-Roll Footage" desc="Insert stock footage" checked={useBroll} onChange={setUseBroll} />
            <ToggleRow label="Hook Glitch" desc="Glitch transition intro" checked={useHookGlitch} onChange={setUseHookGlitch} />
            <ToggleRow label="Background Music" desc="Auto BGM matching" checked={useBgm} onChange={setUseBgm} />
            <ToggleRow label="Karaoke Effect" desc="Word-by-word highlight" checked={useKaraoke} onChange={setUseKaraoke} />
            <ToggleRow label="Hook V2" desc="Multi-hook intro clips" checked={hookV2} onChange={setHookV2} />
            <ToggleRow label="Silence Trim" desc="Remove dead air" checked={silenceTrim} onChange={setSilenceTrim} />
            <ToggleRow label="YouTube Subs" desc="Skip Whisper if available" checked={useDlpSubs} onChange={setUseDlpSubs} />
            <ToggleRow label="No Subtitles" desc="Render without text" checked={noSubs} onChange={setNoSubs} />
            <ToggleRow label="Bypass AI" desc="Reuse gemini JSON (if exist)" checked={loadGeminiJson} onChange={setLoadGeminiJson} />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: 'var(--error-dim)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 'var(--radius-md)', padding: '12px 16px', marginBottom: '16px', color: 'var(--error)', fontSize: '13px' }}>
            ⚠️ {error}
          </div>
        )}

        {/* Submit */}
        <button type="submit" className="btn btn-primary" disabled={submitting} style={{ fontSize: '14px', padding: '12px 28px' }}>
          {submitting ? <><span className="spinner"></span> Processing...</> : '🚀 Start Clipping'}
        </button>
      </form>
    </div>
  )
}

function ToggleRow({ label, desc, checked, onChange }) {
  return (
    <div className="toggle-row">
      <div>
        <div className="toggle-label">{label}</div>
        {desc && <div className="toggle-desc">{desc}</div>}
      </div>
      <label className="toggle-switch">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="toggle-slider"></span>
      </label>
    </div>
  )
}

export default NewJob
