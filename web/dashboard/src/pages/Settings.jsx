import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import { Cookie, Eye, EyeOff, KeyRound, Link2, Save, SlidersHorizontal, Trash2, Upload, UserRound, CheckCircle2, CircleDashed, ShieldCheck, ShieldAlert } from 'lucide-react'
import { api, connection, normalizeBaseUrl } from '../lib/api'
import { useConnection } from '../lib/connection'
import { useDocumentTitle, useLocalState, useResource } from '../lib/hooks'
import { useToast } from '../lib/toast'
import { PageHeader } from '../components/Layout'
import { Badge, Button, Card, CardHeader, ConfirmModal, Field, Input, Select, Skeleton } from '../components/ui'

const TABS = [
  { id: 'profile', label: 'Profile', icon: UserRound },
  { id: 'connection', label: 'Connection', icon: Link2 },
  { id: 'keys', label: 'API keys', icon: KeyRound },
  { id: 'defaults', label: 'Defaults', icon: SlidersHorizontal },
  { id: 'cookies', label: 'YouTube cookies', icon: Cookie },
]

export default function SettingsPage() {
  useDocumentTitle('Settings')
  const [params, setParams] = useSearchParams()
  const tab = TABS.some((t) => t.id === params.get('tab')) ? params.get('tab') : 'profile'
  const settings = useResource((signal) => api.settings({ signal }), [])

  return (
    <>
      <PageHeader title="Settings" description="Your workspace, server connection and pipeline defaults." />
      <div className="settings-layout">
        <nav className="settings-nav" aria-label="Settings sections">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button key={id} type="button" className={`settings-tab ${tab === id ? 'is-active' : ''}`} onClick={() => setParams({ tab: id }, { replace: true })} aria-current={tab === id ? 'page' : undefined}>
              {tab === id && <motion.span layoutId="settings-tab" className="settings-tab-bg" transition={{ type: 'spring', stiffness: 500, damping: 40 }} />}
              <Icon size={16} aria-hidden /><span>{label}</span>
            </button>
          ))}
        </nav>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={tab} className="settings-body" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
            {tab === 'profile' && <ProfileTab />}
            {tab === 'connection' && <ConnectionTab />}
            {tab === 'keys' && <KeysTab settings={settings} />}
            {tab === 'defaults' && <DefaultsTab settings={settings} />}
            {tab === 'cookies' && <CookiesTab settings={settings} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </>
  )
}

function ProfileTab() {
  const toast = useToast()
  const [profile, setProfile] = useLocalState('osc.profile', { name: '', channel: '', niche: '' })
  const [form, setForm] = useState(profile)
  const initials = (form.name || 'You').split(/\s+/).map((s) => s[0]).join('').slice(0, 2).toUpperCase()
  return (
    <Card>
      <CardHeader title="Creator profile" description="Personalizes your workspace. Stored only in this browser." />
      <form className="stack-md" onSubmit={(e) => { e.preventDefault(); setProfile({ name: form.name.trim().slice(0, 60), channel: form.channel.trim().slice(0, 80), niche: form.niche.trim().slice(0, 80) }); toast.success('Profile saved') }}>
        <div className="profile-row">
          <div className="avatar" aria-hidden>{initials}</div>
          <div>
            <strong>{form.name || 'Your name'}</strong>
            <p className="muted small">{form.channel || 'Your channel'}{form.niche ? ` · ${form.niche}` : ''}</p>
          </div>
        </div>
        <div className="form-grid">
          <Field label="Display name" htmlFor="p-name"><Input id="p-name" value={form.name} maxLength={60} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Alex" autoComplete="name" /></Field>
          <Field label="Channel / handle" htmlFor="p-channel"><Input id="p-channel" value={form.channel} maxLength={80} onChange={(e) => setForm({ ...form, channel: e.target.value })} placeholder="@yourchannel" /></Field>
          <Field label="Niche" htmlFor="p-niche"><Input id="p-niche" value={form.niche} maxLength={80} onChange={(e) => setForm({ ...form, niche: e.target.value })} placeholder="Podcasts, finance, gaming…" /></Field>
        </div>
        <div><Button type="submit" variant="primary" icon={Save}>Save profile</Button></div>
      </form>
    </Card>
  )
}

function ConnectionTab() {
  const { configure, status, info, base } = useConnection()
  const toast = useToast()
  const [url, setUrl] = useState(base)
  const [token, setToken] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const hasToken = !!connection.getToken()

  const save = async (e) => {
    e.preventDefault()
    setErr('')
    let normalized = ''
    try { normalized = url.trim() ? normalizeBaseUrl(url) : '' } catch { setErr('Enter a valid http(s) URL.'); return }
    setBusy(true)
    await configure({ baseUrl: normalized, ...(token ? { token } : {}) })
    setBusy(false)
    setToken('')
    toast.success('Connection updated')
  }

  return (
    <div className="stack-md">
      <Card>
        <CardHeader title="Backend server" description="Where the clipping pipeline runs. Empty means this same server." actions={<Badge tone={status === 'ready' ? 'success' : 'danger'} dot>{status === 'ready' ? 'Connected' : status}</Badge>} />
        <form className="stack-md" onSubmit={save}>
          <Field label="Backend URL" htmlFor="c-url" error={err} hint="For Kaggle/Colab, paste the ngrok Public URL printed by the notebook.">
            <Input id="c-url" icon={Link2} value={url} onChange={(e) => setUrl(e.target.value)} placeholder="Same origin" spellCheck={false} inputMode="url" />
          </Field>
          <Field label="Access token" htmlFor="c-token" hint={hasToken ? 'A token is saved. Enter a new one to replace it.' : 'Required only when the server sets OSC_API_TOKEN.'}>
            <Input id="c-token" icon={KeyRound} type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder={hasToken ? '•••••••• saved' : 'None'} autoComplete="off" />
          </Field>
          <div className="row gap-sm wrap">
            <Button type="submit" variant="primary" loading={busy}>Save & reconnect</Button>
            {hasToken && <Button type="button" variant="ghost" icon={Trash2} onClick={() => configure({ token: '' })}>Forget token</Button>}
          </div>
        </form>
      </Card>
      <Card>
        <CardHeader title="Security" />
        {info?.auth_required ? (
          <p className="row gap-sm align-start"><ShieldCheck size={18} className="ok" aria-hidden /><span>Access token required. Only people with the token can use this Studio; media links are signed and expire after 12 hours.</span></p>
        ) : (
          <p className="row gap-sm align-start"><ShieldAlert size={18} className="warn" aria-hidden /><span>This server is <strong>open</strong>. Anyone who can reach its URL can start jobs with your API keys. Before exposing it publicly, restart it with <code>OSC_API_TOKEN</code> set to a long random value.</span></p>
        )}
      </Card>
    </div>
  )
}

function SecretField({ id, label, set, value, onChange, hint, placeholder }) {
  const [show, setShow] = useState(false)
  return (
    <Field label={label} htmlFor={id} hint={hint} trailing={set ? <Badge tone="success"><CheckCircle2 size={11} aria-hidden />Configured</Badge> : <Badge><CircleDashed size={11} aria-hidden />Not set</Badge>}>
      <div className="input-wrap">
        <KeyRound size={16} className="input-icon" aria-hidden />
        <input id={id} className="input has-icon has-trailing mono" type={show ? 'text' : 'password'} value={value} onChange={(e) => onChange(e.target.value)} placeholder={set ? '•••••••• (leave empty to keep)' : placeholder} autoComplete="off" spellCheck={false} maxLength={512} />
        <button type="button" className="input-trailing icon-btn icon-btn-sm" onClick={() => setShow((s) => !s)} aria-label={show ? 'Hide value' : 'Show value'}>{show ? <EyeOff size={14} /> : <Eye size={14} />}</button>
      </div>
    </Field>
  )
}

const KEY_FIELDS = [
  { key: 'google_api_key', flag: 'google_api_key_set', label: 'Google Gemini API key', hint: 'Required for the default AI provider and voice-over scripts.', placeholder: 'AIza…' },
  { key: 'nvidia_api_key', flag: 'nvidia_api_key_set', label: 'NVIDIA NIM API key', hint: 'Only when using NVIDIA as the AI provider.', placeholder: 'nvapi-…' },
  { key: 'pexels_api_key', flag: 'pexels_api_key_set', label: 'Pexels API key', hint: 'Enables contextual B-roll. Free at pexels.com/api.', placeholder: 'Pexels key' },
  { key: 'hf_token', flag: 'hf_token_set', label: 'Hugging Face token', hint: 'Needed for podcast split-screen / camera switch (Pyannote).', placeholder: 'hf_…' },
]

function KeysTab({ settings }) {
  const toast = useToast()
  const [values, setValues] = useState({})
  const [busy, setBusy] = useState(false)
  const [clearKey, setClearKey] = useState(null)
  const s = settings.data

  const save = async (e) => {
    e.preventDefault()
    const payload = Object.fromEntries(Object.entries(values).map(([k, v]) => [k, v.trim()]).filter(([, v]) => v))
    if (!Object.keys(payload).length) { toast.info('Nothing to save — enter a key first.'); return }
    setBusy(true)
    try {
      settings.setData(await api.updateSettings(payload))
      setValues({})
      toast.success('API keys saved for this server session')
    } catch (err) { toast.error(err.message) } finally { setBusy(false) }
  }

  const clear = async () => {
    setBusy(true)
    try {
      settings.setData(await api.updateSettings({ [clearKey]: '' }))
      toast.success('Key removed')
    } catch (err) { toast.error(err.message) } finally { setBusy(false); setClearKey(null) }
  }

  if (settings.loading && !s) return <Skeleton style={{ height: 380 }} />
  if (settings.error) return <p className="alert alert-danger">{settings.error.message}</p>

  return (
    <Card>
      <CardHeader title="API keys" description="Keys are sent to your server only, held in memory for the current session and never shown again. For permanent setup, use environment variables or notebook Secrets." />
      <form className="stack-md" onSubmit={save}>
        {KEY_FIELDS.map((f) => (
          <div key={f.key} className="key-row">
            <SecretField id={`k-${f.key}`} label={f.label} set={s?.[f.flag]} value={values[f.key] || ''} onChange={(v) => setValues((x) => ({ ...x, [f.key]: v }))} hint={f.hint} placeholder={f.placeholder} />
            {s?.[f.flag] && <Button type="button" size="sm" variant="ghost" icon={Trash2} onClick={() => setClearKey(f.key)} aria-label={`Remove ${f.label}`}>Remove</Button>}
          </div>
        ))}
        <div><Button type="submit" variant="primary" icon={Save} loading={busy}>Save keys</Button></div>
      </form>
      <ConfirmModal open={!!clearKey} onClose={() => setClearKey(null)} onConfirm={clear} loading={busy} title="Remove this key?" description="The key is disabled for this server session — including one set by an environment variable — until you add a new key or restart the server. Jobs that need it will fail meanwhile." confirmLabel="Remove key" />
    </Card>
  )
}

function DefaultsTab({ settings }) {
  const toast = useToast()
  const s = settings.data
  const [form, setForm] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (s && !form) setForm({
      default_clips: s.default_clips, default_ratio: s.default_ratio, default_font_style: s.default_font_style,
      default_whisper_model: s.default_whisper_model, default_whisper_device: s.default_whisper_device, default_ai_provider: s.default_ai_provider,
    })
  }, [s, form])

  if (!form) return settings.error ? <p className="alert alert-danger">{settings.error.message}</p> : <Skeleton style={{ height: 320 }} />

  const save = async (e) => {
    e.preventDefault()
    setBusy(true)
    try {
      settings.setData(await api.updateSettings({ ...form, default_clips: Math.max(1, Math.min(30, Number(form.default_clips) || 7)) }))
      toast.success('Defaults saved')
    } catch (err) { toast.error(err.message) } finally { setBusy(false) }
  }

  return (
    <Card>
      <CardHeader title="Pipeline defaults" description="Pre-filled on every new project. Applies to this server session." />
      <form className="stack-md" onSubmit={save}>
        <div className="form-grid">
          <Field label="Clips per project" htmlFor="d-clips"><Input id="d-clips" type="number" min="1" max="30" value={form.default_clips} onChange={(e) => setForm({ ...form, default_clips: e.target.value })} /></Field>
          <Field label="Aspect ratio" htmlFor="d-ratio">
            <Select id="d-ratio" value={form.default_ratio} onChange={(e) => setForm({ ...form, default_ratio: e.target.value })}>
              {['9:16', '4:5', '1:1', '3:4', '16:9'].map((r) => <option key={r} value={r}>{r}</option>)}
            </Select>
          </Field>
          <Field label="Caption style" htmlFor="d-font">
            <Select id="d-font" value={form.default_font_style} onChange={(e) => setForm({ ...form, default_font_style: e.target.value })}>
              <option value="HORMOZI">Hormozi</option><option value="DEFAULT">Montserrat</option><option value="STORYTELLER">Storyteller</option><option value="CINEMATIC">Cinematic</option>
            </Select>
          </Field>
          <Field label="AI provider" htmlFor="d-ai">
            <Select id="d-ai" value={form.default_ai_provider} onChange={(e) => setForm({ ...form, default_ai_provider: e.target.value })}>
              <option value="gemini">Google Gemini</option><option value="nvidia">NVIDIA NIM</option><option value="vllm">vLLM</option>
            </Select>
          </Field>
          <Field label="Whisper model" htmlFor="d-wm">
            <Select id="d-wm" value={form.default_whisper_model} onChange={(e) => setForm({ ...form, default_whisper_model: e.target.value })}>
              <option value="large-v3">large-v3</option><option value="medium">medium</option><option value="small">small</option><option value="base">base</option>
            </Select>
          </Field>
          <Field label="Whisper device" htmlFor="d-wd">
            <Select id="d-wd" value={form.default_whisper_device} onChange={(e) => setForm({ ...form, default_whisper_device: e.target.value })}>
              <option value="cuda">CUDA GPU</option><option value="cpu">CPU</option><option value="auto">Auto</option>
            </Select>
          </Field>
        </div>
        <div><Button type="submit" variant="primary" icon={Save} loading={busy}>Save defaults</Button></div>
      </form>
    </Card>
  )
}

function CookiesTab({ settings }) {
  const toast = useToast()
  const input = useRef(null)
  const [busy, setBusy] = useState(false)
  const s = settings.data

  const upload = async (file) => {
    if (!file) return
    if (file.size > 1024 * 1024) { toast.error('cookies.txt must be under 1 MiB'); return }
    setBusy(true)
    try {
      const status = await api.uploadCookies(file)
      settings.setData((d) => ({ ...d, ...status }))
      toast.success('YouTube cookies uploaded for this session')
    } catch (err) {
      toast.error(err.status === 422 ? 'That file is not a valid Netscape cookies.txt export.' : err.message)
    } finally { setBusy(false) }
  }
  const remove = async () => {
    setBusy(true)
    try {
      const status = await api.removeCookies()
      settings.setData((d) => ({ ...d, ...status }))
      toast.success('Uploaded cookies removed')
    } catch (err) { toast.error(err.message) } finally { setBusy(false) }
  }

  return (
    <Card>
      <CardHeader
        title="YouTube cookies"
        description="Some videos (age-restricted, members-only, or when YouTube asks to “confirm you’re not a bot”) need a signed-in session. Upload a Netscape-format cookies.txt export."
        actions={s && <Badge tone={s.youtube_cookies_set ? 'success' : 'neutral'} dot>{s.youtube_cookies_uploaded ? 'Uploaded' : s.youtube_cookies_set ? 'Server file' : 'Not set'}</Badge>}
      />
      <input ref={input} type="file" accept=".txt,text/plain" hidden onChange={(e) => { upload(e.target.files?.[0]); e.target.value = '' }} />
      <div className="row gap-sm wrap">
        <Button variant="primary" icon={Upload} loading={busy} onClick={() => input.current?.click()}>Upload cookies.txt</Button>
        {s?.youtube_cookies_uploaded && <Button variant="ghost" icon={Trash2} onClick={remove} disabled={busy}>Remove upload</Button>}
      </div>
      <p className="alert alert-warning" style={{ marginTop: 16 }}>Cookies grant access to your account. Use a secondary account, and only upload to a server you control.</p>
    </Card>
  )
}
