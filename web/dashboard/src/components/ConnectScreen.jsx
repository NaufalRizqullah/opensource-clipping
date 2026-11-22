import { useState } from 'react'
import { motion } from 'motion/react'
import { KeyRound, Link2, RefreshCw, ShieldCheck, ServerCrash } from 'lucide-react'
import { useConnection } from '../lib/connection'
import { connection, normalizeBaseUrl } from '../lib/api'
import { Button, Field, Input } from './ui'
import { Logo } from './Layout'

export default function ConnectScreen() {
  const { status, error, configure, check } = useConnection()
  const [baseUrl, setBaseUrl] = useState(connection.getBase())
  const [token, setToken] = useState('')
  const [formError, setFormError] = useState('')
  const [busy, setBusy] = useState(false)
  const unauthorized = status === 'unauthorized'

  const submit = async (e) => {
    e.preventDefault()
    setFormError('')
    let normalized = ''
    try {
      normalized = baseUrl.trim() ? normalizeBaseUrl(baseUrl) : ''
    } catch {
      setFormError('Enter a valid http(s) URL, or leave empty to use this server.')
      return
    }
    if (normalized && window.location.protocol === 'https:' && normalized.startsWith('http://') && !/^http:\/\/(localhost|127\.0\.0\.1)/.test(normalized)) {
      setFormError('This page is served over HTTPS, so the browser will block an http:// backend. Use the https tunnel URL.')
      return
    }
    setBusy(true)
    await configure({ baseUrl: normalized, ...(token.trim() || unauthorized ? { token: token.trim() } : {}) })
    setBusy(false)
  }

  return (
    <div className="connect">
      <div className="connect-glow" aria-hidden />
      <motion.div
        className="connect-card"
        initial={{ opacity: 0, y: 18, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: 'spring', stiffness: 260, damping: 28 }}
      >
        <Logo />
        <div className="connect-head">
          <div className={`connect-icon ${unauthorized ? 'is-auth' : ''}`}>
            {unauthorized ? <ShieldCheck size={20} /> : <ServerCrash size={20} />}
          </div>
          <h1>{unauthorized ? 'This Studio is protected' : 'Connect to your Studio backend'}</h1>
          <p>
            {unauthorized
              ? 'Enter the access token set as OSC_API_TOKEN on the server.'
              : 'Run the backend locally or start the Kaggle/Colab notebook, then paste its public URL here.'}
          </p>
        </div>

        <form onSubmit={submit} className="stack-md">
          <Field label="Backend URL" htmlFor="conn-url" hint="Leave empty when this page is served by the backend itself.">
            <Input id="conn-url" icon={Link2} value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://xxxx.ngrok-free.app" autoComplete="url" inputMode="url" spellCheck={false} />
          </Field>
          <Field label="Access token" htmlFor="conn-token" hint={unauthorized ? undefined : 'Only needed when the server sets OSC_API_TOKEN.'}>
            <Input id="conn-token" icon={KeyRound} type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder={connection.getToken() ? '•••••••• (saved)' : 'Paste token'} autoComplete="off" spellCheck={false} data-autofocus={unauthorized || undefined} />
          </Field>
          {(formError || error) && <p className="alert alert-danger" role="alert">{formError || error}</p>}
          <div className="row gap-sm">
            <Button type="submit" variant="primary" loading={busy || status === 'checking'} className="grow">Connect</Button>
            <Button type="button" variant="ghost" icon={RefreshCw} onClick={check}>Retry</Button>
          </div>
        </form>

        <details className="connect-help">
          <summary>How do I start the backend?</summary>
          <ol>
            <li>Local: <code>uvicorn web.api.app:app --port 8000</code> from the project folder.</li>
            <li>Kaggle/Colab: run <code>notebooks/kaggle-studio-server.ipynb</code> and copy the Public URL.</li>
            <li>Protect public servers by setting <code>OSC_API_TOKEN</code> before starting.</li>
          </ol>
        </details>
      </motion.div>
    </div>
  )
}
