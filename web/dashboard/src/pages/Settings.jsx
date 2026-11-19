import { useState, useEffect } from 'react'
import { fetchSettings, updateSettings } from '../api'

const PasswordInput = ({ value, onChange, placeholder, isSet }) => {
  const [show, setShow] = useState(false)
  return (
    <div style={{ position: 'relative' }}>
      <input
        className="form-input"
        type={show ? "text" : "password"}
        placeholder={isSet ? '••••••••••••••••' : placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{ paddingRight: '40px' }}
      />
      <button
        type="button"
        onClick={() => setShow(!show)}
        style={{
          position: 'absolute',
          right: '12px',
          top: '50%',
          transform: 'translateY(-50%)',
          background: 'none',
          border: 'none',
          color: 'var(--text-secondary)',
          cursor: 'pointer',
          fontSize: '14px',
          padding: '4px'
        }}
        title={show ? "Hide" : "Show"}
      >
        {show ? '👀' : '👁️'}
      </button>
    </div>
  )
}

function Settings() {
  const [settings, setSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')

  // Form fields
  const [googleKey, setGoogleKey] = useState('')
  const [pexelsKey, setPexelsKey] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [nvidiaKey, setNvidiaKey] = useState('')
  const [defaultAiProvider, setDefaultAiProvider] = useState('gemini')

  useEffect(() => {
    fetchSettings()
      .then(data => { setSettings(data); setDefaultAiProvider(data.default_ai_provider || 'gemini'); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const handleSave = async (e) => {
    e.preventDefault()
    setSaving(true)
    setMsg('')
    try {
      const payload = {}
      if (googleKey) payload.google_api_key = googleKey
      if (pexelsKey) payload.pexels_api_key = pexelsKey
      if (hfToken) payload.hf_token = hfToken
      if (nvidiaKey) payload.nvidia_api_key = nvidiaKey
      if (defaultAiProvider !== settings?.default_ai_provider) payload.default_ai_provider = defaultAiProvider

      if (Object.keys(payload).length === 0) {
        setMsg('Tidak ada perubahan')
        setSaving(false)
        return
      }

      const updated = await updateSettings(payload)
      setSettings(updated)
      setGoogleKey('')
      setPexelsKey('')
      setHfToken('')
      setNvidiaKey('')
      setMsg('✅ Settings berhasil diperbarui!')
    } catch (err) {
      setMsg('❌ Gagal menyimpan: ' + err.message)
    } finally {
      setSaving(false)
    }
  }

  if (loading) return <div className="empty-state"><div className="spinner"></div></div>

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h2>Settings</h2>
          <p>Konfigurasi API keys dan default settings</p>
        </div>
      </div>

      <form onSubmit={handleSave}>
        <div className="settings-grid">
          {/* API Keys */}
          <div className="settings-section">
            <h3>🔑 API Keys</h3>

            <div className="form-group">
              <label className="form-label">
                Google Gemini API Key
                {settings?.google_api_key_set && <span style={{ color: 'var(--success)', marginLeft: '8px' }}>✅ Set</span>}
              </label>
              <PasswordInput
                value={googleKey}
                onChange={setGoogleKey}
                placeholder="Paste your Gemini API key"
                isSet={settings?.google_api_key_set}
              />
              <p className="form-hint">
                <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener" style={{ color: 'var(--accent)' }}>Get free key →</a>
              </p>
            </div>

            <div className="form-group">
              <label className="form-label">
                Pexels API Key
                {settings?.pexels_api_key_set && <span style={{ color: 'var(--success)', marginLeft: '8px' }}>✅ Set</span>}
              </label>
              <PasswordInput
                value={pexelsKey}
                onChange={setPexelsKey}
                placeholder="For B-roll footage (optional)"
                isSet={settings?.pexels_api_key_set}
              />
              <p className="form-hint">Required for B-roll stock footage</p>
            </div>

            <div className="form-group">
              <label className="form-label">
                HuggingFace Token
                {settings?.hf_token_set && <span style={{ color: 'var(--success)', marginLeft: '8px' }}>✅ Set</span>}
              </label>
              <PasswordInput
                value={hfToken}
                onChange={setHfToken}
                placeholder="For split-screen mode (optional)"
                isSet={settings?.hf_token_set}
              />
              <p className="form-hint">Required for speaker diarization</p>
            </div>

            <div className="form-group">
              <label className="form-label">
                NVIDIA API Key
                {settings?.nvidia_api_key_set && <span style={{ color: 'var(--success)', marginLeft: '8px' }}>✅ Set</span>}
              </label>
              <PasswordInput
                value={nvidiaKey}
                onChange={setNvidiaKey}
                placeholder="For NVIDIA NIM provider (optional)"
                isSet={settings?.nvidia_api_key_set}
              />
            </div>
          </div>

          {/* System Info */}
          <div className="settings-section">
            <h3>💻 System Info</h3>
            <div className="form-group">
              <label className="form-label" htmlFor="default-ai-provider">Default AI Provider</label>
              <select id="default-ai-provider" className="form-select" value={defaultAiProvider} onChange={(e) => setDefaultAiProvider(e.target.value)}>
                <option value="gemini">Google Gemini</option>
                <option value="nvidia">NVIDIA NIM</option>
                <option value="vllm">vLLM (Self-hosted)</option>
              </select>
              <p className="form-hint">Berlaku selama backend berjalan. Endpoint dan key vLLM dikonfigurasi di notebook/server.</p>
            </div>
            <div className="form-group">
              <label className="form-label">Model default vLLM</label>
              <p style={{ fontSize: '12px', overflowWrap: 'anywhere' }}>{settings?.vllm_model || 'Gunakan konfigurasi server'}</p>
              <p className="form-hint">Key vLLM: {settings?.vllm_api_key_set ? 'tersedia' : 'tidak disetel (opsional)'}. Status ini tidak memeriksa koneksi ke model.</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>GPU</span>
                <span style={{ color: settings?.gpu_available ? 'var(--success)' : 'var(--text-tertiary)' }}>
                  {settings?.gpu_available ? '✅ Available' : '⚪ Not available'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Default Whisper</span>
                <span>{settings?.default_whisper_model}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Default AI</span>
                <span>{settings?.default_ai_provider}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Default Ratio</span>
                <span>{settings?.default_ratio}</span>
              </div>
            </div>
          </div>
        </div>

        {msg && (
          <div style={{ marginTop: '16px', fontSize: '13px', color: msg.startsWith('✅') ? 'var(--success)' : 'var(--error)' }}>
            {msg}
          </div>
        )}

        <button type="submit" className="btn btn-primary" disabled={saving} style={{ marginTop: '20px' }}>
          {saving ? <><span className="spinner"></span> Saving...</> : '💾 Save Settings'}
        </button>
      </form>
    </div>
  )
}

export default Settings
