import { useState, useEffect } from 'react'
import { Eye, EyeOff, Loader2, ExternalLink } from 'lucide-react'
import { fetchSettings, updateSettings } from '../api'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/components/ui/toast'

function PasswordInput({ value, onChange, placeholder, isSet }) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative">
      <Input
        type={show ? 'text' : 'password'}
        placeholder={isSet ? '••••••••••••••••' : placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="pr-10"
      />
      <button
        type="button"
        onClick={() => setShow(!show)}
        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors hover:text-foreground"
        title={show ? 'Hide' : 'Show'}
      >
        {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
      </button>
    </div>
  )
}

function KeyField({ label, isSet, hint, link, ...props }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <Label>{label}</Label>
        {isSet && <Badge variant="success">Set</Badge>}
      </div>
      <PasswordInput isSet={isSet} {...props} />
      {(hint || link) && (
        <p className="text-xs text-muted-foreground">
          {hint}
          {link && (
            <a href={link} target="_blank" rel="noopener" className="ml-1 inline-flex items-center gap-0.5 font-medium text-primary hover:underline">
              Get key <ExternalLink className="size-3" />
            </a>
          )}
        </p>
      )}
    </div>
  )
}

function InfoRow({ label, children }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{children}</span>
    </div>
  )
}

function Settings() {
  const { toast } = useToast()
  const [settings, setSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const [googleKey, setGoogleKey] = useState('')
  const [openaiKey, setOpenaiKey] = useState('')
  const [anthropicKey, setAnthropicKey] = useState('')
  const [pexelsKey, setPexelsKey] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [nvidiaKey, setNvidiaKey] = useState('')

  useEffect(() => {
    fetchSettings()
      .then((data) => { setSettings(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const handleSave = async (e) => {
    e.preventDefault()
    setSaving(true)
    try {
      const payload = {}
      if (googleKey) payload.google_api_key = googleKey
      if (openaiKey) payload.openai_api_key = openaiKey
      if (anthropicKey) payload.anthropic_api_key = anthropicKey
      if (pexelsKey) payload.pexels_api_key = pexelsKey
      if (hfToken) payload.hf_token = hfToken
      if (nvidiaKey) payload.nvidia_api_key = nvidiaKey

      if (Object.keys(payload).length === 0) {
        toast({ variant: 'info', title: 'No changes to save', description: 'Enter a key before saving.' })
        setSaving(false)
        return
      }
      const updated = await updateSettings(payload)
      setSettings(updated)
      setGoogleKey(''); setOpenaiKey(''); setAnthropicKey(''); setPexelsKey(''); setHfToken(''); setNvidiaKey('')
      toast({ variant: 'success', title: 'Settings saved', description: 'Your API keys were updated.' })
    } catch (err) {
      toast({ variant: 'error', title: 'Failed to save', description: err.message })
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return <div className="grid place-items-center py-32 text-muted-foreground"><Loader2 className="size-6 animate-spin" /></div>
  }

  return (
    <div className="animate-in-up space-y-8">
      <div>
        <h1 className="font-display text-4xl font-bold tracking-tight">Settings</h1>
        <p className="mt-1.5 text-muted-foreground">Configure API keys and view system defaults</p>
      </div>

      <form onSubmit={handleSave} className="grid gap-5 lg:grid-cols-[1.5fr_1fr]">
        {/* API Keys */}
        <Card className="p-6">
          <h2 className="mb-5 font-display text-base font-bold">API Keys</h2>
          <div className="space-y-5">
            <KeyField
              label="Google Gemini API Key" isSet={settings?.google_api_key_set}
              value={googleKey} onChange={setGoogleKey}
              placeholder="Paste your Gemini API key"
              hint="For the Gemini provider." link="https://aistudio.google.com/apikey"
            />
            <KeyField
              label="OpenAI API Key" isSet={settings?.openai_api_key_set}
              value={openaiKey} onChange={setOpenaiKey}
              placeholder="For the OpenAI provider"
              hint="Required when AI provider is OpenAI." link="https://platform.openai.com/api-keys"
            />
            <KeyField
              label="Anthropic API Key" isSet={settings?.anthropic_api_key_set}
              value={anthropicKey} onChange={setAnthropicKey}
              placeholder="For the Anthropic Claude provider"
              hint="Required when AI provider is Anthropic." link="https://console.anthropic.com/settings/keys"
            />
            <KeyField
              label="Pexels API Key" isSet={settings?.pexels_api_key_set}
              value={pexelsKey} onChange={setPexelsKey}
              placeholder="For B-roll footage (optional)"
              hint="Required for B-roll stock footage." link="https://www.pexels.com/api/"
            />
            <KeyField
              label="HuggingFace Token" isSet={settings?.hf_token_set}
              value={hfToken} onChange={setHfToken}
              placeholder="For split-screen mode (optional)"
              hint="Required for speaker diarization." link="https://huggingface.co/settings/tokens"
            />
            <KeyField
              label="NVIDIA API Key" isSet={settings?.nvidia_api_key_set}
              value={nvidiaKey} onChange={setNvidiaKey}
              placeholder="For NVIDIA NIM provider (optional)"
              hint="Only needed if using the NVIDIA provider."
            />
          </div>

          <Button type="submit" className="mt-5" disabled={saving}>
            {saving ? <><Loader2 className="size-4 animate-spin" /> Saving…</> : 'Save Settings'}
          </Button>
        </Card>

        {/* System Info */}
        <Card className="h-fit p-6">
          <h2 className="mb-3 font-display text-base font-bold">System</h2>
          <div className="divide-y">
            <InfoRow label="GPU">
              <span className={settings?.gpu_available ? 'text-success' : 'text-muted-foreground'}>
                {settings?.gpu_available ? 'Available' : 'Not available'}
              </span>
            </InfoRow>
            <InfoRow label="Default Whisper">{settings?.default_whisper_model}</InfoRow>
            <InfoRow label="Default AI">{settings?.default_ai_provider}</InfoRow>
            <InfoRow label="Default Ratio">{settings?.default_ratio}</InfoRow>
          </div>
        </Card>
      </form>
    </div>
  )
}

export default Settings
