/**
 * Studio API client.
 *
 * - Backend URL defaults to same-origin (the FastAPI server serves this app)
 *   and can point to a remote Kaggle/Colab tunnel from Settings → Connection.
 * - The optional access token (OSC_API_TOKEN on the server) is sent as a
 *   Bearer header. Media URLs returned by the API are pre-signed.
 */
import { storage } from './storage'

const BASE_KEY = 'osc.backendUrl'
const TOKEN_KEY = 'osc.accessToken'
const TUNNEL_HEADERS = { 'ngrok-skip-browser-warning': 'true' }

export class ApiError extends Error {
  constructor(message, status = 0, detail = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

export function normalizeBaseUrl(value) {
  const raw = String(value || '').trim().replace(/\/+$/, '')
  if (!raw) return ''
  const withScheme = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`
  const url = new URL(withScheme)
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Only http(s) URLs are supported')
  return url.origin + url.pathname.replace(/\/+$/, '')
}

export const connection = {
  getBase: () => storage.get(BASE_KEY, ''),
  setBase: (v) => (v ? storage.set(BASE_KEY, normalizeBaseUrl(v)) : storage.remove(BASE_KEY)),
  getToken: () => storage.get(TOKEN_KEY, ''),
  setToken: (v) => (v ? storage.set(TOKEN_KEY, String(v).trim()) : storage.remove(TOKEN_KEY)),
}

function authHeaders() {
  const token = connection.getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export function apiUrl(path) {
  return `${connection.getBase()}${path}`
}

/** Absolute URL for a media path returned by the API (already signed when auth is on). */
export function mediaUrl(path) {
  if (!path) return ''
  if (/^https?:\/\//i.test(path)) return path
  return apiUrl(path)
}

function describeDetail(detail, fallback) {
  if (!detail) return fallback
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== 'body').join('.') : ''
        return field ? `${field}: ${d.msg}` : d.msg
      })
      .join(' · ')
  }
  return fallback
}

export async function request(path, { method = 'GET', json, body, headers, signal, timeout = 30000 } = {}) {
  const controller = new AbortController()
  const onAbort = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', onAbort, { once: true })
  const timer = timeout ? setTimeout(() => controller.abort(new DOMException('Request timed out', 'TimeoutError')), timeout) : null

  let res
  try {
    res = await fetch(apiUrl(path), {
      method,
      headers: {
        Accept: 'application/json',
        ...TUNNEL_HEADERS,
        ...authHeaders(),
        ...(json !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...headers,
      },
      body: json !== undefined ? JSON.stringify(json) : body,
      signal: controller.signal,
      redirect: 'error',
    })
  } catch (err) {
    if (signal?.aborted) throw err
    const timedOut = controller.signal.reason?.name === 'TimeoutError'
    throw new ApiError(timedOut ? 'The server took too long to respond.' : 'Cannot reach the Studio server.', 0)
  } finally {
    if (timer) clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }

  const type = res.headers.get('content-type') || ''
  const data = type.includes('application/json') ? await res.json().catch(() => null) : null
  if (!res.ok) {
    const fallback = res.status === 401 ? 'Access token missing or invalid.' : `Request failed (${res.status})`
    throw new ApiError(describeDetail(data?.detail, fallback), res.status, data?.detail)
  }
  if (!type.includes('application/json')) {
    // A tunnel interstitial or a static host answering instead of the API.
    throw new ApiError('The server did not return API data. Check the backend URL.', res.status)
  }
  return data
}

export const api = {
  info: (o) => request('/api', o),
  authCheck: (o) => request('/api/auth/check', o),
  health: (o) => request('/api/health', o),
  jobs: (o) => request('/api/jobs', o),
  job: (id, o) => request(`/api/jobs/${encodeURIComponent(id)}`, o),
  createJob: (payload, o) => request('/api/jobs', { ...o, method: 'POST', json: payload, timeout: 160000 }),
  deleteJob: (id, o) => request(`/api/jobs/${encodeURIComponent(id)}`, { ...o, method: 'DELETE' }),
  outputs: (id, o) => request(`/api/outputs/${encodeURIComponent(id)}`, o),
  settings: (o) => request('/api/settings', o),
  updateSettings: (payload, o) => request('/api/settings', { ...o, method: 'PUT', json: payload }),
  uploadCookies: (file, o) =>
    request('/api/settings/youtube-cookies', {
      ...o, method: 'PUT', body: file, headers: { 'Content-Type': 'application/octet-stream' },
    }),
  removeCookies: (o) => request('/api/settings/youtube-cookies', { ...o, method: 'DELETE' }),
  shutdown: (o) => request('/api/shutdown', { ...o, method: 'POST' }),
}

/** /api/ready returns its report with 503 too — fetch it raw. */
export async function fetchReadiness(signal) {
  const res = await fetch(apiUrl('/api/ready'), {
    headers: { Accept: 'application/json', ...TUNNEL_HEADERS, ...authHeaders() },
    signal,
  })
  const data = await res.json().catch(() => null)
  if (!data || typeof data.ready !== 'boolean') throw new ApiError('Readiness report unavailable', res.status)
  return data
}

/** Upload with progress (fetch has no upload progress). */
export function uploadVideo(file, { onProgress, signal } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', apiUrl('/api/upload'))
    Object.entries({ ...TUNNEL_HEADERS, ...authHeaders() }).forEach(([k, v]) => xhr.setRequestHeader(k, v))
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total)
    xhr.onload = () => {
      let data = null
      try { data = JSON.parse(xhr.responseText) } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300 && data) resolve(data)
      else reject(new ApiError(describeDetail(data?.detail, `Upload failed (${xhr.status})`), xhr.status))
    }
    xhr.onerror = () => reject(new ApiError('Upload interrupted — check your connection.', 0))
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
    signal?.addEventListener('abort', () => xhr.abort(), { once: true })
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'deleted'])

/**
 * Live job progress over SSE (via fetch so auth headers work), reconnecting
 * with backoff. Returns a stop() function.
 */
export function streamJob(jobId, onEvent) {
  const controller = new AbortController()
  let attempt = 0
  let stopped = false

  const run = async () => {
    while (!stopped) {
      try {
        const res = await fetch(apiUrl(`/api/jobs/${encodeURIComponent(jobId)}/status`), {
          headers: { Accept: 'text/event-stream', ...TUNNEL_HEADERS, ...authHeaders() },
          signal: controller.signal,
        })
        if (!res.ok || !res.body || !(res.headers.get('content-type') || '').includes('text/event-stream')) {
          throw new ApiError('Stream unavailable', res.status)
        }
        attempt = 0
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        for (;;) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const frames = buffer.split(/\r?\n\r?\n/)
          buffer = frames.pop() || ''
          for (const frame of frames) {
            const data = frame.split(/\r?\n/).filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trimStart()).join('\n')
            if (!data) continue
            let event
            try { event = JSON.parse(data) } catch { continue }
            onEvent(event)
            if (TERMINAL.has(event.type)) { stopped = true; return }
          }
        }
      } catch (err) {
        if (stopped || controller.signal.aborted) return
        if (err.status === 404 || err.status === 401) { onEvent({ type: 'stream_error', status: err.status }); return }
      }
      if (stopped) return
      attempt += 1
      await new Promise((r) => setTimeout(r, Math.min(15000, 1000 * 2 ** Math.min(attempt, 4))))
    }
  }
  run()
  return () => { stopped = true; controller.abort() }
}
