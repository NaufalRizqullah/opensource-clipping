import { useCallback, useEffect, useRef, useState } from 'react'
import { api, streamJob } from './api'
import { isActive } from './format'
import { storage } from './storage'

/** Fetch with loading/error state, abort on unmount, optional visibility-aware polling. */
export function useResource(loader, deps = [], { interval } = {}) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const loaderRef = useRef(loader)
  loaderRef.current = loader

  const load = useCallback(async (signal) => {
    try {
      const result = await loaderRef.current(signal)
      if (signal?.aborted) return
      setData(result)
      setError(null)
    } catch (err) {
      if (signal?.aborted || err?.name === 'AbortError') return
      setError(err)
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    load(controller.signal)
    return () => controller.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  const ms = typeof interval === 'function' ? interval(data) : interval
  useEffect(() => {
    if (!ms) return
    let controller = new AbortController()
    const tick = () => {
      if (document.visibilityState !== 'visible') return
      controller.abort()
      controller = new AbortController()
      load(controller.signal)
    }
    const id = setInterval(tick, ms)
    document.addEventListener('visibilitychange', tick)
    return () => { clearInterval(id); controller.abort(); document.removeEventListener('visibilitychange', tick) }
  }, [ms, load])

  const reload = useCallback(() => load(), [load])
  return { data, error, loading, reload, setData }
}

/** Job detail kept live by SSE while active, with a polling safety net. */
export function useLiveJob(jobId) {
  const res = useResource((signal) => api.job(jobId, { signal }), [jobId], {
    interval: (job) => (job && isActive(job) ? 8000 : 0),
  })
  const { data: job, setData, reload } = res
  const active = isActive(job)

  useEffect(() => {
    if (!active) return
    const stop = streamJob(jobId, (event) => {
      if (event.type === 'deleted') { reload(); return }
      if (event.type === 'stream_error') return
      setData((prev) => prev && {
        ...prev,
        status: event.status || prev.status,
        progress: event.progress ?? prev.progress,
        error: event.error ?? prev.error,
        cancel_requested: event.cancel_requested ?? prev.cancel_requested,
        ...(event.clips ? { clips: event.clips } : {}),
      })
      if (['completed', 'failed', 'cancelled'].includes(event.type)) reload()
    })
    return stop
  }, [jobId, active, setData, reload])

  return res
}

export function useLocalState(key, initial) {
  const [value, setValue] = useState(() => storage.getJSON(key, initial))
  useEffect(() => { storage.setJSON(key, value) }, [key, value])
  return [value, setValue]
}

export function useDocumentTitle(title) {
  useEffect(() => {
    document.title = title ? `${title} · Clipping Studio` : 'Clipping Studio'
  }, [title])
}

export function useHotkeys(map, enabled = true) {
  const ref = useRef(map)
  ref.current = map
  useEffect(() => {
    if (!enabled) return
    const onKey = (e) => {
      const t = e.target
      if (t && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName))) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const fn = ref.current[e.key] || ref.current[e.key.toLowerCase()]
      if (fn) { e.preventDefault(); fn(e) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enabled])
}
