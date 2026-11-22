/** localStorage wrapper that never throws (private mode, blocked storage). */
export const storage = {
  get(key, fallback = null) {
    try {
      const v = window.localStorage.getItem(key)
      return v === null ? fallback : v
    } catch { return fallback }
  },
  set(key, value) {
    try { window.localStorage.setItem(key, value) } catch { /* ignore */ }
  },
  remove(key) {
    try { window.localStorage.removeItem(key) } catch { /* ignore */ }
  },
  getJSON(key, fallback = null) {
    const raw = this.get(key)
    if (raw === null) return fallback
    try { return JSON.parse(raw) } catch { return fallback }
  },
  setJSON(key, value) { this.set(key, JSON.stringify(value)) },
}
