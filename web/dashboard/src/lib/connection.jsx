import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { api, connection } from './api'

const ConnectionContext = createContext(null)

/**
 * status: 'checking' | 'ready' | 'unauthorized' | 'unreachable'
 */
export function ConnectionProvider({ children }) {
  const [status, setStatus] = useState('checking')
  const [info, setInfo] = useState(null)
  const [error, setError] = useState('')
  const [base, setBaseState] = useState(connection.getBase())
  const seq = useRef(0)

  const check = useCallback(async () => {
    const id = ++seq.current
    setStatus((s) => (s === 'ready' ? s : 'checking'))
    try {
      let meta = null
      try {
        meta = await api.info({ timeout: 12000 })
      } catch (err) {
        // Older backends have no /api endpoint; fall back to liveness.
        if (err.status !== 404) throw err
        await api.health({ timeout: 12000 })
        meta = { auth_required: false }
      }
      if (meta.auth_required) await api.authCheck({ timeout: 12000 })
      else await api.jobs({ timeout: 12000 })
      if (id !== seq.current) return
      setInfo(meta)
      setError('')
      setStatus('ready')
    } catch (err) {
      if (id !== seq.current) return
      setError(err.message)
      setStatus(err.status === 401 ? 'unauthorized' : 'unreachable')
    }
  }, [])

  useEffect(() => { check() }, [check])

  const configure = useCallback(async ({ baseUrl, token }) => {
    if (baseUrl !== undefined) connection.setBase(baseUrl)
    if (token !== undefined) connection.setToken(token)
    setBaseState(connection.getBase())
    setStatus('checking')
    await check()
  }, [check])

  const value = useMemo(() => ({
    status, info, error, base, check, configure,
    hasToken: !!connection.getToken(),
  }), [status, info, error, base, check, configure])

  return <ConnectionContext.Provider value={value}>{children}</ConnectionContext.Provider>
}

export function useConnection() {
  const ctx = useContext(ConnectionContext)
  if (!ctx) throw new Error('useConnection must be used within ConnectionProvider')
  return ctx
}
