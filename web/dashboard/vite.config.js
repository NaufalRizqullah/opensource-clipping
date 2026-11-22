import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // Docker compose sets VITE_API_PROXY=http://backend:8000
  const target = env.VITE_API_PROXY || 'http://127.0.0.1:8000'
  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: { '/api': { target, changeOrigin: true } },
      watch: env.VITE_POLLING ? { usePolling: true } : undefined,
    },
    build: {
      target: 'es2022',
      sourcemap: false,
      chunkSizeWarningLimit: 700,
    },
  }
})
