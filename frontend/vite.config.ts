import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server proxies /api -> the local volatility_lab_server (default
// port 8787, see apps/server/main.cpp). This keeps the frontend's own API
// client talking to a single origin (no CORS round-trip to reason about in
// dev) while the production build still just does `fetch('/api/...')`
// against whatever reverse-proxies it in front of the C++ service.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
      },
    },
  },
})
