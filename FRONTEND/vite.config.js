import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import fs from 'fs';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    https: {
      key: fs.readFileSync('../localhost+1-key.pem'),
      cert: fs.readFileSync('../localhost+1.pem'),
    },
    port: 5173,
  },
})
