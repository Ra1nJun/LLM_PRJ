import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import fs from 'fs';

const isCI = process.env.CI === 'true';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: isCI ? {} : {
      https: {
        key: fs.readFileSync('../localhost+1-key.pem'),
        cert: fs.readFileSync('../localhost+1.pem'),
      },
      port: 5173,
    }
})
