import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Cloudera AI / CDSW Application routes require binding to CDSW_APP_PORT.
const uiPort = Number(process.env.CDSW_APP_PORT || process.env.PORT || 5173);

export default defineConfig({
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: uiPort,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/v1": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: uiPort,
    strictPort: true,
  },
});
