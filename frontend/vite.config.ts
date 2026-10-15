import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Use PLATFORM_UI_PORT for dev; CAI production uses built assets + ui_gateway (CDSW_APP_PORT).
const uiPort = Number(process.env.PLATFORM_UI_PORT || process.env.PORT || 5173);

export default defineConfig({
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  plugins: [react()],
  server: {
    host: "127.0.0.1",
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
    host: "127.0.0.1",
    port: uiPort,
    strictPort: true,
  },
});
