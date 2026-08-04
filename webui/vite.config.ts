import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 開発時 (npm run dev) は /api を FastAPI (rageval web) に中継する。
// 本番は FastAPI が dist/ を配信するのでプロキシ不要。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 800,
  },
});
