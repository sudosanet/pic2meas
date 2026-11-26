/// <reference types="vitest/config" />
import babel from "@rolldown/plugin-babel";
import { cloudflare } from "@cloudflare/vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

// Vite emits onnxruntime's wasm (28 MB, over the 25 MiB asset limit); the app loads it from R2 instead.
const dropOrtWasm = (): Plugin => ({
  name: "drop-ort-wasm",
  generateBundle(_, bundle) { for (const k of Object.keys(bundle)) if (/ort-wasm.*\.wasm$/.test(k)) delete bundle[k]; },
});

const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};

export default defineConfig({
  plugins: [react(), babel({ presets: [reactCompilerPreset()] }), tailwindcss(), cloudflare(), dropOrtWasm()],
  worker: { format: "es", plugins: () => [dropOrtWasm()] },
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  test: { include: ["src/**/*.test.ts"] },
});
