import { Hono } from "hono";

type Env = { Bindings: { ASSETS: Fetcher; MODELS: R2Bucket } };

// Cross-origin isolation enables WASM threads. public/_headers covers static assets.
const ISOLATION = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};
const TYPES: Record<string, string> = { wasm: "application/wasm", mjs: "text/javascript", onnx: "application/octet-stream", txt: "text/plain; charset=utf-8" };

const app = new Hono<Env>();

// Keys are versioned, so responses are immutable.
app.get("/models/*", async (c) => {
  const key = c.req.path.slice(1);
  const obj = await c.env.MODELS.get(key, { onlyIf: c.req.raw.headers });
  if (!obj) return c.notFound();
  const headers = new Headers(ISOLATION);
  obj.writeHttpMetadata(headers);
  headers.set("ETag", obj.httpEtag);
  headers.set("Cache-Control", "public, max-age=31536000, immutable");
  headers.set("Content-Type", TYPES[key.split(".").pop() ?? ""] ?? "application/octet-stream");
  if (!("body" in obj)) return new Response(null, { status: 304, headers });
  headers.set("Content-Length", String(obj.size));
  return new Response(obj.body, { headers });
});

app.all("*", (c) => c.env.ASSETS.fetch(c.req.raw));

export default app;
