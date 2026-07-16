/**
 * server.ts
 * ---------
 * Serveur web Bun + Hono pour Boeuf Tracker.
 *
 * Rôle :
 *  1. Sert l'UI statique (HTML/CSS/JS) depuis public/
 *  2. Proxy transparent vers le worker Python (port 8100) pour /api/* et /video_feed
 *  3. Logique applicative : noms, dashboard, timeline (phases ultérieures)
 *
 * L'IA (YOLO + DINOv2 + race) reste dans le worker Python — Bun ne fait que
 * router et présenter.
 */
import { Hono } from "hono";
import { serveStatic } from "hono/bun";

const PYTHON_WORKER = process.env.PYTHON_WORKER ?? "http://localhost:8100";
const PORT = Number(process.env.PORT ?? 8000);

const app = new Hono();

// ─── Proxy vers le worker Python ────────────────────────────────────────
// Tout /api/* et /video_feed est forwardé tel quel (JSON, binaire, etc.).
async function proxyToPython(path: string, init?: RequestInit): Promise<Response> {
  const url = `${PYTHON_WORKER}${path}`;
  try {
    const resp = await fetch(url, init);
    // On retransmet la réponse telle quelle (headers + body binaire/JSON)
    const headers = new Headers();
    resp.headers.forEach((v, k) => headers.set(k, v));
    const body = await resp.arrayBuffer();
    return new Response(body, { status: resp.status, headers });
  } catch (e) {
    return Response.json(
      { ok: false, error: `Worker Python injoignable (${PYTHON_WORKER}). Lancez: python app.py --mlx` },
      { status: 502 }
    );
  }
}

// Proxy GET /api/* et /video_feed
app.all("/api/*", (c) => {
  const qs = c.req.raw.url.split("?")[1] ?? "";
  const path = c.req.path + (qs ? `?${qs}` : "");
  return proxyToPython(path, {
    method: c.req.method,
    headers: c.req.raw.headers,
    body: c.req.method !== "GET" && c.req.method !== "HEAD" ? c.req.raw.body : undefined,
  });
});

app.all("/video_feed", (c) => {
  return proxyToPython("/video_feed", {
    method: c.req.method,
    headers: c.req.raw.headers,
    body: c.req.method !== "GET" && c.req.method !== "HEAD" ? c.req.raw.body : undefined,
  });
});

// ─── Fichiers statiques (UI) ────────────────────────────────────────────
app.use("/*", serveStatic({ root: "./public" }));

// Fallback : index.html pour la route racine
app.get("/", (c) => {
  return serveStatic({ root: "./public" })(
    new Request("http://localhost/index.html"),
    c.env
  );
});

console.log(`╔══════════════════════════════════════════════╗`);
console.log(`║  BOEUF TRACKER — Serveur web Bun + Hono     ║`);
console.log(`║  URL       : http://localhost:${PORT}          ║`);
console.log(`║  Worker PY : ${PYTHON_WORKER.padEnd(28)} ║`);
console.log(`╚══════════════════════════════════════════════╝`);

export default {
  port: PORT,
  fetch: app.fetch,
};
