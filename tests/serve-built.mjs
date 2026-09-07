/** A test-only loopback adapter for the same production worker used by SSR tests. */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(fileURLToPath(new URL("../dist/client/", import.meta.url)));
const contentTypes = { ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".woff": "font/woff", ".woff2": "font/woff2" };

export async function serveBuilt() {
  const { default: worker } = await import("../dist/server/index.js");
  const assets = {
    async fetch(request) {
      const url = new URL(typeof request === "string" ? request : request.url);
      const file = path.resolve(root, `.${decodeURIComponent(url.pathname)}`);
      if (!file.startsWith(`${root}${path.sep}`) && file !== root) return new Response("Not found", { status: 404 });
      try {
        const data = await readFile(file);
        return new Response(data, { headers: { "content-type": contentTypes[path.extname(file)] ?? "application/octet-stream" } });
      } catch {
        return new Response("Not found", { status: 404 });
      }
    },
  };
  const server = createServer(async (req, res) => {
    try {
      const request = new Request(`http://${req.headers.host}${req.url}`, { headers: req.headers });
      let response = await assets.fetch(request);
      if (response.status === 404) response = await worker.fetch(request, { ASSETS: assets }, { waitUntil() {}, passThroughOnException() {} });
      res.writeHead(response.status, Object.fromEntries(response.headers));
      res.end(Buffer.from(await response.arrayBuffer()));
    } catch (error) {
      console.error(error);
      res.writeHead(500);
      res.end("Production render failed");
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return { url: `http://127.0.0.1:${server.address().port}`, close: () => new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve())) };
}
