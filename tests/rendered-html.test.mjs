import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the lyricwave workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>lyricwave — Your song, word for word<\/title>/i);
  assert.match(html, /Turn any song into live lyrics/);
  assert.match(html, /100% on-device/);
  assert.match(html, /Drop your song here/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("keeps audio processing client-local", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /@browserai\/browserai\/demucs/);
  assert.match(page, /whisper-base_timestamped/);
  assert.match(page, /return_timestamps:\s*"word"/);
  assert.match(page, /URL\.createObjectURL/);
  assert.doesNotMatch(page, /fetch\(["'`]\/api|FormData\(\)/);
});
