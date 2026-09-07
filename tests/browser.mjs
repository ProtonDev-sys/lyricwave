/** Real production-browser checks. All songs/lyrics/API responses are synthetic. */
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium, firefox, webkit } from "playwright";
import AxeBuilder from "@axe-core/playwright";
import { serveBuilt } from "./serve-built.mjs";

const reportDir = path.resolve("test-results/browser");
await mkdir(reportDir, { recursive: true });
const app = await serveBuilt();
const browserNames = (process.env.LYRICWAVE_TEST_BROWSERS ?? "chromium,firefox,webkit").split(",");
const results = [];
const phrases = ["Amber light across the room", "Every sound has somewhere to go", "Open windows let the morning in", "We trace a rhythm of our own", "A quiet pause before the chorus", "And then the room comes alive", "Another note another new beginning", "Carry this moment through the night"];
const lines = phrases.map((phrase, lineIndex) => {
  const start = 1 + lineIndex * 6;
  const words = phrase.split(" ").map((text, i) => ({ text, start: start + i * .55, end: start + i * .55 + .48, kind: "lead", timing_source: lineIndex === 4 ? "estimated" : "qwen" }));
  return { id: `line-${lineIndex}`, start, end: words.at(-1).end, words, kind: "lead" };
});
const completed = { id: "fixture", stage: "complete", progress: 100, status: "Ready", duration: 60, words: lines.flatMap((line) => line.words), lines, vocal_url: "/api/jobs/fixture/vocals", quality: "accurate", language: "english", device: "Synthetic test engine", transcription_model: "fixture-asr", alignment_model: "fixture-aligner" };
function wav() {
  const samples = 8000 * 60;
  const data = Buffer.alloc(44 + samples * 2);
  data.write("RIFF", 0); data.writeUInt32LE(data.length - 8, 4); data.write("WAVEfmt ", 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22);
  data.writeUInt32LE(8000, 24); data.writeUInt32LE(16000, 28); data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
  data.write("data", 36); data.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) data.writeInt16LE(Math.round(Math.sin(i * 2 * Math.PI * 220 / 8000) * 400), 44 + i * 2);
  return data;
}
const audio = wav();
const file = { name: "Studio Band - Night Drive.wav", mimeType: "audio/wav", buffer: audio };
const health = { ok: true, ready: true, cuda: true, device: "Synthetic test engine", ffmpeg: true, demucs: true, transformers: true, request_token: "fixture-token", default_quality: "balanced" };
const types = { chromium, firefox, webkit };
let failures = 0;

try {
  for (const name of browserNames) {
    const browser = await types[name].launch();
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: "reduce", acceptDownloads: true });
    const page = await context.newPage();
    page.setDefaultTimeout(15000);
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    let online = true;
    let jobState = "separating";
    let submissions = 0;
    let cancellations = 0;
    let uploadBody = "";
    await page.route("http://127.0.0.1:8008/**", async (route) => {
      const request = route.request();
      const pathname = new URL(request.url()).pathname;
      const headers = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "GET,POST,DELETE,OPTIONS" };
      if (request.method() === "OPTIONS") return route.fulfill({ status: 204, headers });
      if (pathname.endsWith("/vocals")) return route.fulfill({ body: audio, contentType: "audio/wav", headers });
      if (pathname.endsWith("/health")) return route.fulfill({ json: { ...health, ready: online }, headers });
      if (request.method() === "POST") {
        assert.equal(request.headers()["x-lyricwave-token"], "fixture-token");
        uploadBody = request.postDataBuffer().toString();
        submissions++;
        return route.fulfill({ json: { id: "fixture" }, status: 202, headers });
      }
      if (request.method() === "DELETE") {
        cancellations++;
        return route.fulfill({ json: { id: "fixture", stage: "cancelled" }, headers });
      }
      if (jobState === "error") return route.fulfill({ json: { ...completed, stage: "error", error: "Synthetic engine failure" }, headers });
      return route.fulfill({ json: jobState === "complete" ? completed : { id: "fixture", stage: jobState, progress: 35, status: "Processing synthetic fixture", duration: 60 }, headers });
    });
    async function check(label, action) {
      try { await action(); results.push({ browser: name, test: label, status: "passed" }); console.log(`PASS ${name}: ${label}`); }
      catch (error) { failures++; results.push({ browser: name, test: label, status: "failed", error: String(error) }); await page.screenshot({ path: path.join(reportDir, `${name}-failure-${failures}.png`), fullPage: true }); throw error; }
    }
    async function accessible(label) {
      const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
      await writeFile(path.join(reportDir, `${name}-${label}-axe.json`), JSON.stringify(result.violations, null, 2));
      assert.deepEqual(result.violations.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.map((n) => ({ target: n.target, summary: n.failureSummary })) })), []);
    }
    try {
      await check("production hydration, ready engine, keyboard skip link", async () => {
        await page.goto(app.url);
        await page.getByRole("button", { name: "Local GPU ready" }).waitFor();
        await page.keyboard.press("Tab");
        assert.equal(await page.locator(":focus").textContent(), "Skip to lyrics");
        assert.equal(await page.getByRole("heading", { name: /Hear the music/ }).count(), 1);
        await page.locator("body").click({ position: { x: 1, y: 1 } });
        await accessible("empty");
        await page.screenshot({ path: path.join(reportDir, `${name}-desktop.png`), fullPage: true });
      });
      await check("responsive 320/390/768/1440 layouts retain upload and profile controls", async () => {
        for (const width of [320, 390, 768, 1440]) {
          await page.setViewportSize({ width, height: 900 });
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `overflow at ${width}`);
          await page.getByRole("button", { name: "Choose audio" }).scrollIntoViewIfNeeded();
          assert.ok(await page.getByRole("button", { name: "Choose audio" }).isVisible());
          assert.equal(await page.getByRole("radio").count(), 3);
          if (width === 390) await page.screenshot({ path: path.join(reportDir, `${name}-mobile.png`), fullPage: true });
        }
        await page.evaluate(() => scrollTo(0, 0));
      });
      await check("invalid upload is rejected before a job is sent", async () => {
        await page.getByLabel("Choose an audio file").setInputFiles({ name: "not-a-song.txt", mimeType: "text/plain", buffer: Buffer.from("test") });
        await page.getByRole("alert").waitFor();
        assert.equal(submissions, 0);
        await page.getByRole("button", { name: "Dismiss message" }).click();
      });
      await check("quality/language selection, authenticated upload and progress", async () => {
        await page.getByRole("radio", { name: /Best quality/ }).check();
        await page.getByLabel("Lyrics language").selectOption("english");
        await page.getByLabel("Choose an audio file").setInputFiles(file);
        await page.getByText("Separating vocals", { exact: true }).waitFor();
        assert.match(uploadBody, /name="quality"\r?\n\r?\naccurate/);
        assert.match(uploadBody, /name="language"\r?\n\r?\nenglish/);
        assert.equal(submissions, 1);
        await page.screenshot({ path: path.join(reportDir, `${name}-processing.png`), fullPage: true });
      });
      await check("completed lyrics preserve filename metadata and measured timestamps", async () => {
        jobState = "complete";
        await page.getByRole("button", { name: "Word JSON" }).waitFor();
        await page.waitForFunction(() => document.querySelector("audio").readyState >= 1);
        assert.equal(await page.getByRole("heading", { name: "Night Drive" }).count(), 1);
        assert.ok((await page.locator(".track-heading").textContent()).includes("Studio Band"));
        assert.equal(await page.locator(".lyric-line").count(), lines.length);
        await accessible("loaded");
        await page.screenshot({ path: path.join(reportDir, `${name}-lyrics.png`), fullPage: true });
      });
      await check("calibrated word seeking, playback speed, and focused-input keyboard safety", async () => {
        const offset = page.getByLabel("Lyric timing offset in milliseconds");
        await offset.fill("250");
        await page.getByText("Amber", { exact: true }).click();
        assert.ok(Math.abs(await page.locator("audio").evaluate((element) => element.currentTime) - 1.25) < .1);
        await offset.focus();
        const before = await page.locator("audio").evaluate((element) => element.currentTime);
        await page.keyboard.press("ArrowRight");
        assert.equal(await page.locator("audio").evaluate((element) => element.currentTime), before);
        await page.getByLabel("Playback speed").selectOption("0.75");
        assert.equal(await page.locator("audio").evaluate((element) => element.playbackRate), .75);
      });
      await check("play/pause, mute and source-switch preserve playback settings", async () => {
        await page.getByRole("button", { name: "Play", exact: true }).click();
        await page.getByRole("button", { name: "Pause", exact: true }).waitFor();
        await page.waitForFunction(() => document.querySelector("audio").currentTime > 1.3);
        await page.getByRole("button", { name: "Pause", exact: true }).click();
        await page.getByRole("button", { name: "Mute", exact: true }).click();
        assert.ok(await page.locator("audio").evaluate((element) => element.muted));
        const before = await page.locator("audio").evaluate((element) => element.currentTime);
        await page.getByRole("button", { name: "Vocal", exact: true }).click();
        await page.waitForFunction(() => document.querySelector("audio").src.includes("/vocals") && document.querySelector("audio").readyState >= 1);
        assert.ok(Math.abs(await page.locator("audio").evaluate((element) => element.currentTime) - before) < .2);
        assert.equal(await page.locator("audio").evaluate((element) => element.playbackRate), .75);
        assert.ok(await page.locator("audio").evaluate((element) => element.muted));
      });
      await check("manual lyric scrolling can resume and does not scroll the document", async () => {
        await page.locator(".lyrics-scroll").dispatchEvent("wheel", { deltaY: 200 });
        await page.getByRole("button", { name: "Resume auto-follow" }).waitFor();
        const before = await page.evaluate(() => scrollY);
        await page.getByRole("button", { name: "Resume auto-follow" }).click();
        await page.getByRole("button", { name: "Auto-follow on" }).waitFor();
        assert.equal(await page.evaluate(() => scrollY), before);
      });
      await check("downloaded LRC is calibrated while JSON retains acoustic provenance", async () => {
        for (const [button, extension] of [["Word JSON", "json"], ["LRC", "lrc"]]) {
          const pending = page.waitForEvent("download");
          await page.getByRole("button", { name: button, exact: true }).click();
          const download = await pending;
          const content = await readFile(await download.path(), "utf8");
          assert.ok(download.suggestedFilename().endsWith(`.${extension}`));
          if (extension === "json") {
            const exported = JSON.parse(content);
            assert.equal(exported.playbackOffsetMs, 250);
            assert.equal(exported.lines[0].words[0].start, 1);
            assert.equal(exported.lines[0].words[0].timing_source, "qwen");
          } else assert.match(content, /^\[00:01\.25\]Amber light/);
        }
      });
      await check("mobile loaded state keeps vocal, volume and calibration controls", async () => {
        await page.setViewportSize({ width: 390, height: 844 });
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        assert.ok(await page.getByRole("button", { name: "Vocal", exact: true }).isVisible());
        assert.ok(await page.getByLabel("Volume", { exact: true }).isVisible());
        assert.ok(await page.getByLabel("Lyric timing offset in milliseconds").isVisible());
        await accessible("mobile-loaded");
        await page.screenshot({ path: path.join(reportDir, `${name}-mobile-lyrics.png`), fullPage: true });
      });
      await check("processing errors surface and retry succeeds", async () => {
        jobState = "error";
        await page.getByLabel("Choose an audio file").setInputFiles(file);
        await page.getByRole("button", { name: "Try again" }).waitFor();
        assert.ok((await page.locator(".error-card").textContent()).includes("Synthetic engine failure"));
        jobState = "complete";
        await page.getByRole("button", { name: "Try again" }).click();
        await page.getByRole("button", { name: "Word JSON" }).waitFor();
      });
      await check("reset cancels active job and clears audio, lyrics and calibration", async () => {
        jobState = "transcribing";
        await page.getByLabel("Choose an audio file").setInputFiles(file);
        await page.getByText("Transcribing and aligning", { exact: true }).waitFor();
        await page.getByRole("button", { name: "Load another audio file" }).click();
        await page.getByRole("button", { name: "Choose audio" }).waitFor();
        assert.equal(await page.locator("audio").getAttribute("src"), null);
        await page.waitForTimeout(200);
        assert.ok(cancellations >= 1);
      });
      await check("offline engine has an explicit recheck path", async () => {
        online = false;
        await page.reload();
        await page.getByRole("button", { name: "Local engine offline" }).click();
        online = true;
        await page.getByRole("button", { name: "Check engine again" }).click();
        await page.getByRole("button", { name: "Local GPU ready" }).waitFor();
        assert.deepEqual(pageErrors, []);
      });
    } catch (error) {
      console.error(`${name}:`, error);
    } finally {
      await context.close();
      await browser.close();
    }
  }
} finally {
  await app.close();
  await writeFile(path.join(reportDir, "results.json"), JSON.stringify({ failures, results }, null, 2));
}
if (failures) process.exitCode = 1;
