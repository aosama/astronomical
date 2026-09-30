#!/usr/bin/env node
/**
 * Browser preview server for the Thin Talk web client.
 *
 * The production client runs inside a WKWebView with two native supplies this
 * server reproduces in the browser, so the real bundle can be exercised
 * visually without building or launching the app:
 *
 * - The app's strict Content-Security-Policy forbids inline scripts, so the two
 *   native supplies are served as external same-origin scripts. The config is
 *   `window.__thintalkConfig` (read by `boot.ts`) and the session-bridge double
 *   is the prebuilt `preview/preview-shim.js` (the compiled `installPreviewShim`
 *   glue). Both load as `<script src=...>` and pass the CSP.
 * - The supervisor endpoints (`/v1/status`, `/v1/models`, `/v1/chat/completions`)
 *   are served by this same process, with the chat completion streamed over real
 *   SSE so token rendering is visible in the browser.
 * - The port is taken from an environment variable and, if busy, falls back to a
 *   free ephemeral port instead of crashing; a source edit is picked up on the
 *   next request because every asset is read from disk per request.
 *
 * Usage: THINTALK_PREVIEW_PORT=6790 node dev-server.mjs [--port 6790]
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const resourcesWebRoot = path.join(webRoot, "..", "Sources", "ThinTalkCanvas", "Resources", "web");
const indexHTMLPath = path.join(resourcesWebRoot, "index.html");
const previewShimPath = path.join(resourcesWebRoot, "preview", "preview-shim.js");
const DEFAULT_PORT = 6790;

const expectedChannel = process.env.THINTALK_PREVIEW_CHANNEL || "development";
const expectedStateDirectory = process.env.THINTALK_PREVIEW_STATE_DIR || "preview-state";

const preferredPortArg =
  process.argv.includes("--port") && Number.isInteger(Number(process.argv[process.argv.indexOf("--port") + 1]))
    ? Number(process.argv[process.argv.indexOf("--port") + 1])
    : 0;
const preferredPort = preferredPortArg || Number(process.env.THINTALK_PREVIEW_PORT) || DEFAULT_PORT;

let supervisorBaseURL = `http://127.0.0.1:${preferredPort}`;
let actualPort = preferredPort;
const previewConfig = {
  supervisorBaseURL,
  expectedChannel,
  expectedStateDirectory,
};

/**
 * The app's Content-Security-Policy forbids inline scripts, so the two native
 * supplies are injected as external same-origin `<script src=...>` tags. Order
 * matters: `preview-config.js` then `preview/preview-shim.js` load in the head
 * before the bundle, so `__thintalkConfig` exists and the bridge is installed
 * before `boot.ts` runs.
 */
const injectedScripts =
  `<script src="/preview-config.js"></script>` +
  `<script src="/preview/preview-shim.js"></script>`;

function readIndexHTML() {
  return fs.readFileSync(indexHTMLPath, "utf8")
    .replaceAll("thintalk-asset://shell/", "/")
    .replace("<head>", `<head>${injectedScripts}`);
}

const chatModel = {
  id: "preview-chat-model",
  input_modalities: ["text"],
  supported_endpoints: ["/v1/chat/completions"],
};

/**
 * Streams a scripted reply that quotes the user's message, so the streaming
 * render is visibly tied to the request: the server you are watching arrived
 * over server-sent events, one token at a time.
 */
function streamChatCompletion(requestBody, response) {
  const turns = Array.isArray(requestBody?.messages) ? requestBody.messages : [];
  const lastUserTurn = [...turns].reverse().find((turn) => turn?.role === "user");
  const userText = typeof lastUserTurn?.content === "string" ? lastUserTurn.content.trim() : "";
  const replyWords = (
    `You asked: "${userText}". Here is the prologue's answer, streamed word by word: ` +
    "Two households, both alike in dignity, in fair Verona, where we lay our scene. " +
    "The stream you are watching arrived over server-sent events, one token at a time."
  ).split(/(?<=\s)/);
  response.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-store",
    connection: "keep-alive",
  });
  let wordIndex = 0;
  const streamTimer = setInterval(() => {
    if (wordIndex < replyWords.length) {
      response.write(`data: ${JSON.stringify({ choices: [{ delta: { content: replyWords[wordIndex] } }] })}\n\n`);
      wordIndex += 1;
      return;
    }
    clearInterval(streamTimer);
    response.write("data: [DONE]\n\n");
    response.end();
  }, 70);
}

const contentTypes = {
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".json": "application/json",
  ".woff2": "font/woff2",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
};

const server = http.createServer((request, response) => {
  const url = new URL(request.url ?? "/", supervisorBaseURL);
  const requestStartedAtMs = performance.now();
  response.on("finish", () => {
    const durationMs = (performance.now() - requestStartedAtMs).toFixed(1);
    console.log(`[dev-server] ${request.method} ${url.pathname} -> ${response.statusCode} in ${durationMs}ms`);
  });

  if (url.pathname === "/v1/status") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({
      status: "ready",
      application: { channel: expectedChannel, state_directory: expectedStateDirectory },
    }));
    return;
  }
  if (url.pathname === "/v1/models") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ data: [chatModel] }));
    return;
  }
  if (url.pathname === "/v1/chat/completions" && request.method === "POST") {
    let rawBody = "";
    request.on("data", (segment) => { rawBody += segment; });
    request.on("end", () => {
      try {
        streamChatCompletion(JSON.parse(rawBody), response);
      } catch {
        response.writeHead(400, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: { message: "the request body was not valid JSON" } }));
      }
    });
    return;
  }
  if (url.pathname === "/preview-config.js") {
    // The API base must match the browsing origin exactly, or the boot fetches
    // become cross-origin and CORS blocks them ("Load failed" in WebKit).
    // Browsing via http://localhost:6791 and http://127.0.0.1:6791 are
    // different origins, so the config is derived per request from the Host
    // header rather than fixed at startup.
    const requestHost = request.headers.host ?? `127.0.0.1:${actualPort}`;
    const requestConfig = { ...previewConfig, supervisorBaseURL: `http://${requestHost}` };
    response.writeHead(200, { "content-type": contentTypes[".js"], "cache-control": "no-store" });
    response.end(`window.__thintalkConfig = ${JSON.stringify(requestConfig)};`);
    return;
  }
  if (url.pathname === "/preview/preview-shim.js") {
    if (!fs.existsSync(previewShimPath)) {
      response.writeHead(500, { "content-type": "text/plain" });
      response.end("preview-shim.js is missing; run `node build-web.mjs` to build it");
      return;
    }
    response.writeHead(200, { "content-type": contentTypes[".js"], "cache-control": "no-store" });
    fs.createReadStream(previewShimPath).pipe(response);
    return;
  }
  if (url.pathname === "/" || url.pathname === "/index.html") {
    response.writeHead(200, { "content-type": contentTypes[".html"], "cache-control": "no-store" });
    response.end(readIndexHTML());
    return;
  }
  const relativeAsset = path.normalize(url.pathname).replace(/^[/\\]+/, "");
  const assetPath = path.join(resourcesWebRoot, relativeAsset);
  if (!assetPath.startsWith(resourcesWebRoot) || !fs.existsSync(assetPath) || !fs.statSync(assetPath).isFile()) {
    response.writeHead(404, { "content-type": "text/plain" });
    response.end("no such asset on the preview server");
    return;
  }
  response.writeHead(200, { "content-type": contentTypes[path.extname(assetPath)] ?? "application/octet-stream", "cache-control": "no-store" });
  fs.createReadStream(assetPath).pipe(response);
});

function listenOnce(target) {
  return new Promise((resolve, reject) => {
    const onError = (error) => reject(error);
    server.once("error", onError);
    server.listen(target, "127.0.0.1", () => {
      server.removeListener("error", onError);
      resolve();
    });
  });
}

/**
 * Start on the preferred port, but if it is taken, fall back to a free ephemeral
 * port so the server never dies because a previous instance left the port bound
 * (the earlier crash was this server exiting on EADDRINUSE with no fallback).
 */
async function start() {
  try {
    await listenOnce(preferredPort);
  } catch (error) {
    if (error.code !== "EADDRINUSE") {
      console.error("[dev-server] failed to start:", error.message);
      process.exitCode = 1;
      return;
    }
    console.warn(`[dev-server] preferred port ${preferredPort} is busy; using a free ephemeral port`);
    await listenOnce(0);
  }

  const boundPort = server.address().port;
  actualPort = boundPort;
  supervisorBaseURL = `http://127.0.0.1:${boundPort}`;
  console.log(`[dev-server] Thin Talk preview running at ${supervisorBaseURL}`);
  console.log("[dev-server] Sessions persist in localStorage; clear site data for a fresh start.");
}

server.on("error", (error) => {
  console.error("[dev-server] server error:", error.message);
});

start();
