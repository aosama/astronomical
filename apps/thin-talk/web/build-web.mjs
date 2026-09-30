// Builds the Thin Talk web client into the SwiftPM canvas resources.
//
// The compiled bundle is committed so `swift build` and `swift test` work on a
// machine without Node; this script regenerates it whenever the TypeScript
// sources change. KaTeX and mermaid stay as vendored script tags (mermaid for
// startup size, KaTeX for its font asset tree) and are not bundled here.

import { build } from "esbuild";
import { mkdir, rm, stat } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const resourcesWebRoot = path.join(
  webRoot,
  "..",
  "Sources",
  "ThinTalkCanvas",
  "Resources",
  "web",
);
const bundleScriptPath = path.join(resourcesWebRoot, "bundle.js");
const bundleStylesheetPath = path.join(resourcesWebRoot, "bundle.css");
const previewShimScriptPath = path.join(resourcesWebRoot, "preview", "preview-shim.js");

const shouldCleanOnly = process.argv.includes("--clean");

function reportProgress(step, detail) {
  const timestamp = new Date().toISOString();
  console.log(`[build-web] ${timestamp} status=${step} ${detail}`);
}

async function fileExists(candidatePath) {
  try {
    await stat(candidatePath);
    return true;
  } catch {
    return false;
  }
}

async function cleanOutputs() {
  for (const outputPath of [bundleScriptPath, bundleStylesheetPath, previewShimScriptPath]) {
    if (await fileExists(outputPath)) {
      await rm(outputPath);
      reportProgress("clean", `removed ${path.basename(outputPath)}`);
    }
  }
}

async function bundleScript() {
  reportProgress("bundle-script", "start");
  const startedAtMs = Date.now();
  const result = await build({
    entryPoints: [path.join(webRoot, "src", "main.ts")],
    bundle: true,
    format: "iife",
    target: ["safari17"],
    outfile: bundleScriptPath,
    legalComments: "inline",
    logLevel: "silent",
    sourcemap: false,
    minify: false,
  });
  const elapsedMs = Date.now() - startedAtMs;
  if (result.errors.length > 0) {
    throw new Error(`esbuild reported ${result.errors.length} error(s)`);
  }
  reportProgress(
    "bundle-script",
    `complete elapsed_ms=${elapsedMs} output=bundle.js`,
  );
}

async function bundleStylesheet() {
  reportProgress("bundle-stylesheet", "start");
  const startedAtMs = Date.now();
  const result = await build({
    entryPoints: [path.join(webRoot, "src", "main.css")],
    bundle: true,
    outfile: bundleStylesheetPath,
    logLevel: "silent",
    sourcemap: false,
    minify: false,
  });
  const elapsedMs = Date.now() - startedAtMs;
  if (result.errors.length > 0) {
    throw new Error(`esbuild reported ${result.errors.length} error(s)`);
  }
  reportProgress(
    "bundle-stylesheet",
    `complete elapsed_ms=${elapsedMs} output=bundle.css`,
  );
}

async function bundlePreviewShim() {
  reportProgress("bundle-preview-shim", "start");
  const startedAtMs = Date.now();
  await mkdir(path.dirname(previewShimScriptPath), { recursive: true });
  const result = await build({
    entryPoints: [path.join(webRoot, "src", "preview", "previewShimEntry.ts")],
    bundle: true,
    format: "iife",
    target: ["safari17"],
    outfile: previewShimScriptPath,
    logLevel: "silent",
    sourcemap: false,
    minify: false,
  });
  const elapsedMs = Date.now() - startedAtMs;
  if (result.errors.length > 0) {
    throw new Error(`esbuild reported ${result.errors.length} error(s)`);
  }
  reportProgress(
    "bundle-preview-shim",
    `complete elapsed_ms=${elapsedMs} output=preview/preview-shim.js`,
  );
}

if (shouldCleanOnly) {
  await cleanOutputs();
} else {
  await cleanOutputs();
  await bundleScript();
  await bundleStylesheet();
  await bundlePreviewShim();
  reportProgress("done", "web client bundle is up to date");
}
