// Builds the Thin Talk web client into the SwiftPM canvas resources.
//
// The compiled bundle.js/bundle.css are NOT committed: the root .gitignore
// excludes them, so a fresh clone must run `npm install && npm run build`
// before any conversation renders — without them the canvas opens blank.
// CanvasWebBundleContractTests fails `swift test` with the recovery command
// when the bundle is missing, and run-thin-talk.sh aborts the same way.
// KaTeX and mermaid stay as vendored script tags (mermaid for
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
// The export feature is its own script, loaded the first time someone exports
// (src/ExportLauncher.ts): it carries the Word writer, which alone is about as
// large as the rest of the client, and most launches never need it.
const exportScriptPath = path.join(resourcesWebRoot, "export.js");
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
  for (const outputPath of [bundleScriptPath, bundleStylesheetPath, exportScriptPath, previewShimScriptPath]) {
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

// The shipped bundle stays unminified for readable stack traces, so the budget
// is measured on a minified build of the same entry: that tracks real code
// weight without counting comments and whitespace. A second in-memory pass is
// used instead of a metafile from the shipped build because a metafile reports
// unminified input sizes. The agent loop imports only Pi's agent core and the
// pi-ai event-stream utility; a stray import of one of Pi's provider adapters
// would pull in a vendor SDK (the OpenAI one is 275 KB minified) and trip this
// budget.
const MAXIMUM_MINIFIED_BUNDLE_BYTES = 600 * 1024;

async function checkBundleBudget() {
  reportProgress("bundle-budget", "start");
  const startedAtMs = Date.now();
  const minifiedBuild = await build({
    entryPoints: [path.join(webRoot, "src", "main.ts")],
    bundle: true,
    format: "iife",
    target: ["safari17"],
    write: false,
    logLevel: "silent",
    minify: true,
  });
  const minifiedByteCount = minifiedBuild.outputFiles.reduce(
    (byteCount, outputFile) => byteCount + outputFile.contents.byteLength,
    0,
  );
  const elapsedMs = Date.now() - startedAtMs;
  const budgetReport = `elapsed_ms=${elapsedMs} minified_bytes=${minifiedByteCount} budget_bytes=${MAXIMUM_MINIFIED_BUNDLE_BYTES}`;
  if (minifiedByteCount > MAXIMUM_MINIFIED_BUNDLE_BYTES) {
    reportProgress("bundle-budget", `exceeded ${budgetReport}`);
    throw new Error(
      `bundle.js is ${minifiedByteCount} bytes minified, over the ${MAXIMUM_MINIFIED_BUNDLE_BYTES}-byte budget; look for a newly bundled dependency`,
    );
  }
  reportProgress("bundle-budget", `within ${budgetReport}`);
}

// The export script's budget, on the same minified measure. docx and its HTML
// reader measured 598 KB minified when the script was introduced; the margin
// covers a minor release, and a budget broken by a new dependency points at
// it here.
const MAXIMUM_MINIFIED_EXPORT_SCRIPT_BYTES = 1024 * 1024;

async function bundleExportScript() {
  reportProgress("bundle-export-script", "start");
  const startedAtMs = Date.now();
  const exportEntryPath = path.join(webRoot, "src", "export", "exportEntry.ts");
  await build({
    entryPoints: [exportEntryPath],
    bundle: true,
    format: "iife",
    target: ["safari17"],
    outfile: exportScriptPath,
    legalComments: "inline",
    logLevel: "silent",
    sourcemap: false,
    minify: false,
  });
  const minifiedBuild = await build({
    entryPoints: [exportEntryPath],
    bundle: true,
    format: "iife",
    target: ["safari17"],
    write: false,
    logLevel: "silent",
    minify: true,
  });
  const minifiedByteCount = minifiedBuild.outputFiles.reduce((byteCount, outputFile) => byteCount + outputFile.contents.byteLength, 0);
  const budgetReport = `elapsed_ms=${Date.now() - startedAtMs} minified_bytes=${minifiedByteCount} budget_bytes=${MAXIMUM_MINIFIED_EXPORT_SCRIPT_BYTES}`;
  if (minifiedByteCount > MAXIMUM_MINIFIED_EXPORT_SCRIPT_BYTES) {
    reportProgress("bundle-export-script", `exceeded ${budgetReport}`);
    throw new Error(`export.js is ${minifiedByteCount} bytes minified, over the ${MAXIMUM_MINIFIED_EXPORT_SCRIPT_BYTES}-byte budget`);
  }
  reportProgress("bundle-export-script", `complete output=export.js ${budgetReport}`);
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
  await checkBundleBudget();
  await bundleExportScript();
  await bundleStylesheet();
  await bundlePreviewShim();
  reportProgress("done", "web client bundle is up to date");
}
