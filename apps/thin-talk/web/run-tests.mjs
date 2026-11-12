import { build } from "esbuild";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const indexHTML = path.join(webRoot, "..", "Sources", "ThinTalkCanvas", "Resources", "web", "index.html");
const bundlePath = path.join(webRoot, ".test-build", "tests.cjs");

await build({
  entryPoints: [path.join(webRoot, "test", "index.ts")],
  bundle: true,
  platform: "node",
  format: "cjs",
  outfile: bundlePath,
  sourcemap: "inline",
  logLevel: "warning",
});

// The heap limit makes a runaway test crash on its own within seconds. The
// whole suite peaks far below it, so a test that reaches it is broken: without
// a limit one such test once grew until the Mac ran out of application memory.
const TEST_PROCESS_HEAP_LIMIT_MEGABYTES = 1024;

const result = spawnSync(process.execPath, ["--test", `--max-old-space-size=${TEST_PROCESS_HEAP_LIMIT_MEGABYTES}`, bundlePath], {
  stdio: "inherit",
  env: { ...process.env, THINTALK_INDEX_HTML: indexHTML },
});
const testExitStatus = result.status != null ? result.status : 1;
process.exit(testExitStatus);
