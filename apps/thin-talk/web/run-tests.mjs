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

const result = spawnSync(process.execPath, ["--test", bundlePath], {
  stdio: "inherit",
  env: { ...process.env, THINTALK_INDEX_HTML: indexHTML },
});
process.exit(result.status ?? 1);
