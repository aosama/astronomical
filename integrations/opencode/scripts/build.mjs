import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, extname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

// npm workspaces hoist every dependency to the root node_modules, so a
// "../node_modules/..." path no longer resolves. Walk up to the first
// ancestor carrying node_modules and resolve packages from there — works
// whether deps are hoisted (root) or kept per-plugin.
function nodeModuleDir(pkgName) {
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = join(dir, "node_modules", pkgName);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(dir);
    if (parent === dir) {
      throw new Error(`Could not locate node_modules/${pkgName} from ${dir}`);
    }
    dir = parent;
  }
}

const { transformSolidSource } = await import(
  pathToFileURL(join(nodeModuleDir("@opentui/solid"), "scripts", "solid-transform.js")).href
);

const executeFile = promisify(execFile);
const repositoryRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const tscPath = join(nodeModuleDir("typescript"), "bin", "tsc");
const sourceDirectory = join(repositoryRoot, "src");
const outputDirectory = join(repositoryRoot, "dist");

async function listFiles(directory) {
  const directoryEntries = await readdir(directory, { withFileTypes: true });
  const nestedFiles = await Promise.all(
    directoryEntries.map((directoryEntry) => {
      const entryPath = join(directory, directoryEntry.name);
      return directoryEntry.isDirectory() ? listFiles(entryPath) : [entryPath];
    }),
  );
  return nestedFiles.flat();
}

// Keep the last complete output in place while TypeScript builds: OpenCode
// watches this directory for the installed local plugin, so deleting it first
// makes a normal rebuild look like a broken plugin to the running service.
// JSX is preserved so the Solid universal compiler, not TypeScript's eager
// react-jsx transform, owns reactive expression output.
await executeFile(process.execPath, [
  tscPath,
  "--project",
  join(repositoryRoot, "tsconfig.json"),
]);

const sourceFiles = await listFiles(sourceDirectory);
for (const sourceFile of sourceFiles) {
  if (extname(sourceFile) !== ".tsx") continue;
  const relativeSourcePath = relative(sourceDirectory, sourceFile);
  const emittedJsxPath = join(
    outputDirectory,
    dirname(relativeSourcePath),
    `${basename(relativeSourcePath, ".tsx")}.jsx`,
  );
  const transformedJavaScript = await transformSolidSource(
    await readFile(sourceFile, "utf8"),
    {
      filename: sourceFile,
      moduleName: "@opentui/solid",
    },
  );
  const emittedJavaScriptPath = emittedJsxPath.replace(/\.jsx$/, ".js");
  await writeFile(emittedJavaScriptPath, `${transformedJavaScript}\n`, "utf8");
  await rm(emittedJsxPath, { force: true });
}

// TypeScript emits declaration filenames from TSX sources correctly, but this
// defensive rename keeps package exports stable if a compiler version emits a
// .d.jsx declaration artifact in the future.
for (const emittedFile of await listFiles(outputDirectory)) {
  if (!emittedFile.endsWith(".d.jsx")) continue;
  await rename(emittedFile, emittedFile.replace(/\.d\.jsx$/, ".d.ts"));
}
