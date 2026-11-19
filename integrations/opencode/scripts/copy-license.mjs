// The repository has one LICENSE at its root, and npm only packs files inside
// the package folder. Copying it in for the pack (and removing it after) keeps
// one license file to maintain while every published package carries it.
// Node rather than `cp`, so packing works the same in any shell npm uses.

import { copyFileSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";

const repositoryLicenseUrl = new URL("../../../LICENSE", import.meta.url);
const packagedLicenseUrl = new URL("../LICENSE", import.meta.url);

if (process.argv.includes("--remove")) {
  rmSync(fileURLToPath(packagedLicenseUrl), { force: true });
} else {
  copyFileSync(
    fileURLToPath(repositoryLicenseUrl),
    fileURLToPath(packagedLicenseUrl),
  );
}
