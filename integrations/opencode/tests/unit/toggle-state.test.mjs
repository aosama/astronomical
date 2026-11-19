// Shared state tests cover new V2 toggle persistence and the one-time fallback
// to the V1 terminal preference during upgrade.

import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import {
  readAstronomicalPluginEnabled,
  resolveAstronomicalPluginStateFilePath,
  writeAstronomicalPluginEnabled,
} from "../../dist/toggle-state.js";

describe("the shared Astronomical provider toggle", () => {
  it("defaults to enabled when neither new nor legacy state exists", async () => {
    const directory = await mkdtemp(join(tmpdir(), "astronomical-state-"));
    try {
      assert.equal(
        await readAstronomicalPluginEnabled({
          stateFilePath: join(directory, "provider.json"),
          legacyStateFilePath: join(directory, "legacy-kv.json"),
        }),
        true,
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("persists the new state atomically and reads it back", async () => {
    const directory = await mkdtemp(
      join(tmpdir(), "astronomical-state-write-"),
    );
    const stateFilePath = join(directory, "nested", "provider.json");
    try {
      await writeAstronomicalPluginEnabled(false, { stateFilePath });

      assert.equal(
        await readAstronomicalPluginEnabled({ stateFilePath }),
        false,
      );
      assert.deepEqual(JSON.parse(await readFile(stateFilePath, "utf8")), {
        enabled: false,
      });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("honors the old terminal toggle only until new state is written", async () => {
    const directory = await mkdtemp(
      join(tmpdir(), "astronomical-state-legacy-"),
    );
    const stateFilePath = join(directory, "provider.json");
    const legacyStateFilePath = join(directory, "legacy-kv.json");
    try {
      await writeFile(
        legacyStateFilePath,
        JSON.stringify({ plugin_enabled: { Astronomical: false } }),
        "utf8",
      );

      assert.equal(
        await readAstronomicalPluginEnabled({
          stateFilePath,
          legacyStateFilePath,
        }),
        false,
      );

      await writeAstronomicalPluginEnabled(true, { stateFilePath });
      assert.equal(
        await readAstronomicalPluginEnabled({
          stateFilePath,
          legacyStateFilePath,
        }),
        true,
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("ignores another plugin's legacy toggle and tolerates malformed state", async () => {
    const directory = await mkdtemp(
      join(tmpdir(), "astronomical-state-invalid-"),
    );
    const stateFilePath = join(directory, "provider.json");
    const legacyStateFilePath = join(directory, "legacy-kv.json");
    try {
      await writeFile(stateFilePath, "{ not json", "utf8");
      await writeFile(
        legacyStateFilePath,
        JSON.stringify({ plugin_enabled: { "Other Plugin": false } }),
        "utf8",
      );

      assert.equal(
        await readAstronomicalPluginEnabled({
          stateFilePath,
          legacyStateFilePath,
        }),
        true,
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("resolves under XDG_STATE_HOME and otherwise under the home directory", () => {
    assert.equal(
      resolveAstronomicalPluginStateFilePath(
        { XDG_STATE_HOME: "/xdg/state" },
        "/home/tester",
      ),
      "/xdg/state/opencode/astronomical-plugin-state.json",
    );
    assert.equal(
      resolveAstronomicalPluginStateFilePath({}, "/home/tester"),
      "/home/tester/.local/state/opencode/astronomical-plugin-state.json",
    );
  });
});
