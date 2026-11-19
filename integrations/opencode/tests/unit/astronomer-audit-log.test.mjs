// Audit-log tests: JSONL entries land in the right shape, failures to write are
// swallowed rather than propagated, and an oversized log is truncated.

import assert from "node:assert/strict";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import {
  createAstronomicalAuditLogger,
  resolveAstronomicalAuditLogPath,
} from "../../dist/astronomer-audit-log.js";

describe("the audit log records the V2 plugin lifecycle as JSONL", () => {
  it("appends one parseable entry per event, with detail fields merged in", async () => {
    const directory = await mkdtemp(join(tmpdir(), "astronomical-audit-"));
    const logPath = join(directory, "opencode", "astronomical-plugin.log");
    let clock = 0;
    const audit = createAstronomicalAuditLogger({
      logPath,
      now: () => new Date(1_700_000_000_000 + clock++),
    });

    await audit("setup-start", { entry: "server" });
    await audit("provider-registered", {
      channel: "http://127.0.0.1:6732/v1",
      modelCount: 9,
    });

    const lines = (await readFile(logPath, "utf8")).trim().split("\n");
    assert.equal(lines.length, 2);

    const imported = JSON.parse(lines[0]);
    assert.equal(imported.plugin, "astronomical");
    assert.equal(imported.event, "setup-start");
    assert.equal(imported.entry, "server");
    assert.equal(imported.timestamp, "2023-11-14T22:13:20.000Z");

    const registered = JSON.parse(lines[1]);
    assert.equal(registered.event, "provider-registered");
    assert.equal(registered.modelCount, 9);
    assert.equal(registered.channel, "http://127.0.0.1:6732/v1");

    await rm(directory, { recursive: true, force: true });
  });
});

describe("a logging failure must never break the plugin", () => {
  it("swallows an unwritable log path instead of throwing", async () => {
    const directory = await mkdtemp(
      join(tmpdir(), "astronomical-audit-blocked-"),
    );
    // A plain file where the log's parent directory should be makes mkdir fail.
    const blocker = join(directory, "blocker");
    await writeFile(blocker, "not a directory", "utf8");

    const audit = createAstronomicalAuditLogger({
      logPath: join(blocker, "nested", "astronomical-plugin.log"),
    });

    // Must resolve, not reject.
    await audit("setup-start", { entry: "server" });
    await rm(directory, { recursive: true, force: true });
  });
});

describe("an oversized audit log is truncated rather than grown without bound", () => {
  it("replaces the log with the new entry once it exceeds the ceiling", async () => {
    const directory = await mkdtemp(join(tmpdir(), "astronomical-audit-cap-"));
    const logPath = join(directory, "astronomical-plugin.log");
    // A line bigger than the 1MB ceiling.
    await writeFile(logPath, `x`.repeat(1_100_000), "utf8");

    const audit = createAstronomicalAuditLogger({ logPath });
    await audit("provider-registered", { modelCount: 9 });

    const contents = await readFile(logPath, "utf8");
    const stats = await stat(logPath);
    assert.ok(stats.size < 10_000, "log should have been truncated");
    assert.equal(JSON.parse(contents.trim()).event, "provider-registered");

    await rm(directory, { recursive: true, force: true });
  });
});

describe("the audit log lives in opencode's state directory", () => {
  it("resolves under XDG_STATE_HOME when set", () => {
    assert.equal(
      resolveAstronomicalAuditLogPath({ XDG_STATE_HOME: "/xdg/state" }),
      "/xdg/state/opencode/astronomical-plugin.log",
    );
  });

  it("falls back to ~/.local/state when XDG_STATE_HOME is unset", () => {
    assert.equal(
      resolveAstronomicalAuditLogPath({}, "/home/tester"),
      "/home/tester/.local/state/opencode/astronomical-plugin.log",
    );
  });
});
