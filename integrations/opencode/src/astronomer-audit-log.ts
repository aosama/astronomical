// Append-only JSONL audit trail for setup, catalogue refresh, provider
// registration, and terminal actions. A missing setup event means the module
// did not reach the V2 setup lifecycle.

import { appendFile, mkdir, stat, truncate } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { ASTRONOMICAL_PLUGIN_ID } from "./constants.js";

// Beyond this size the log is truncated rather than grown without bound. The
// newest loads overwrite from the top, which is acceptable for an audit of
// recent starts.
const AUDIT_LOG_MAX_BYTES = 1_000_000;

export type AstronomicalAuditEvent =
  | "setup-start"
  | "setup-skipped"
  | "provider-registered"
  | "provider-refreshed"
  | "provider-removed"
  | "diagnostic"
  | "status-command"
  | "toggle-command";

export type AstronomicalAuditLogger = (
  event: AstronomicalAuditEvent,
  detail?: Record<string, unknown>,
) => Promise<void>;

export function resolveAstronomicalAuditLogPath(
  env: NodeJS.ProcessEnv = process.env,
  homeDirectory: string = homedir(),
): string {
  const stateHome =
    env.XDG_STATE_HOME?.trim() || join(homeDirectory, ".local", "state");
  return join(stateHome, "opencode", "astronomical-plugin.log");
}

export function createAstronomicalAuditLogger(
  input: { logPath?: string; now?: () => Date } = {},
): AstronomicalAuditLogger {
  const logPath = input.logPath ?? resolveAstronomicalAuditLogPath();
  const now = input.now ?? (() => new Date());

  return async (event, detail) => {
    try {
      // A missing file is the normal first-run case, so a stat failure must
      // not stop the write.
      try {
        const stats = await stat(logPath);
        if (stats.size > AUDIT_LOG_MAX_BYTES) await truncate(logPath, 0);
      } catch {
        // File does not exist yet; fall through and create it.
      }

      await mkdir(dirname(logPath), { recursive: true });

      const entry = {
        timestamp: now().toISOString(),
        plugin: ASTRONOMICAL_PLUGIN_ID,
        event,
        ...detail,
      };
      await appendFile(logPath, `${JSON.stringify(entry)}\n`, "utf8");
    } catch {
      // A logging failure must never break plugin loading or registration.
    }
  };
}
