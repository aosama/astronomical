// Append-only JSONL audit trail for setup, model list refreshes, provider
// registration, and terminal actions. A missing setup event means the module
// did not reach OpenCode's setup lifecycle.

import { appendFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { ASTRONOMICAL_PLUGIN_ID } from "./constants.ts";
import { resolveOpenCodeStateDirectory } from "./opencode-state-directory.ts";

// Beyond this size the log is trimmed rather than grown without bound.
const AUDIT_LOG_MAX_BYTES = 1_000_000;

// Trimming keeps the newest complete lines up to half the ceiling, so the
// recent history that explains a problem survives a trim, and the next trim is
// another half a megabyte of entries away rather than one entry away.
const AUDIT_LOG_KEPT_BYTES = AUDIT_LOG_MAX_BYTES / 2;

export type AstronomicalAuditEvent =
  | "setup-start"
  | "setup-skipped"
  | "provider-registered"
  | "provider-refreshed"
  | "provider-removed"
  | "diagnostic"
  | "status-command";

export type AstronomicalAuditLogger = (
  event: AstronomicalAuditEvent,
  detail?: Record<string, unknown>,
) => Promise<void>;

export function resolveAstronomicalAuditLogPath(
  environment: NodeJS.ProcessEnv = process.env,
  homeDirectory: string = homedir(),
): string {
  return join(
    resolveOpenCodeStateDirectory(environment, homeDirectory),
    "astronomical-plugin.log",
  );
}

/** The newest complete lines of `logText` whose total fits `keptBytes`. */
export function newestLinesWithin(logText: string, keptBytes: number): string {
  const logLines: string[] = logText.split("\n").filter(Boolean);
  const keptLines: string[] = [];
  let keptByteCount = 0;
  for (let lineIndex = logLines.length - 1; lineIndex >= 0; lineIndex -= 1) {
    const lineByteCount = Buffer.byteLength(logLines[lineIndex]!) + 1;
    if (keptByteCount + lineByteCount > keptBytes) break;
    keptLines.unshift(logLines[lineIndex]!);
    keptByteCount += lineByteCount;
  }
  return keptLines.map((keptLine: string): string => `${keptLine}\n`).join("");
}

async function trimOversizedLog(logPath: string): Promise<void> {
  try {
    const logStats = await stat(logPath);
    if (logStats.size <= AUDIT_LOG_MAX_BYTES) return;
    const logText: string = await readFile(logPath, "utf8");
    await writeFile(logPath, newestLinesWithin(logText, AUDIT_LOG_KEPT_BYTES));
  } catch {
    // A missing file is the normal first-run case; there is nothing to trim.
  }
}

export function createAstronomicalAuditLogger(
  input: { logPath?: string; now?: () => Date } = {},
): AstronomicalAuditLogger {
  const logPath: string = input.logPath ?? resolveAstronomicalAuditLogPath();
  const now: () => Date = input.now ?? ((): Date => new Date());

  return async (event, detail): Promise<void> => {
    try {
      await trimOversizedLog(logPath);
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

/** Records one event, ignoring any failure of a caller-supplied logger. */
export async function recordAuditEvent(
  audit: AstronomicalAuditLogger,
  event: AstronomicalAuditEvent,
  detail?: Record<string, unknown>,
): Promise<void> {
  try {
    await audit(event, detail);
  } catch {
    // Diagnostics must not prevent provider registration or a terminal command.
  }
}
