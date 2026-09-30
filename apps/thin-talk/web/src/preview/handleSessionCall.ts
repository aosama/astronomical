import { SessionBridgeOp } from "../SessionBridgeOp";
import { SessionBridgeError } from "../SessionBridgeError";

/** A session record as stored by the adapter (the persisted shape). */
export interface SessionStoreRecord {
  id: string;
  title: string;
  updatedAt: string;
  messages: unknown[];
}

/**
 * The read/write operations the preview shim answers. Each method is one axis
 * of the on-disk (here, localStorage-backed) session store; the pure logic that
 * maps a bridge op onto these — validation, ordering, null-for-missing, error
 * words — lives in {@link handleSessionCall} so the preview shim and any fake
 * host share exactly one implementation instead of hand-written string glue
 * that can silently drift.
 */
export interface SessionStoreAdapter {
  listSessions(): SessionStoreRecord[];
  loadSession(id: string): SessionStoreRecord | null;
  saveSession(record: SessionStoreRecord): void;
  deleteSession(id: string): void;
  renameSession(id: string, title: string): void;
  loadPreferences(): unknown;
  savePreferences(value: unknown): void;
}

function stringField(payload: unknown, key: string): string {
  if (typeof payload !== "object" || payload === null) {
    throw new SessionBridgeError(`payload for "${key}" must be an object`);
  }
  const value = (payload as Record<string, unknown>)[key];
  if (typeof value !== "string") {
    throw new SessionBridgeError(`payload field "${key}" must be a string`);
  }
  return value;
}

function normalizeSave(payload: unknown): SessionStoreRecord {
  if (typeof payload !== "object" || payload === null) {
    throw new SessionBridgeError("save payload must be an object");
  }
  const record = payload as Record<string, unknown>;
  return {
    id: stringField(record, "id"),
    title: typeof record["title"] === "string" ? record["title"] : "",
    updatedAt: typeof record["updatedAt"] === "string" ? record["updatedAt"] : "",
    messages: Array.isArray(record["messages"]) ? record["messages"] : [],
  };
}

/**
 * Maps one bridge op onto the store adapter, mirroring the store's semantics
 * exactly: newest-first listing, `null` for a missing load, and the exact error
 * messages a real store uses, so drift between the preview shim and a fake host
 * is caught by the contract test rather than surfacing in a running app.
 */
export function handleSessionCall(op: string, payload: unknown, adapter: SessionStoreAdapter): unknown {
  switch (op) {
    case SessionBridgeOp.LIST:
      return adapter.listSessions();
    case SessionBridgeOp.LOAD:
      return adapter.loadSession(stringField(payload, "id"));
    case SessionBridgeOp.SAVE: {
      adapter.saveSession(normalizeSave(payload));
      return null;
    }
    case SessionBridgeOp.DELETE:
      adapter.deleteSession(stringField(payload, "id"));
      return null;
    case SessionBridgeOp.RENAME:
      adapter.renameSession(stringField(payload, "id"), stringField(payload, "title"));
      return null;
    case SessionBridgeOp.LOAD_PREFS:
      return adapter.loadPreferences();
    case SessionBridgeOp.SAVE_PREFS:
      adapter.savePreferences(payload);
      return null;
    default:
      throw new SessionBridgeError(`host does not know op "${op}"`);
  }
}
