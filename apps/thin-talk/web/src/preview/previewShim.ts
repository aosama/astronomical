import { handleSessionCall } from "./handleSessionCall";
import type { SessionStoreAdapter, SessionStoreRecord } from "./handleSessionCall";

/**
 * The minimal host bridge shape the shim reads/writes: the single
 * `webkit.messageHandlers.thintalk.postMessage` entry the native shell listens
 * on (see `vendor-globals.d.ts`), plus the op-resolver the native side calls
 * back into. The shim installs only this surface — nothing about `thintalk`
 * itself is referenced, so this lives in preview and the production shim does
 * not either.
 */
export interface PreviewWindowLike {
  webkit?: { messageHandlers?: { thintalk?: { postMessage(payload: unknown): void } } };
  __thintalkBridge?: {
    resolve?(callId: string, payload: unknown): void;
    reject?(callId: string, message: string): void;
  };
}

const SESSIONS_KEY = "thintalk.preview.sessions";
const PREFERENCES_KEY = "thintalk.preview.preferences";

/** JSON-backed persistence the adapter reads and writes, keyed by session id. */
function makeStorageAdapter(storage: Storage): SessionStoreAdapter {
  const readSessions = (): Record<string, SessionStoreRecord> =>
    storage.getItem(SESSIONS_KEY) ? JSON.parse(storage.getItem(SESSIONS_KEY) as string) : {};
  const writeSessions = (sessions: Record<string, SessionStoreRecord>): void => {
    if (Object.keys(sessions).length === 0) {
      storage.removeItem(SESSIONS_KEY);
    } else {
      storage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
    }
  };

  return {
    listSessions: () =>
      Object.values(readSessions())
        .map((record) => ({ ...record }))
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
    loadSession: (id) => {
      const record = readSessions()[id];
      return record ? { ...record } : null;
    },
    saveSession: (record) => {
      const sessions = readSessions();
      sessions[record.id] = { ...record };
      writeSessions(sessions);
    },
    deleteSession: (id) => {
      const sessions = readSessions();
      delete sessions[id];
      writeSessions(sessions);
    },
    renameSession: (id, title) => {
      const sessions = readSessions();
      if (sessions[id]) {
        sessions[id] = { ...sessions[id], title };
        writeSessions(sessions);
      }
    },
    loadPreferences: () => {
      return storage.getItem(PREFERENCES_KEY) ? JSON.parse(storage.getItem(PREFERENCES_KEY) as string) : {};
    },
    savePreferences: (value) => {
      storage.setItem(PREFERENCES_KEY, JSON.stringify(value));
    },
  };
}

/**
 * Installs the single `webkit.messageHandlers.thintalk.postMessage` surface the
 * native shell dispatches to, routing each `sessionCall` envelope through the
 * shared {@link handleSessionCall} contract and resolving or rejecting it on
 * the native side via `window.__thintalkBridge`.
 */
export function installPreviewShim(win: PreviewWindowLike, storage: Storage): void {
  const adapter = makeStorageAdapter(storage);

  const postMessage = (payload: unknown): void => {
    const envelope = payload as { kind: string; callId?: string; op?: string; payload?: unknown };
    if (envelope.kind !== "sessionCall" || !envelope.callId || !envelope.op) {
      return;
    }

    try {
      const reply = handleSessionCall(envelope.op, envelope.payload ?? null, adapter);
      win.__thintalkBridge?.resolve?.(envelope.callId, reply);
    } catch (error) {
      win.__thintalkBridge?.reject?.(envelope.callId, error instanceof Error ? error.message : String(error));
    }
  };

  win.webkit = win.webkit || {};
  win.webkit.messageHandlers = win.webkit.messageHandlers || {};
  win.webkit.messageHandlers.thintalk = { postMessage };
}
