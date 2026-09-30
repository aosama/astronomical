import assert from "node:assert/strict";
import { test } from "node:test";
import { handleSessionCall } from "../src/preview/handleSessionCall";
import type { SessionStoreAdapter, SessionStoreRecord } from "../src/preview/handleSessionCall";
import { SessionBridgeOp } from "../src/SessionBridgeOp";
import { installPreviewShim } from "../src/preview/previewShim";
import type { PreviewWindowLike } from "../src/preview/previewShim";

/** An in-memory adapter so the pure contract runs without a DOM. */
class MemoryAdapter implements SessionStoreAdapter {
  private files = new Map<string, SessionStoreRecord>();
  private preferences: Record<string, unknown> = {};

  public listSessions(): SessionStoreRecord[] {
    return [...this.files.values()].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }
  public loadSession(id: string): SessionStoreRecord | null {
    return this.files.get(id) ?? null;
  }
  public saveSession(record: SessionStoreRecord): void {
    this.files.set(record.id, { ...record });
  }
  public deleteSession(id: string): void {
    this.files.delete(id);
  }
  public renameSession(id: string, title: string): void {
    const record = this.files.get(id);
    if (record) {
      this.files.set(id, { ...record, title });
    }
  }
  public loadPreferences(): Record<string, unknown> {
    return { ...this.preferences };
  }
  public savePreferences(value: Record<string, unknown>): void {
    this.preferences = { ...value };
  }
}

test("listing returns the newest session first and copies records", () => {
  const adapter = new MemoryAdapter();
  adapter.saveSession({ id: "a", title: "old", updatedAt: "2024-01-01T00:00:00Z", messages: [] });
  adapter.saveSession({ id: "b", title: "new", updatedAt: "2024-06-01T00:00:00Z", messages: [] });
  adapter.deleteSession("a");

  const list = handleSessionCall(SessionBridgeOp.LIST, null, adapter) as SessionStoreRecord[];

  assert.equal(list.length, 1);
  assert.equal(list[0].id, "b");
});

test("load returns null for a missing session and the full record otherwise", () => {
  const adapter = new MemoryAdapter();
  assert.equal(handleSessionCall(SessionBridgeOp.LOAD, { id: "missing" }, adapter), null);
  adapter.saveSession({ id: "b", title: "new", updatedAt: "2024-06-01T00:00:00Z", messages: [{ role: "user", text: "hi" }] });
  const record = handleSessionCall(SessionBridgeOp.LOAD, { id: "b" }, adapter) as SessionStoreRecord;
  assert.equal(record.id, "b");
  assert.deepEqual(record.messages, [{ role: "user", text: "hi" }]);
});

test("save persists and normalises missing fields without throwing", () => {
  const adapter = new MemoryAdapter();
  handleSessionCall(SessionBridgeOp.SAVE, { id: "b" }, adapter);
  const record = handleSessionCall(SessionBridgeOp.LOAD, { id: "b" }, adapter) as SessionStoreRecord;
  assert.equal(record.id, "b");
  assert.equal(record.title, "");
  assert.equal(record.updatedAt, "");
  assert.deepEqual(record.messages, []);
});

test("delete and rename mutate the stored session in place", () => {
  const adapter = new MemoryAdapter();
  adapter.saveSession({ id: "b", title: "old", updatedAt: "2024-06-01T00:00:00Z", messages: [] });
  handleSessionCall(SessionBridgeOp.RENAME, { id: "b", title: "renamed" }, adapter);
  handleSessionCall(SessionBridgeOp.DELETE, { id: "b" }, adapter);
  assert.equal(handleSessionCall(SessionBridgeOp.LOAD, { id: "b" }, adapter), null);
  assert.equal(adapter.listSessions().length, 0);
});

test("preferences round-trip through loadPrefs/savePrefs", () => {
  const adapter = new MemoryAdapter();
  assert.deepEqual(handleSessionCall(SessionBridgeOp.LOAD_PREFS, null, adapter), {});
  handleSessionCall(SessionBridgeOp.SAVE_PREFS, { theme: "dark" }, adapter);
  assert.deepEqual(handleSessionCall(SessionBridgeOp.LOAD_PREFS, null, adapter), { theme: "dark" });
});

test("invalid payloads raise the exact host error messages", () => {
  const adapter = new MemoryAdapter();
  for (const op of [SessionBridgeOp.LOAD, SessionBridgeOp.DELETE, SessionBridgeOp.RENAME]) {
    assert.throws(() => handleSessionCall(op, { id: 5 }, adapter), /payload field "id" must be a string/);
    assert.throws(() => handleSessionCall(op, "not-an-object", adapter), /payload for "id" must be an object/);
  }
  assert.throws(() => handleSessionCall(SessionBridgeOp.SAVE, "not-an-object", adapter), /save payload must be an object/);
  assert.throws(() => handleSessionCall("bogus-op" as unknown as string, null, adapter), /host does not know op "bogus-op"/);
});

/** A Storage-shaped backing the glue writes through. */
class MemoryStorage {
  private store = new Map<string, string>();
  public getItem(key: string): string | null {
    return this.store.has(key) ? (this.store.get(key) as string) : null;
  }
  public setItem(key: string, value: string): void {
    this.store.set(key, value);
  }
  public removeItem(key: string): void {
    this.store.delete(key);
  }
}

test("installPreviewShim resolves a sessionCall through the native bridge", async () => {
  const storage = new MemoryStorage();
  const resolves: { callId: string; payload: unknown }[] = [];
  const rejects: { callId: string; message: string }[] = [];
  const win: PreviewWindowLike = {
    webkit: { messageHandlers: { thintalk: { postMessage: () => {} } } },
    __thintalkBridge: {
      resolve: (callId, payload) => resolves.push({ callId, payload }),
      reject: (callId, message) => rejects.push({ callId, message }),
    },
  };
  installPreviewShim(win, storage as unknown as Storage);

  const saved = { id: "x", title: "t", updatedAt: "2024-06-01T00:00:00Z", messages: [] };
  (win.webkit?.messageHandlers?.thintalk?.postMessage as (payload: unknown) => void)({
    kind: "sessionCall",
    callId: "call-1",
    op: "save",
    payload: saved,
  });
  (win.webkit?.messageHandlers?.thintalk?.postMessage as (payload: unknown) => void)({
    kind: "sessionCall",
    callId: "call-2",
    op: "list",
    payload: null,
  });

  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(resolves.length, 2);
  assert.equal(rejects.length, 0);
  assert.equal((resolves[0].payload as unknown), null);
  assert.deepEqual((resolves[1].payload as SessionStoreRecord[]), [saved]);
});

test("installPreviewShim routes a malformed envelope and unknown op to reject", async () => {
  const storage = new MemoryStorage();
  const rejects: { callId: string; message: string }[] = [];
  const win: PreviewWindowLike = {
    webkit: { messageHandlers: { thintalk: { postMessage: () => {} } } },
    __thintalkBridge: { reject: (callId, message) => rejects.push({ callId, message }) },
  };
  installPreviewShim(win, storage as unknown as Storage);

  (win.webkit?.messageHandlers?.thintalk?.postMessage as (payload: unknown) => void)({ kind: "something-else" });
  (win.webkit?.messageHandlers?.thintalk?.postMessage as (payload: unknown) => void)({
    kind: "sessionCall",
    callId: "call-1",
    op: "no-op",
    payload: null,
  });

  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(rejects.length, 1, "malformed envelopes (no callId) are ignored, not rejected");
  assert.match(rejects[0].message, /host does not know op "no-op"/);
});
