import type { SessionBridgeRequest } from "../../src/SessionBridgeRequest";

/**
 * An in-memory stand-in for the Swift session host. It answers `sessionCall`
 * envelopes exactly the way SessionFileStore does — same plain shapes, same
 * newest-first ordering, same null-for-missing semantics — so a test boot
 * exercises the real bridge, store, and controller against a faithful double.
 */
export class FakeSessionHost {
  private readonly files = new Map<string, { title: string; updatedAt: string; messages: unknown[] }>();
  private preferences: unknown = null;
  private nextIdentifier = 1;
  public readonly receivedRequests: SessionBridgeRequest[] = [];

  public constructor(
    private readonly resolveCall: (callId: string, payload: unknown) => void,
    private readonly rejectCall: (callId: string, message: string) => void,
  ) {}

  /** Installs the fake WKWebView message handler onto the live window. */
  public install(): void {
    window.webkit = {
      messageHandlers: {
        thintalk: {
          postMessage: (payload: unknown) => this.receive(payload),
        },
      },
    };
  }

  private receive(payload: unknown): void {
    if (typeof payload !== "object" || payload === null) {
      return;
    }
    const record = payload as Record<string, unknown>;
    if (record["kind"] !== "sessionCall") {
      return;
    }
    const request = record as unknown as SessionBridgeRequest;
    this.receivedRequests.push(request);
    try {
      const reply = this.perform(request.op, request.payload);
      queueMicrotask(() => this.resolveCall(request.callId, reply));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      queueMicrotask(() => this.rejectCall(request.callId, message));
    }
  }

  private perform(op: string, payload: unknown): unknown {
    switch (op) {
      case "list":
        return this.list();
      case "load":
        return this.load(this.stringField(payload, "id"));
      case "save":
        this.save(payload);
        return null;
      case "delete":
        this.files.delete(this.stringField(payload, "id"));
        return null;
      case "rename":
        this.rename(this.stringField(payload, "id"), this.stringField(payload, "title"));
        return null;
      case "loadPrefs":
        return this.preferences;
      case "savePrefs":
        this.preferences = payload;
        return null;
      default:
        throw new Error(`fake host does not know op "${op}"`);
    }
  }

  private list(): unknown[] {
    const summaries = [...this.files.entries()].map(([id, file]) => ({
      id,
      title: file.title,
      updatedAt: file.updatedAt,
    }));
    summaries.sort((first, second) => second.updatedAt.localeCompare(first.updatedAt));
    return summaries;
  }

  private load(id: string): unknown {
    const file = this.files.get(id);
    if (!file) {
      return null;
    }
    return { id, title: file.title, updatedAt: file.updatedAt, messages: file.messages };
  }

  private save(payload: unknown): void {
    if (typeof payload !== "object" || payload === null) {
      throw new Error("save payload must be an object");
    }
    const record = payload as Record<string, unknown>;
    const id = this.stringField(payload, "id");
    this.files.set(id, {
      title: typeof record["title"] === "string" ? record["title"] : "",
      updatedAt: typeof record["updatedAt"] === "string" ? record["updatedAt"] : "",
      messages: Array.isArray(record["messages"]) ? record["messages"] : [],
    });
  }

  private rename(id: string, title: string): void {
    const file = this.files.get(id);
    if (file) {
      file.title = title;
    }
  }

  private stringField(payload: unknown, key: string): string {
    if (typeof payload !== "object" || payload === null) {
      throw new Error(`payload for "${key}" must be an object`);
    }
    const value = (payload as Record<string, unknown>)[key];
    if (typeof value !== "string") {
      throw new Error(`payload field "${key}" must be a string`);
    }
    return value;
  }

  /** Seeds a session file directly, as if a previous run had written it. */
  public seedSession(id: string, title: string, updatedAt: string, messages: unknown[]): void {
    this.files.set(id, { title, updatedAt, messages });
  }

  public seedPreferences(preferences: unknown): void {
    this.preferences = preferences;
  }

  public fileCount(): number {
    return this.files.size;
  }

  public nextSessionIdentifier(): string {
    return `seeded-${this.nextIdentifier++}`;
  }
}
