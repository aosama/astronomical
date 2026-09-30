import { SessionBridgeOp } from "./SessionBridgeOp";
import { SessionBridgeRequest } from "./SessionBridgeRequest";
import { SessionBridgeError, SessionBridgeTimeoutError } from "./SessionBridgeError";
import { SessionDocument } from "./SessionDocument";
import { SessionSummary } from "./SessionSummary";
import { SessionPreferences } from "./SessionPreferences";

interface PendingSessionCall {
  resolve: (payload: unknown) => void;
  reject: (error: Error) => void;
  timer: number;
}

/** How long a session call may run before the page gives up on the host. */
const CALL_TIMEOUT_MILLISECONDS = 10_000;

/**
 * The typed client side of the session file bridge. Every durable read and write
 * goes through here as a promise: the page posts a `sessionCall` to the host and
 * the host answers by invoking `window.__thintalkBridge.resolve/reject` with the
 * call id. Installing that resolver is this class's contract with the host, so
 * it happens at construction.
 */
export class SessionBridge {
  private readonly post: (request: SessionBridgeRequest) => void;
  private readonly pendingCalls = new Map<string, PendingSessionCall>();
  private nextCallNumber = 0;

  public constructor(post: (request: SessionBridgeRequest) => void) {
    this.post = post;
    window.__thintalkBridge = {
      resolve: (callId: string, payload: unknown) => this.resolveCall(callId, payload),
      reject: (callId: string, message: string) => this.rejectCall(callId, message),
    };
  }

  public list(): Promise<SessionSummary[]> {
    return this.call(SessionBridgeOp.LIST, null).then((payload) =>
      SessionBridge.summariesFromPlain(payload),
    );
  }

  /** Loads one session, or null when the host has no file under that id. */
  public load(id: string): Promise<SessionDocument | null> {
    return this.call(SessionBridgeOp.LOAD, { id }).then((payload) =>
      payload === null ? null : SessionDocument.fromPlain(payload),
    );
  }

  public save(document: SessionDocument): Promise<void> {
    return this.call(SessionBridgeOp.SAVE, document.toPlain()).then(() => undefined);
  }

  public delete(id: string): Promise<void> {
    return this.call(SessionBridgeOp.DELETE, { id }).then(() => undefined);
  }

  public rename(id: string, title: string): Promise<void> {
    return this.call(SessionBridgeOp.RENAME, { id, title }).then(() => undefined);
  }

  public loadPrefs(): Promise<SessionPreferences> {
    return this.call(SessionBridgeOp.LOAD_PREFS, null).then((payload) =>
      SessionPreferences.fromPlain(payload),
    );
  }

  public savePrefs(preferences: SessionPreferences): Promise<void> {
    return this.call(SessionBridgeOp.SAVE_PREFS, preferences.toPlain()).then(() => undefined);
  }

  private call(op: SessionBridgeOp, payload: unknown): Promise<unknown> {
    return new Promise<unknown>((resolve, reject) => {
      this.nextCallNumber += 1;
      const callId = `call-${this.nextCallNumber}`;
      const timer = window.setTimeout(() => {
        this.pendingCalls.delete(callId);
        reject(new SessionBridgeTimeoutError(op, CALL_TIMEOUT_MILLISECONDS));
      }, CALL_TIMEOUT_MILLISECONDS);
      this.pendingCalls.set(callId, { resolve, reject, timer });
      this.post(new SessionBridgeRequest({ callId, op, payload }));
    });
  }

  private resolveCall(callId: string, payload: unknown): void {
    const pendingCall = this.pendingCalls.get(callId);
    if (!pendingCall) {
      return;
    }
    this.pendingCalls.delete(callId);
    window.clearTimeout(pendingCall.timer);
    pendingCall.resolve(payload);
  }

  private rejectCall(callId: string, message: string): void {
    const pendingCall = this.pendingCalls.get(callId);
    if (!pendingCall) {
      return;
    }
    this.pendingCalls.delete(callId);
    window.clearTimeout(pendingCall.timer);
    pendingCall.reject(new SessionBridgeError(message));
  }

  private static summariesFromPlain(payload: unknown): SessionSummary[] {
    if (!Array.isArray(payload)) {
      throw new SessionBridgeError("the session list reply is not an array");
    }
    return payload.map((rawSummary) => {
      if (typeof rawSummary !== "object" || rawSummary === null) {
        throw new SessionBridgeError("a session list entry is not an object");
      }
      const record = rawSummary as Record<string, unknown>;
      if (typeof record["id"] !== "string" || typeof record["updatedAt"] !== "string") {
        throw new SessionBridgeError("a session list entry is missing its id or timestamp");
      }
      const title = typeof record["title"] === "string" ? record["title"] : "";
      return new SessionSummary({ id: record["id"], title, updatedAt: record["updatedAt"] });
    });
  }
}
