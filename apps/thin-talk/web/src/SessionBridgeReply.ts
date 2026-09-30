/**
 * One host→page session reply, delivered by the host calling
 * `window.__thintalkBridge.resolve(callId, payload)` or
 * `window.__thintalkBridge.reject(callId, message)`.
 */
export class SessionBridgeReply {
  public readonly callId: string;
  /** null means the op succeeded with no result (save/delete/rename). */
  public readonly payload: unknown;

  public constructor(init: { callId: string; payload: unknown }) {
    this.callId = init.callId;
    this.payload = init.payload;
  }
}
