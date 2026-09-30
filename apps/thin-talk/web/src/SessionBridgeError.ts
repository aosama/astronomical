/**
 * A session call the host refused or could not complete. The host's message
 * travels as the error message so the surface can show the real cause.
 */
export class SessionBridgeError extends Error {
  public constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "SessionBridgeError";
  }
}

/** A session call that outlived its timeout without a host reply. */
export class SessionBridgeTimeoutError extends SessionBridgeError {
  public constructor(op: string, timeoutMilliseconds: number) {
    super(`session call "${op}" timed out after ${timeoutMilliseconds}ms`);
    this.name = "SessionBridgeTimeoutError";
  }
}
