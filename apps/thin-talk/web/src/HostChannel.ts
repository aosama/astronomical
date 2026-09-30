import { SessionBridgeRequest } from "./SessionBridgeRequest";

interface ThintalkMessageHandlers {
  thintalk?: { postMessage: (payload: unknown) => void };
}

/**
 * The one place the page talks to its Swift host: session bridge calls, user
 * actions that must leave the page (copy, open a link), and error reports.
 * Everything else — chat traffic — goes straight to the supervisor over REST.
 */
export class HostChannel {
  private readonly handlers: ThintalkMessageHandlers | null;

  public constructor() {
    this.handlers = window.webkit?.messageHandlers ?? null;
  }

  public postSessionCall(request: SessionBridgeRequest): void {
    this.deliver(request);
  }

  public postAction(action: string, messageID: string, detail: string): void {
    this.deliver({ kind: "action", action, messageId: messageID, detail });
  }

  public postError(detail: string): void {
    this.deliver({ kind: "error", detail });
  }

  private deliver(payload: unknown): void {
    this.handlers?.thintalk?.postMessage(payload);
  }
}
