/**
 * One interaction the transcript reported: copying a message's text or opening a
 * link outside the app. Both leave the page — the host owns the pasteboard and
 * the system link opener.
 */
export class TranscriptAction {
  public readonly kind: "copy" | "openExternal";
  public readonly messageID: string;
  public readonly detail: string;

  public constructor(init: { kind: "copy" | "openExternal"; messageID: string; detail: string }) {
    this.kind = init.kind;
    this.messageID = init.messageID;
    this.detail = init.detail;
  }
}
