/** What a transcript interaction asks the host to do. */
export type TranscriptActionKind = "copy" | "openExternal" | "saveImage" | "copyImage";

/**
 * One interaction the transcript reported: copying text, opening a link
 * outside the app, or saving or copying a generated picture. All of them leave
 * the page — the host owns the pasteboard, the system link opener and the
 * files. For a picture, `detail` is the stored image's name, never a path.
 */
export class TranscriptAction {
  public readonly kind: TranscriptActionKind;
  public readonly messageID: string;
  public readonly detail: string;

  public constructor(init: { kind: TranscriptActionKind; messageID: string; detail: string }) {
    this.kind = init.kind;
    this.messageID = init.messageID;
    this.detail = init.detail;
  }
}
