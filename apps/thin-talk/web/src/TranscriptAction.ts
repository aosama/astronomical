/** What a transcript interaction asks the host to do. */
export type TranscriptActionKind = "copy" | "openExternal" | "saveImage" | "copyImage" | "exportAnswer" | "speakAnswer";

/**
 * One interaction the transcript reported: copying text, opening a link
 * outside the app, saving or copying a generated picture, or exporting an
 * answer. All of them leave the transcript — the host owns the pasteboard, the
 * system link opener and the files, and the export sheet owns exporting. For
 * a picture, `detail` is the stored image's name, never a path.
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
