/**
 * One intent the composer reported: sending the draft, stopping the stream,
 * retrying after a failure, or choosing a thinking level.
 */
export class ComposerAction {
  public readonly kind: "send" | "stop" | "retry" | "setEffort";
  public readonly detail: string;

  public constructor(init: { kind: "send" | "stop" | "retry" | "setEffort"; detail?: string }) {
    this.kind = init.kind;
    this.detail = init.detail ?? "";
  }
}
