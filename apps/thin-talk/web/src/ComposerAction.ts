/**
 * One intent the composer reported: sending the draft, stopping the stream,
 * retrying after a failure, choosing a thinking level, or selecting the chat
 * model. The detail carries the effort value or the model id.
 */
export class ComposerAction {
  public readonly kind: "send" | "stop" | "retry" | "setEffort" | "selectModel";
  public readonly detail: string;

  public constructor(init: {
    kind: "send" | "stop" | "retry" | "setEffort" | "selectModel";
    detail?: string;
  }) {
    const configuredDetail: string | undefined = init.detail;
    this.kind = init.kind;
    this.detail = configuredDetail != null ? configuredDetail : "";
  }
}
