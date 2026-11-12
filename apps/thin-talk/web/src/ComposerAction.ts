/**
 * One intent the composer reported: sending the draft, stopping the stream,
 * retrying after a failure, selecting the chat or image model, flipping the
 * Tools switch, or asking for the Models page. The detail carries the model id, or "on"/"off" for the
 * switch.
 */
export class ComposerAction {
  public readonly kind: "send" | "stop" | "retry" | "selectModel" | "selectImageModel" | "openModelPanel" | "setTools" | "openModelsPage";
  public readonly detail: string;

  public constructor(init: {
    kind: "send" | "stop" | "retry" | "selectModel" | "selectImageModel" | "openModelPanel" | "setTools" | "openModelsPage";
    detail?: string;
  }) {
    const configuredDetail: string | undefined = init.detail;
    this.kind = init.kind;
    this.detail = configuredDetail != null ? configuredDetail : "";
  }
}
