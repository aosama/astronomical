/**
 * One intent the composer reported: sending the draft, stopping the stream,
 * retrying after a failure, selecting the chat or image model, or flipping
 * the Tools switch. The detail carries the model id, or "on"/"off" for the
 * switch.
 */
export class ComposerAction {
  public readonly kind: "send" | "stop" | "retry" | "selectModel" | "selectImageModel" | "openImageModels" | "setTools";
  public readonly detail: string;

  public constructor(init: {
    kind: "send" | "stop" | "retry" | "selectModel" | "selectImageModel" | "openImageModels" | "setTools";
    detail?: string;
  }) {
    const configuredDetail: string | undefined = init.detail;
    this.kind = init.kind;
    this.detail = configuredDetail != null ? configuredDetail : "";
  }
}
