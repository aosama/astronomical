import { ChatFailureKind } from "./ChatFailureKind";

/**
 * One failure the conversation can hit, plus the human-readable reason the
 * supervisor reported. The reason is shown verbatim where it is safe; the next
 * action is derived from the kind.
 */
export class ChatFailure extends Error {
  public readonly kind: ChatFailureKind;
  public readonly reason: string | null;

  public constructor(kind: ChatFailureKind, reason: string | null = null) {
    super(reason ?? kind);
    this.name = "ChatFailure";
    this.kind = kind;
    this.reason = reason;
  }

  /** The one specific, plain-language action the user can take after this
   * failure. This is the failure→action contract kept in one place so every
   * surface stays consistent. */
  public get nextAction(): string {
    switch (this.kind) {
      case ChatFailureKind.NO_USABLE_MODEL:
        return "Get a model from the Library, then try again.";
      case ChatFailureKind.MODEL_LOAD_FAILED:
        return "Pick a smaller model, or raise the memory ceiling, then try again.";
      case ChatFailureKind.WORKER_UNAVAILABLE:
        return "The runner is not running. Start it and try again.";
      case ChatFailureKind.ENGINE_BUSY:
        return "The runner is busy. Wait a moment and try again.";
      case ChatFailureKind.INVALID_REQUEST:
        return "Shorten the message and try again.";
      case ChatFailureKind.MALFORMED_MODEL_OUTPUT:
        return "Something went wrong producing that reply. Try again.";
      case ChatFailureKind.STALL:
        return "The reply stalled. Try again.";
      case ChatFailureKind.UNKNOWN:
        return "Try again.";
    }
  }
}
