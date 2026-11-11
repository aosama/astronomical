import { ChatFailureKind } from "./ChatFailureKind";

/**
 * One failure the conversation can hit, plus the human-readable reason the
 * daemon reported. The reason is shown verbatim where it is safe; the next
 * action is derived from the kind.
 */
export class ChatFailure extends Error {
  public readonly kind: ChatFailureKind;
  public readonly reason: string | null;

  public constructor(kind: ChatFailureKind, reason: string | null = null) {
    const failureMessage: string = reason != null ? reason : kind;
    super(failureMessage);
    this.name = "ChatFailure";
    this.kind = kind;
    this.reason = reason;
  }

  /** What happened, in words for the person and not for the log. The
   * engine's own sentence is used only where it is the specific thing they
   * need (which model is missing, why a message was refused); elsewhere it
   * names workers and log files, so it is kept as `reason` for the banner's
   * tooltip and a plain sentence is shown. */
  public get headline(): string {
    switch (this.kind) {
      case ChatFailureKind.NO_USABLE_MODEL:
        return this.reason ?? "No chat model is ready.";
      case ChatFailureKind.MODEL_LOAD_FAILED:
        return this.reason ?? "That model would not load.";
      case ChatFailureKind.INVALID_REQUEST:
        return this.reason ?? "That message could not be sent.";
      case ChatFailureKind.OUT_OF_MEMORY:
        return "This Mac ran out of memory for that one.";
      case ChatFailureKind.WORKER_UNAVAILABLE:
        return "The engine stopped answering.";
      case ChatFailureKind.ENGINE_BUSY:
        return "The engine is busy with something else.";
      case ChatFailureKind.MALFORMED_MODEL_OUTPUT:
        return "That reply came out garbled.";
      case ChatFailureKind.STALL:
        return "The reply stopped arriving.";
      case ChatFailureKind.UNKNOWN:
        return "Something went wrong.";
    }
  }

  /** The one specific, plain-language action the user can take after this
   * failure. This is the failure→action contract kept in one place so every
   * surface stays consistent. */
  public get nextAction(): string {
    switch (this.kind) {
      case ChatFailureKind.NO_USABLE_MODEL:
        return "Get a model from the Models page in the Observatory, then try again.";
      case ChatFailureKind.MODEL_LOAD_FAILED:
        return "Pick a smaller model, or raise the memory ceiling, then try again.";
      case ChatFailureKind.OUT_OF_MEMORY:
        return "Close other apps that use a lot of memory, or pick a smaller model, then try again.";
      case ChatFailureKind.WORKER_UNAVAILABLE:
        // True for both ways the engine goes quiet: a crashed worker comes
        // back on the next message, and an engine that was stopped needs the
        // menu bar.
        return "Try again in a few seconds. If it still does not answer, choose Restart server from the Astronomical icon in the menu bar.";
      case ChatFailureKind.ENGINE_BUSY:
        return "Wait a moment and try again.";
      case ChatFailureKind.INVALID_REQUEST:
        return "Shorten the message and try again.";
      case ChatFailureKind.MALFORMED_MODEL_OUTPUT:
      case ChatFailureKind.STALL:
      case ChatFailureKind.UNKNOWN:
        return "Try again.";
    }
  }
}
