import type { FailureGuidance } from "./ChatFailure";
import { ChatFailureKind } from "./ChatFailureKind";

/**
 * Maps a daemon HTTP status and raw body — or a streaming error chunk — to a
 * classified failure kind. Recognises the public chat codes first, then falls
 * back to status-code heuristics so no failure is left unclassified.
 */
export class ChatFailureClassifier {
  /** Classifies a human-readable reason (from an HTTP body or an error chunk). */
  public static classifyReason(rawReason: string): ChatFailureKind {
    const reason = rawReason.toLowerCase();
    if (reason.includes("chat_out_of_memory")) return ChatFailureKind.OUT_OF_MEMORY;
    if (reason.includes("model_not_found")) return ChatFailureKind.NO_USABLE_MODEL;
    if (reason.includes("model_load_failed")) return ChatFailureKind.MODEL_LOAD_FAILED;
    if (reason.includes("chat_malformed_model_output")) return ChatFailureKind.MALFORMED_MODEL_OUTPUT;
    if (reason.includes("chat_worker_unavailable")) return ChatFailureKind.WORKER_UNAVAILABLE;
    if (reason.includes("chat_engine_busy") || reason.includes("server_capacity") || reason.includes("busy")) {
      return ChatFailureKind.ENGINE_BUSY;
    }
    if (reason.includes("context_length") || reason.includes("context") || reason.includes("invalid_request")) {
      return ChatFailureKind.INVALID_REQUEST;
    }
    return ChatFailureKind.UNKNOWN;
  }

  /** Classifies a streaming error chunk. The daemon sends its public code in
   * a field of its own, beside the sentence meant for people, so the code is
   * read first: classifying the sentence alone left every streamed failure
   * unrecognised, and the person was shown the engine's raw wording. */
  public static classifyErrorChunk(errorCode: string | null, errorMessage: string | null): ChatFailureKind {
    const codeKind = errorCode !== null ? ChatFailureClassifier.classifyReason(errorCode) : ChatFailureKind.UNKNOWN;
    if (codeKind !== ChatFailureKind.UNKNOWN) {
      return codeKind;
    }
    return ChatFailureClassifier.classifyReason(errorMessage !== null ? errorMessage : "");
  }

  /** Classifies an error answer from its body: the public code in
   * `{error:{code}}` first, then the sentence, then the HTTP status. Reading
   * the sentence alone sent a model that would not load to the status
   * fallback, which named it "the engine stopped answering". */
  public static classifyErrorDocument(status: number, bodyText: string): ChatFailureKind {
    const errorCode = ChatFailureClassifier.codeFromErrorDocument(bodyText);
    const codeKind = errorCode !== null ? ChatFailureClassifier.classifyReason(errorCode) : ChatFailureKind.UNKNOWN;
    if (codeKind !== ChatFailureKind.UNKNOWN) {
      return codeKind;
    }
    const reason = ChatFailureClassifier.reasonFromErrorDocument(bodyText);
    return ChatFailureClassifier.classify(status, reason !== null ? reason : "");
  }

  /** The engine's own sentence and next step for this failure, from
   * `{error:{plain_message, next_step}}`, or null when it gave none. */
  public static guidanceFromErrorDocument(bodyText: string): FailureGuidance | null {
    try {
      const document: unknown = JSON.parse(bodyText);
      const errorField = typeof document === "object" && document !== null ? (document as Record<string, unknown>)["error"] : null;
      if (typeof errorField !== "object" || errorField === null) {
        return null;
      }
      const plainMessage = (errorField as Record<string, unknown>)["plain_message"];
      const nextStep = (errorField as Record<string, unknown>)["next_step"];
      if (typeof plainMessage !== "string" || plainMessage.length === 0 || typeof nextStep !== "string" || nextStep.length === 0) {
        return null;
      }
      return { plainMessage, nextStep };
    } catch {
      return null;
    }
  }

  /** The public code of an `{error:{code}}` answer, or null. */
  public static codeFromErrorDocument(bodyText: string): string | null {
    try {
      const document: unknown = JSON.parse(bodyText);
      const errorField = typeof document === "object" && document !== null ? (document as Record<string, unknown>)["error"] : null;
      const errorCode = typeof errorField === "object" && errorField !== null ? (errorField as Record<string, unknown>)["code"] : null;
      return typeof errorCode === "string" && errorCode.length > 0 ? errorCode : null;
    } catch {
      return null;
    }
  }

  public static classify(status: number, reason: string): ChatFailureKind {
    const reasonKind = ChatFailureClassifier.classifyReason(reason);
    if (reasonKind !== ChatFailureKind.UNKNOWN) {
      return reasonKind;
    }
    switch (status) {
      case 404:
        return ChatFailureKind.NO_USABLE_MODEL;
      case 429:
        return ChatFailureKind.ENGINE_BUSY;
      case 503:
        return ChatFailureKind.WORKER_UNAVAILABLE;
      case 400:
        return ChatFailureKind.INVALID_REQUEST;
      case 500:
        return ChatFailureKind.MALFORMED_MODEL_OUTPUT;
      default:
        return ChatFailureKind.UNKNOWN;
    }
  }

  /** Extracts the failure reason from a lenient daemon error document. Both
   * `{status,message}` and `{error:{message}}` shapes are accepted. */
  public static reasonFromErrorDocument(bodyText: string): string | null {
    let document: unknown;
    try {
      document = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (typeof document !== "object" || document === null) {
      return null;
    }
    const record = document as Record<string, unknown>;
    if (typeof record["status"] === "string" && record["status"].length > 0) {
      return record["status"];
    }
    if (typeof record["message"] === "string" && record["message"].length > 0) {
      return record["message"];
    }
    const errorField = record["error"];
    if (typeof errorField === "object" && errorField !== null) {
      const errorMessage = (errorField as Record<string, unknown>)["message"];
      if (typeof errorMessage === "string" && errorMessage.length > 0) {
        return errorMessage;
      }
    }
    return null;
  }
}
