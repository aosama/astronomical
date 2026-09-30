import { ChatFailureKind } from "./ChatFailureKind";

/**
 * Maps a supervisor HTTP status and raw body — or a streaming error chunk — to a
 * classified failure kind. Recognises the public chat codes first, then falls
 * back to status-code heuristics so no failure is left unclassified.
 */
export class ChatFailureClassifier {
  /** Classifies a human-readable reason (from an HTTP body or an error chunk). */
  public static classifyReason(rawReason: string): ChatFailureKind {
    const reason = rawReason.toLowerCase();
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

  /** Extracts the failure reason from a lenient supervisor error document. Both
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
