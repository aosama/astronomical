import { ChatFailure } from "./ChatFailure";
import { ChatFailureClassifier } from "./ChatFailureClassifier";

/**
 * One decoded `data:` line of the daemon's streaming chat completion: the
 * visible text and reasoning it adds, and the finish reason when the line
 * closes the choice. Decoding is kept apart from the network so the wire
 * format is testable line by line.
 */
export class ChatCompletionChunk {
  public readonly textDelta: string;
  public readonly reasoningDelta: string;
  /** The daemon's `finish_reason` ("stop", "length", "tool_calls"), or null
   * while the choice is still streaming. */
  public readonly finishReason: string | null;

  public constructor(init: { textDelta: string; reasoningDelta: string; finishReason: string | null }) {
    this.textDelta = init.textDelta;
    this.reasoningDelta = init.reasoningDelta;
    this.finishReason = init.finishReason;
  }

  /** Decodes one server-sent-events line. Returns null for lines that carry
   * nothing (comments, blank keep-alives, `[DONE]`, unparseable payloads) and
   * throws a classified ChatFailure when the daemon streams an error object. */
  public static fromServerSentEventLine(line: string): ChatCompletionChunk | null {
    const trimmedLine = line.trim();
    if (!trimmedLine.startsWith("data:")) {
      return null;
    }
    const dataFieldText = trimmedLine.slice("data:".length).trim();
    if (dataFieldText.length === 0 || dataFieldText === "[DONE]") {
      return null;
    }
    let decodedPayload: unknown;
    try {
      decodedPayload = JSON.parse(dataFieldText);
    } catch {
      return null;
    }
    if (typeof decodedPayload !== "object" || decodedPayload === null) {
      return null;
    }
    const chunkRecord = decodedPayload as Record<string, unknown>;
    ChatCompletionChunk.throwIfErrorChunk(chunkRecord);
    return ChatCompletionChunk.fromChoices(chunkRecord["choices"]);
  }

  private static throwIfErrorChunk(chunkRecord: Record<string, unknown>): void {
    const errorField = chunkRecord["error"];
    if (typeof errorField !== "object" || errorField === null) {
      return;
    }
    const errorMessage = (errorField as Record<string, unknown>)["message"];
    const reason = typeof errorMessage === "string" ? errorMessage : null;
    const classifiedReason: string = reason != null ? reason : "";
    throw new ChatFailure(ChatFailureClassifier.classifyReason(classifiedReason), reason);
  }

  /** Thin Talk always asks for one choice; every choice in a chunk is folded in
   * order so a server that splits one chunk across choices loses nothing. */
  private static fromChoices(rawChoices: unknown): ChatCompletionChunk {
    let textDelta = "";
    let reasoningDelta = "";
    let finishReason: string | null = null;
    const choices: unknown[] = Array.isArray(rawChoices) ? rawChoices : [];
    for (const rawChoice of choices) {
      if (typeof rawChoice !== "object" || rawChoice === null) {
        continue;
      }
      const choiceRecord = rawChoice as Record<string, unknown>;
      const choiceFinishReason = choiceRecord["finish_reason"];
      if (typeof choiceFinishReason === "string" && choiceFinishReason.length > 0) {
        finishReason = choiceFinishReason;
      }
      const delta = choiceRecord["delta"];
      if (typeof delta !== "object" || delta === null) {
        continue;
      }
      const deltaRecord = delta as Record<string, unknown>;
      const content = deltaRecord["content"];
      if (typeof content === "string") {
        textDelta += content;
      }
      const reasoning = deltaRecord["reasoning_content"];
      if (typeof reasoning === "string") {
        reasoningDelta += reasoning;
      }
    }
    return new ChatCompletionChunk({ textDelta, reasoningDelta, finishReason });
  }
}
