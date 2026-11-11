import { ChatFailure } from "./ChatFailure";
import { ChatFailureClassifier } from "./ChatFailureClassifier";

/**
 * One piece of a tool call as it streams. The first piece for a call index
 * carries the call id and function name; later pieces for the same index
 * append to the arguments text. The daemon sends each call whole, but the
 * OpenAI wire format allows the split, so both are accepted.
 */
export interface ToolCallFragment {
  readonly toolCallIndex: number;
  readonly toolCallID: string | null;
  readonly toolName: string | null;
  readonly argumentsFragment: string;
}

/**
 * One decoded `data:` line of the daemon's streaming chat completion: the
 * visible text, reasoning and tool-call pieces it adds, and the finish reason
 * when the line closes the choice. Decoding is kept apart from the network so
 * the wire format is testable line by line.
 */
export class ChatCompletionChunk {
  public readonly textDelta: string;
  public readonly reasoningDelta: string;
  public readonly toolCallFragments: readonly ToolCallFragment[];
  /** The daemon's `finish_reason` ("stop", "length", "tool_calls"), or null
   * while the choice is still streaming. */
  public readonly finishReason: string | null;

  public constructor(init: {
    textDelta: string;
    reasoningDelta: string;
    finishReason: string | null;
    toolCallFragments?: readonly ToolCallFragment[];
  }) {
    this.textDelta = init.textDelta;
    this.reasoningDelta = init.reasoningDelta;
    this.finishReason = init.finishReason;
    const configuredFragments: readonly ToolCallFragment[] | undefined = init.toolCallFragments;
    this.toolCallFragments = configuredFragments != null ? configuredFragments : [];
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
    const errorRecord = errorField as Record<string, unknown>;
    const errorMessage = errorRecord["message"];
    const errorCode = errorRecord["code"];
    const reason = typeof errorMessage === "string" ? errorMessage : null;
    const code = typeof errorCode === "string" ? errorCode : null;
    throw new ChatFailure(ChatFailureClassifier.classifyErrorChunk(code, reason), reason);
  }

  /** Thin Talk always asks for one choice; every choice in a chunk is folded in
   * order so a server that splits one chunk across choices loses nothing. */
  private static fromChoices(rawChoices: unknown): ChatCompletionChunk {
    let textDelta = "";
    let reasoningDelta = "";
    let finishReason: string | null = null;
    const toolCallFragments: ToolCallFragment[] = [];
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
      toolCallFragments.push(...ChatCompletionChunk.toolCallFragmentsFrom(deltaRecord["tool_calls"]));
    }
    return new ChatCompletionChunk({ textDelta, reasoningDelta, finishReason, toolCallFragments });
  }

  /** Reads the `tool_calls` array of one delta. A piece without a numeric
   * index cannot be matched to its call, so it is skipped. */
  private static toolCallFragmentsFrom(rawToolCalls: unknown): ToolCallFragment[] {
    const toolCallFragments: ToolCallFragment[] = [];
    const toolCalls: unknown[] = Array.isArray(rawToolCalls) ? rawToolCalls : [];
    for (const rawToolCall of toolCalls) {
      if (typeof rawToolCall !== "object" || rawToolCall === null) {
        continue;
      }
      const toolCallRecord = rawToolCall as Record<string, unknown>;
      const toolCallIndex = toolCallRecord["index"];
      if (typeof toolCallIndex !== "number" || !Number.isInteger(toolCallIndex) || toolCallIndex < 0) {
        continue;
      }
      const functionField = toolCallRecord["function"];
      const functionRecord =
        typeof functionField === "object" && functionField !== null ? (functionField as Record<string, unknown>) : {};
      const toolCallID = toolCallRecord["id"];
      const toolName = functionRecord["name"];
      const argumentsFragment = functionRecord["arguments"];
      toolCallFragments.push({
        toolCallIndex,
        toolCallID: typeof toolCallID === "string" && toolCallID.length > 0 ? toolCallID : null,
        toolName: typeof toolName === "string" && toolName.length > 0 ? toolName : null,
        argumentsFragment: typeof argumentsFragment === "string" ? argumentsFragment : "",
      });
    }
    return toolCallFragments;
  }
}
