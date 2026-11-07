import type { Api, AssistantMessage, Model, StopReason } from "@earendil-works/pi-ai";
import type { AssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import { ChatCompletionChunk, type ToolCallFragment } from "./ChatCompletionChunk";
import { ChatFailure } from "./ChatFailure";
import { ChatFailureKind } from "./ChatFailureKind";
import { emptyAssistantMessage } from "./DaemonModelDescription";

type DoneStopReason = Extract<StopReason, "stop" | "length" | "toolUse">;

/** A tool call being assembled: where its block sits and its arguments text
 * so far. Arguments are parsed once, when the response ends, because a
 * fragment is rarely valid JSON on its own. */
interface ToolCallInProgress {
  readonly blockIndex: number;
  argumentsText: string;
}

/**
 * Folds the daemon's streamed chunks into the event sequence Pi's agent loop
 * expects for one assistant response: `start`, then a start/delta/end run for
 * each reasoning, text or tool-call block, then exactly one `done` or `error`.
 *
 * The daemon interleaves reasoning and text freely, so a block closes as soon
 * as the other kind arrives and a fresh block opens; Pi keeps them as
 * separate, ordered content blocks.
 */
export class AssistantEventAssembler {
  private readonly eventStream: AssistantMessageEventStream;
  private readonly assistantMessage: AssistantMessage;
  private openBlockIndex: number | null = null;
  private finishReason: string | null = null;
  private readonly toolCallsByFragmentIndex = new Map<number, ToolCallInProgress>();
  private readonly responseStartedAtMilliseconds = Date.now();

  public constructor(eventStream: AssistantMessageEventStream, model: Model<Api>) {
    this.eventStream = eventStream;
    this.assistantMessage = emptyAssistantMessage(model);
  }

  public start(): void {
    this.eventStream.push({ type: "start", partial: this.snapshot() });
  }

  public apply(chunk: ChatCompletionChunk): void {
    if (chunk.reasoningDelta.length > 0) {
      this.appendReasoning(chunk.reasoningDelta);
    }
    if (chunk.textDelta.length > 0) {
      this.appendText(chunk.textDelta);
    }
    for (const toolCallFragment of chunk.toolCallFragments) {
      this.appendToolCallFragment(toolCallFragment);
    }
    if (chunk.finishReason != null) {
      this.finishReason = chunk.finishReason;
    }
  }

  /** Ends a response the daemon closed normally. A stream that stops without
   * a finish reason was cut off (the daemon always closes the choice with
   * one), so it is reported as malformed output instead of a short answer. */
  public finish(): void {
    if (this.finishReason == null) {
      throw new ChatFailure(ChatFailureKind.MALFORMED_MODEL_OUTPUT, "the stream ended without a finish reason");
    }
    this.closeOpenBlock();
    this.finishToolCalls();
    // A server that ends a tool-calling response with "stop" is reported as
    // "toolUse", matching what Pi's own adapters report. "length" is kept: Pi
    // then refuses to run calls whose arguments may have been cut off and
    // asks the model to send them again.
    const mappedStopReason = AssistantEventAssembler.doneStopReason(this.finishReason);
    const doneStopReason = this.toolCallsByFragmentIndex.size > 0 && mappedStopReason === "stop" ? "toolUse" : mappedStopReason;
    this.assistantMessage.stopReason = doneStopReason;
    this.eventStream.push({ type: "done", reason: doneStopReason, message: this.assistantMessage });
    this.eventStream.end(this.assistantMessage);
  }

  /** Ends the response with a stop (`aborted`) or a failure (`error`),
   * keeping whatever text and reasoning already arrived. */
  public fail(stopReason: "aborted" | "error", errorMessage: string): void {
    this.closeOpenBlock();
    this.finishToolCalls();
    this.assistantMessage.stopReason = stopReason;
    this.assistantMessage.errorMessage = errorMessage;
    this.eventStream.push({ type: "error", reason: stopReason, error: this.assistantMessage });
    this.eventStream.end(this.assistantMessage);
  }

  private appendReasoning(reasoningDelta: string): void {
    let blockIndex = this.openBlockIndex;
    if (blockIndex == null || this.assistantMessage.content[blockIndex]?.type !== "thinking") {
      this.closeOpenBlock();
      blockIndex = this.assistantMessage.content.push({ type: "thinking", thinking: "" }) - 1;
      this.openBlockIndex = blockIndex;
      this.eventStream.push({ type: "thinking_start", contentIndex: blockIndex, partial: this.snapshot() });
    }
    const thinkingBlock = this.assistantMessage.content[blockIndex];
    if (thinkingBlock?.type === "thinking") {
      thinkingBlock.thinking += reasoningDelta;
    }
    this.eventStream.push({ type: "thinking_delta", contentIndex: blockIndex, delta: reasoningDelta, partial: this.snapshot() });
  }

  private appendText(textDelta: string): void {
    let blockIndex = this.openBlockIndex;
    if (blockIndex == null || this.assistantMessage.content[blockIndex]?.type !== "text") {
      this.closeOpenBlock();
      blockIndex = this.assistantMessage.content.push({ type: "text", text: "" }) - 1;
      this.openBlockIndex = blockIndex;
      this.eventStream.push({ type: "text_start", contentIndex: blockIndex, partial: this.snapshot() });
    }
    const textBlock = this.assistantMessage.content[blockIndex];
    if (textBlock?.type === "text") {
      textBlock.text += textDelta;
    }
    this.eventStream.push({ type: "text_delta", contentIndex: blockIndex, delta: textDelta, partial: this.snapshot() });
  }

  private appendToolCallFragment(toolCallFragment: ToolCallFragment): void {
    let toolCallInProgress = this.toolCallsByFragmentIndex.get(toolCallFragment.toolCallIndex);
    if (toolCallInProgress == null) {
      this.closeOpenBlock();
      // A server that sends no id gets one unique to this response, so calls
      // from different rounds of a turn never share an id.
      const toolCallID =
        toolCallFragment.toolCallID != null ? toolCallFragment.toolCallID : `call_${this.responseStartedAtMilliseconds.toString(36)}_${toolCallFragment.toolCallIndex}`;
      const toolName = toolCallFragment.toolName != null ? toolCallFragment.toolName : "";
      const blockIndex = this.assistantMessage.content.push({ type: "toolCall", id: toolCallID, name: toolName, arguments: {} }) - 1;
      toolCallInProgress = { blockIndex, argumentsText: "" };
      this.toolCallsByFragmentIndex.set(toolCallFragment.toolCallIndex, toolCallInProgress);
      this.eventStream.push({ type: "toolcall_start", contentIndex: blockIndex, partial: this.snapshot() });
    }
    const toolCallBlock = this.assistantMessage.content[toolCallInProgress.blockIndex];
    if (toolCallBlock?.type === "toolCall" && toolCallBlock.name.length === 0 && toolCallFragment.toolName != null) {
      toolCallBlock.name = toolCallFragment.toolName;
    }
    if (toolCallFragment.argumentsFragment.length > 0) {
      toolCallInProgress.argumentsText += toolCallFragment.argumentsFragment;
      this.eventStream.push({
        type: "toolcall_delta",
        contentIndex: toolCallInProgress.blockIndex,
        delta: toolCallFragment.argumentsFragment,
        partial: this.snapshot(),
      });
    }
  }

  /** Parses each tool call's complete arguments and closes its block. Text
   * that is not a JSON object leaves the arguments empty and records why, so
   * the loop reports a failed call to the model instead of running the tool
   * with arguments it did not mean. */
  private finishToolCalls(): void {
    for (const toolCallInProgress of this.toolCallsByFragmentIndex.values()) {
      const toolCallBlock = this.assistantMessage.content[toolCallInProgress.blockIndex];
      if (toolCallBlock?.type !== "toolCall") {
        continue;
      }
      const argumentsText = toolCallInProgress.argumentsText.trim();
      try {
        const decodedArguments: unknown = argumentsText.length === 0 ? {} : JSON.parse(argumentsText);
        if (typeof decodedArguments !== "object" || decodedArguments === null || Array.isArray(decodedArguments)) {
          throw new Error("not an object");
        }
        toolCallBlock.arguments = decodedArguments as typeof toolCallBlock.arguments;
      } catch {
        toolCallBlock.arguments = {};
        toolCallBlock.argumentsError = `the arguments were not a JSON object: ${argumentsText}`;
      }
      this.eventStream.push({ type: "toolcall_end", contentIndex: toolCallInProgress.blockIndex, toolCall: toolCallBlock, partial: this.snapshot() });
    }
  }

  private closeOpenBlock(): void {
    const blockIndex = this.openBlockIndex;
    if (blockIndex == null) {
      return;
    }
    this.openBlockIndex = null;
    const openBlock = this.assistantMessage.content[blockIndex];
    if (openBlock?.type === "thinking") {
      this.eventStream.push({ type: "thinking_end", contentIndex: blockIndex, content: openBlock.thinking, partial: this.snapshot() });
    } else if (openBlock?.type === "text") {
      this.eventStream.push({ type: "text_end", contentIndex: blockIndex, content: openBlock.text, partial: this.snapshot() });
    }
  }

  /** Pi keeps the latest partial as its streaming message. Only the block
   * list is copied: an event then lists exactly the blocks that existed when
   * it fired, while the blocks themselves are shared and keep growing. Pi and
   * the turn runner read only the newest partial and each event's own delta,
   * so the shared blocks are never read stale. */
  private snapshot(): AssistantMessage {
    return { ...this.assistantMessage, content: [...this.assistantMessage.content] };
  }

  /** The daemon's finish reasons in Pi's terms. Anything unrecognised is
   * treated as a normal stop so an added server reason never fails a turn. */
  private static doneStopReason(finishReason: string): DoneStopReason {
    switch (finishReason) {
      case "length":
        return "length";
      case "tool_calls":
        return "toolUse";
      default:
        return "stop";
    }
  }
}
