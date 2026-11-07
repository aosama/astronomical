import type { Api, AssistantMessage, Model, StopReason } from "@earendil-works/pi-ai";
import type { AssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import { ChatCompletionChunk } from "./ChatCompletionChunk";
import { ChatFailure } from "./ChatFailure";
import { ChatFailureKind } from "./ChatFailureKind";
import { emptyAssistantMessage } from "./DaemonModelDescription";

type DoneStopReason = Extract<StopReason, "stop" | "length" | "toolUse">;

/**
 * Folds the daemon's streamed chunks into the event sequence Pi's agent loop
 * expects for one assistant response: `start`, then a start/delta/end run for
 * each reasoning or text block, then exactly one `done` or `error`.
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
    const doneStopReason = AssistantEventAssembler.doneStopReason(this.finishReason);
    this.assistantMessage.stopReason = doneStopReason;
    this.eventStream.push({ type: "done", reason: doneStopReason, message: this.assistantMessage });
    this.eventStream.end(this.assistantMessage);
  }

  /** Ends the response with a stop (`aborted`) or a failure (`error`),
   * keeping whatever text and reasoning already arrived. */
  public fail(stopReason: "aborted" | "error", errorMessage: string): void {
    this.closeOpenBlock();
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
