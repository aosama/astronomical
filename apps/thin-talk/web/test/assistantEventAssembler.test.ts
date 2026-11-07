import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessageEvent } from "@earendil-works/pi-ai";
import { AssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import { AssistantEventAssembler } from "../src/AssistantEventAssembler";
import { ChatCompletionChunk } from "../src/ChatCompletionChunk";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { daemonModel } from "../src/DaemonModelDescription";

const DAEMON_MODEL = daemonModel("fake-chat-model");

function chunk(init: { text?: string; reasoning?: string; finishReason?: string }): ChatCompletionChunk {
  return new ChatCompletionChunk({
    textDelta: init.text != null ? init.text : "",
    reasoningDelta: init.reasoning != null ? init.reasoning : "",
    finishReason: init.finishReason != null ? init.finishReason : null,
  });
}

async function collectEvents(eventStream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
  const collectedEvents: AssistantMessageEvent[] = [];
  for await (const streamEvent of eventStream) {
    collectedEvents.push(streamEvent);
  }
  return collectedEvents;
}

test("should_emit_reasoning_then_text_blocks_and_finish_with_done", async () => {
  const eventStream = new AssistantMessageEventStream();
  const assembler = new AssistantEventAssembler(eventStream, DAEMON_MODEL);
  assembler.start();
  assembler.apply(chunk({ reasoning: "Let me ", text: "" }));
  assembler.apply(chunk({ reasoning: "think." }));
  assembler.apply(chunk({ text: "Answer." }));
  assembler.apply(chunk({ finishReason: "stop" }));
  assembler.finish();
  const eventTypes = (await collectEvents(eventStream)).map((streamEvent) => streamEvent.type);
  assert.deepEqual(eventTypes, [
    "start", "thinking_start", "thinking_delta", "thinking_delta", "thinking_end",
    "text_start", "text_delta", "text_end", "done",
  ]);
  const finalMessage = await eventStream.result();
  assert.equal(finalMessage.stopReason, "stop");
  assert.deepEqual(finalMessage.content, [
    { type: "thinking", thinking: "Let me think." },
    { type: "text", text: "Answer." },
  ]);
});

test("should_open_a_new_block_each_time_reasoning_and_text_alternate", async () => {
  const eventStream = new AssistantMessageEventStream();
  const assembler = new AssistantEventAssembler(eventStream, DAEMON_MODEL);
  assembler.start();
  assembler.apply(chunk({ text: "a" }));
  assembler.apply(chunk({ reasoning: "b" }));
  assembler.apply(chunk({ text: "c", finishReason: "stop" }));
  assembler.finish();
  const finalMessage = await eventStream.result();
  assert.deepEqual(finalMessage.content.map((contentBlock) => contentBlock.type), ["text", "thinking", "text"]);
});

test("should_map_the_daemon_finish_reasons_to_pi_stop_reasons", async () => {
  const expectedStopReasons: [string, string][] = [["stop", "stop"], ["length", "length"], ["tool_calls", "toolUse"], ["content_filter", "stop"]];
  for (const [finishReason, expectedStopReason] of expectedStopReasons) {
    const eventStream = new AssistantMessageEventStream();
    const assembler = new AssistantEventAssembler(eventStream, DAEMON_MODEL);
    assembler.start();
    assembler.apply(chunk({ text: "x", finishReason }));
    assembler.finish();
    assert.equal((await eventStream.result()).stopReason, expectedStopReason, finishReason);
  }
});

test("should_report_a_stream_without_a_finish_reason_as_malformed", () => {
  const assembler = new AssistantEventAssembler(new AssistantMessageEventStream(), DAEMON_MODEL);
  assembler.start();
  assembler.apply(chunk({ text: "cut off" }));
  assert.throws(
    () => assembler.finish(),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.MALFORMED_MODEL_OUTPUT,
  );
});

test("should_keep_partial_text_when_a_response_is_stopped", async () => {
  const eventStream = new AssistantMessageEventStream();
  const assembler = new AssistantEventAssembler(eventStream, DAEMON_MODEL);
  assembler.start();
  assembler.apply(chunk({ text: "half an ans" }));
  assembler.fail("aborted", "Stopped by the user.");
  const collectedEvents = await collectEvents(eventStream);
  assert.deepEqual(collectedEvents.slice(-2).map((streamEvent) => streamEvent.type), ["text_end", "error"]);
  const finalMessage = await eventStream.result();
  assert.equal(finalMessage.stopReason, "aborted");
  assert.deepEqual(finalMessage.content, [{ type: "text", text: "half an ans" }]);
});

test("should_end_with_a_lone_error_when_nothing_arrived", async () => {
  const eventStream = new AssistantMessageEventStream();
  const assembler = new AssistantEventAssembler(eventStream, DAEMON_MODEL);
  assembler.start();
  assembler.fail("error", "the daemon is not running");
  const eventTypes = (await collectEvents(eventStream)).map((streamEvent) => streamEvent.type);
  assert.deepEqual(eventTypes, ["start", "error"]);
  assert.equal((await eventStream.result()).errorMessage, "the daemon is not running");
});

test("should_finish_a_reasoning_only_response_with_done", async () => {
  const eventStream = new AssistantMessageEventStream();
  const assembler = new AssistantEventAssembler(eventStream, DAEMON_MODEL);
  assembler.start();
  assembler.apply(chunk({ reasoning: "Nothing to say.", finishReason: "stop" }));
  assembler.finish();
  const eventTypes = (await collectEvents(eventStream)).map((streamEvent) => streamEvent.type);
  assert.deepEqual(eventTypes, ["start", "thinking_start", "thinking_delta", "thinking_end", "done"]);
});
