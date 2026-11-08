import assert from "node:assert/strict";
import { test } from "node:test";
import { AgentTurnRunner } from "../src/AgentTurnRunner";
import { AppConfig } from "../src/AppConfig";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { ChatMessage } from "../src/ChatMessage";
import { ChatRole } from "../src/ChatRole";
import type { ChatStreamRequest } from "../src/ChatStreamer";
import { DaemonClient } from "../src/DaemonClient";
import type { ConversationTool } from "../src/ConversationTool";
import { ToolCatalog } from "../src/ToolCatalog";
import { DAEMON_BASE_URL, ScriptedChatFetch, chunkLine, toolCallLine, waitUntil } from "./support/scriptedChatFetch";
import { MAXIMUM_TOOL_ROUNDS_PER_TURN, TOOL_ROUND_LIMIT_REASON } from "../src/ToolLimits";
import { CurrentTimeTool } from "../src/CurrentTimeTool";
import { ToolStep, ToolStepState } from "../src/ToolStep";

interface ForwardedDeltas {
  text: string;
  reasoning: string;
}

function turnRunner(availableTools: ConversationTool[] = []): AgentTurnRunner {
  return new AgentTurnRunner(
    new DaemonClient(
      new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
    ),
    new ToolCatalog(availableTools),
  );
}

function turnRequest(messages: ChatMessage[], forwardedDeltas: ForwardedDeltas, signal: AbortSignal): ChatStreamRequest {
  return {
    modelID: "fake-chat-model",
    messages,
    toolsEnabled: true,
    modelSupportsToolCalls: false,
    modelAcceptsImages: false,
    conversationID: null,
    onText: (delta) => { forwardedDeltas.text += delta; },
    onReasoning: (delta) => { forwardedDeltas.reasoning += delta; },
    onToolStep: () => undefined,
    onQueuedMessageAnswered: () => undefined,
    contextWindowTokens: null,
    signal,
  };
}

function userMessage(content: string): ChatMessage {
  return new ChatMessage({ id: `user-${content}`, role: ChatRole.USER, content });
}

test("should_stream_reasoning_and_text_of_one_turn_to_the_conversation", async () => {
  new ScriptedChatFetch([{
    lines: [chunkLine({ reasoning_content: "Thinking." }), chunkLine({ content: "Shakespeare." }), chunkLine({}, "stop")],
  }]).install();
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  await turnRunner().streamChat(turnRequest([userMessage("Who wrote Hamlet?")], forwardedDeltas, new AbortController().signal));
  assert.deepEqual(forwardedDeltas, { text: "Shakespeare.", reasoning: "Thinking." });
});

test("should_replay_earlier_turns_and_ask_the_latest_question_once", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "1606." }, "stop")] }]);
  scriptedFetch.install();
  const history = [
    userMessage("Who wrote Macbeth?"),
    new ChatMessage({ id: "answer-1", role: ChatRole.ASSISTANT, content: "Shakespeare.", reasoning: "Known." }),
    userMessage("When?"),
  ];
  await turnRunner().streamChat(turnRequest(history, { text: "", reasoning: "" }, new AbortController().signal));
  assert.equal(scriptedFetch.requests.length, 1);
  assert.deepEqual(scriptedFetch.requests[0]?.body["messages"], [
    { role: "user", content: "Who wrote Macbeth?" },
    { role: "assistant", content: "Shakespeare." },
    { role: "user", content: "When?" },
  ]);
  assert.equal("thinking_budget" in (scriptedFetch.requests[0]?.body ?? {}), false, "no cap on thinking");
});

test("should_refuse_a_turn_whose_history_does_not_end_with_a_question", async () => {
  const scriptedFetch = new ScriptedChatFetch([]);
  scriptedFetch.install();
  const endsWithAnswer = [userMessage("Hi"), new ChatMessage({ id: "answer", role: ChatRole.ASSISTANT, content: "Hello." })];
  for (const history of [[], endsWithAnswer]) {
    await assert.rejects(
      turnRunner().streamChat(turnRequest(history, { text: "", reasoning: "" }, new AbortController().signal)),
      (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.INVALID_REQUEST,
    );
  }
  assert.equal(scriptedFetch.requests.length, 0);
});

test("should_end_quietly_when_the_user_stops_mid_answer", async () => {
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "Once upon a time" })], holdOpenUntilAborted: true }]).install();
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  const stopController = new AbortController();
  const turnFinished = turnRunner().streamChat(turnRequest([userMessage("Tell a story")], forwardedDeltas, stopController.signal));
  await waitUntil(() => forwardedDeltas.text.length > 0, "the first words of the answer");
  stopController.abort();
  await turnFinished;
  assert.equal(forwardedDeltas.text, "Once upon a time");
});

test("should_send_nothing_when_stopped_before_the_turn_starts", async () => {
  const scriptedFetch = new ScriptedChatFetch([]);
  scriptedFetch.install();
  const stopController = new AbortController();
  stopController.abort();
  await turnRunner().streamChat(turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, stopController.signal));
  assert.equal(scriptedFetch.requests.length, 0);
});

test("should_rethrow_the_daemons_classified_failure", async () => {
  new ScriptedChatFetch([{ lines: ['data: {"error":{"message":"chat_engine_busy: another request is running"}}'] }]).install();
  await assert.rejects(
    turnRunner().streamChat(turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, new AbortController().signal)),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.ENGINE_BUSY,
  );
});

test("should_report_an_unreachable_daemon_as_unavailable", async () => {
  new ScriptedChatFetch([{ networkFailure: true }]).install();
  await assert.rejects(
    turnRunner().streamChat(turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, new AbortController().signal)),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.WORKER_UNAVAILABLE,
  );
});

test("should_fail_a_turn_whose_stream_was_cut_off", async () => {
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "Half" })] }]).install();
  await assert.rejects(
    turnRunner().streamChat(turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, new AbortController().signal)),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.MALFORMED_MODEL_OUTPUT,
  );
});

test("should_finish_a_turn_cut_at_the_length_limit_without_failing", async () => {
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "A long answer that" }, "length")] }]).install();
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  await turnRunner().streamChat(turnRequest([userMessage("Explain")], forwardedDeltas, new AbortController().signal));
  assert.equal(forwardedDeltas.text, "A long answer that");
});

test("should_cancel_the_daemon_request_when_the_conversation_listener_throws", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Hello" })], holdOpenUntilAborted: true }]);
  scriptedFetch.install();
  const throwingRequest: ChatStreamRequest = {
    ...turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, new AbortController().signal),
    onText: () => {
      throw new Error("the transcript failed to draw");
    },
  };
  await assert.rejects(
    turnRunner().streamChat(throwingRequest),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.UNKNOWN,
  );
  await waitUntil(() => scriptedFetch.requests[0]?.wasCancelled === true, "the daemon request to be cancelled");
});

function clockTool(answer = "Saturday, 2 PM"): ConversationTool {
  return {
    name: "get_current_time",
    label: "Current time",
    description: "Reads the clock.",
    parameters: { type: "object", properties: {} },
    execute: async () => answer,
  };
}

function toolTurnRequest(messages: ChatMessage[], recordedSteps: ToolStep[], forwardedDeltas: ForwardedDeltas): ChatStreamRequest {
  return {
    ...turnRequest(messages, forwardedDeltas, new AbortController().signal),
    modelSupportsToolCalls: true,
    onToolStep: (toolStep) => { recordedSteps.push(toolStep); },
  };
}

test("should_run_a_tool_report_its_steps_and_answer_with_the_result", async () => {
  const scriptedFetch = new ScriptedChatFetch([
    { lines: [toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls")] },
    { lines: [chunkLine({ content: "It is 2 PM." }, "stop")] },
  ]);
  scriptedFetch.install();
  const recordedSteps: ToolStep[] = [];
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  await turnRunner([clockTool()]).streamChat(toolTurnRequest([userMessage("What time is it?")], recordedSteps, forwardedDeltas));
  assert.equal(forwardedDeltas.text, "It is 2 PM.");
  assert.deepEqual(recordedSteps.map((toolStep) => [toolStep.label, toolStep.state, toolStep.resultText]), [
    ["Current time", ToolStepState.RUNNING, ""],
    ["Current time", ToolStepState.SUCCEEDED, "Saturday, 2 PM"],
  ]);
  const followUpMessages = scriptedFetch.requests[1]?.body["messages"] as { role: string; content?: string }[];
  assert.deepEqual(followUpMessages.map((requestMessage) => requestMessage.role), ["user", "assistant", "tool"]);
  assert.equal(followUpMessages[2]?.content, "Saturday, 2 PM");
});

test("should_report_a_failing_tool_to_the_model_and_mark_its_step_failed", async () => {
  const scriptedFetch = new ScriptedChatFetch([
    { lines: [toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls")] },
    { lines: [chunkLine({ content: "I could not read the clock." }, "stop")] },
  ]);
  scriptedFetch.install();
  const brokenClock: ConversationTool = { ...clockTool(), execute: async () => { throw new Error("clock unavailable"); } };
  const recordedSteps: ToolStep[] = [];
  await turnRunner([brokenClock]).streamChat(toolTurnRequest([userMessage("What time is it?")], recordedSteps, { text: "", reasoning: "" }));
  assert.equal(recordedSteps[recordedSteps.length - 1]?.state, ToolStepState.FAILED);
  const toolMessage = (scriptedFetch.requests[1]?.body["messages"] as { role: string; content?: string }[])[2];
  assert.match(toolMessage?.content ?? "", /clock unavailable/);
});

test("should_refuse_calls_past_the_round_limit_and_end_if_the_model_keeps_calling", async () => {
  const toolCallReply = { lines: [toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls")] };
  const scriptedFetch = new ScriptedChatFetch(Array.from({ length: MAXIMUM_TOOL_ROUNDS_PER_TURN + 4 }, () => toolCallReply));
  scriptedFetch.install();
  const recordedSteps: ToolStep[] = [];
  await turnRunner([clockTool()]).streamChat(toolTurnRequest([userMessage("Loop")], recordedSteps, { text: "", reasoning: "" }));
  // Eight rounds run, the ninth is refused with the reason, and the model
  // gets one more request; calling tools again then ends the turn.
  assert.equal(scriptedFetch.requests.length, MAXIMUM_TOOL_ROUNDS_PER_TURN + 2);
  const refusedRoundMessages = scriptedFetch.requests[MAXIMUM_TOOL_ROUNDS_PER_TURN + 1]?.body["messages"] as { role: string; content?: string }[];
  assert.equal(refusedRoundMessages[refusedRoundMessages.length - 1]?.content, TOOL_ROUND_LIMIT_REASON);
  assert.equal(recordedSteps[recordedSteps.length - 1]?.state, ToolStepState.FAILED);
});

test("should_deliver_the_answer_after_exactly_the_round_limit", async () => {
  const toolCallReply = { lines: [toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls")] };
  const scriptedFetch = new ScriptedChatFetch([
    ...Array.from({ length: MAXIMUM_TOOL_ROUNDS_PER_TURN }, () => toolCallReply),
    { lines: [chunkLine({ content: "Done." }, "stop")] },
  ]);
  scriptedFetch.install();
  const recordedSteps: ToolStep[] = [];
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  await turnRunner([clockTool()]).streamChat(toolTurnRequest([userMessage("Eight checks")], recordedSteps, forwardedDeltas));
  assert.equal(forwardedDeltas.text, "Done.");
  assert.equal(recordedSteps.filter((toolStep) => toolStep.state === ToolStepState.FAILED).length, 0);
  assert.equal(recordedSteps[recordedSteps.length - 1]?.roundIndex, MAXIMUM_TOOL_ROUNDS_PER_TURN - 1);
});

test("should_offer_no_tools_when_the_switch_is_off_or_the_model_cannot_use_them", async () => {
  for (const [toolsEnabled, modelSupportsToolCalls] of [[false, true], [true, false]] as const) {
    const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Hi." }, "stop")] }]);
    scriptedFetch.install();
    const request = { ...toolTurnRequest([userMessage("Hi")], [], { text: "", reasoning: "" }), toolsEnabled, modelSupportsToolCalls };
    await turnRunner([clockTool()]).streamChat(request);
    assert.equal("tools" in (scriptedFetch.requests[0]?.body ?? {}), false, `${toolsEnabled}/${modelSupportsToolCalls}`);
  }
});

test("should_start_a_new_paragraph_when_text_resumes_after_a_tool_round", async () => {
  new ScriptedChatFetch([
    { lines: [chunkLine({ content: "Let me check." }), toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls")] },
    { lines: [chunkLine({ content: "It is 2 PM." }, "stop")] },
  ]).install();
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  await turnRunner([clockTool()]).streamChat(toolTurnRequest([userMessage("Time?")], [], forwardedDeltas));
  assert.equal(forwardedDeltas.text, "Let me check.\n\nIt is 2 PM.");
});

test("should_run_the_real_clock_tool_when_the_model_sends_stray_arguments", async () => {
  new ScriptedChatFetch([
    { lines: [toolCallLine("call-1", "get_current_time", '{"type":"object","properties":{}}'), chunkLine({}, "tool_calls")] },
    { lines: [chunkLine({ content: "Noted." }, "stop")] },
  ]).install();
  const recordedSteps: ToolStep[] = [];
  await turnRunner([new CurrentTimeTool(() => new Date("2026-11-07T19:30:00.000Z"))])
    .streamChat(toolTurnRequest([userMessage("Time?")], recordedSteps, { text: "", reasoning: "" }));
  const finishedStep = recordedSteps[recordedSteps.length - 1];
  assert.equal(finishedStep?.state, ToolStepState.SUCCEEDED);
  assert.match(finishedStep?.resultText ?? "", /2026-11-07T19:30:00\.000Z/);
  assert.equal(finishedStep?.argumentsJSON, '{"type":"object","properties":{}}');
});

test("should_fail_a_call_to_a_tool_that_does_not_exist_and_label_it_by_name", async () => {
  new ScriptedChatFetch([
    { lines: [toolCallLine("call-1", "open_the_pod_bay_doors", "{}"), chunkLine({}, "tool_calls")] },
    { lines: [chunkLine({ content: "I cannot." }, "stop")] },
  ]).install();
  const recordedSteps: ToolStep[] = [];
  await turnRunner([clockTool()]).streamChat(toolTurnRequest([userMessage("Open them")], recordedSteps, { text: "", reasoning: "" }));
  const finishedStep = recordedSteps[recordedSteps.length - 1];
  assert.equal(finishedStep?.state, ToolStepState.FAILED);
  assert.equal(finishedStep?.label, "open_the_pod_bay_doors");
});

test("should_fail_the_running_step_and_end_quietly_when_stopped_mid_tool", async () => {
  new ScriptedChatFetch([
    { lines: [toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls")] },
  ]).install();
  const stopController = new AbortController();
  const slowClock: ConversationTool = {
    ...clockTool(),
    execute: (_toolArguments, signal) => new Promise<string>((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new Error("stopped")));
      stopController.abort();
    }),
  };
  const recordedSteps: ToolStep[] = [];
  const request = { ...toolTurnRequest([userMessage("Time?")], recordedSteps, { text: "", reasoning: "" }), signal: stopController.signal };
  await turnRunner([slowClock]).streamChat(request);
  assert.equal(recordedSteps[recordedSteps.length - 1]?.state, ToolStepState.FAILED);
});

test("should_refuse_a_queued_message_when_no_turn_is_running", () => {
  assert.equal(turnRunner().queueUserMessage("q1", "Later"), false);
});

test("should_answer_a_message_queued_mid_answer_after_the_answer", async () => {
  const scriptedFetch = new ScriptedChatFetch([
    { lines: [chunkLine({ content: "Shakespeare." }, "stop")] },
    { lines: [chunkLine({ content: "Around 1600." }, "stop")] },
  ]);
  scriptedFetch.install();
  const runner = turnRunner();
  const deliveryOrder: string[] = [];
  let hasQueued = false;
  const request: ChatStreamRequest = {
    ...turnRequest([userMessage("Who wrote Hamlet?")], { text: "", reasoning: "" }, new AbortController().signal),
    onText: (delta) => {
      deliveryOrder.push(`text:${delta}`);
      if (!hasQueued) {
        hasQueued = true;
        assert.equal(runner.queueUserMessage("q1", "And when?"), true);
      }
    },
    onQueuedMessageAnswered: (queuedMessageID) => deliveryOrder.push(`answering:${queuedMessageID}`),
  };
  await runner.streamChat(request);
  assert.deepEqual(deliveryOrder, ["text:Shakespeare.", "answering:q1", "text:Around 1600."]);
  const followUpMessages = scriptedFetch.requests[1]?.body["messages"] as { role: string; content: string }[];
  assert.deepEqual(followUpMessages.slice(-2), [{ role: "assistant", content: "Shakespeare." }, { role: "user", content: "And when?" }]);
});

test("should_steer_a_message_queued_during_a_tool_into_the_next_request", async () => {
  const scriptedFetch = new ScriptedChatFetch([
    { lines: [toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls")] },
    { lines: [chunkLine({ content: "It is 2 PM in Verona." }, "stop")] },
  ]);
  scriptedFetch.install();
  let runner: AgentTurnRunner | null = null;
  const steeringClock: ConversationTool = {
    ...clockTool(),
    execute: async () => {
      assert.equal(runner?.queueUserMessage("q1", "In Verona, please."), true);
      return "Saturday, 2 PM";
    },
  };
  runner = turnRunner([steeringClock]);
  const answeredIDs: string[] = [];
  await runner.streamChat({
    ...toolTurnRequest([userMessage("What time is it?")], [], { text: "", reasoning: "" }),
    onQueuedMessageAnswered: (queuedMessageID) => answeredIDs.push(queuedMessageID),
  });
  assert.equal(scriptedFetch.requests.length, 2, "the steered message rides on the request after the tool");
  const secondRequestMessages = scriptedFetch.requests[1]?.body["messages"] as { role: string; content: string }[];
  assert.deepEqual(secondRequestMessages.slice(-2).map((requestMessage) => requestMessage.role), ["tool", "user"]);
  assert.equal(secondRequestMessages[secondRequestMessages.length - 1]?.content, "In Verona, please.");
  assert.deepEqual(answeredIDs, ["q1"]);
});

test("should_trim_old_turns_to_the_models_context_window", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Fine." }, "stop")] }]);
  scriptedFetch.install();
  const longHistory = [
    userMessage("x".repeat(4_000)),
    new ChatMessage({ id: "a1", role: ChatRole.ASSISTANT, content: "y".repeat(4_000) }),
    userMessage("Short question?"),
  ];
  await turnRunner().streamChat({ ...turnRequest(longHistory, { text: "", reasoning: "" }, new AbortController().signal), contextWindowTokens: 200 });
  assert.deepEqual(scriptedFetch.requests[0]?.body["messages"], [{ role: "user", content: "Short question?" }]);
});

test("should_answer_queued_messages_in_the_order_they_were_sent_even_across_a_tool", async () => {
  const scriptedFetch = new ScriptedChatFetch([
    { lines: [chunkLine({ content: "Checking." }), toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls")] },
    { lines: [chunkLine({ content: "Answer to A." }, "stop")] },
    { lines: [chunkLine({ content: "Answer to B." }, "stop")] },
    { lines: [chunkLine({ content: "Done." }, "stop")] },
  ]);
  scriptedFetch.install();
  let runner: AgentTurnRunner | null = null;
  const queueingClock: ConversationTool = {
    ...clockTool(),
    execute: async () => {
      runner?.queueUserMessage("b", "Message B");
      return "Saturday, 2 PM";
    },
  };
  runner = turnRunner([queueingClock]);
  const answeredIDs: string[] = [];
  let hasQueuedA = false;
  await runner.streamChat({
    ...toolTurnRequest([userMessage("Time?")], [], { text: "", reasoning: "" }),
    onText: () => {
      if (!hasQueuedA) {
        hasQueuedA = true;
        runner?.queueUserMessage("a", "Message A");
      }
    },
    onQueuedMessageAnswered: (queuedMessageID) => answeredIDs.push(queuedMessageID),
  });
  assert.deepEqual(answeredIDs, ["a", "b"]);
  const requestEndings = scriptedFetch.requests.slice(1, 3).map((sentRequest) => {
    const requestMessages = sentRequest.body["messages"] as { content: string }[];
    return requestMessages[requestMessages.length - 1]?.content;
  });
  assert.deepEqual(requestEndings, ["Message A", "Message B"]);
});

function drawnImageStep(imageNumber: number): ToolStep {
  return new ToolStep({
    toolCallID: `draw-${imageNumber}`,
    toolName: "generate_image",
    label: "Image generation",
    argumentsJSON: "{}",
    resultText: `Generated image ${imageNumber}.`,
    state: ToolStepState.SUCCEEDED,
    roundIndex: imageNumber,
    generatedImageName: `00000000-0000-4000-8000-00000000000${imageNumber}.png`,
  });
}

test("should_show_a_seeing_model_only_the_two_latest_generated_images_again", async () => {
  const loadedImageNames: string[] = [];
  const runner = new AgentTurnRunner(
    new DaemonClient(new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" })),
    new ToolCatalog([new CurrentTimeTool()]),
    async (imageName) => {
      loadedImageNames.push(imageName);
      return { base64: `IMAGE-${imageName.slice(-5, -4)}`, mimeType: "image/png" };
    },
  );
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Three roses." }, "stop")] }]);
  scriptedFetch.install();
  const earlierAnswer = new ChatMessage({
    id: "assistant-1", role: ChatRole.ASSISTANT, content: "Drawn.", toolSteps: [drawnImageStep(1), drawnImageStep(2), drawnImageStep(3)],
  });
  const forwardedDeltas = { text: "", reasoning: "" };
  const request = turnRequest([userMessage("Draw three roses."), earlierAnswer, userMessage("Which is reddest?")], forwardedDeltas, new AbortController().signal);
  await runner.streamChat({ ...request, modelAcceptsImages: true, modelSupportsToolCalls: true });
  assert.deepEqual(loadedImageNames.map((imageName) => imageName.slice(-5, -4)), ["2", "3"]);
  const sentImageURLs = [...JSON.stringify(scriptedFetch.requests[0]?.body["messages"]).matchAll(/data:image\/png;base64,(IMAGE-\d)/g)].map((urlMatch) => urlMatch[1]);
  assert.deepEqual(sentImageURLs, ["IMAGE-2", "IMAGE-3"]);
});

test("should_load_no_images_for_a_model_that_cannot_see_or_when_tools_are_off", async () => {
  const loadedImageNames: string[] = [];
  const runner = new AgentTurnRunner(
    new DaemonClient(new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" })),
    new ToolCatalog([new CurrentTimeTool()]),
    async (imageName) => {
      loadedImageNames.push(imageName);
      return null;
    },
  );
  const earlierAnswer = new ChatMessage({ id: "assistant-1", role: ChatRole.ASSISTANT, content: "Drawn.", toolSteps: [drawnImageStep(1)] });
  for (const capabilities of [
    { modelAcceptsImages: false, modelSupportsToolCalls: true, toolsEnabled: true },
    { modelAcceptsImages: true, modelSupportsToolCalls: true, toolsEnabled: false },
  ]) {
    new ScriptedChatFetch([{ lines: [chunkLine({ content: "A rose." }, "stop")] }]).install();
    const request = turnRequest([userMessage("Draw a rose."), earlierAnswer, userMessage("Is it red?")], { text: "", reasoning: "" }, new AbortController().signal);
    await runner.streamChat({ ...request, ...capabilities });
  }
  assert.deepEqual(loadedImageNames, []);
});

test("should_answer_even_when_a_recalled_image_cannot_be_read", async () => {
  const runner = new AgentTurnRunner(
    new DaemonClient(new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" })),
    new ToolCatalog([new CurrentTimeTool()]),
    async () => { throw new Error("the image was deleted"); },
  );
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "It was a rose." }, "stop")] }]).install();
  const earlierAnswer = new ChatMessage({ id: "assistant-1", role: ChatRole.ASSISTANT, content: "Drawn.", toolSteps: [drawnImageStep(1)] });
  const forwardedDeltas = { text: "", reasoning: "" };
  const request = turnRequest([userMessage("Draw a rose."), earlierAnswer, userMessage("What was it?")], forwardedDeltas, new AbortController().signal);
  await runner.streamChat({ ...request, modelAcceptsImages: true, modelSupportsToolCalls: true });
  assert.equal(forwardedDeltas.text, "It was a rose.");
});

test("should_send_nothing_when_stopped_while_earlier_images_load", async () => {
  const stopController = new AbortController();
  const runner = new AgentTurnRunner(
    new DaemonClient(new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" })),
    new ToolCatalog([new CurrentTimeTool()]),
    async () => {
      stopController.abort();
      return { base64: "IMAGE", mimeType: "image/png" };
    },
  );
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Too late." }, "stop")] }]);
  scriptedFetch.install();
  const earlierAnswer = new ChatMessage({ id: "assistant-1", role: ChatRole.ASSISTANT, content: "Drawn.", toolSteps: [drawnImageStep(1)] });
  const request = turnRequest([userMessage("Draw a rose."), earlierAnswer, userMessage("Is it red?")], { text: "", reasoning: "" }, stopController.signal);
  await runner.streamChat({ ...request, modelAcceptsImages: true, modelSupportsToolCalls: true });
  assert.equal(scriptedFetch.requests.length, 0, "a stop during the image load sends no request");
});

test("should_load_no_images_when_the_catalog_offers_the_turn_no_tools", async () => {
  const loadedImageNames: string[] = [];
  const runner = new AgentTurnRunner(
    new DaemonClient(new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" })),
    new ToolCatalog([]),
    async (imageName) => {
      loadedImageNames.push(imageName);
      return null;
    },
  );
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "A rose." }, "stop")] }]).install();
  const earlierAnswer = new ChatMessage({ id: "assistant-1", role: ChatRole.ASSISTANT, content: "Drawn.", toolSteps: [drawnImageStep(1)] });
  const request = turnRequest([userMessage("Draw a rose."), earlierAnswer, userMessage("Is it red?")], { text: "", reasoning: "" }, new AbortController().signal);
  await runner.streamChat({ ...request, modelAcceptsImages: true, modelSupportsToolCalls: true });
  assert.deepEqual(loadedImageNames, []);
});
