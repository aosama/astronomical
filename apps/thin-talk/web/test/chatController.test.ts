import { ImageModelDirectory } from "../src/ImageModelDirectory";
import { ImageModelSelection } from "../src/ImageModelSelection";
import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { ChatController } from "../src/ChatController";
import { ChatMessageState } from "../src/ChatMessageState";
import type { ChatStreamer, ChatStreamRequest } from "../src/ChatStreamer";
import { DaemonClient } from "../src/DaemonClient";
import { HostChannel } from "../src/HostChannel";
import { SessionBridge } from "../src/SessionBridge";
import { SessionStore } from "../src/SessionStore";
import { ToolStep, ToolStepState } from "../src/ToolStep";
import { installDom, waitFor } from "./support/dom";
import { FakeDaemon } from "./support/fakeDaemon";
import { FAKE_STORED_IMAGE_NAME, FakeSessionHost } from "./support/fakeHost";
import { DAEMON_BASE_URL } from "./support/journey";

/**
 * Controller behaviour that depends on tool steps: stall detection pauses
 * while a tool runs, and deleting a conversation deletes the images its tools
 * generated. A scripted streamer stands in for the agent loop.
 */

const SHORT_STALL_TIMEOUT_MILLISECONDS = 50;

function imageStep(state: ToolStepState): ToolStep {
  return new ToolStep({
    toolCallID: "call-1",
    toolName: "generate_image",
    label: "Image generation",
    argumentsJSON: '{"prompt":"A rose"}',
    resultText: state === ToolStepState.RUNNING ? "" : "Generated.",
    state,
    generatedImageName: state === ToolStepState.SUCCEEDED ? FAKE_STORED_IMAGE_NAME : null,
  });
}

/** A turn whose one tool runs well past the stall timeout, then answers. */
class SlowToolStreamer implements ChatStreamer {
  public readonly stallTimeoutMilliseconds = SHORT_STALL_TIMEOUT_MILLISECONDS;

  public queueUserMessage(): boolean {
    return false;
  }

  public async streamChat(request: ChatStreamRequest): Promise<void> {
    request.onToolStep(imageStep(ToolStepState.RUNNING));
    // The watchdog polls once a second, so this outlasts at least one poll.
    await new Promise((resolve) => setTimeout(resolve, 1_300));
    request.onToolStep(imageStep(ToolStepState.SUCCEEDED));
    request.onText("Here is your rose.");
  }
}

async function bootController(streamer: ChatStreamer = new SlowToolStreamer()): Promise<{ controller: ChatController; host: FakeSessionHost }> {
  installDom();
  const host = new FakeSessionHost(
    (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
    (callId, message) => window.__thintalkBridge?.reject(callId, message),
  );
  host.install();
  new FakeDaemon({ daemonBaseURL: DAEMON_BASE_URL, channel: "development", stateDirectory: "~/.astronomical-dev" }).install();
  const config = new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" });
  const channel = new HostChannel();
  const bridge = new SessionBridge((request) => channel.postSessionCall(request));
  const client = new DaemonClient(config);
  const controller = new ChatController({
    client,
    streamer,
    bridge,
    store: new SessionStore(bridge),
    imageModelDirectory: new ImageModelDirectory(client),
    imageModelSelection: new ImageModelSelection(),
  });
  await controller.load();
  return { controller, host };
}

test("should_not_report_a_stall_while_a_tool_runs_past_the_stall_timeout", async () => {
  const { controller } = await bootController();
  let latestMessages: { state: ChatMessageState; markdown: string }[] = [];
  let latestNotice: string | null = null;
  controller.subscribe((state) => {
    latestMessages = state.transcript.messages.map((message) => ({ state: message.state, markdown: message.markdown }));
    latestNotice = state.composer.failure != null ? state.composer.failure.message : null;
  });
  controller.send("Draw a rose.");
  await waitFor(() => latestMessages[1]?.state === ChatMessageState.COMPLETE, "the answer to complete");
  assert.equal(latestNotice, null, "no stall failure was shown");
  assert.equal(latestMessages[1]?.markdown, "Here is your rose.");
}, { timeout: 30_000 });

test("should_delete_a_conversations_generated_images_with_it", async () => {
  const { controller, host } = await bootController();
  let activeSessionID: string | null = null;
  let isAnswered = false;
  controller.subscribe((state) => {
    activeSessionID = state.activeSessionID;
    isAnswered = state.transcript.messages[1]?.state === ChatMessageState.COMPLETE;
  });
  controller.send("Draw a rose.");
  await waitFor(() => isAnswered && activeSessionID != null, "the answer to be saved");
  await controller.deleteSession(activeSessionID ?? "");
  assert.deepEqual(host.deletedGeneratedImageNames, [FAKE_STORED_IMAGE_NAME]);
}, { timeout: 30_000 });

/** A turn that answers each request after a short wait and never takes queued
 * messages itself, so the controller must send them as the next turns. */
class DecliningStreamer implements ChatStreamer {
  public readonly stallTimeoutMilliseconds = 60_000;
  public readonly askedQuestions: string[] = [];

  public queueUserMessage(): boolean {
    return false;
  }

  public async streamChat(request: ChatStreamRequest): Promise<void> {
    const question = request.messages[request.messages.length - 1]?.content ?? "";
    this.askedQuestions.push(question);
    await new Promise((resolve) => setTimeout(resolve, 50));
    if (!request.signal.aborted) {
      request.onText(`Answer to ${question}`);
    }
  }
}

function transcriptOf(controller: ChatController): () => { markdown: string; state: ChatMessageState }[] {
  let latestMessages: { markdown: string; state: ChatMessageState }[] = [];
  controller.subscribe((state) => {
    latestMessages = state.transcript.messages.map((message) => ({ markdown: message.markdown, state: message.state }));
  });
  return () => latestMessages;
}

test("should_queue_a_message_sent_while_answering_and_send_it_next", async () => {
  const streamer = new DecliningStreamer();
  const { controller } = await bootController(streamer);
  const currentTranscript = transcriptOf(controller);
  controller.send("First?");
  controller.send("Second?");
  assert.equal(currentTranscript()[2]?.state, ChatMessageState.QUEUED);
  await waitFor(() => currentTranscript().length === 4 && currentTranscript()[3]?.state === ChatMessageState.COMPLETE, "both answers");
  assert.deepEqual(streamer.askedQuestions, ["First?", "Second?"]);
  assert.deepEqual(currentTranscript().map((message) => message.markdown), ["First?", "Answer to First?", "Second?", "Answer to Second?"]);
}, { timeout: 30_000 });

test("should_mark_queued_messages_not_sent_when_the_user_stops", async () => {
  const streamer = new DecliningStreamer();
  const { controller } = await bootController(streamer);
  const currentTranscript = transcriptOf(controller);
  controller.send("First?");
  controller.send("Never mind?");
  await controller.stop();
  assert.deepEqual(streamer.askedQuestions, ["First?"]);
  assert.equal(currentTranscript().find((message) => message.markdown === "Never mind?")?.state, ChatMessageState.FAILED);
}, { timeout: 30_000 });

/** A turn that fails after a moment, as a daemon refusal would. */
class FailingStreamer implements ChatStreamer {
  public readonly stallTimeoutMilliseconds = 60_000;

  public queueUserMessage(): boolean {
    return false;
  }

  public async streamChat(): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, 30));
    throw new Error("the daemon refused");
  }
}

test("should_mark_queued_messages_not_sent_when_the_answer_fails", async () => {
  const { controller } = await bootController(new FailingStreamer());
  const currentTranscript = transcriptOf(controller);
  controller.send("First?");
  controller.send("Then this?");
  await waitFor(() => currentTranscript().some((message) => message.markdown === "Then this?" && message.state === ChatMessageState.FAILED), "the queued message to be marked not sent");
}, { timeout: 30_000 });

test("should_not_send_queued_messages_after_starting_a_new_chat", async () => {
  const streamer = new DecliningStreamer();
  const { controller } = await bootController(streamer);
  controller.send("First?");
  controller.send("Queued?");
  await controller.newChat();
  await new Promise((resolve) => setTimeout(resolve, 150));
  assert.deepEqual(streamer.askedQuestions, ["First?"]);
}, { timeout: 30_000 });

test("should_ignore_an_image_model_that_is_not_installed_without_saving", async () => {
  const { controller, host } = await bootController();
  const savesBefore = host.receivedRequests.filter((request) => request.op === "savePrefs").length;

  controller.selectImageModel("not-installed");
  await new Promise((resolve) => setTimeout(resolve, 32));

  assert.equal(host.receivedRequests.filter((request) => request.op === "savePrefs").length, savesBefore);
});
