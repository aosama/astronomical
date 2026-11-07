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

  public async streamChat(request: ChatStreamRequest): Promise<void> {
    request.onToolStep(imageStep(ToolStepState.RUNNING));
    // The watchdog polls once a second, so this outlasts at least one poll.
    await new Promise((resolve) => setTimeout(resolve, 1_300));
    request.onToolStep(imageStep(ToolStepState.SUCCEEDED));
    request.onText("Here is your rose.");
  }
}

async function bootController(): Promise<{ controller: ChatController; host: FakeSessionHost }> {
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
  const controller = new ChatController({
    client: new DaemonClient(config),
    streamer: new SlowToolStreamer(),
    bridge,
    store: new SessionStore(bridge),
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
