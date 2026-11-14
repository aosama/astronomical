import { ImageModelDirectory } from "../src/ImageModelDirectory";
import { OTHER_COPY_RUNNING_NOTICE } from "../src/OtherCopyAnsweredError";
import { ImageModelSelection } from "../src/ImageModelSelection";
import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { ChatController, ENGINE_START_PATIENCE } from "../src/ChatController";
import { WAITING_FOR_ENGINE_NOTICE } from "../src/LoadNotice";
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

/** Boots a controller against an engine the test can silence and wake, with
 * retries short enough to watch several of them within a test. */
function bootControllerWithEngine(engineChannel = "development"): { controller: ChatController; daemon: FakeDaemon } {
  installDom();
  new FakeSessionHost(
    (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
    (callId, message) => window.__thintalkBridge?.reject(callId, message),
  ).install();
  const daemon = new FakeDaemon({ daemonBaseURL: DAEMON_BASE_URL, channel: engineChannel, stateDirectory: "~/.astronomical-dev" });
  daemon.install();
  const config = new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" });
  const channel = new HostChannel();
  const bridge = new SessionBridge((request) => channel.postSessionCall(request));
  const client = new DaemonClient(config);
  const controller = new ChatController({
    client,
    streamer: new SlowToolStreamer(),
    bridge,
    store: new SessionStore(bridge),
    imageModelDirectory: new ImageModelDirectory(client),
    imageModelSelection: new ImageModelSelection(),
  });
  return { controller, daemon };
}

/** Start-up attempts the engine has seen. The handshake is counted because
 * only a start-up makes one; the model list is also read for the picture pill. */
function startUpAttemptCount(daemon: FakeDaemon): number {
  return daemon.requests.filter((request) => request.url.endsWith("/v1/status")).length;
}

async function withQuickEngineRetries(journey: () => Promise<void>): Promise<void> {
  const realPatience = { ...ENGINE_START_PATIENCE };
  Object.assign(ENGINE_START_PATIENCE, { quickRetryMilliseconds: 10, slowRetryMilliseconds: 10 });
  try {
    await journey();
  } finally {
    Object.assign(ENGINE_START_PATIENCE, realPatience);
  }
}

test("should_list_the_models_by_itself_once_a_silent_engine_starts_answering", async () => {
  await withQuickEngineRetries(async () => {
    const { controller, daemon } = bootControllerWithEngine();
    let listedModelCount = 0;
    let notice: string | null = null;
    let silentEngineNoticeAsksForAction: boolean | undefined;
    controller.subscribe((state) => {
      listedModelCount = state.composer.models.length;
      notice = state.transcript.notice;
      silentEngineNoticeAsksForAction = state.transcript.noticeAsksForAction;
    });
    daemon.modelsUnreachable = true;
    await controller.load();
    assert.equal(listedModelCount, 0, "a silent engine lists no model");
    assert.equal(notice, WAITING_FOR_ENGINE_NOTICE, "the window says it is waiting, not the browser's error text");
    assert.equal(silentEngineNoticeAsksForAction, false, "a wait that ends by itself stays in quiet print");

    daemon.modelsUnreachable = false;

    await waitFor(() => listedModelCount > 0, "the models to appear without a relaunch");
    assert.equal(notice, null, "the failure notice goes once the engine answers");
  });
}, { timeout: 30_000 });

test("should_keep_asking_while_the_engine_stays_silent", async () => {
  await withQuickEngineRetries(async () => {
    const { controller, daemon } = bootControllerWithEngine();
    let listedModelCount = 0;
    controller.subscribe((state) => {
      listedModelCount = state.composer.models.length;
    });
    daemon.modelsUnreachable = true;
    await controller.load();

    await waitFor(() => startUpAttemptCount(daemon) >= 4, "several more asks of a silent engine");

    assert.equal(listedModelCount, 0, "no model is listed while the engine stays silent");
    daemon.modelsUnreachable = false;
    await waitFor(() => listedModelCount > 0, "the retries to end with the engine's answer");
  });
}, { timeout: 30_000 });

test("should_stop_asking_once_the_engine_has_answered", async () => {
  await withQuickEngineRetries(async () => {
    const { controller, daemon } = bootControllerWithEngine();
    let listedModelCount = 0;
    controller.subscribe((state) => {
      listedModelCount = state.composer.models.length;
    });
    daemon.modelsUnreachable = true;
    await controller.load();
    daemon.modelsUnreachable = false;
    await waitFor(() => listedModelCount > 0, "the engine's answer");
    const attemptCountWhenAnswered = startUpAttemptCount(daemon);

    await new Promise((resolve) => setTimeout(resolve, 100));

    assert.equal(startUpAttemptCount(daemon), attemptCountWhenAnswered, "an answered window asks no more");
  });
}, { timeout: 30_000 });

test("should_run_one_start_up_when_a_model_lands_while_a_retry_is_due", async () => {
  await withQuickEngineRetries(async () => {
    const { controller, daemon } = bootControllerWithEngine();
    let listedModelCount = 0;
    controller.subscribe((state) => {
      listedModelCount = state.composer.models.length;
    });
    daemon.modelsUnreachable = true;
    await controller.load();
    const attemptCountWhileSilent = startUpAttemptCount(daemon);
    daemon.modelsUnreachable = false;

    await Promise.all([controller.refreshChatModels(), controller.refreshChatModels()]);

    assert.ok(listedModelCount > 0, "the landed model is listed");
    assert.equal(startUpAttemptCount(daemon), attemptCountWhileSilent + 1, "the two callers shared one start-up");
  });
}, { timeout: 30_000 });

test("should_not_retry_a_start_up_that_succeeded_first_time", async () => {
  await withQuickEngineRetries(async () => {
    const { controller, daemon } = bootControllerWithEngine();
    await controller.load();
    const attemptCountAfterLoad = startUpAttemptCount(daemon);

    await new Promise((resolve) => setTimeout(resolve, 100));

    assert.equal(startUpAttemptCount(daemon), attemptCountAfterLoad);
  });
}, { timeout: 30_000 });

test("should_say_another_copy_is_running_and_carry_on_once_it_lets_go_of_the_address", async () => {
  await withQuickEngineRetries(async () => {
    const { controller, daemon } = bootControllerWithEngine();
    let listedModelCount = 0;
    let notice: string | null = null;
    let noticeAsksForAction: boolean | undefined;
    controller.subscribe((state) => {
      listedModelCount = state.composer.models.length;
      notice = state.transcript.notice;
      noticeAsksForAction = state.transcript.noticeAsksForAction;
    });
    // The direct download and the App Store version share one address but
    // keep their files apart, so the other copy reports another folder.
    daemon.reportedStateDirectory = "~/.astronomical";
    await controller.load();
    assert.equal(notice, OTHER_COPY_RUNNING_NOTICE, "the window explains it in plain words");
    assert.equal(noticeAsksForAction, true, "only the person can end this, so the notice is drawn to be read");
    assert.equal(listedModelCount, 0, "the other copy's models are not shown as this one's");

    daemon.reportedStateDirectory = "~/.astronomical-dev";

    await waitFor(() => listedModelCount > 0, "the models to appear once this app's own engine answers");
    assert.equal(notice, null);
    assert.equal(noticeAsksForAction, false);
  });
}, { timeout: 30_000 });

test("should_treat_an_engine_of_another_channel_as_another_copy", async () => {
  await withQuickEngineRetries(async () => {
    const { controller } = bootControllerWithEngine("stable");
    let notice: string | null = null;
    controller.subscribe((state) => {
      notice = state.transcript.notice;
    });
    await controller.load();
    assert.equal(notice, OTHER_COPY_RUNNING_NOTICE);
  });
}, { timeout: 30_000 });

test("should_not_keep_asking_an_engine_whose_answer_makes_no_sense", async () => {
  await withQuickEngineRetries(async () => {
    const { controller, daemon } = bootControllerWithEngine();
    let notice: string | null = null;
    controller.subscribe((state) => {
      notice = state.transcript.notice;
    });
    daemon.answersStatusWithNonsense = true;
    await controller.load();
    const attemptCountAfterLoad = startUpAttemptCount(daemon);

    await new Promise((resolve) => setTimeout(resolve, 100));

    assert.equal(startUpAttemptCount(daemon), attemptCountAfterLoad, "nonsense is answered the same every time");
    assert.match(notice ?? "", /unexpected response/);
  });
}, { timeout: 30_000 });
