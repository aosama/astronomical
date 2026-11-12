import assert from "node:assert/strict";
import { test } from "node:test";
import { MODEL_LOADING_CLASS, ModelLoadNotice, STATUS_POLL_MILLISECONDS, isWaitingForFirstWords } from "../src/ModelLoadNotice";
import type { ChatSurfaceState } from "../src/ChatSurfaceState";
import { ChatMessageState } from "../src/ChatMessageState";
import { ChatRole } from "../src/ChatRole";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { ClassicMessageChrome } from "../src/MessageChrome";
import { AstroMessageChrome } from "../src/astro/AstroMessageChrome";
import { TranscriptPersona } from "../src/TranscriptPersona";
import fs from "node:fs";
import path from "node:path";
import { waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";

// A large model takes tens of seconds to load and idle models are unloaded,
// so "thinking…" for the whole wait read as a stuck app.

function stateWithNewest(role: ChatRole, state: ChatMessageState, markdown = "", reasoning = ""): ChatSurfaceState {
  const messages = [
    new TranscriptMessage({ id: "q", role: ChatRole.USER, markdown: "hi", reasoning: "", state: ChatMessageState.COMPLETE }),
    new TranscriptMessage({ id: "newest", role, markdown, reasoning, state }),
  ];
  return { transcript: { messages, notice: null } } as unknown as ChatSurfaceState;
}

test("should_wait_for_first_words_only_while_the_newest_answer_has_shown_nothing", () => {
  assert.equal(isWaitingForFirstWords(stateWithNewest(ChatRole.ASSISTANT, ChatMessageState.STREAMING)), true);
  assert.equal(isWaitingForFirstWords(stateWithNewest(ChatRole.ASSISTANT, ChatMessageState.STREAMING, "Hel")), false);
  assert.equal(isWaitingForFirstWords(stateWithNewest(ChatRole.ASSISTANT, ChatMessageState.STREAMING, "", "The user asks")), false);
  assert.equal(isWaitingForFirstWords(stateWithNewest(ChatRole.ASSISTANT, ChatMessageState.COMPLETE)), false);
  assert.equal(isWaitingForFirstWords(stateWithNewest(ChatRole.USER, ChatMessageState.COMPLETE, "hello")), false);
  assert.equal(isWaitingForFirstWords({ transcript: { messages: [], notice: null } } as unknown as ChatSurfaceState), false);
});

async function settle(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve));
}

test("should_mark_the_transcript_while_a_model_loads_and_clear_it_when_words_arrive", async (testContext) => {
  testContext.mock.timers.enable({ apis: ["setInterval"] });
  const transcriptElement = document.createElement("main");
  let isLoading = true;
  let statusRequests = 0;
  const notice = new ModelLoadNotice({
    isModelLoading: async () => {
      statusRequests += 1;
      return isLoading;
    },
    transcriptElement,
  });

  notice.setWaitingForFirstWords(true);
  await settle();
  assert.equal(transcriptElement.classList.contains(MODEL_LOADING_CLASS), true, "asked at once, not after the first interval");
  isLoading = false;
  testContext.mock.timers.tick(STATUS_POLL_MILLISECONDS);
  await settle();
  assert.equal(transcriptElement.classList.contains(MODEL_LOADING_CLASS), false, "the load ended; back to thinking");
  isLoading = true;
  testContext.mock.timers.tick(STATUS_POLL_MILLISECONDS);
  await settle();
  assert.equal(transcriptElement.classList.contains(MODEL_LOADING_CLASS), true);

  notice.setWaitingForFirstWords(false);
  assert.equal(transcriptElement.classList.contains(MODEL_LOADING_CLASS), false);
  const requestsWhenStopped = statusRequests;
  testContext.mock.timers.tick(6000);
  await settle();
  assert.equal(statusRequests, requestsWhenStopped, "nothing is asked once the words arrived");
});

test("should_ask_once_per_interval_however_often_the_state_is_published", async (testContext) => {
  testContext.mock.timers.enable({ apis: ["setInterval"] });
  let statusRequests = 0;
  const notice = new ModelLoadNotice({ isModelLoading: async () => ((statusRequests += 1), false), transcriptElement: document.createElement("main") });

  notice.setWaitingForFirstWords(true);
  notice.setWaitingForFirstWords(true);
  notice.setWaitingForFirstWords(true);
  await settle();

  assert.equal(statusRequests, 1);
  notice.setWaitingForFirstWords(false);
});

test("should_show_no_notice_when_the_status_cannot_be_read_or_arrives_after_the_wait", async (testContext) => {
  testContext.mock.timers.enable({ apis: ["setInterval"] });
  const transcriptElement = document.createElement("main");
  const failingNotice = new ModelLoadNotice({ isModelLoading: async () => Promise.reject(new Error("engine gone")), transcriptElement });
  failingNotice.setWaitingForFirstWords(true);
  await settle();
  assert.equal(transcriptElement.classList.contains(MODEL_LOADING_CLASS), false);
  failingNotice.setWaitingForFirstWords(false);

  let answerStatus: (isLoading: boolean) => void = () => undefined;
  const lateNotice = new ModelLoadNotice({ isModelLoading: () => new Promise<boolean>((resolve) => (answerStatus = resolve)), transcriptElement });
  lateNotice.setWaitingForFirstWords(true);
  lateNotice.setWaitingForFirstWords(false);
  answerStatus(true);
  await settle();
  assert.equal(transcriptElement.classList.contains(MODEL_LOADING_CLASS), false, "a late answer must not mark a finished wait");
});

const WAITING_ANSWER = new TranscriptMessage({ id: "a", role: ChatRole.ASSISTANT, markdown: "", reasoning: "", state: ChatMessageState.STREAMING });
const TALKING_ANSWER = new TranscriptMessage({ id: "a", role: ChatRole.ASSISTANT, markdown: "Hello", reasoning: "", state: ChatMessageState.STREAMING });
const escapeHtml = (text: string): string => text;

test("should_carry_loading_words_beside_the_thinking_words_only_before_the_first_words", () => {
  const astro = new AstroMessageChrome(new TranscriptPersona({ isAstroShown: true, nickname: "Maya" }), escapeHtml);
  const classic = new ClassicMessageChrome(escapeHtml);

  assert.match(astro.assistantHeaderHtml(WAITING_ANSWER), /<span class="message__state-words">thinking…<\/span><span class="message__loading-words">loading the model…<\/span>/);
  assert.match(classic.assistantHeaderHtml(WAITING_ANSWER), /<span class="message__state-words">generating<\/span><span class="message__loading-words">loading the model…<\/span>/);
  for (const chrome of [astro, classic]) {
    assert.doesNotMatch(chrome.assistantHeaderHtml(TALKING_ANSWER), /message__loading-words/, "an answer already talking is not loading");
  }
});

/** Every value the transcript's loading class took during a turn, recorded
 * as it changed: the load is over long before a test could look. */
function recordLoadingClass(): boolean[] {
  const transcriptElement = document.querySelector("#transcript") as HTMLElement;
  const seenStates: boolean[] = [];
  new window.MutationObserver(() => seenStates.push(transcriptElement.classList.contains(MODEL_LOADING_CLASS))).observe(transcriptElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  return seenStates;
}

test("should_show_the_loading_words_in_the_real_page_while_the_engine_reports_a_load", async () => {
  const journey = bootJourney();
  await waitUntilReady();
  journey.daemon.modelLoadInFlight = { model_id: "Example-35B" };
  const seenStates = recordLoadingClass();

  sendUserMessage("Explain black holes in three sentences");
  await waitFor(() => journey.host.fileCount() === 1, "the finished turn to be saved");

  assert.ok(seenStates.includes(true), `the transcript was marked as loading: ${JSON.stringify(seenStates)}`);
  assert.equal(document.querySelector("#transcript")?.classList.contains(MODEL_LOADING_CLASS), false, "and cleared once words arrived");
}, { timeout: 30_000 });

test("should_never_mark_the_real_page_as_loading_when_the_engine_reports_none", async () => {
  const journey = bootJourney();
  await waitUntilReady();
  const seenStates = recordLoadingClass();

  sendUserMessage("Explain black holes in three sentences");
  await waitFor(() => journey.host.fileCount() === 1, "the finished turn to be saved");

  assert.ok(!seenStates.includes(true), JSON.stringify(seenStates));
}, { timeout: 30_000 });

// The DOM tests see the class, never the paint, so the stylesheet is held to
// swapping the words on the newest answer only.
test("should_swap_thinking_for_the_loading_words_on_the_newest_answer_only", () => {
  const transcriptStylesheet = fs.readFileSync(path.join(__dirname, "..", "src", "transcript.css"), "utf8").replace(/\s+/g, " ");

  assert.match(transcriptStylesheet, /\.message__loading-words \{ display: none; \}/);
  assert.match(transcriptStylesheet, /\.transcript--model-loading \.message--latest \.message__state-words \{ display: none; \}/);
  assert.match(transcriptStylesheet, /\.transcript--model-loading \.message--latest \.message__loading-words \{ display: inline; \}/);
});
