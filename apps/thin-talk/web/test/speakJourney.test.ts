import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";

// A boot builds a fresh page, so the fake Web Audio is installed after it.
// The Speak button end to end against the real wiring: the page, the daemon's
// speech endpoint and Web Audio are faked, everything between is production.

class FakeSource {
  public buffer: { duration: number } | null = null;
  public onended: (() => void) | null = null;
  public stopped = false;
  public connect(): void {}
  public start(): void {}
  public stop(): void { this.stopped = true; }
}

class FakeAudioContext {
  public static readonly sources: FakeSource[] = [];
  public currentTime = 0;
  public state = "suspended";
  public readonly destination = {};
  public async resume(): Promise<void> { this.state = "running"; }
  public async suspend(): Promise<void> { this.state = "suspended"; }
  public async decodeAudioData(): Promise<{ duration: number }> { return { duration: 2 }; }
  public createBufferSource(): FakeSource {
    const source = new FakeSource();
    FakeAudioContext.sources.push(source);
    return source;
  }
}

function installFakeAudio(): void {
  FakeAudioContext.sources.length = 0;
  (window as unknown as { AudioContext: unknown }).AudioContext = FakeAudioContext;
}

function speakButton(): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>('#transcript button[data-action="speak-answer"]');
}

function speechRequests(journey: ReturnType<typeof bootJourney>): string[] {
  return journey.daemon.requests.filter((request) => request.url.endsWith("/v1/audio/speech")).map((request) => request.body ?? "");
}

async function askAndWaitForAnswer(question: string): Promise<void> {
  sendUserMessage(question);
  await waitFor(() => speakButton() !== null, "the finished answer to offer Speak");
}

test("should_speak_a_finished_answer_and_rest_again_when_it_ends", async () => {
  const journey = bootJourney();
  installFakeAudio();
  await waitUntilReady();
  await askAndWaitForAnswer("Summarize the prologue.");
  assert.equal(speakButton()?.textContent, "Speak");

  speakButton()?.dispatchEvent(new window.Event("click", { bubbles: true }));
  await waitFor(() => speakButton()?.textContent === "Stop", "the reply to start sounding");
  const spoken = JSON.parse(speechRequests(journey)[0]) as Record<string, unknown>;
  assert.match(String(spoken["input"]), /Hello from the fake daemon\./);
  assert.equal(spoken["model"], undefined, "the daemon picks the voice model");

  for (const source of FakeAudioContext.sources) {
    source.onended?.();
  }
  await waitFor(() => speakButton()?.textContent === "Speak", "the button to rest when the reply ends");
}, { timeout: 30_000 });

test("should_stop_the_sound_when_the_button_is_pressed_again", async () => {
  bootJourney();
  installFakeAudio();
  await waitUntilReady();
  await askAndWaitForAnswer("Summarize the prologue.");
  speakButton()?.dispatchEvent(new window.Event("click", { bubbles: true }));
  await waitFor(() => speakButton()?.textContent === "Stop", "the reply to start sounding");
  speakButton()?.dispatchEvent(new window.Event("click", { bubbles: true }));
  await waitFor(() => speakButton()?.textContent === "Speak", "Stop to end the reading");
  assert.ok(FakeAudioContext.sources.every((source) => source.stopped), "every scheduled piece was stopped");
}, { timeout: 30_000 });

test("should_end_the_reading_when_a_new_question_is_sent", async () => {
  bootJourney();
  installFakeAudio();
  await waitUntilReady();
  await askAndWaitForAnswer("Summarize the prologue.");
  speakButton()?.dispatchEvent(new window.Event("click", { bubbles: true }));
  await waitFor(() => speakButton()?.textContent === "Stop", "the reply to start sounding");
  sendUserMessage("And the epilogue?");
  await waitFor(() => FakeAudioContext.sources.every((source) => source.stopped), "the sound to stop at the send");
  await waitFor(() => document.querySelectorAll('#transcript button[data-action="speak-answer"]').length === 2, "the second answer to arrive");
  assert.ok(
    [...document.querySelectorAll('#transcript button[data-action="speak-answer"]')].every((button) => button.textContent === "Speak"),
    "no reply is left showing Stop",
  );
}, { timeout: 30_000 });

test("should_offer_no_speak_button_for_an_answer_with_nothing_to_hear", async () => {
  const codeOnlyReply = [
    'data: {"choices":[{"delta":{"content":"```py\\nprint(1)\\n```"}}]}',
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    "data: [DONE]",
  ];
  bootJourney(undefined, { chatReplySequence: [codeOnlyReply] });
  installFakeAudio();
  await waitUntilReady();
  sendUserMessage("Show me code.");
  await waitFor(() => document.querySelector('#transcript article.message--assistant button[data-action="copy"]') !== null, "the finished answer");
  assert.equal(speakButton(), null);
}, { timeout: 30_000 });

test("should_offer_a_voice_when_no_voice_model_is_installed", async () => {
  const journey = bootJourney(undefined, { speechRefusal: { status: 400, code: "no_speech_model" } });
  installFakeAudio();
  await waitUntilReady();
  await askAndWaitForAnswer("Summarize the prologue.");
  speakButton()?.dispatchEvent(new window.Event("click", { bubbles: true }));
  await waitFor(() => speakButton()?.textContent === "Get a voice…", "the button to offer a voice");
  assert.equal(speechRequests(journey).length, 1);
  assert.equal(FakeAudioContext.sources.length, 0, "nothing was played");
}, { timeout: 30_000 });

test("should_send_only_the_words_of_a_reply_with_a_list_code_a_link_and_an_emoji", async () => {
  const markdown = "Here is the plan 🎉\n\n- First step\n- Second step\n\n```py\nprint(1)\n```\n\nSee [the docs](https://example.com/docs) now.";
  const mixedReply = [
    `data: ${JSON.stringify({ choices: [{ delta: { content: markdown } }] })}`,
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    "data: [DONE]",
  ];
  const journey = bootJourney(undefined, { chatReplySequence: [mixedReply] });
  installFakeAudio();
  await waitUntilReady();
  await askAndWaitForAnswer("Plan it.");
  speakButton()?.dispatchEvent(new window.Event("click", { bubbles: true }));
  await waitFor(() => speechRequests(journey).length === 1, "the speech request to be sent");
  const spoken = JSON.parse(speechRequests(journey)[0]) as Record<string, unknown>;
  assert.equal(spoken["input"], "Here is the plan. First step. Second step. See the docs now.");
}, { timeout: 30_000 });

test("should_grey_an_earlier_replys_button_while_a_new_answer_is_being_written", async () => {
  bootJourney();
  installFakeAudio();
  await waitUntilReady();
  await askAndWaitForAnswer("Summarize the prologue.");
  sendUserMessage("And the epilogue?");
  assert.equal(speakButton()?.disabled, true, "the button waits while the engine is busy");
  await waitFor(() => document.querySelectorAll('#transcript button[data-action="speak-answer"]').length === 2, "the second answer to arrive");
  assert.ok(
    [...document.querySelectorAll<HTMLButtonElement>('#transcript button[data-action="speak-answer"]')].every((button) => !button.disabled),
    "both buttons work again once the answer is finished",
  );
}, { timeout: 30_000 });
