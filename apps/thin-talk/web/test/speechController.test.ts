import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { SpeechController } from "../src/speech/SpeechController";
import { SpeakPhase } from "../src/speech/SpeakPhase";
import type { SpeechState } from "../src/speech/SpeechState";
import type { ConversationSnapshot } from "../src/speech/SpeechController";
import { SpeechFetchFailure } from "../src/speech/SpeechFetchFailure";
import { SpeechFetchFailureKind } from "../src/speech/SpeechFetchFailureKind";
import type { SpeechPieceRequest } from "../src/speech/SpeechClient";
import type { SpeechAudioCallbacks, SpeechAudioOutput } from "../src/speech/SpeechAudioOutput";

class FakeOutput {
  public readonly enqueued: ArrayBuffer[] = [];
  public completed = 0;
  public silenced = 0;
  public onFirstSound: () => void = () => undefined;
  public onFinished: () => void = () => undefined;
  public onFailed: () => void = () => undefined;
  public wakeThrows = false;
  public wake(callbacks: SpeechAudioCallbacks): void {
    if (this.wakeThrows) {
      throw new Error("no Web Audio");
    }
    this.onFirstSound = callbacks.onFirstSound;
    this.onFinished = callbacks.onFinished;
    this.onFailed = callbacks.onFailed;
  }
  public enqueue(bytes: ArrayBuffer): void { this.enqueued.push(bytes); }
  public markComplete(): void { this.completed += 1; }
  public silence(): void { this.silenced += 1; }
}

const REPLY = "Two households, both alike in dignity. In fair Verona, where we lay our scene.";

function build(fetchSpokenPiece: (request: SpeechPieceRequest) => Promise<ArrayBuffer>, voiceModels: string[] = []) {
  const output = new FakeOutput();
  let modelsPageOpened = 0;
  const controller = new SpeechController({
    output: output as unknown as SpeechAudioOutput,
    fetchSpokenPiece,
    listVoiceModels: async () => voiceModels,
    openModelsPage: () => { modelsPageOpened += 1; },
  });
  const states: SpeechState[] = [];
  controller.subscribe((state) => states.push(state));
  return { controller, output, states, modelsPageOpened: () => modelsPageOpened };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test("should_prepare_then_play_and_rest_when_the_reply_ends", async () => {
  const { controller, output, states } = build(async () => new ArrayBuffer(4));
  controller.press("m1", REPLY);
  assert.equal(controller.state.phase, SpeakPhase.PREPARING);
  await settle();
  assert.ok(output.enqueued.length >= 1 && output.completed === 1);
  output.onFirstSound();
  assert.equal(controller.state.phase, SpeakPhase.PLAYING);
  output.onFinished();
  assert.equal(controller.state.phase, SpeakPhase.RESTING);
  assert.deepEqual(states.map((state) => state.phase), [SpeakPhase.PREPARING, SpeakPhase.PLAYING, SpeakPhase.RESTING]);
});

test("should_use_one_seed_for_every_piece_of_a_reply", async () => {
  const seeds: number[] = [];
  const longReply = Array(30).fill("The quality of mercy is not strained, it droppeth as the gentle rain.").join(" ");
  const { controller } = build(async (request) => { seeds.push(request.seed); return new ArrayBuffer(2); });
  controller.press("m1", longReply);
  await settle();
  assert.ok(seeds.length > 1);
  assert.ok(seeds.every((seed) => seed === seeds[0]));
});

test("should_stop_when_the_playing_reply_is_pressed_again", async () => {
  const { controller, output } = build(async () => new ArrayBuffer(4));
  controller.press("m1", REPLY);
  await settle();
  output.onFirstSound();
  controller.press("m1", REPLY);
  assert.equal(controller.state.phase, SpeakPhase.RESTING);
  assert.ok(output.silenced >= 1);
});

test("should_move_the_reading_to_another_reply_when_its_button_is_pressed", async () => {
  const { controller, output } = build(async () => new ArrayBuffer(4));
  controller.press("m1", REPLY);
  await settle();
  output.onFirstSound();
  controller.press("m2", REPLY);
  assert.deepEqual(controller.state, { messageID: "m2", phase: SpeakPhase.PREPARING });
});

test("should_offer_a_voice_when_none_is_installed_and_open_models_on_the_next_press", async () => {
  const { controller, modelsPageOpened } = build(async () => {
    throw new SpeechFetchFailure(SpeechFetchFailureKind.NO_VOICE, "none");
  });
  controller.press("m1", REPLY);
  await settle();
  assert.equal(controller.state.phase, SpeakPhase.NO_VOICE);
  controller.press("m1", REPLY);
  assert.equal(modelsPageOpened(), 1);
  assert.equal(controller.state.phase, SpeakPhase.RESTING);
});

test("should_name_kokoro_when_several_voices_are_installed_and_none_is_the_default", async () => {
  const models: Array<string | null> = [];
  const { controller, output } = build(async (request) => {
    models.push(request.model);
    if (request.model === null) {
      throw new SpeechFetchFailure(SpeechFetchFailureKind.AMBIGUOUS_VOICE, "several");
    }
    return new ArrayBuffer(2);
  }, ["phonon-31m-en", "Kokoro-82M-bf16"]);
  controller.press("m1", REPLY);
  await settle();
  assert.deepEqual(models.slice(0, 2), [null, "Kokoro-82M-bf16"]);
  assert.ok(output.enqueued.length >= 1);
});

test("should_show_a_passing_busy_note_and_then_rest", async () => {
  const { controller } = build(async () => { throw new SpeechFetchFailure(SpeechFetchFailureKind.BUSY, "busy"); });
  controller.press("m1", REPLY);
  await settle();
  assert.equal(controller.state.phase, SpeakPhase.BUSY);
  controller.stop();
  assert.equal(controller.state.phase, SpeakPhase.RESTING);
});

test("should_play_what_was_fetched_even_when_a_later_piece_fails_before_the_first_sound", async () => {
  let fetched = 0;
  const longReply = Array(30).fill("The quality of mercy is not strained, it droppeth as the gentle rain.").join(" ");
  const { controller, output } = build(async () => {
    fetched += 1;
    if (fetched === 2) {
      throw new SpeechFetchFailure(SpeechFetchFailureKind.FAILED, "boom");
    }
    return new ArrayBuffer(2);
  });
  controller.press("m1", longReply);
  await settle();
  output.onFirstSound();
  assert.equal(output.enqueued.length, 1);
  assert.equal(output.completed, 1, "the audio already fetched plays to its end");
  assert.equal(controller.state.phase, SpeakPhase.PLAYING);
});

function conversation(overrides: Partial<ConversationSnapshot>): ConversationSnapshot {
  return { isGenerating: false, activeSessionID: "chat-1", shownMessageIDs: new Set(["m1", "m2"]), ...overrides };
}

test("should_keep_reading_while_the_conversation_stays_as_it_is", () => {
  const { controller } = build(async () => new ArrayBuffer(2));
  controller.followConversation(conversation({}));
  controller.press("m1", REPLY);
  controller.followConversation(conversation({}));
  assert.equal(controller.state.phase, SpeakPhase.PREPARING);
});

test("should_stop_reading_when_its_reply_leaves_the_screen", () => {
  const { controller } = build(async () => new ArrayBuffer(2));
  controller.followConversation(conversation({}));
  controller.press("m1", REPLY);
  controller.followConversation(conversation({ shownMessageIDs: new Set(["m2"]) }));
  assert.equal(controller.state.phase, SpeakPhase.RESTING);
});

test("should_stop_reading_when_an_answer_starts_being_written", () => {
  const { controller } = build(async () => new ArrayBuffer(2));
  controller.followConversation(conversation({}));
  controller.press("m1", REPLY);
  controller.followConversation(conversation({ isGenerating: true }));
  assert.equal(controller.state.phase, SpeakPhase.RESTING);
});

test("should_stop_reading_when_another_chat_opens_even_if_message_ids_repeat", () => {
  const { controller } = build(async () => new ArrayBuffer(2));
  controller.followConversation(conversation({}));
  controller.press("m1", REPLY);
  controller.followConversation(conversation({ activeSessionID: "chat-2" }));
  assert.equal(controller.state.phase, SpeakPhase.RESTING);
});

test("should_publish_nothing_when_the_conversation_changes_and_nothing_is_read", () => {
  const { controller, states } = build(async () => new ArrayBuffer(2));
  controller.followConversation(conversation({}));
  controller.followConversation(conversation({ isGenerating: true, activeSessionID: "chat-2" }));
  assert.equal(states.length, 0);
});

test("should_say_so_instead_of_staying_silent_when_no_audio_context_can_be_made", () => {
  const { controller, output } = build(async () => new ArrayBuffer(2));
  output.wakeThrows = true;
  controller.press("m1", REPLY);
  assert.deepEqual(controller.state, { messageID: "m1", phase: SpeakPhase.FAILED });
});

test("should_note_a_failure_when_the_audio_output_cannot_run", async () => {
  const { controller, output } = build(async () => new ArrayBuffer(2));
  controller.press("m1", REPLY);
  output.onFailed();
  assert.equal(controller.state.phase, SpeakPhase.FAILED);
});

test("should_let_a_passing_note_expire_and_the_button_rest", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const { controller, output } = build(async () => new ArrayBuffer(2));
    controller.press("m1", REPLY);
    output.onFailed();
    assert.equal(controller.state.phase, SpeakPhase.FAILED);
    mock.timers.tick(2500);
    assert.equal(controller.state.phase, SpeakPhase.RESTING);
  } finally {
    mock.timers.reset();
  }
});

test("should_remember_the_named_voice_for_later_presses_and_forget_it_when_it_fails", async () => {
  const models: Array<string | null> = [];
  let kokoroInstalled = true;
  const { controller } = build(async (request) => {
    models.push(request.model);
    if (request.model === null) {
      throw new SpeechFetchFailure(SpeechFetchFailureKind.AMBIGUOUS_VOICE, "several");
    }
    if (!kokoroInstalled) {
      throw new SpeechFetchFailure(SpeechFetchFailureKind.FAILED, "model_not_found");
    }
    return new ArrayBuffer(2);
  }, ["phonon-31m-en", "Kokoro-82M-bf16"]);
  controller.press("m1", REPLY);
  await settle();
  controller.stop();
  controller.press("m1", REPLY);
  await settle();
  assert.equal(models[2], "Kokoro-82M-bf16", "the second press names the remembered voice at once");
  controller.stop();
  kokoroInstalled = false;
  controller.press("m1", REPLY);
  await settle();
  controller.stop();
  controller.press("m1", REPLY);
  await settle();
  assert.equal(models[models.length - 2], null, "after a failure the model is not named again blindly");
});

test("should_fail_when_voices_are_ambiguous_and_none_can_be_listed", async () => {
  const { controller } = build(async () => {
    throw new SpeechFetchFailure(SpeechFetchFailureKind.AMBIGUOUS_VOICE, "several");
  }, []);
  controller.press("m1", REPLY);
  await settle();
  assert.equal(controller.state.phase, SpeakPhase.FAILED);
});
