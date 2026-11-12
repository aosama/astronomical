import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { SpeechAudioOutput } from "../src/speech/SpeechAudioOutput";
import type { SpeechAudioBuffer, SpeechAudioCallbacks, SpeechAudioContext, SpeechAudioSource } from "../src/speech/SpeechAudioOutput";

class FakeSource implements SpeechAudioSource {
  public buffer: SpeechAudioBuffer | null = null;
  public onended: (() => void) | null = null;
  public startedAt: number | null = null;
  public stopped = false;
  public connect(): void {}
  public start(when: number): void { this.startedAt = when; }
  public stop(): void { this.stopped = true; }
}

class FakeContext implements SpeechAudioContext {
  public currentTime = 0;
  public state = "suspended";
  public suspendCalls = 0;
  public resumeFails = false;
  public readonly destination = {};
  public readonly sources: FakeSource[] = [];
  public async resume(): Promise<void> {
    if (this.resumeFails) {
      throw new Error("the page may not make sound");
    }
    this.state = "running";
  }
  public async suspend(): Promise<void> { this.suspendCalls += 1; this.state = "suspended"; }
  public async decodeAudioData(audioBytes: ArrayBuffer): Promise<SpeechAudioBuffer> {
    if (audioBytes.byteLength === 0) {
      throw new Error("not audio");
    }
    return { duration: audioBytes.byteLength };
  }
  public createBufferSource(): SpeechAudioSource {
    const source = new FakeSource();
    this.sources.push(source);
    return source;
  }
}

function callbacks(overrides: Partial<SpeechAudioCallbacks>): SpeechAudioCallbacks {
  return { onFirstSound: () => undefined, onFinished: () => undefined, onFailed: () => undefined, ...overrides };
}

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test("should_schedule_each_piece_exactly_where_the_previous_one_ends", async () => {
  const context = new FakeContext();
  const output = new SpeechAudioOutput(() => context);
  let firstSounds = 0;
  output.wake(callbacks({ onFirstSound: () => { firstSounds += 1; } }));
  output.enqueue(new ArrayBuffer(3));
  output.enqueue(new ArrayBuffer(4));
  await settle();
  assert.equal(context.state, "running", "the click woke the output");
  assert.equal(context.sources.length, 2);
  const firstStart = context.sources[0].startedAt as number;
  assert.equal(context.sources[1].startedAt, firstStart + 3, "no gap and no overlap");
  assert.equal(firstSounds, 1, "first sound is reported once");
});

test("should_finish_only_after_every_piece_has_played_and_none_are_coming", async () => {
  const context = new FakeContext();
  const output = new SpeechAudioOutput(() => context);
  let finished = 0;
  output.wake(callbacks({ onFinished: () => { finished += 1; } }));
  output.enqueue(new ArrayBuffer(2));
  await settle();
  output.markComplete();
  assert.equal(finished, 0, "a piece is still playing");
  context.sources[0].onended?.();
  assert.equal(finished, 1);
});

test("should_silence_at_once_and_ignore_pieces_still_being_decoded", async () => {
  const context = new FakeContext();
  const output = new SpeechAudioOutput(() => context);
  output.wake(callbacks({}));
  output.enqueue(new ArrayBuffer(2));
  await settle();
  output.enqueue(new ArrayBuffer(5));
  output.silence();
  await settle();
  assert.ok(context.sources[0].stopped, "the playing piece is stopped");
  assert.equal(context.sources.length, 1, "the piece in flight was dropped");
  assert.equal(context.state, "suspended");
});

test("should_skip_a_piece_that_cannot_be_decoded_and_play_the_rest", async () => {
  const context = new FakeContext();
  const output = new SpeechAudioOutput(() => context);
  output.wake(callbacks({}));
  output.enqueue(new ArrayBuffer(0));
  output.enqueue(new ArrayBuffer(2));
  await settle();
  assert.equal(context.sources.length, 1);
});

test("should_finish_an_empty_reply_without_sound", async () => {
  const context = new FakeContext();
  const output = new SpeechAudioOutput(() => context);
  let finished = 0;
  output.wake(callbacks({ onFinished: () => { finished += 1; } }));
  output.markComplete();
  assert.equal(finished, 1);
});

test("should_have_the_third_piece_ready_before_the_second_ends_when_pieces_last_a_minute", async () => {
  const context = new FakeContext();
  const output = new SpeechAudioOutput(() => context);
  output.wake(callbacks({}));
  output.enqueue(new ArrayBuffer(10));
  output.enqueue(new ArrayBuffer(60));
  output.enqueue(new ArrayBuffer(60));
  await settle();
  assert.equal(context.sources.length, 2, "only the playing piece and the one behind it are decoded");
  const firstStart = context.sources[0].startedAt as number;
  context.currentTime = firstStart + 10;
  context.sources[0].onended?.();
  await settle();
  assert.equal(context.sources.length, 3, "the third is decoded as soon as the first ends");
  assert.equal(context.sources[2].startedAt, firstStart + 70, "it starts exactly where the second ends");
});

test("should_report_failure_and_play_nothing_when_the_context_refuses_to_run", async () => {
  const context = new FakeContext();
  context.resumeFails = true;
  const output = new SpeechAudioOutput(() => context);
  let failures = 0;
  let firstSounds = 0;
  output.wake(callbacks({ onFailed: () => { failures += 1; }, onFirstSound: () => { firstSounds += 1; } }));
  output.enqueue(new ArrayBuffer(3));
  await settle();
  assert.equal(failures, 1);
  assert.equal(firstSounds, 0, "the reply is never reported as playing");
  assert.equal(context.sources.length, 0);
});

test("should_suspend_on_silence_even_while_the_context_still_reads_suspended", async () => {
  const context = new FakeContext();
  const output = new SpeechAudioOutput(() => context);
  output.wake(callbacks({}));
  const suspendCallsBeforeStop = context.suspendCalls;
  context.state = "suspended";
  output.silence();
  assert.equal(context.suspendCalls, suspendCallsBeforeStop + 1);
});

test("should_report_failure_when_a_context_never_answers_but_only_after_its_timeout", async () => {
  const context = new FakeContext();
  context.resume = () => new Promise<void>(() => undefined);
  const output = new SpeechAudioOutput(() => context);
  let failures = 0;
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    output.wake(callbacks({ onFailed: () => { failures += 1; } }));
    output.enqueue(new ArrayBuffer(3));
    await Promise.resolve();
    assert.equal(failures, 0, "still waiting for the context");
    mock.timers.tick(10_000);
    await Promise.resolve();
    await Promise.resolve();
  } finally {
    mock.timers.reset();
  }
  assert.equal(failures, 1);
});

test("should_throw_from_wake_when_no_audio_context_can_be_made", () => {
  const output = new SpeechAudioOutput(() => { throw new Error("no Web Audio"); });
  assert.throws(() => output.wake(callbacks({})), /no Web Audio/);
});
