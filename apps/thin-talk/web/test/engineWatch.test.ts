import assert from "node:assert/strict";
import { test } from "node:test";
import type { AstroLibrary } from "../src/astro/AstroLibrary";
import { ENGINE_PATIENCE, EngineWatch } from "../src/astro/EngineWatch";

/** A library whose engine gives the scripted answers in turn, then repeats the last. */
function scriptedLibrary(answers: (boolean | null)[]): { library: AstroLibrary; askCount: () => number } {
  let asks = 0;
  const library = {
    hasChatModel: async (): Promise<boolean | null> => {
      const answer = answers[Math.min(asks, answers.length - 1)];
      asks += 1;
      return answer === undefined ? null : answer;
    },
  } as unknown as AstroLibrary;
  return { library, askCount: () => asks };
}

async function withQuickPatience(run: () => Promise<void>): Promise<void> {
  const savedAskEvery = ENGINE_PATIENCE.askEveryMilliseconds;
  ENGINE_PATIENCE.askEveryMilliseconds = 5;
  try {
    await run();
  } finally {
    ENGINE_PATIENCE.askEveryMilliseconds = savedAskEvery;
  }
}

async function until(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!predicate()) {
    if (Date.now() > deadline) {
      throw new Error("timed out");
    }
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
}

test("should_answer_at_once_and_ask_fresh_every_time", async () => {
  const { library, askCount } = scriptedLibrary([false, true]);
  const engineWatch = new EngineWatch(library);
  assert.equal(await engineWatch.ask(), false);
  assert.equal(await engineWatch.ask(), true, "a brain downloaded since is not missed");
  assert.equal(askCount(), 2);
});

test("should_keep_asking_a_silent_engine_while_someone_waits_and_tell_them_once", async () => {
  await withQuickPatience(async () => {
    const { library } = scriptedLibrary([null, null, true]);
    const engineWatch = new EngineWatch(library);
    const heardAnswers: boolean[] = [];
    engineWatch.onceAnswered((hasBrain) => heardAnswers.push(hasBrain));
    assert.equal(await engineWatch.ask(), null);
    await until(() => heardAnswers.length > 0);
    assert.deepEqual(heardAnswers, [true]);
    await engineWatch.ask();
    assert.deepEqual(heardAnswers, [true], "a listener hears one answer only");
  });
});

test("should_stop_asking_a_silent_engine_once_nobody_waits", async () => {
  await withQuickPatience(async () => {
    const { library, askCount } = scriptedLibrary([null]);
    const engineWatch = new EngineWatch(library);
    let wasCalled = false;
    const stopWaiting = engineWatch.onceAnswered(() => {
      wasCalled = true;
    });
    await until(() => askCount() >= 2);
    stopWaiting();
    const asksWhenStopped = askCount();
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.ok(askCount() <= asksWhenStopped + 1, "at most the ask already under way finishes");
    assert.equal(wasCalled, false);
  });
});

test("should_not_keep_asking_when_nobody_waits_for_the_answer", async () => {
  await withQuickPatience(async () => {
    const { library, askCount } = scriptedLibrary([null]);
    const engineWatch = new EngineWatch(library);
    assert.equal(await engineWatch.ask(), null);
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.equal(askCount(), 1);
  });
});

test("should_ask_once_when_asked_twice_at_the_same_time", async () => {
  const { library, askCount } = scriptedLibrary([true]);
  const engineWatch = new EngineWatch(library);
  assert.deepEqual(await Promise.all([engineWatch.ask(), engineWatch.ask()]), [true, true]);
  assert.equal(askCount(), 1);
});
