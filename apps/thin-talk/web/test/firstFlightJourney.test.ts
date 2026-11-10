import assert from "node:assert/strict";
import { test } from "node:test";
import { FirstFlightJourney, FirstFlightStep } from "../src/astro/FirstFlightJourney";

function journeyAtStyle(hasBrain: boolean): FirstFlightJourney {
  const journey = new FirstFlightJourney({ hasBrain });
  journey.greet();
  journey.giveName();
  return journey;
}

test("should_go_hello_name_style_skills_finished_on_a_mac_that_has_a_brain", () => {
  const journey = new FirstFlightJourney({ hasBrain: true });
  const visitedSteps = [journey.step];
  for (const move of [() => journey.greet(), () => journey.giveName(), () => journey.chooseStyle(), () => journey.finish()]) {
    move();
    visitedSteps.push(journey.step);
  }
  assert.deepEqual(visitedSteps, [FirstFlightStep.HELLO, FirstFlightStep.NAME, FirstFlightStep.STYLE, FirstFlightStep.SKILLS, FirstFlightStep.FINISHED]);
});

test("should_visit_the_wake_up_step_only_when_no_brain_is_installed", () => {
  const journey = journeyAtStyle(false);
  journey.chooseStyle();
  assert.equal(journey.step, FirstFlightStep.WAKE);
  journey.leaveWake();
  assert.equal(journey.step, FirstFlightStep.SKILLS);
});

test("should_stay_on_wake_up_when_the_brain_arrives_so_astro_can_celebrate", () => {
  const journey = journeyAtStyle(false);
  journey.chooseStyle();
  journey.brainArrived();
  assert.equal(journey.step, FirstFlightStep.WAKE);
  journey.leaveWake();
  assert.equal(journey.step, FirstFlightStep.SKILLS);
});

test("should_return_from_the_grown_up_note_to_the_style_choice", () => {
  const journey = journeyAtStyle(true);
  journey.visitGrownUpNote();
  assert.equal(journey.step, FirstFlightStep.GROWN_UP);
  journey.leaveGrownUpNote();
  assert.equal(journey.step, FirstFlightStep.STYLE);
});

test("should_ignore_a_move_that_belongs_to_another_step", () => {
  const journey = new FirstFlightJourney({ hasBrain: true });
  journey.chooseStyle();
  journey.finish();
  journey.leaveWake();
  assert.equal(journey.step, FirstFlightStep.HELLO, "a stray late click must not skip screens");
  journey.greet();
  journey.greet();
  assert.equal(journey.step, FirstFlightStep.NAME, "a double click on Hi Astro moves one step only");
});

test("should_skip_to_finished_from_any_step", () => {
  for (const hasBrain of [true, false]) {
    const journey = journeyAtStyle(hasBrain);
    journey.skip();
    assert.equal(journey.step, FirstFlightStep.FINISHED);
  }
  const freshJourney = new FirstFlightJourney({ hasBrain: false });
  freshJourney.skip();
  assert.equal(freshJourney.step, FirstFlightStep.FINISHED);
});

test("should_wait_for_a_silent_engine_after_the_style_and_route_on_its_answer", () => {
  for (const [hasBrain, expectedStep] of [[true, FirstFlightStep.SKILLS], [false, FirstFlightStep.WAKE]] as const) {
    const journey = new FirstFlightJourney({ hasBrain: null });
    journey.greet();
    journey.giveName();
    journey.chooseStyle();
    assert.equal(journey.step, FirstFlightStep.ENGINE_STARTING);
    journey.engineAnswered(hasBrain);
    assert.equal(journey.step, expectedStep);
  }
});

test("should_record_an_early_engine_answer_without_moving_the_flight", () => {
  const journey = new FirstFlightJourney({ hasBrain: null });
  journey.greet();
  journey.engineAnswered(false);
  assert.equal(journey.step, FirstFlightStep.NAME, "an answer during the name step changes nothing on screen");
  journey.giveName();
  journey.chooseStyle();
  assert.equal(journey.step, FirstFlightStep.WAKE, "no wait once the answer is known");
});

test("should_say_the_engine_stayed_silent_and_wait_again_after_a_retry", () => {
  const journey = new FirstFlightJourney({ hasBrain: null });
  journey.engineStayedSilent();
  assert.equal(journey.step, FirstFlightStep.HELLO, "silence only matters while waiting");
  journey.greet();
  journey.giveName();
  journey.chooseStyle();
  journey.engineStayedSilent();
  assert.equal(journey.step, FirstFlightStep.ENGINE_SILENT);
  journey.retryEngine();
  assert.equal(journey.step, FirstFlightStep.ENGINE_STARTING);
  journey.engineStayedSilent();
  journey.engineAnswered(true);
  assert.equal(journey.step, FirstFlightStep.SKILLS, "a late answer still moves a silent flight on");
});
