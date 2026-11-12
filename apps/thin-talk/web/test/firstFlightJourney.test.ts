import assert from "node:assert/strict";
import { test } from "node:test";
import { FirstFlightJourney, FirstFlightStep } from "../src/astro/FirstFlightJourney";

test("should_go_from_the_welcome_straight_to_finished_on_a_mac_that_has_a_chat_model", () => {
  const journey = new FirstFlightJourney({ hasChatModel: true });
  assert.equal(journey.step, FirstFlightStep.WELCOME);
  journey.begin();
  assert.equal(journey.step, FirstFlightStep.FINISHED);
});

test("should_visit_the_download_step_only_when_no_chat_model_is_installed", () => {
  const journey = new FirstFlightJourney({ hasChatModel: false });
  journey.begin();
  assert.equal(journey.step, FirstFlightStep.DOWNLOAD);
  journey.leaveDownload();
  assert.equal(journey.step, FirstFlightStep.FINISHED);
});

test("should_stay_on_the_download_step_when_the_model_arrives_so_the_screen_can_say_so", () => {
  const journey = new FirstFlightJourney({ hasChatModel: false });
  journey.begin();
  journey.chatModelArrived();
  assert.equal(journey.step, FirstFlightStep.DOWNLOAD);
  journey.leaveDownload();
  assert.equal(journey.step, FirstFlightStep.FINISHED);
});

test("should_ignore_a_move_that_belongs_to_another_step", () => {
  const journey = new FirstFlightJourney({ hasChatModel: false });
  journey.leaveDownload();
  journey.retryEngine();
  assert.equal(journey.step, FirstFlightStep.WELCOME, "a stray late click must not skip the welcome");
  journey.begin();
  journey.begin();
  assert.equal(journey.step, FirstFlightStep.DOWNLOAD, "a double click on Get started moves one step only");
});

test("should_skip_to_finished_from_any_step", () => {
  const welcomeJourney = new FirstFlightJourney({ hasChatModel: false });
  welcomeJourney.skip();
  assert.equal(welcomeJourney.step, FirstFlightStep.FINISHED);
  const downloadJourney = new FirstFlightJourney({ hasChatModel: false });
  downloadJourney.begin();
  downloadJourney.skip();
  assert.equal(downloadJourney.step, FirstFlightStep.FINISHED);
  const waitingJourney = new FirstFlightJourney({ hasChatModel: null });
  waitingJourney.begin();
  waitingJourney.skip();
  assert.equal(waitingJourney.step, FirstFlightStep.FINISHED);
});

test("should_wait_for_a_silent_engine_after_the_welcome_and_route_on_its_answer", () => {
  for (const [hasChatModel, expectedStep] of [[true, FirstFlightStep.FINISHED], [false, FirstFlightStep.DOWNLOAD]] as const) {
    const journey = new FirstFlightJourney({ hasChatModel: null });
    journey.begin();
    assert.equal(journey.step, FirstFlightStep.ENGINE_STARTING);
    journey.engineAnswered(hasChatModel);
    assert.equal(journey.step, expectedStep);
  }
});

test("should_record_an_early_engine_answer_without_moving_the_journey", () => {
  const journey = new FirstFlightJourney({ hasChatModel: null });
  journey.engineAnswered(false);
  assert.equal(journey.step, FirstFlightStep.WELCOME, "an answer during the welcome changes nothing on screen");
  journey.begin();
  assert.equal(journey.step, FirstFlightStep.DOWNLOAD, "no wait once the answer is known");
});

test("should_say_the_engine_stayed_silent_and_wait_again_after_a_retry", () => {
  const journey = new FirstFlightJourney({ hasChatModel: null });
  journey.engineStayedSilent();
  assert.equal(journey.step, FirstFlightStep.WELCOME, "silence only matters while waiting");
  journey.begin();
  journey.engineStayedSilent();
  assert.equal(journey.step, FirstFlightStep.ENGINE_SILENT);
  journey.retryEngine();
  assert.equal(journey.step, FirstFlightStep.ENGINE_STARTING);
  journey.engineStayedSilent();
  journey.engineAnswered(false);
  assert.equal(journey.step, FirstFlightStep.DOWNLOAD, "a late answer still moves a silent journey on");
});
