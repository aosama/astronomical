import assert from "node:assert/strict";
import { test } from "node:test";
import { GetStartedJourney, GetStartedStep } from "../src/astro/GetStartedJourney";

test("should_go_from_the_welcome_straight_to_finished_on_a_mac_that_has_a_chat_model", () => {
  const journey = new GetStartedJourney({ hasChatModel: true });
  assert.equal(journey.step, GetStartedStep.WELCOME);
  journey.begin();
  assert.equal(journey.step, GetStartedStep.FINISHED);
});

test("should_visit_the_download_step_only_when_no_chat_model_is_installed", () => {
  const journey = new GetStartedJourney({ hasChatModel: false });
  journey.begin();
  assert.equal(journey.step, GetStartedStep.DOWNLOAD);
  journey.leaveDownload();
  assert.equal(journey.step, GetStartedStep.FINISHED);
});

test("should_stay_on_the_download_step_when_the_model_arrives_so_the_screen_can_say_so", () => {
  const journey = new GetStartedJourney({ hasChatModel: false });
  journey.begin();
  journey.chatModelArrived();
  assert.equal(journey.step, GetStartedStep.DOWNLOAD);
  journey.leaveDownload();
  assert.equal(journey.step, GetStartedStep.FINISHED);
});

test("should_ignore_a_move_that_belongs_to_another_step", () => {
  const journey = new GetStartedJourney({ hasChatModel: false });
  journey.leaveDownload();
  journey.retryEngine();
  assert.equal(journey.step, GetStartedStep.WELCOME, "a stray late click must not skip the welcome");
  journey.begin();
  journey.begin();
  assert.equal(journey.step, GetStartedStep.DOWNLOAD, "a double click on Get started moves one step only");
});

test("should_skip_to_finished_from_any_step", () => {
  const welcomeJourney = new GetStartedJourney({ hasChatModel: false });
  welcomeJourney.skip();
  assert.equal(welcomeJourney.step, GetStartedStep.FINISHED);
  const downloadJourney = new GetStartedJourney({ hasChatModel: false });
  downloadJourney.begin();
  downloadJourney.skip();
  assert.equal(downloadJourney.step, GetStartedStep.FINISHED);
  const waitingJourney = new GetStartedJourney({ hasChatModel: null });
  waitingJourney.begin();
  waitingJourney.skip();
  assert.equal(waitingJourney.step, GetStartedStep.FINISHED);
});

test("should_wait_for_a_silent_engine_after_the_welcome_and_route_on_its_answer", () => {
  for (const [hasChatModel, expectedStep] of [[true, GetStartedStep.FINISHED], [false, GetStartedStep.DOWNLOAD]] as const) {
    const journey = new GetStartedJourney({ hasChatModel: null });
    journey.begin();
    assert.equal(journey.step, GetStartedStep.ENGINE_STARTING);
    journey.engineAnswered(hasChatModel);
    assert.equal(journey.step, expectedStep);
  }
});

test("should_record_an_early_engine_answer_without_moving_the_journey", () => {
  const journey = new GetStartedJourney({ hasChatModel: null });
  journey.engineAnswered(false);
  assert.equal(journey.step, GetStartedStep.WELCOME, "an answer during the welcome changes nothing on screen");
  journey.begin();
  assert.equal(journey.step, GetStartedStep.DOWNLOAD, "no wait once the answer is known");
});

test("should_say_the_engine_stayed_silent_and_wait_again_after_a_retry", () => {
  const journey = new GetStartedJourney({ hasChatModel: null });
  journey.engineStayedSilent();
  assert.equal(journey.step, GetStartedStep.WELCOME, "silence only matters while waiting");
  journey.begin();
  journey.engineStayedSilent();
  assert.equal(journey.step, GetStartedStep.ENGINE_SILENT);
  journey.retryEngine();
  assert.equal(journey.step, GetStartedStep.ENGINE_STARTING);
  journey.engineStayedSilent();
  journey.engineAnswered(false);
  assert.equal(journey.step, GetStartedStep.DOWNLOAD, "a late answer still moves a silent journey on");
});

test("should_finish_from_the_download_step_when_a_chat_model_turns_up_from_elsewhere", () => {
  const journey = new GetStartedJourney({ hasChatModel: false });
  journey.begin();
  journey.chatModelFoundElsewhere();
  assert.equal(journey.step, GetStartedStep.FINISHED);
});

test("should_not_leave_the_welcome_when_a_chat_model_turns_up_before_the_person_began", () => {
  const journey = new GetStartedJourney({ hasChatModel: false });
  journey.chatModelFoundElsewhere();
  assert.equal(journey.step, GetStartedStep.WELCOME, "the welcome is still the first screen");
  journey.begin();
  assert.equal(journey.step, GetStartedStep.FINISHED, "and with a model here there is no download step");
});
