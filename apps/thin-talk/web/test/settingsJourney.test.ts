import assert from "node:assert/strict";
import { test } from "node:test";
import { SettingsJourney, SettingsStep } from "../src/astro/SettingsJourney";

function openJourney(): SettingsJourney {
  const journey = new SettingsJourney();
  journey.open();
  return journey;
}

test("should_start_closed_and_open_on_the_page", () => {
  const journey = new SettingsJourney();
  assert.equal(journey.step, SettingsStep.CLOSED);
  journey.open();
  assert.equal(journey.step, SettingsStep.PAGE);
});

test("should_open_each_answer_on_its_own_and_come_back_to_the_page", () => {
  for (const [move, expectedStep] of [
    [(journey: SettingsJourney) => journey.editName(), SettingsStep.NAME],
    [(journey: SettingsJourney) => journey.readPrivacyNote(), SettingsStep.PRIVACY],
  ] as const) {
    const journey = openJourney();
    move(journey);
    assert.equal(journey.step, expectedStep);
    journey.backToPage();
    assert.equal(journey.step, SettingsStep.PAGE, `${expectedStep} returns to the page`);
  }
});

test("should_ignore_a_move_that_belongs_to_another_step", () => {
  const closedJourney = new SettingsJourney();
  closedJourney.editName();
  closedJourney.backToPage();
  assert.equal(closedJourney.step, SettingsStep.CLOSED, "nothing opens a closed page but open()");
  const journey = openJourney();
  journey.editName();
  journey.readPrivacyNote();
  assert.equal(journey.step, SettingsStep.NAME, "a late click on another row does not jump screens");
});

test("should_back_out_one_level_with_escape", () => {
  const journey = openJourney();
  journey.readPrivacyNote();
  journey.escape();
  assert.equal(journey.step, SettingsStep.PAGE);
  journey.escape();
  assert.equal(journey.step, SettingsStep.CLOSED);
  journey.escape();
  assert.equal(journey.step, SettingsStep.CLOSED, "escape on a closed page stays closed");
});

test("should_reopen_on_the_page_whatever_was_on_show_last_time", () => {
  const journey = openJourney();
  journey.editName();
  journey.close();
  assert.equal(journey.step, SettingsStep.CLOSED);
  journey.open();
  assert.equal(journey.step, SettingsStep.PAGE);
});
