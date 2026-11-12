import assert from "node:assert/strict";
import { test } from "node:test";
import { YourAstroJourney, YourAstroStep } from "../src/astro/YourAstroJourney";

function openJourney(): YourAstroJourney {
  const journey = new YourAstroJourney();
  journey.open();
  return journey;
}

test("should_start_closed_and_open_on_the_page", () => {
  const journey = new YourAstroJourney();
  assert.equal(journey.step, YourAstroStep.CLOSED);
  journey.open();
  assert.equal(journey.step, YourAstroStep.PAGE);
});

test("should_open_each_answer_on_its_own_and_come_back_to_the_page", () => {
  for (const [move, expectedStep] of [
    [(journey: YourAstroJourney) => journey.editName(), YourAstroStep.NAME],
    [(journey: YourAstroJourney) => journey.readPrivacyNote(), YourAstroStep.PRIVACY],
  ] as const) {
    const journey = openJourney();
    move(journey);
    assert.equal(journey.step, expectedStep);
    journey.backToPage();
    assert.equal(journey.step, YourAstroStep.PAGE, `${expectedStep} returns to the page`);
  }
});

test("should_ignore_a_move_that_belongs_to_another_step", () => {
  const closedJourney = new YourAstroJourney();
  closedJourney.editName();
  closedJourney.backToPage();
  assert.equal(closedJourney.step, YourAstroStep.CLOSED, "nothing opens a closed page but open()");
  const journey = openJourney();
  journey.editName();
  journey.readPrivacyNote();
  assert.equal(journey.step, YourAstroStep.NAME, "a late click on another row does not jump screens");
});

test("should_back_out_one_level_with_escape", () => {
  const journey = openJourney();
  journey.readPrivacyNote();
  journey.escape();
  assert.equal(journey.step, YourAstroStep.PAGE);
  journey.escape();
  assert.equal(journey.step, YourAstroStep.CLOSED);
  journey.escape();
  assert.equal(journey.step, YourAstroStep.CLOSED, "escape on a closed page stays closed");
});

test("should_reopen_on_the_page_whatever_was_on_show_last_time", () => {
  const journey = openJourney();
  journey.editName();
  journey.close();
  assert.equal(journey.step, YourAstroStep.CLOSED);
  journey.open();
  assert.equal(journey.step, YourAstroStep.PAGE);
});
