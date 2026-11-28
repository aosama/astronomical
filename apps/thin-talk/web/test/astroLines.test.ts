import assert from "node:assert/strict";
import { test } from "node:test";
import { AstroMoment, HomeIdeaKind, astroLine, homeIdeasFor, timeOfDayOpening } from "../src/astro/AstroLines";

const EVERY_MOMENT = Object.values(AstroMoment);
/** Words from the younger-audience wording the app moved away from, and
 * internals no person should meet. */
const UNWANTED_WORDS = [/\bbrain\b/i, /\bsketchbook\b/i, /\bgrown-?ups?\b/i, /\bgifts?\b/i, /\bLibrary\b/, /\bdaemon\b/i, /\bMLX\b/, /\bHugging\s*Face\b/i];

test("should_have_a_line_for_every_moment", () => {
  for (const moment of EVERY_MOMENT) {
    assert.ok(astroLine(moment, "Maya").trim().length > 0, moment);
  }
});

test("should_greet_by_name_when_given_one_and_never_leave_a_dangling_comma_without", () => {
  assert.match(astroLine(AstroMoment.HOME_GREETING, "Maya"), /^Good (morning|afternoon|evening), Maya$/);
  assert.match(astroLine(AstroMoment.HOME_GREETING, ""), /^Good (morning|afternoon|evening)$/);
  for (const moment of EVERY_MOMENT) {
    assert.doesNotMatch(astroLine(moment, ""), /, [!.?]|,\s*$/, moment);
  }
});

test("should_greet_by_the_time_of_day_at_each_boundary", () => {
  assert.equal(timeOfDayOpening(4), "Good evening", "the small hours are still evening");
  assert.equal(timeOfDayOpening(5), "Good morning");
  assert.equal(timeOfDayOpening(11), "Good morning");
  assert.equal(timeOfDayOpening(12), "Good afternoon");
  assert.equal(timeOfDayOpening(17), "Good afternoon");
  assert.equal(timeOfDayOpening(18), "Good evening");
  assert.equal(timeOfDayOpening(0), "Good evening");
});

test("should_offer_one_idea_of_each_kind_and_keep_them_for_the_whole_day", () => {
  const morning = homeIdeasFor(new Date(2026, 10, 28, 8, 0));
  const lateEvening = homeIdeasFor(new Date(2026, 10, 28, 23, 30));
  assert.deepEqual(morning.map((idea) => idea.kind), [HomeIdeaKind.WRITE, HomeIdeaKind.PICTURES, HomeIdeaKind.LOOK, HomeIdeaKind.READ]);
  assert.deepEqual(lateEvening, morning, "the ideas do not change within one day");
  assert.deepEqual(morning.map((idea) => idea.needsImageModel), [false, true, false, false], "only the picture idea needs an image model");
});

test("should_change_the_ideas_at_local_midnight_and_not_before", () => {
  const justBeforeMidnight = homeIdeasFor(new Date(2026, 10, 28, 23, 59));
  const justAfterMidnight = homeIdeasFor(new Date(2026, 10, 29, 0, 1));
  const earlierThatDay = homeIdeasFor(new Date(2026, 10, 28, 0, 1));
  assert.deepEqual(justBeforeMidnight, earlierThatDay, "the whole local day shows one set");
  assert.notDeepEqual(justAfterMidnight, justBeforeMidnight);
});

test("should_change_the_ideas_from_one_day_to_the_next", () => {
  const firstDayPrompts = homeIdeasFor(new Date(2026, 10, 28, 9, 0)).map((idea) => idea.prompt);
  const nextDayPrompts = homeIdeasFor(new Date(2026, 10, 29, 9, 0)).map((idea) => idea.prompt);
  assert.notDeepEqual(nextDayPrompts, firstDayPrompts);
});

test("should_ask_the_model_to_open_the_picker_in_every_photo_and_file_idea", () => {
  for (let dayOffset = 0; dayOffset < 6; dayOffset += 1) {
    for (const idea of homeIdeasFor(new Date(2026, 10, 28 + dayOffset, 9, 0))) {
      assert.ok(idea.prompt.length > 0 && !idea.prompt.endsWith("."), `"${idea.prompt}" reads as a chip, without a full stop`);
      if (idea.kind === HomeIdeaKind.LOOK || idea.kind === HomeIdeaKind.READ) {
        assert.match(idea.prompt, /I choose/, `"${idea.prompt}" must lead the model to the picker`);
      }
    }
  }
});

test("should_name_the_models_plainly_and_avoid_the_retired_wording", () => {
  assert.match(astroLine(AstroMoment.NEEDS_CHAT_MODEL, ""), /chat model/);
  assert.match(astroLine(AstroMoment.NEEDS_IMAGE_MODEL, ""), /image model/);
  for (const moment of EVERY_MOMENT) {
    const line = astroLine(moment, "Maya");
    for (const unwantedWord of UNWANTED_WORDS) {
      assert.doesNotMatch(line, unwantedWord, `${moment}: "${line}"`);
    }
  }
});

test("should_never_exclaim", () => {
  for (const moment of EVERY_MOMENT) {
    assert.doesNotMatch(astroLine(moment, "Maya"), /!/, moment);
  }
});
