import assert from "node:assert/strict";
import { test } from "node:test";
import { AstroMoment, HOME_IDEAS, astroLine } from "../src/astro/AstroLines";

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
  assert.equal(astroLine(AstroMoment.HOME_GREETING, "Maya"), "Hey, Maya");
  assert.equal(astroLine(AstroMoment.HOME_GREETING, ""), "Hey");
  for (const moment of EVERY_MOMENT) {
    assert.doesNotMatch(astroLine(moment, ""), /, [!.?]|,\s*$/, moment);
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

test("should_offer_four_distinct_home_ideas", () => {
  assert.equal(HOME_IDEAS.length, 4);
  assert.equal(new Set(HOME_IDEAS).size, 4, "no idea repeats");
});
