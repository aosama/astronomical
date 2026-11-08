import assert from "node:assert/strict";
import { test } from "node:test";
import { AstroMoment, astroLine, homeIdeas } from "../src/astro/AstroLines";
import { AstroStyle } from "../src/astro/AstroStyle";

const EVERY_MOMENT = Object.values(AstroMoment);
const EVERY_STYLE = [AstroStyle.EXPLORER, AstroStyle.PILOT];
/** Words a young person should never meet from Astro. */
const JARGON = [/\bmodel\b/i, /\bGB\b/, /\bLibrary\b/, /\bdaemon\b/i, /\bMLX\b/, /\berror\b/i, /\bHugging\s*Face\b/i];

test("should_have_a_line_for_every_moment_in_both_styles", () => {
  for (const moment of EVERY_MOMENT) {
    for (const style of EVERY_STYLE) {
      assert.ok(astroLine(moment, style, "Maya").trim().length > 0, `${moment} in ${style}`);
    }
  }
});

test("should_greet_by_name_when_given_one_and_never_leave_a_dangling_comma_without", () => {
  assert.equal(astroLine(AstroMoment.HOME_GREETING, AstroStyle.EXPLORER, "Maya"), "Hi, Maya!");
  assert.equal(astroLine(AstroMoment.HOME_GREETING, AstroStyle.EXPLORER, ""), "Hi!");
  assert.equal(astroLine(AstroMoment.HOME_GREETING, AstroStyle.PILOT, ""), "Hey");
  for (const style of EVERY_STYLE) {
    for (const moment of EVERY_MOMENT) {
      assert.doesNotMatch(astroLine(moment, style, ""), /, [!.?]|,\s*$/, `${moment} in ${style}`);
    }
  }
});

test("should_speak_differently_in_each_style_where_it_matters", () => {
  assert.notEqual(astroLine(AstroMoment.HELLO, AstroStyle.EXPLORER, ""), astroLine(AstroMoment.HELLO, AstroStyle.PILOT, ""));
  assert.notEqual(astroLine(AstroMoment.CHARGING, AstroStyle.EXPLORER, ""), astroLine(AstroMoment.CHARGING, AstroStyle.PILOT, ""));
});

test("should_keep_astro_free_of_technical_jargon", () => {
  for (const moment of EVERY_MOMENT) {
    for (const style of EVERY_STYLE) {
      const line = astroLine(moment, style, "Maya");
      for (const jargonPattern of JARGON) {
        assert.doesNotMatch(line, jargonPattern, `${moment} in ${style}: "${line}"`);
      }
    }
  }
});

test("should_offer_four_home_ideas_in_each_style", () => {
  for (const style of EVERY_STYLE) {
    const ideas = homeIdeas(style);
    assert.equal(ideas.length, 4);
    assert.equal(new Set(ideas).size, 4, "no idea repeats");
  }
});
