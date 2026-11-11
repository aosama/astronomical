import assert from "node:assert/strict";
import { test } from "node:test";
import { AstroProfile, MAXIMUM_NICKNAME_CHARACTERS } from "../src/astro/AstroProfile";
import { AstroStyle, parseAstroStyle } from "../src/astro/AstroStyle";
import { AstroColourLook, parseAstroColourLook } from "../src/astro/AstroColourLook";

test("should_start_a_newcomer_with_no_name_explorer_style_astro_shown_and_the_flight_ahead", () => {
  const newcomer = AstroProfile.newcomer();
  assert.equal(newcomer.nickname, "");
  assert.equal(newcomer.style, AstroStyle.EXPLORER);
  assert.equal(newcomer.colourLook, AstroColourLook.SILVER);
  assert.equal(newcomer.hasFinishedFirstFlight, false);
  assert.equal(newcomer.isAstroShown, true);
});

test("should_round_trip_every_field_through_its_saved_form", () => {
  const savedProfile = AstroProfile.newcomer().withNickname("Maya").withStyle(AstroStyle.PILOT).withColourLook(AstroColourLook.SUNSET).withFirstFlightFinished().withAstroShown(false);
  const restoredProfile = AstroProfile.fromPlain(JSON.parse(JSON.stringify(savedProfile.toPlain())));
  assert.deepEqual(restoredProfile.toPlain(), { nickname: "Maya", style: "pilot", colourLook: "sunset", hasFinishedFirstFlight: true, isAstroShown: false });
});

test("should_treat_a_missing_or_damaged_profile_as_a_newcomer", () => {
  for (const damagedProfile of [null, undefined, "Maya", 7, []]) {
    assert.deepEqual(AstroProfile.fromPlain(damagedProfile).toPlain(), AstroProfile.newcomer().toPlain());
  }
});

test("should_read_wrongly_typed_fields_as_their_safe_defaults", () => {
  const restoredProfile = AstroProfile.fromPlain({ nickname: 12, style: "captain", colourLook: "plaid", hasFinishedFirstFlight: "yes", isAstroShown: "no" });
  assert.equal(restoredProfile.nickname, "");
  assert.equal(restoredProfile.style, AstroStyle.EXPLORER, "an unknown style falls back to the gentler voice");
  assert.equal(restoredProfile.colourLook, AstroColourLook.SILVER, "an unknown colour look costs only the colour choice");
  assert.equal(restoredProfile.hasFinishedFirstFlight, false);
  assert.equal(restoredProfile.isAstroShown, true, "only an explicit false hides Astro");
});

test("should_tidy_a_typed_nickname_to_one_printable_line", () => {
  assert.equal(AstroProfile.cleanNickname("  Maya  "), "Maya");
  assert.equal(AstroProfile.cleanNickname("Maya\n\tRose"), "Maya Rose");
  assert.equal(AstroProfile.cleanNickname("Ma\u0007ya"), "Maya");
  assert.equal(AstroProfile.cleanNickname("   "), "");
});

test("should_cap_a_nickname_at_the_limit_without_splitting_an_emoji", () => {
  const exactlyAtLimit = "a".repeat(MAXIMUM_NICKNAME_CHARACTERS);
  assert.equal(AstroProfile.cleanNickname(exactlyAtLimit), exactlyAtLimit);
  assert.equal(AstroProfile.cleanNickname(`${exactlyAtLimit}b`), exactlyAtLimit);
  const rocketName = "🚀".repeat(MAXIMUM_NICKNAME_CHARACTERS + 3);
  assert.equal(Array.from(AstroProfile.cleanNickname(rocketName)).length, MAXIMUM_NICKNAME_CHARACTERS);
  assert.ok(AstroProfile.cleanNickname(rocketName).endsWith("🚀"), "the last character is a whole emoji");
});

test("should_leave_the_original_profile_unchanged_when_deriving_a_new_one", () => {
  const original = AstroProfile.newcomer();
  original.withNickname("Sam").withStyle(AstroStyle.PILOT);
  assert.equal(original.nickname, "");
  assert.equal(original.style, AstroStyle.EXPLORER);
});

test("should_parse_only_the_two_known_styles", () => {
  assert.equal(parseAstroStyle("pilot"), AstroStyle.PILOT);
  assert.equal(parseAstroStyle("explorer"), AstroStyle.EXPLORER);
  assert.equal(parseAstroStyle("PILOT"), AstroStyle.EXPLORER);
  assert.equal(parseAstroStyle(undefined), AstroStyle.EXPLORER);
});

// Profiles saved while Astro was a character kept its space trail's colour
// under "trail", and the galaxy's first build under "colours". The galaxy
// starts everyone on silver, so neither key is read.
test("should_start_a_profile_saved_under_an_older_key_on_silver", () => {
  const restoredProfile = AstroProfile.fromPlain({ nickname: "Maya", style: "pilot", trail: "sunset", colours: "aurora", hasFinishedFirstFlight: true, isAstroShown: true });
  assert.equal(restoredProfile.colourLook, AstroColourLook.SILVER);
  assert.equal(restoredProfile.nickname, "Maya", "the older fields survive");
});

test("should_parse_only_the_four_known_colour_looks", () => {
  assert.equal(parseAstroColourLook("silver"), AstroColourLook.SILVER);
  assert.equal(parseAstroColourLook("aurora"), AstroColourLook.AURORA);
  assert.equal(parseAstroColourLook("sunset"), AstroColourLook.SUNSET);
  assert.equal(parseAstroColourLook("rainbow"), AstroColourLook.RAINBOW);
  assert.equal(parseAstroColourLook("Aurora"), AstroColourLook.SILVER, "the stored word is exact");
  assert.equal(parseAstroColourLook(""), AstroColourLook.SILVER);
  assert.equal(parseAstroColourLook(null), AstroColourLook.SILVER);
});
