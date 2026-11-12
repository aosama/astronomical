import assert from "node:assert/strict";
import { test } from "node:test";
import { AstroProfile, MAXIMUM_NICKNAME_CHARACTERS } from "../src/astro/AstroProfile";

test("should_start_a_newcomer_with_no_name_astro_shown_and_the_journey_ahead", () => {
  const newcomer = AstroProfile.newcomer();
  assert.equal(newcomer.nickname, "");
  assert.equal(newcomer.hasFinishedGetStarted, false);
  assert.equal(newcomer.isAstroShown, true);
});

test("should_round_trip_every_field_through_its_saved_form", () => {
  const savedProfile = AstroProfile.newcomer().withNickname("Maya").withGetStartedFinished().withAstroShown(false);
  const restoredProfile = AstroProfile.fromPlain(JSON.parse(JSON.stringify(savedProfile.toPlain())));
  assert.deepEqual(restoredProfile.toPlain(), { nickname: "Maya", hasFinishedFirstFlight: true, isAstroShown: false });
});

test("should_treat_a_missing_or_damaged_profile_as_a_newcomer", () => {
  for (const damagedProfile of [null, undefined, "Maya", 7, []]) {
    assert.deepEqual(AstroProfile.fromPlain(damagedProfile).toPlain(), AstroProfile.newcomer().toPlain());
  }
});

test("should_read_wrongly_typed_fields_as_their_safe_defaults", () => {
  const restoredProfile = AstroProfile.fromPlain({ nickname: 12, hasFinishedFirstFlight: "yes", isAstroShown: "no" });
  assert.equal(restoredProfile.nickname, "");
  assert.equal(restoredProfile.hasFinishedGetStarted, false);
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
  original.withNickname("Sam").withGetStartedFinished();
  assert.equal(original.nickname, "");
  assert.equal(original.hasFinishedGetStarted, false);
});

// An older build saved a voice ("style") and a colour choice ("colourLook")
// with the profile. Neither exists now, so neither is read or written back.
test("should_keep_the_name_and_drop_the_keys_an_older_build_saved", () => {
  const restoredProfile = AstroProfile.fromPlain({ nickname: "Maya", style: "pilot", colourLook: "aurora", hasFinishedFirstFlight: true, isAstroShown: true });
  assert.deepEqual(restoredProfile.toPlain(), { nickname: "Maya", hasFinishedFirstFlight: true, isAstroShown: true });
});

// Profiles already on disk carry the key they were first saved under. A new
// key would read every one of them as unfinished and show the welcome again.
test("should_save_and_read_the_finished_journey_under_the_key_already_on_disk", () => {
  assert.equal(AstroProfile.newcomer().withGetStartedFinished().toPlain()["hasFinishedFirstFlight"], true);
  assert.equal(AstroProfile.fromPlain({ hasFinishedFirstFlight: true }).hasFinishedGetStarted, true);
  assert.equal(AstroProfile.fromPlain({ hasFinishedGetStarted: true }).hasFinishedGetStarted, false, "an unknown key is not read");
});
