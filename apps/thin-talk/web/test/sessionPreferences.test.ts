import assert from "node:assert/strict";
import { test } from "node:test";
import { MAX_SIDEBAR_WIDTH_PIXELS, MIN_SIDEBAR_WIDTH_PIXELS, SessionPreferences } from "../src/SessionPreferences";

test("should_default_to_no_chosen_model", () => {
  const preferences = SessionPreferences.default();
  assert.equal(preferences.modelID, null);
});

test("should_drop_a_thinking_level_saved_by_an_earlier_version", () => {
  const preferences = SessionPreferences.fromPlain({ thinkingEffort: "high", modelID: "model-beta" });
  assert.equal(preferences.modelID, "model-beta");
  assert.equal("thinkingEffort" in preferences.toPlain(), false, "the level is no longer written back");
});

test("should_ignore_a_message_box_font_size_saved_by_an_earlier_version", () => {
  const preferences = SessionPreferences.fromPlain({ composerFontSize: 14, modelID: "model-beta" });
  assert.equal(preferences.modelID, "model-beta");
  assert.equal("composerFontSize" in preferences.toPlain(), false, "the size is no longer written back");
});

test("should_fall_back_to_defaults_on_malformed_input", () => {
  const preferences = SessionPreferences.fromPlain(null);
  assert.equal(preferences.modelID, null);
  const unknownModel = SessionPreferences.fromPlain({ modelID: 42 });
  assert.equal(unknownModel.modelID, null);
});

test("should_round_trip_through_plain_values", () => {
  const preferences = new SessionPreferences({ modelID: "model-beta" });
  const decoded = SessionPreferences.fromPlain(preferences.toPlain());
  assert.equal(decoded.modelID, "model-beta");
});

test("should_turn_tools_on_by_default_and_for_preferences_saved_before_the_switch", () => {
  assert.equal(SessionPreferences.default().toolsEnabled, true);
  assert.equal(SessionPreferences.fromPlain({ modelID: "model-beta" }).toolsEnabled, true);
  assert.equal(SessionPreferences.fromPlain({ toolsEnabled: "no" }).toolsEnabled, true);
});

test("should_remember_tools_switched_off", () => {
  const preferences = SessionPreferences.default();
  preferences.toolsEnabled = false;
  assert.equal(SessionPreferences.fromPlain(preferences.toPlain()).toolsEnabled, false);
});

test("should_keep_a_saved_sidebar_width_within_its_range", () => {
  assert.equal(SessionPreferences.fromPlain({ sidebarWidthPixels: 300 }).sidebarWidthPixels, 300);
  assert.equal(SessionPreferences.fromPlain({ sidebarWidthPixels: 20 }).sidebarWidthPixels, MIN_SIDEBAR_WIDTH_PIXELS);
  assert.equal(SessionPreferences.fromPlain({ sidebarWidthPixels: 5000 }).sidebarWidthPixels, MAX_SIDEBAR_WIDTH_PIXELS);
});

test("should_use_the_default_sidebar_width_when_none_or_a_damaged_one_was_saved", () => {
  assert.equal(SessionPreferences.fromPlain({}).sidebarWidthPixels, null);
  assert.equal(SessionPreferences.fromPlain({ sidebarWidthPixels: "wide" }).sidebarWidthPixels, null);
  assert.equal(SessionPreferences.fromPlain({ sidebarWidthPixels: Number.NaN }).sidebarWidthPixels, null);
  assert.equal(SessionPreferences.default().toPlain().sidebarWidthPixels, null);
});

test("should_round_trip_the_chosen_image_model_and_drop_one_that_is_not_text", () => {
  const preferences = new SessionPreferences({ imageModelID: "flux2-klein-4b-4bit" });
  assert.equal(SessionPreferences.fromPlain(preferences.toPlain()).imageModelID, "flux2-klein-4b-4bit");
  assert.equal(SessionPreferences.default().imageModelID, null);
  assert.equal(SessionPreferences.fromPlain({ imageModelID: 42 }).imageModelID, null);
  assert.equal(SessionPreferences.fromPlain({ modelID: "chat" }).imageModelID, null, "preferences saved before the picker load without one");
});
