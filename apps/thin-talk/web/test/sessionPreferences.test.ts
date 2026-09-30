import assert from "node:assert/strict";
import { test } from "node:test";
import { SessionPreferences } from "../src/SessionPreferences";
import { ThinkingEffort } from "../src/ThinkingEffort";

test("session preferences default to quick effort and the base font size", () => {
  const preferences = SessionPreferences.default();
  assert.equal(preferences.thinkingEffort, ThinkingEffort.QUICK);
  assert.equal(preferences.composerFontSize, 14);
});

test("session preferences clamp out-of-range font sizes", () => {
  const preferences = SessionPreferences.default();
  assert.equal(preferences.clampComposerFontSize(1), 10);
  assert.equal(preferences.clampComposerFontSize(99), 24);
  assert.equal(preferences.clampComposerFontSize(16), 16);
});

test("session preferences fall back to defaults on malformed input", () => {
  const preferences = SessionPreferences.fromPlain(null);
  assert.equal(preferences.thinkingEffort, ThinkingEffort.QUICK);
  assert.equal(preferences.composerFontSize, 14);
  const unknownEffort = SessionPreferences.fromPlain({ thinkingEffort: "nonsense", composerFontSize: 16 });
  assert.equal(unknownEffort.thinkingEffort, ThinkingEffort.QUICK);
  assert.equal(unknownEffort.composerFontSize, 16);
});

test("session preferences round-trip through toPlain", () => {
  const preferences = new SessionPreferences({ thinkingEffort: ThinkingEffort.HIGH, composerFontSize: 18 });
  const decoded = SessionPreferences.fromPlain(preferences.toPlain());
  assert.equal(decoded.thinkingEffort, ThinkingEffort.HIGH);
  assert.equal(decoded.composerFontSize, 18);
});
