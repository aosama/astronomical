import assert from "node:assert/strict";
import { test } from "node:test";
import { assertFocusOn, assertNoElement, installDom } from "./support/dom";

// The helpers exist so a failing DOM assertion stays small: Node printing a
// happy-dom node walks the whole document. Each failure message is checked
// for its size as well as its words.
const LONGEST_SAFE_MESSAGE_CHARACTERS = 300;

function failureMessage(failingAssertion: () => void): string {
  try {
    failingAssertion();
  } catch (thrown) {
    return thrown instanceof Error ? thrown.message : String(thrown);
  }
  return assert.fail("the assertion was expected to fail");
}

test("should_pass_when_no_element_matches_or_there_is_no_root", () => {
  installDom();
  assertNoElement(document, ".never-in-the-page");
  assertNoElement(null, "button");
  assertNoElement(undefined, "button");
});

test("should_fail_with_a_short_description_when_an_element_matches", () => {
  installDom();
  const message = failureMessage(() => assertNoElement(document, "#composer-input", "no message field"));
  assert.match(message, /no message field \(found textarea#composer-input/);
  assert.ok(message.length < LONGEST_SAFE_MESSAGE_CHARACTERS, `the message stays short (${message.length} characters)`);
});

test("should_pass_when_focus_is_on_the_expected_element", () => {
  installDom();
  const messageField = document.getElementById("composer-input") as HTMLElement;
  messageField.focus();
  assertFocusOn(messageField);
});

test("should_fail_with_both_elements_named_briefly_when_focus_is_elsewhere", () => {
  installDom();
  const messageField = document.getElementById("composer-input") as HTMLElement;
  const message = failureMessage(() => assertFocusOn(messageField, "after opening"));
  assert.match(message, /after opening: expected textarea#composer-input.*focus is on body/);
  assert.ok(message.length < LONGEST_SAFE_MESSAGE_CHARACTERS, `the message stays short (${message.length} characters)`);
});
