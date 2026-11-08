import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";

/**
 * The composer stylesheet gives its pills, icons and send button a display rule,
 * and an author display rule beats the browser's own [hidden] style. These
 * tests hold the stylesheet to hiding each of them when the code sets
 * `hidden`, since the DOM tests only see the attribute, never the paint.
 */

const composerStylesheet = fs.readFileSync(path.join(__dirname, "..", "src", "composer.css"), "utf8");

/** The selectors of every rule whose body hides the element. */
function selectorsThatHide(stylesheet: string): string[] {
  const hidingSelectors: string[] = [];
  // Comments go first: their prose holds commas that would split a selector list.
  const stylesheetWithoutComments = stylesheet.replace(/\/\*[\s\S]*?\*\//g, "");
  for (const ruleMatch of stylesheetWithoutComments.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
    if (/display:\s*none/.test(ruleMatch[2]!)) {
      hidingSelectors.push(...ruleMatch[1]!.split(",").map((selector) => selector.trim()));
    }
  }
  return hidingSelectors;
}

test("should_hide_a_hidden_composer_pill_such_as_the_single_image_model_pill", () => {
  assert.ok(selectorsThatHide(composerStylesheet).includes(".composer__pill[hidden]"));
});

test("should_hide_every_hidden_composer_control_that_sets_its_own_display", () => {
  const hidingSelectors = selectorsThatHide(composerStylesheet);
  for (const controlClass of [".composer__icon", ".composer__pill", ".composer__send"]) {
    assert.ok(hidingSelectors.includes(`${controlClass}[hidden]`), `${controlClass} needs a [hidden] rule`);
  }
});

test("should_not_count_a_rule_that_only_styles_a_hidden_control_as_hiding_it", () => {
  const stylesheetWithoutDisplayNone = ".composer__pill[hidden] { opacity: 0.5; }";

  assert.deepEqual(selectorsThatHide(stylesheetWithoutDisplayNone), []);
});
