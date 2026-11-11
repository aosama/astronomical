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

test("should_hide_the_brain_control_while_no_chat_model_is_installed", () => {
  assert.ok(selectorsThatHide(composerStylesheet).includes(".composer__brain[hidden]"));
});

test("should_hide_every_hidden_composer_control_that_sets_its_own_display", () => {
  const hidingSelectors = selectorsThatHide(composerStylesheet);
  for (const controlClass of [".composer__icon", ".composer__brain", ".composer__send"]) {
    assert.ok(hidingSelectors.includes(`${controlClass}[hidden]`), `${controlClass} needs a [hidden] rule`);
  }
});

test("should_drop_the_brain_name_and_keep_the_icon_in_a_narrow_composer", () => {
  // A narrow window must never push Send out of the composer: the name goes,
  // and the brain icon keeps its own size so flex cannot squeeze it.
  const narrowRuleMatch = composerStylesheet.match(/@container \(max-width: (\d+)px\) \{([\s\S]*?)\n\}/);
  assert.ok(narrowRuleMatch, "the composer has a narrow-width rule");
  assert.ok(Number(narrowRuleMatch[1]) > 0);
  assert.ok(selectorsThatHide(narrowRuleMatch[2]!).includes(".composer__brain-label"));
  assert.match(narrowRuleMatch[2]!, /\.composer__brain \{[^}]*flex: none;/);
  assert.match(composerStylesheet, /\.composer \{\s*container-type: inline-size;/);
});

test("should_not_count_a_rule_that_only_styles_a_hidden_control_as_hiding_it", () => {
  const stylesheetWithoutDisplayNone = ".composer__brain[hidden] { opacity: 0.5; }";

  assert.deepEqual(selectorsThatHide(stylesheetWithoutDisplayNone), []);
});

/** The body of the one rule whose selector is exactly `selector`. */
function ruleBody(stylesheet: string, selector: string): string {
  const stylesheetWithoutComments = stylesheet.replace(/\/\*[\s\S]*?\*\//g, "");
  for (const ruleMatch of stylesheetWithoutComments.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
    if (ruleMatch[1]!.trim() === selector) {
      return ruleMatch[2]!;
    }
  }
  return "";
}

// A model's only name is its ID, and IDs share long prefixes, so a fixed-width
// panel cut three different models down to the same "Ornith-1.5-35B-A3B-MLX-…".
test("should_size_the_brain_panel_to_its_longest_model_id_within_the_window", () => {
  const brainPanelRule = ruleBody(composerStylesheet, ".composer__menu--brains");

  assert.match(brainPanelRule, /width:\s*max-content;/, "the panel grows to its longest row");
  assert.match(brainPanelRule, /min-width:\s*300px;/, "a short list keeps the panel's usual width");
  assert.match(brainPanelRule, /max-width:\s*calc\(100vw - 32px\);/, "a narrow window still bounds it");
  assert.doesNotMatch(brainPanelRule, /(^|[^-])width:\s*\d+px;/, "no fixed width that would cut an ID short");
});
