import assert from "node:assert/strict";
import { test } from "node:test";
import { installDom, waitFor } from "./support/dom";
import { ThinkingEffort } from "../src/ThinkingEffort";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";

// Issue 752: every interactive control of the conversation canvas carries a
// stable, domain-meaningful identifier, so automation agents and assistive
// technology share deterministic selectors that never derive from visible text
// or layout position. The shell document carries an id equal to the
// data-testid for each singleton control; the effort menu mints one mirrored
// identifier per level at render time; each transcript bubble is keyed on the
// message's own id as `message:<role>:<message id>`.

const SINGLETON_CONTROL_IDENTIFIERS = [
  "composer-input",
  "composer-send",
  "composer-stop",
  "composer-retry",
  "composer-effort",
  "composer-effort-menu",
  "sidebar-new-chat",
] as const;

// Message ids are UUIDs minted by the conversation log, so a bubble identifier
// can never collapse into visible text or an index.
const BUBBLE_IDENTIFIER_PATTERN =
  /^message:(user|assistant):[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

function renderedBubbles(): HTMLElement[] {
  return [...document.querySelectorAll<HTMLElement>("#transcript article.message")];
}

test("should_resolve_every_core_control_identifier_to_exactly_one_element", () => {
  installDom();
  for (const controlIdentifier of SINGLETON_CONTROL_IDENTIFIERS) {
    const byTestid = document.querySelectorAll(`[data-testid="${controlIdentifier}"]`);
    assert.equal(
      byTestid.length, 1,
      `data-testid "${controlIdentifier}" must resolve to exactly one element`);
    const byId = document.querySelectorAll(`#${controlIdentifier}`);
    assert.equal(
      byId.length, 1,
      `id "${controlIdentifier}" must resolve to exactly one element`);
    const element = byTestid[0] as HTMLElement;
    assert.equal(element.id, controlIdentifier, `id and data-testid must mirror on ${controlIdentifier}`);
  }
});

test("should_give_every_thinking_effort_choice_a_mirrored_identifier", async () => {
  bootJourney();
  await waitUntilReady();

  // The effort choices are minted at render time, not in the shell document,
  // so the mirrored-identifier rule for them is enforced against the live page.
  for (const effortOption of ThinkingEffort.ALL) {
    const choiceIdentifier = `composer-effort-${effortOption}`;
    const byTestid = document.querySelectorAll(`[data-testid="${choiceIdentifier}"]`);
    assert.equal(
      byTestid.length, 1,
      `the thinking-effort choice ${choiceIdentifier} must resolve to exactly one element`);
    const choice = byTestid[0] as HTMLElement;
    assert.equal(
      choice.id, choiceIdentifier,
      `the thinking-effort choice ${choiceIdentifier} must mirror its id and data-testid`);
  }
});

test("should_key_every_transcript_bubble_on_each_message_own_stable_identifier", async () => {
  const journey = bootJourney();
  await waitUntilReady();

  sendUserMessage("Summarize the prologue of Romeo and Juliet.");
  await waitFor(() => renderedBubbles().length === 2, "the first turn to render both bubbles");
  const firstTurnBubbles = renderedBubbles();
  const firstBubbleIdentifier = firstTurnBubbles[0]?.dataset.testid;
  assert.ok(firstBubbleIdentifier, "the first bubble carries an identifier");
  for (const bubble of firstTurnBubbles) {
    const bubbleIdentifier: string | undefined = bubble.dataset.testid;
    assert.ok(bubbleIdentifier, "every bubble carries a data-testid");
    assert.match(bubbleIdentifier ?? "", BUBBLE_IDENTIFIER_PATTERN);
    assert.equal(bubble.id, bubbleIdentifier, "the bubble id mirrors its data-testid");
  }
  const firstTurnIdentifiers = firstTurnBubbles.map((bubble) => bubble.dataset.testid);
  assert.equal(new Set(firstTurnIdentifiers).size, firstTurnIdentifiers.length, "bubble identifiers are unique");

  // A second turn must not churn the earlier bubble's identifier: the locator
  // an agent resolved before the turn stays valid across the re-render.
  sendUserMessage("What does the prologue ask of the audience?");
  await waitFor(() => renderedBubbles().length === 4, "the second turn to render both bubbles");
  const untouchedBubble = document.querySelector<HTMLElement>(`[data-testid="${firstBubbleIdentifier}"]`);
  assert.ok(untouchedBubble, "the earlier bubble's identifier survives the re-render");
  const allBubbleIdentifiers = renderedBubbles().map((bubble) => bubble.dataset.testid ?? "");
  assert.equal(new Set(allBubbleIdentifiers).size, allBubbleIdentifiers.length, "bubble identifiers stay unique across turns");
  assert.ok(
    allBubbleIdentifiers.every((bubbleIdentifier) => BUBBLE_IDENTIFIER_PATTERN.test(bubbleIdentifier)),
    "every bubble identifier is keyed on the message's own id",
  );

  // The whole app-owned page stays unambiguous for identifier-based lookups.
  // Diagram SVGs mint their own internal ids (clip paths, gradients); they are
  // rendered content, not app chrome, so they stay outside this contract.
  const appOwnedIdentifiers = [...document.querySelectorAll("[id]")]
    .filter((element) => element.namespaceURI !== "http://www.w3.org/2000/svg")
    .map((element) => element.id);
  assert.equal(
    new Set(appOwnedIdentifiers).size, appOwnedIdentifiers.length,
    "app-owned identifiers are unique document-wide");
  assert.ok(journey.host.fileCount() >= 1, "the journey persisted its session");
}, { timeout: 30_000 });
