import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";

// Copy must carry the reply and nothing else: the bubble also renders the role
// label, the state label, and the action buttons, and an earlier version pasted
// that chrome into the clipboard because it read the whole article's text.

function assistantMessageID(): string {
  const article = document.querySelector<HTMLElement>("#transcript article.message--assistant");
  const messageID: string | undefined = article?.dataset.messageId;
  return messageID != null ? messageID : "";
}

function copiedAnswerText(): string {
  const testEntry = window.__thintalk;
  return testEntry != null ? testEntry.renderedText(assistantMessageID()) : "";
}

test("should_copy_only_the_answer_body_without_bubble_chrome", async () => {
  const journey = bootJourney();
  await waitUntilReady();

  sendUserMessage("Summarize the prologue of Romeo and Juliet.");
  await waitFor(
    () => copiedAnswerText().includes("Hello from the fake daemon."),
    "the streamed reply to become copyable",
  );
  // The Copy button (and its label) only render on a finished turn, so waiting
  // for the save means the assertion below reads the same DOM the real copy
  // action would.
  await waitFor(() => journey.host.fileCount() === 1, "the finished turn to be saved");

  const copiedText = copiedAnswerText();
  assert.match(copiedText, /Hello from the fake daemon\./);
  assert.ok(!copiedText.includes("Assistant"), "the role label must not be copied");
  assert.ok(!copiedText.includes("Copy"), "the action button label must not be copied");
  assert.ok(!copiedText.includes("Reasoning"), "the reasoning disclosure must not be copied");
}, { timeout: 30_000 });

test("should_copy_a_user_turn_without_its_role_label", async () => {
  bootJourney();
  await waitUntilReady();

  sendUserMessage("What does the prologue ask of the audience?");
  await waitFor(() => {
    const article = document.querySelector<HTMLElement>("#transcript article.message--user");
    return article != null && article.dataset.messageId != null;
  }, "the sent user turn to render");
  const userArticle = document.querySelector<HTMLElement>("#transcript article.message--user");
  const messageID: string | undefined = userArticle?.dataset.messageId;

  const testEntry = window.__thintalk;
  assert.ok(testEntry != null && messageID != null, "the test entry and message id exist");
  const copiedText = testEntry.renderedText(messageID);
  assert.match(copiedText, /What does the prologue ask of the audience\?/);
  assert.ok(!copiedText.includes("You"), "the sender label must not be copied");
}, { timeout: 30_000 });
