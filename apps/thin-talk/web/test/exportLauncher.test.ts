import assert from "node:assert/strict";
import { test } from "node:test";
import { assertNoElement, waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";
import type { Journey } from "./support/journey";
import { createExportFeature } from "../src/export/exportEntry";
import { NOTHING_TO_EXPORT_NOTICE } from "../src/ExportLauncher";

// The three ways into exporting (an answer's Export button, File ▸ Export…
// and the toolbar, a conversation's right-click menu) all open one sheet, on
// the conversation the person is looking at.

const FAKE_REPLY = "Hello from the fake daemon.";

/** A chat with one finished turn. The export script is registered by hand,
 * since the test window cannot load it from the app's private scheme. */
async function chatWithOneTurn(): Promise<Journey> {
  const journey = bootJourney();
  window.__thintalkExport = createExportFeature();
  await waitUntilReady();
  sendUserMessage("Explain mitosis for my biology homework.");
  await waitFor(() => journey.host.fileCount() === 1, "the finished turn to be saved");
  await waitFor(() => document.querySelector('[data-testid="action-export"]') !== null, "the answer's Export button");
  return journey;
}

function element<T extends HTMLElement>(testID: string): T {
  const found = document.querySelector<T>(`[data-testid="${testID}"]`);
  assert.ok(found, `the page has ${testID}`);
  return found;
}

function click(target: Element): void {
  target.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true }));
}

function previewText(): string {
  return element("export-preview").shadowRoot?.textContent ?? "";
}

test("should_offer_export_on_answers_only", async () => {
  await chatWithOneTurn();

  const exportButtons = [...document.querySelectorAll('[data-testid="action-export"]')];
  assert.equal(exportButtons.length, 1);
  assert.ok(exportButtons[0]?.closest(".message--assistant"), "a question has no Export button of its own");
}, { timeout: 30_000 });

test("should_export_one_answer_as_a_pdf_and_say_it_was_saved", async () => {
  const journey = await chatWithOneTurn();
  journey.host.exportSaveAnswer = { outcome: "saved", fileName: "Mitosis.pdf" };

  click(element("action-export"));
  await waitFor(() => !element("export-sheet").hidden && previewText().includes(FAKE_REPLY), "the sheet on this answer");
  assert.equal(element("export-scope-oneAnswer").getAttribute("aria-checked"), "true");
  assertNoElement(element("export-preview").shadowRoot, ".export-question", "one answer starts without its question");
  click(element("export-save"));

  await waitFor(() => !element("brief-notice").hidden, "the saved notice");
  assert.equal(element("brief-notice").textContent, "Saved “Mitosis.pdf”");
  assert.equal(element("export-sheet").hidden, true);
  const savedExport = journey.host.receivedExports[0] ?? {};
  assert.equal(savedExport["format"], "pdf");
  assert.match(String(savedExport["html"]), new RegExp(FAKE_REPLY));
  assert.doesNotMatch(String(savedExport["html"]), /data-testid|message__action|thintalk-asset/, "nothing of the window travels");
}, { timeout: 30_000 });

test("should_export_the_whole_chat_from_the_menu_command_with_questions", async () => {
  const journey = await chatWithOneTurn();

  assert.equal(window.__thintalkCommands?.run({ name: "exportConversation" }), true);
  await waitFor(() => !element("export-sheet").hidden && previewText().includes("biology homework"), "the sheet on the whole chat");
  click(element("export-format-html"));
  click(element("export-save"));

  await waitFor(() => journey.host.receivedExports.length === 1, "the web page to be saved");
  assert.equal(journey.host.receivedExports[0]?.["format"], "html");
  assert.match(String(journey.host.receivedExports[0]?.["html"]), /biology homework[\s\S]*Hello from the fake daemon/);
}, { timeout: 30_000 });

test("should_say_there_is_nothing_to_export_on_an_empty_chat_instead_of_opening_the_sheet", async () => {
  bootJourney();
  window.__thintalkExport = createExportFeature();
  await waitUntilReady();

  window.__thintalkCommands?.run({ name: "exportConversation" });

  await waitFor(() => !element("brief-notice").hidden, "the notice");
  assert.equal(element("brief-notice").textContent, NOTHING_TO_EXPORT_NOTICE);
  assert.equal(element("export-sheet").hidden, true);
}, { timeout: 30_000 });

test("should_open_a_chat_and_its_export_sheet_from_the_sidebars_right_click_menu", async () => {
  const journey = await chatWithOneTurn();
  await new Promise<void>((resolve) => {
    window.__thintalkCommands?.run({ name: "newChat" });
    setTimeout(resolve, 50);
  });
  await waitFor(() => document.querySelector(".message--assistant") === null, "a new empty chat");
  journey.host.conversationMenuChoice = "export";
  const savedChatRow = document.querySelector("#session-list [data-session-id]");
  assert.ok(savedChatRow, "the saved chat is listed");

  savedChatRow.dispatchEvent(new window.MouseEvent("contextmenu", { bubbles: true, cancelable: true }));

  await waitFor(() => !element("export-sheet").hidden && previewText().includes(FAKE_REPLY), "the sheet on the chosen chat");
  assert.equal(element("export-scope-wholeChat").getAttribute("aria-checked"), "true");
}, { timeout: 30_000 });

test("should_keep_the_window_behind_the_sheet_out_of_reach_until_it_closes", async () => {
  await chatWithOneTurn();
  window.__thintalkCommands?.run({ name: "exportConversation" });
  await waitFor(() => !element("export-sheet").hidden, "the sheet");

  assert.equal(document.querySelector("#sidebar")?.hasAttribute("inert"), true);
  assert.equal(document.querySelector("#main")?.hasAttribute("inert"), true);
  click(element("export-cancel"));
  assert.equal(document.querySelector("#main")?.hasAttribute("inert"), false);
}, { timeout: 30_000 });

test("should_pick_messages_in_the_conversation_and_come_back_to_the_sheet", async () => {
  await chatWithOneTurn();
  window.__thintalkCommands?.run({ name: "exportConversation" });
  await waitFor(() => !element("export-sheet").hidden, "the sheet");

  click(element("export-scope-pickedMessages"));
  await waitFor(() => !element("pick-bar").hidden, "the pick bar");
  assert.equal(element("export-sheet").hidden, true, "the sheet steps aside while picking");
  assert.equal(element("composer-input").closest("#app")?.classList.contains("app--picking"), true);
  const answer = document.querySelector(".message--assistant");
  assert.ok(answer);
  click(answer);
  click(element("pick-done"));

  await waitFor(() => !element("export-sheet").hidden && element("export-scope-pickedMessages").textContent === "1 picked…", "the sheet with the pick");
  await waitFor(
    () => previewText().includes(FAKE_REPLY) && element("export-preview").shadowRoot?.querySelector(".export-question") === null,
    "the preview of the picked answer alone",
  );
  assert.equal(element("pick-bar").hidden, true);
}, { timeout: 30_000 });

test("should_say_when_the_file_could_not_be_saved", async () => {
  const journey = await chatWithOneTurn();
  journey.host.exportSaveAnswer = "fails";

  click(element("action-export"));
  await waitFor(() => !element("export-sheet").hidden && previewText().includes(FAKE_REPLY), "the sheet");
  click(element("export-save"));

  await waitFor(() => element("export-status").textContent !== "", "the failure to be said");
  assert.equal(element("export-sheet").hidden, false);
  assert.equal(element("brief-notice").hidden, true);
}, { timeout: 30_000 });
