import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { ExportSheet, pageCountText, previewTextHeightPixels } from "../src/export/ExportSheet";
import type { ExportEnvironment, ExportSaveOutcome, ExportSaveRequest, ExportableMessage } from "../src/export/ExportEnvironment";
import { ExportFormat } from "../src/export/ExportFormat";

interface SheetScene {
  sheet: ExportSheet;
  sheetElement: HTMLElement;
  saveRequests: ExportSaveRequest[];
  announcedFileNames: string[];
  setSaveAnswer(answer: ExportSaveOutcome | Error): void;
  setPickAnswer(answer: readonly string[] | null): void;
}

function exportableMessage(messageID: string, isQuestion: boolean, text: string): ExportableMessage {
  const renderedBody = document.createElement("div");
  renderedBody.innerHTML = `<p>${text}</p>`;
  return { messageID, isQuestion, renderedBody, generatedImageNames: [], generatedImageDescriptions: new Map() };
}

function sheetScene(): SheetScene {
  const sheetElement = document.createElement("section");
  sheetElement.hidden = true;
  document.body.appendChild(sheetElement);
  const saveRequests: ExportSaveRequest[] = [];
  const announcedFileNames: string[] = [];
  let saveAnswer: ExportSaveOutcome | Error = { outcome: "saved", fileName: "Mitosis explained.pdf" };
  let pickAnswer: readonly string[] | null = null;
  const environment: ExportEnvironment = {
    sheetElement,
    conversationTitle: () => "Mitosis explained",
    assistantName: () => "Astro",
    exportableMessages: () => [exportableMessage("q1", true, "What is mitosis?"), exportableMessage("a1", false, "Cell division."), exportableMessage("q2", true, "Thanks"), exportableMessage("a2", false, "You're welcome.")],
    loadPicture: async () => null,
    drawDiagramForPaper: async () => null,
    saveDocument: async (saveRequest) => {
      saveRequests.push(saveRequest);
      if (saveAnswer instanceof Error) {
        throw saveAnswer;
      }
      return saveAnswer;
    },
    pickMessages: async () => pickAnswer,
    announceSaved: (fileName) => announcedFileNames.push(fileName),
  };
  const sheet = new ExportSheet(environment, async () => null);
  return {
    sheet, sheetElement, saveRequests, announcedFileNames,
    setSaveAnswer: (answer) => { saveAnswer = answer; },
    setPickAnswer: (answer) => { pickAnswer = answer; },
  };
}

function control<T extends HTMLElement>(scene: SheetScene, testID: string): T {
  const found = scene.sheetElement.querySelector<T>(`[data-testid="${testID}"]`);
  assert.ok(found, `the sheet has ${testID}`);
  return found;
}

function previewText(scene: SheetScene): string {
  return control(scene, "export-preview").shadowRoot?.textContent ?? "";
}

function click(element: Element): void {
  element.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true }));
}

test("should_open_on_the_whole_chat_as_a_pdf_with_every_message_in_the_preview", async () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "wholeChat" });

  await waitFor(() => previewText(scene).includes("You're welcome."), "the preview to show the chat");
  assert.equal(scene.sheetElement.hidden, false);
  assert.equal(control(scene, "export-scope-wholeChat").getAttribute("aria-checked"), "true");
  assert.equal(control(scene, "export-scope-oneAnswer").hidden, true, "there is no answer to export on its own");
  assert.equal(control(scene, "export-format-pdf").getAttribute("aria-checked"), "true");
  assert.equal(control(scene, "export-save").textContent, "Save as PDF…");
  assert.match(previewText(scene), /Mitosis explained[\s\S]*What is mitosis\?[\s\S]*Written with help from Astro/);
});

test("should_show_one_answer_without_its_question_and_bring_the_question_back_when_ticked", async () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "oneAnswer", messageID: "a1" });
  await waitFor(() => previewText(scene).includes("Cell division."), "the answer in the preview");
  assert.doesNotMatch(previewText(scene), /What is mitosis\?|You're welcome/);

  const questionsBox = control<HTMLInputElement>(scene, "export-include-includesQuestions");
  questionsBox.checked = true;
  questionsBox.dispatchEvent(new window.Event("change"));

  await waitFor(() => previewText(scene).includes("What is mitosis?"), "the question to come back");
  assert.doesNotMatch(previewText(scene), /You're welcome/);
});

test("should_save_a_word_file_and_say_where_it_went", async () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "wholeChat" });
  click(control(scene, "export-format-docx"));
  assert.equal(control(scene, "export-save").textContent, "Save as Word…");
  await waitFor(() => previewText(scene).includes("You're welcome."), "the preview");
  scene.setSaveAnswer({ outcome: "saved", fileName: "Mitosis explained.docx" });

  click(control(scene, "export-save"));

  await waitFor(() => scene.announcedFileNames.length === 1, "the save to be announced");
  assert.equal(scene.saveRequests[0]?.format, ExportFormat.WORD);
  assert.equal(scene.saveRequests[0]?.suggestedName, "Mitosis explained");
  assert.ok((scene.saveRequests[0]?.base64 ?? "").length > 100, "the Word file travels as its bytes");
  assert.deepEqual(scene.announcedFileNames, ["Mitosis explained.docx"]);
  assert.equal(scene.sheetElement.hidden, true);
});

test("should_stay_open_when_the_save_panel_is_cancelled_and_say_when_saving_fails", async () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "wholeChat" });
  await waitFor(() => previewText(scene).includes("You're welcome."), "the preview");
  scene.setSaveAnswer({ outcome: "cancelled" });

  click(control(scene, "export-save"));
  await waitFor(() => scene.saveRequests.length === 1 && !control<HTMLButtonElement>(scene, "export-save").disabled, "the cancelled save to settle");
  assert.equal(scene.sheetElement.hidden, false);
  assert.equal(control(scene, "export-status").textContent, "");

  scene.setSaveAnswer(new Error("disk full"));
  click(control(scene, "export-save"));
  await waitFor(() => control(scene, "export-status").textContent !== "", "the failure to be said");
  assert.equal(control(scene, "export-status").textContent, "That file couldn't be saved. Try again.");
  assert.equal(scene.sheetElement.hidden, false);
  assert.deepEqual(scene.announcedFileNames, []);
});

test("should_not_save_when_the_choices_leave_nothing_to_export", async () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "oneAnswer", messageID: "missing" });

  await waitFor(() => control(scene, "export-status").textContent !== "", "the empty choice to be explained");
  assert.equal(control<HTMLButtonElement>(scene, "export-save").disabled, true);
  click(control(scene, "export-save"));
  assert.deepEqual(scene.saveRequests, []);
});

test("should_export_the_picked_messages_and_keep_the_sheet_as_it_was_when_picking_is_cancelled", async () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "wholeChat" });
  scene.setPickAnswer(null);
  click(control(scene, "export-scope-pickedMessages"));
  await waitFor(() => !scene.sheetElement.hidden, "the sheet to come back");
  assert.equal(control(scene, "export-scope-wholeChat").getAttribute("aria-checked"), "true");

  scene.setPickAnswer(["a2"]);
  click(control(scene, "export-scope-pickedMessages"));

  await waitFor(() => control(scene, "export-scope-pickedMessages").textContent === "1 picked…", "the picked count");
  await waitFor(() => !previewText(scene).includes("Cell division.") && previewText(scene).includes("You're welcome."), "the preview of the pick");
  assert.equal(control(scene, "export-scope-pickedMessages").getAttribute("aria-checked"), "true");
});

test("should_close_on_cancel_and_on_escape", () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "wholeChat" });
  click(control(scene, "export-cancel"));
  assert.equal(scene.sheetElement.hidden, true);

  scene.sheet.open({ kind: "wholeChat" });
  scene.sheetElement.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  assert.equal(scene.sheetElement.hidden, true);
});

test("should_estimate_pages_for_paper_formats_and_none_for_a_web_page", () => {
  assert.equal(pageCountText(ExportFormat.PDF, 0), "1 page");
  assert.equal(pageCountText(ExportFormat.PDF, 864), "1 page");
  assert.equal(pageCountText(ExportFormat.PDF, 865), "About 2 pages");
  assert.equal(pageCountText(ExportFormat.WORD, 864 * 3), "About 3 pages");
  assert.equal(pageCountText(ExportFormat.WEB_PAGE, 864 * 3), "One page that scrolls");
});

test("should_measure_the_previewed_text_at_full_size_whatever_scale_the_page_was_drawn_at", () => {
  const fullPageWidth = 8.5 * 96;
  const fiveInchesOfText = 5 * 96;
  const fullPageHeight = fiveInchesOfText + 2 * 96;
  assert.equal(previewTextHeightPixels(fullPageWidth, fullPageHeight), fiveInchesOfText, "reported unzoomed, as WebKit does");
  assert.equal(previewTextHeightPixels(fullPageWidth * 0.3, fullPageHeight * 0.3), fiveInchesOfText, "reported zoomed");
  assert.equal(previewTextHeightPixels(0, 500), 0, "a preview not drawn yet");
  assert.equal(previewTextHeightPixels(fullPageWidth, 100), 0, "margins alone are no text");
});

function pressKey(target: Element, key: string): void {
  target.dispatchEvent(new window.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
}

test("should_move_through_a_radio_group_with_the_arrow_keys_without_choosing", async () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "wholeChat" });
  const pdf = control(scene, "export-format-pdf");
  const word = control(scene, "export-format-docx");
  const webPage = control(scene, "export-format-html");

  pdf.focus();
  pressKey(pdf, "ArrowRight");
  assert.equal(document.activeElement, word);
  pressKey(word, "ArrowDown");
  assert.equal(document.activeElement, webPage);
  pressKey(webPage, "ArrowRight");
  assert.equal(document.activeElement, pdf, "the arrows wrap round");
  pressKey(pdf, "ArrowLeft");
  assert.equal(document.activeElement, webPage);

  assert.equal(pdf.getAttribute("aria-checked"), "true", "moving the focus chooses nothing");
  assert.deepEqual([pdf.tabIndex, word.tabIndex, webPage.tabIndex], [0, -1, -1], "Tab stops on the chosen one only");
});

test("should_skip_a_hidden_choice_and_never_start_picking_from_an_arrow_key", async () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "wholeChat" });
  const wholeChat = control(scene, "export-scope-wholeChat");
  const pickMessages = control(scene, "export-scope-pickedMessages");

  wholeChat.focus();
  pressKey(wholeChat, "ArrowRight");

  assert.equal(document.activeElement, pickMessages, "the hidden 'This answer' is skipped");
  assert.equal(scene.sheetElement.hidden, false, "arrowing onto Pick messages does not leave the sheet");
  pressKey(pickMessages, "a");
  assert.equal(document.activeElement, pickMessages, "other keys are left alone");
});

test("should_move_the_tab_stop_to_the_newly_chosen_format", () => {
  const scene = sheetScene();
  scene.sheet.open({ kind: "wholeChat" });

  click(control(scene, "export-format-docx"));

  assert.deepEqual(
    [control(scene, "export-format-pdf").tabIndex, control(scene, "export-format-docx").tabIndex],
    [-1, 0],
  );
});
