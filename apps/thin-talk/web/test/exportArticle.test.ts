import assert from "node:assert/strict";
import { test } from "node:test";
import { exportArticleHtml, exportPreviewMarkup, standaloneExportPage } from "../src/export/ExportArticle";
import type { ExportContent } from "../src/export/ExportContentBuilder";

function contentWithTitle(title: string, aiNote: string | null): ExportContent {
  const answerBody = document.createElement("div");
  answerBody.innerHTML = "<p>Cell division.</p>";
  return { title, dateText: "November 10, 2026", entries: [{ isQuestion: false, body: answerBody }], aiNote };
}

test("should_make_a_saved_page_that_loads_nothing_from_anywhere", () => {
  const page = standaloneExportPage(contentWithTitle("Mitosis", null));

  assert.match(page, /^<!DOCTYPE html>/);
  assert.match(page, /Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"/);
  assert.doesNotMatch(page, /<script/);
  assert.doesNotMatch(page, /thintalk-asset:/);
});

test("should_escape_a_title_so_it_can_never_become_markup", () => {
  const page = standaloneExportPage(contentWithTitle(`<img src=x onerror="alert(1)">`, null));

  assert.doesNotMatch(page, /<img src=x/);
  assert.match(page, /<title>&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;<\/title>/);
});

test("should_end_with_the_ai_note_only_when_it_is_included", () => {
  assert.match(exportArticleHtml(contentWithTitle("Mitosis", "Written with help from Astro, an AI on this Mac.")), /<p class="export-note">Written with help from Astro/);
  assert.doesNotMatch(exportArticleHtml(contentWithTitle("Mitosis", null)), /export-note/);
});

test("should_mark_questions_and_answers_apart", () => {
  const questionBody = document.createElement("div");
  questionBody.textContent = "What is mitosis?";
  const content: ExportContent = { ...contentWithTitle("Mitosis", null), entries: [{ isQuestion: true, body: questionBody }, ...contentWithTitle("Mitosis", null).entries] };

  const article = exportArticleHtml(content);

  assert.ok(article.indexOf(`<div class="export-question">What is mitosis?</div>`) < article.indexOf(`<div class="export-answer">`));
});

// The preview wrapped code where the paper did not and guessed 3 pages for a
// 2-page PDF while its text column was narrower than the printed one.
test("should_preview_on_the_printed_pages_text_column_and_margins", () => {
  const previewMarkup = exportPreviewMarkup(contentWithTitle("Mitosis", null));

  assert.match(previewMarkup, /\.export-article\{box-sizing:content-box;max-width:none;width:6\.5in;margin:0;padding:1in\}/);
  assert.match(previewMarkup, /\.export-preview-scaler\{width:8\.5in;transform:scale\(0\.3\);transform-origin:0 0\}/);
  assert.doesNotMatch(previewMarkup, /zoom:/, "zoomed text is held to WebKit's minimum size and wraps wrongly");
  assert.ok(previewMarkup.lastIndexOf("width:6.5in") > previewMarkup.lastIndexOf("max-width: 6.5in"), "the page geometry overrides the screen layout");
  assert.match(previewMarkup, /<article class="export-article">/);
});
