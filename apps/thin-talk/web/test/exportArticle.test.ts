import assert from "node:assert/strict";
import { test } from "node:test";
import { exportArticleHtml, exportPreviewMarkup, hostPrintedExportPage, standaloneExportPage } from "../src/export/ExportArticle";
import type { ExportContent } from "../src/export/ExportContentBuilder";
import { A4_PAPER, LETTER_PAPER } from "../src/export/PaperSize";

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
  const previewMarkup = exportPreviewMarkup(contentWithTitle("Mitosis", null), LETTER_PAPER);

  assert.match(previewMarkup, /\.export-article\{box-sizing:content-box;max-width:none;width:6\.5in;margin:0;padding:1in\}/);
  assert.match(previewMarkup, /\.export-preview-scaler\{width:8\.5in;transform:scale\(0\.3\);transform-origin:0 0\}/);
  assert.doesNotMatch(previewMarkup, /zoom:/, "zoomed text is held to WebKit's minimum size and wraps wrongly");
  assert.ok(previewMarkup.lastIndexOf("width:6.5in") > previewMarkup.lastIndexOf("max-width: 6.5in"), "the page geometry overrides the screen layout");
  assert.match(previewMarkup, /<article class="export-article">/);
});

function contentWithAnswerMarkup(answerMarkup: string): ExportContent {
  const answerBody = document.createElement("div");
  answerBody.innerHTML = answerMarkup;
  return { title: "Water", dateText: "November 10, 2026", entries: [{ isQuestion: false, body: answerBody }], aiNote: null };
}

// A printed PDF left "Water Cycle Flowchart" alone at the foot of page 2 with
// its diagram on page 3: WebKit ignores break-after on the heading.
test("should_group_a_heading_with_the_block_it_introduces_on_a_printed_page", () => {
  const page = standaloneExportPage(contentWithAnswerMarkup(`<p>Intro.</p><h2>Flowchart</h2><figure class="export-diagram"></figure><p>After.</p>`));

  assert.match(page, /<p>Intro\.<\/p><div class="export-keep"><h2>Flowchart<\/h2><figure class="export-diagram"><\/figure><\/div><p>After\.<\/p>/);
  assert.match(page, /\.export-keep \{ break-inside: avoid; \}|, \.export-keep \{ break-inside: avoid; \}/);
});

test("should_group_headings_that_follow_one_another_with_one_block", () => {
  const preview = exportPreviewMarkup(contentWithAnswerMarkup(`<h2>Stages</h2><h3>Evaporation</h3><p>Sun heats water.</p><p>More.</p>`), LETTER_PAPER);

  assert.match(preview, /<div class="export-keep"><h2>Stages<\/h2><h3>Evaporation<\/h3><p>Sun heats water\.<\/p><\/div><p>More\.<\/p>/);
});

test("should_group_every_heading_in_an_answer_not_only_the_first", () => {
  const page = standaloneExportPage(contentWithAnswerMarkup(`<h2>One</h2><p>a</p><h2>Two</h2><p>b</p>`));

  assert.equal(page.split(`<div class="export-keep">`).length - 1, 2);
});

test("should_not_hold_a_long_list_to_its_heading_or_keep_it_whole", () => {
  const sevenItems = "<li>Step</li>".repeat(7);
  const page = standaloneExportPage(contentWithAnswerMarkup(`<h2>Steps</h2><ul>${sevenItems}</ul>`));

  assert.match(page, /<div class="export-answer"><h2>Steps<\/h2><ul><li>/);
  assert.doesNotMatch(page, /class="export-keep"/);
});

// A three-item checklist printed two items at the foot of one page and the
// third at the top of the next.
test("should_keep_a_short_list_whole_alone_or_under_its_heading", () => {
  const sixItems = "<li>Step</li>".repeat(6);
  const underHeading = standaloneExportPage(contentWithAnswerMarkup(`<h2>Checklist</h2><ul>${sixItems}</ul><p>After.</p>`));
  const alone = standaloneExportPage(contentWithAnswerMarkup(`<p>Before.</p><ol>${sixItems}</ol>`));

  assert.match(underHeading, /<div class="export-keep"><h2>Checklist<\/h2><ul>.*<\/ul><\/div><p>After\.<\/p>/);
  assert.match(alone, /<p>Before\.<\/p><ol class="export-keep">/);
});

test("should_treat_a_list_with_few_items_but_much_text_as_long", () => {
  const wordyItem = `<li>${"word ".repeat(70)}</li>`;
  const page = standaloneExportPage(contentWithAnswerMarkup(`<ul>${wordyItem}${wordyItem}</ul>`));

  assert.doesNotMatch(page, /class="export-keep"/);
});

// "…packed into a given volume:" ended one page and its formula began the next.
test("should_keep_a_sentence_ending_in_a_colon_with_the_block_it_leads_into", () => {
  const page = standaloneExportPage(contentWithAnswerMarkup(`<p>Intro.</p><p>The density is:</p><math display="block"></math><p>Where m is mass.</p>`));

  assert.match(page, /<p>Intro\.<\/p><div class="export-keep"><p>The density is:<\/p><math display="block"><\/math><\/div><p>Where m is mass\.<\/p>/);
});

test("should_keep_a_heading_its_lead_in_sentence_and_the_block_after_it_as_one_group", () => {
  const page = standaloneExportPage(contentWithAnswerMarkup(`<h2>Density</h2><p>The formula is:</p><math display="block"></math><p>After.</p>`));

  assert.match(page, /<div class="export-keep"><h2>Density<\/h2><p>The formula is:<\/p><math display="block"><\/math><\/div><p>After\.<\/p>/);
  assert.equal(page.split(`class="export-keep"`).length - 1, 1, "one group, not one inside another");
});

test("should_leave_a_lead_in_sentence_alone_before_a_long_list_a_heading_or_nothing", () => {
  const sevenItems = "<li>Step</li>".repeat(7);
  const beforeLongList = standaloneExportPage(contentWithAnswerMarkup(`<p>Do these:</p><ul>${sevenItems}</ul>`));
  const beforeHeading = standaloneExportPage(contentWithAnswerMarkup(`<p>Next:</p><h2>Part two</h2>`));
  const atTheEnd = standaloneExportPage(contentWithAnswerMarkup(`<p>Text.</p><p>In short:</p>`));

  for (const page of [beforeLongList, beforeHeading, atTheEnd]) {
    assert.doesNotMatch(page, /class="export-keep"/);
  }
});

test("should_not_take_a_colon_inside_a_sentence_for_a_lead_in", () => {
  const page = standaloneExportPage(contentWithAnswerMarkup(`<p>Note: water boils at 100 degrees.</p><p>More.</p>`));

  assert.doesNotMatch(page, /class="export-keep"/);
});


test("should_leave_a_heading_with_nothing_after_it_ungrouped", () => {
  const page = standaloneExportPage(contentWithAnswerMarkup(`<p>Text.</p><h2>The end</h2>`));

  assert.doesNotMatch(page, /<div class="export-keep">/);
});

test("should_give_word_the_article_without_page_groups_and_leave_the_body_untouched", () => {
  const content = contentWithAnswerMarkup(`<h2>Flowchart</h2><p>Text.</p>`);

  standaloneExportPage(content);

  assert.doesNotMatch(exportArticleHtml(content), /export-keep/);
  assert.equal(content.entries[0]?.body.innerHTML, `<h2>Flowchart</h2><p>Text.</p>`);
});

// The printed PDF drew a 22-point title at 23.47 points and broke pages
// before the preview did; the host's print test pins the engine side.
test("should_zoom_only_the_page_the_host_prints_so_sizes_on_paper_are_the_sizes_asked_for", () => {
  const content = contentWithTitle("Mitosis", null);

  assert.match(hostPrintedExportPage(content), /@media print \{ body \{ zoom: 0\.9375; \} \}<\/style>/);
  assert.doesNotMatch(standaloneExportPage(content), /zoom:/, "Chrome prints a saved page at true size already");
  assert.doesNotMatch(exportPreviewMarkup(content, LETTER_PAPER), /zoom:/);
  assert.equal(hostPrintedExportPage(content).replace(/@media print \{ body \{ zoom: 0\.9375; \} \}/, ""), standaloneExportPage(content), "nothing else differs");
});

// On A4 the column is a quarter of an inch narrower, which moves line and
// page breaks; a Letter preview there showed pages the printer did not make.
test("should_preview_on_a4_when_the_mac_prints_on_a4", () => {
  const previewMarkup = exportPreviewMarkup(contentWithTitle("Mitosis", null), A4_PAPER);

  assert.match(previewMarkup, /\.export-preview-scaler\{width:8\.27in;/);
  assert.match(previewMarkup, /max-width:none;width:6\.27\d*in;margin:0;padding:1in\}/);
});
