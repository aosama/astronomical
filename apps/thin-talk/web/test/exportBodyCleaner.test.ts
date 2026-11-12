import assert from "node:assert/strict";
import { assertNoElement } from "./support/dom";
import { test } from "node:test";
import { cleanExportBody, EXPORT_DARK_DIAGRAM_CLASS, EXPORT_DIAGRAM_CLASS } from "../src/export/ExportBodyCleaner";

// An exported file has none of the window around it: no app stylesheet, no
// KaTeX fonts, no private image scheme, no working buttons. The cleaner turns
// a rendered answer into markup that stands on its own.

function renderedBody(innerHtml: string): HTMLElement {
  const body = document.createElement("div");
  body.className = "markdown-body";
  body.innerHTML = innerHtml;
  return body;
}

const HIGHLIGHTED_CODE_BLOCK =
  `<div class="code-block" data-md-key="code-1"><pre><code class="language-python hljs" data-highlighted="true"><span class="hljs-keyword">print</span>("hi")</code></pre>` +
  `<div class="code-block__tools"><span class="code-block__label">python</span><button class="code-block__copy" data-action="copy-code">Copy</button></div></div>`;

test("should_turn_a_highlighted_code_block_into_plain_code_without_its_label_or_copy_button", () => {
  const cleanedBody = cleanExportBody(renderedBody(HIGHLIGHTED_CODE_BLOCK), false);

  assert.equal(cleanedBody.innerHTML, `<pre><code>print("hi")</code></pre>`);
});

test("should_keep_a_drawn_diagram_and_mark_it_dark_only_when_drawn_in_the_dark", () => {
  const diagramBlock = `<div class="code-block code-block--diagram"><figure class="diagram"><svg viewBox="0 0 10 10"><text>A</text></svg></figure></div>`;

  const lightBody = cleanExportBody(renderedBody(diagramBlock), false);
  const darkBody = cleanExportBody(renderedBody(diagramBlock), true);

  assert.equal(lightBody.querySelector("figure")?.className, EXPORT_DIAGRAM_CLASS);
  assert.ok(lightBody.querySelector("figure > svg"), "the drawing is kept, not its source");
  assert.equal(darkBody.querySelector("figure")?.className, `${EXPORT_DIAGRAM_CLASS} ${EXPORT_DARK_DIAGRAM_CLASS}`);
});

test("should_keep_katex_maths_as_mathml_and_mark_display_maths_as_a_block", () => {
  const katexMaths =
    `<p>Inline <span class="katex"><span class="katex-mathml"><math><mi>x</mi></math></span><span class="katex-html" aria-hidden="true">x</span></span></p>` +
    `<span class="katex-display"><span class="katex"><span class="katex-mathml"><math><mi>y</mi></math></span><span class="katex-html">y</span></span></span>`;

  const cleanedBody = cleanExportBody(renderedBody(katexMaths), false);

  assert.equal(cleanedBody.querySelectorAll("math").length, 2);
  assertNoElement(cleanedBody, ".katex, .katex-html", "the positioned spans need KaTeX's stylesheet");
  assert.equal(cleanedBody.querySelectorAll("math")[1]?.getAttribute("display"), "block");
  assert.equal(cleanedBody.querySelectorAll("math")[0]?.getAttribute("display"), null);
});

test("should_fall_back_to_the_latex_source_when_katex_left_no_mathml", () => {
  const cleanedBody = cleanExportBody(renderedBody(`<p><span class="katex"><span class="katex-html">\\frac{a}{b}</span></span></p>`), false);

  assert.equal(cleanedBody.textContent, "\\frac{a}{b}");
});

test("should_replace_a_picture_the_file_cannot_carry_with_its_description", () => {
  const cleanedBody = cleanExportBody(
    renderedBody(`<p><img src="thintalk-asset://attachment/abc" alt="A cat"><img src="https://example.com/x.png" alt=""><img src="data:image/png;base64,AAAA" alt="kept"></p>`),
    false,
  );

  assert.equal(cleanedBody.querySelectorAll("img").length, 1, "only inline image data travels with the file");
  assert.equal(cleanedBody.querySelector("img")?.getAttribute("alt"), "kept");
  assert.match(cleanedBody.textContent ?? "", /\[Picture: A cat\]/);
});

test("should_open_folded_sections_flat_with_their_summary_as_a_bold_line", () => {
  const cleanedBody = cleanExportBody(renderedBody(`<details><summary>Hint</summary><p>Look closer.</p></details>`), false);

  assertNoElement(cleanedBody, "details");
  assert.equal(cleanedBody.querySelector("strong")?.textContent, "Hint");
  assert.match(cleanedBody.textContent ?? "", /Look closer\./);
});

test("should_drop_the_pages_own_attributes_and_any_link_a_reader_could_not_open", () => {
  const cleanedBody = cleanExportBody(
    renderedBody(`<p id="p1" class="x" data-md-key="k"><a href="https://example.com" rel="noopener" target="_blank">site</a> <a href="thintalk-asset://shell/x">inside</a> <a href="javascript:alert(1)">bad</a></p><table><tr><td colspan="2" class="c">cell</td></tr></table>`),
    false,
  );

  const paragraph = cleanedBody.querySelector("p");
  assert.deepEqual(paragraph?.getAttributeNames(), []);
  const linkTargets = [...cleanedBody.querySelectorAll("a")].map((link) => link.getAttribute("href"));
  assert.deepEqual(linkTargets, ["https://example.com", null, null]);
  assert.deepEqual(cleanedBody.querySelector("a")?.getAttributeNames(), ["href"]);
  assert.deepEqual(cleanedBody.querySelector("td")?.getAttributeNames(), ["colspan"]);
  assert.deepEqual(cleanedBody.getAttributeNames(), [], "the body's own class goes too");
});

test("should_leave_an_answer_of_plain_prose_as_it_was", () => {
  const cleanedBody = cleanExportBody(renderedBody("<p>Mitosis makes <strong>two</strong> cells.</p>"), false);

  assert.equal(cleanedBody.innerHTML, "<p>Mitosis makes <strong>two</strong> cells.</p>");
});

test("should_leave_an_empty_body_empty", () => {
  assert.equal(cleanExportBody(renderedBody(""), false).innerHTML, "");
});

// The transcript's drawn boxes need its stylesheet; a file has only text.
test("should_turn_checklist_boxes_into_characters_and_drop_the_items_bullets", () => {
  const cleanedBody = cleanExportBody(
    renderedBody(`<ul><li><span class="task-box task-box--done" title="Done"></span> Done</li><li><span class="task-box" title="Not done"></span> Todo</li><li>Plain</li></ul>`),
    false,
  );

  assertNoElement(cleanedBody, ".task-box");
  const items = [...cleanedBody.querySelectorAll("li")];
  assert.deepEqual(items.map((item) => (item.textContent ?? "").replace(/\s+/g, " ").trim()), ["☑ Done", "☐ Todo", "Plain"]);
  assert.deepEqual(items.map((item) => item.className), ["export-task", "export-task", ""]);
});

test("should_give_a_dark_ground_only_to_a_dark_diagram_not_drawn_again_for_paper", () => {
  const paperDrawn = `<div class="code-block"><figure class="diagram" data-paper-drawn=""><svg></svg></figure></div>`;

  assert.equal(cleanExportBody(renderedBody(paperDrawn), true).querySelector("figure")?.className, "export-diagram");
});
