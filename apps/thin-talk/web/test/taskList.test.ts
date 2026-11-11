import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { SANITIZE_OPTIONS, TrustBoundary } from "../src/TrustBoundary";
import { MathPipeline } from "../src/MathPipeline";
import { DiagramPipeline } from "../src/DiagramPipeline";
import { DONE_TASK_BOX_CLASS, Renderer, TASK_BOX_CLASS } from "../src/Renderer";
import { TranscriptView } from "../src/TranscriptView";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";
import { cleanExportBody } from "../src/export/ExportBodyCleaner";

// The sanitiser reduced markdown's checkbox to a bare input, so the chat drew
// an empty text field in front of every checklist item and lost every tick.

async function renderedAnswer(markdown: string): Promise<{ view: TranscriptView; body: HTMLElement }> {
  const trust = new TrustBoundary();
  const transcriptHost = document.createElement("div");
  const view = new TranscriptView({
    transcript: transcriptHost,
    emptyState: document.createElement("div"),
    renderer: new Renderer({ trust, math: new MathPipeline(trust), diagrams: new DiagramPipeline() }),
    onAction: () => undefined,
    onError: () => undefined,
  });
  view.render({ messages: [new TranscriptMessage({ id: "a", role: ChatRole.ASSISTANT, markdown, reasoning: "", state: ChatMessageState.COMPLETE })], notice: null });
  await waitFor(() => transcriptHost.querySelector(".markdown-body") !== null, "the answer to be drawn");
  return { view, body: transcriptHost.querySelector(".markdown-body") as HTMLElement };
}

test("should_draw_a_checklist_with_ticked_and_empty_boxes_and_no_form_control", async () => {
  const { body } = await renderedAnswer("- [x] Explain where it happens\n- [ ] Draw the two stages\n- A plain item");

  const taskBoxes = [...body.querySelectorAll(`.${TASK_BOX_CLASS}`)];
  assert.equal(body.querySelector("input"), null, "no form control in model output");
  assert.deepEqual(taskBoxes.map((taskBox) => taskBox.classList.contains(DONE_TASK_BOX_CLASS)), [true, false]);
  assert.deepEqual(taskBoxes.map((taskBox) => taskBox.getAttribute("title")), ["Done", "Not done"]);
  assert.equal(body.querySelectorAll("li").length, 3);
  assert.match(body.textContent ?? "", /Explain where it happens[\s\S]*Draw the two stages[\s\S]*A plain item/);
});

// The test window's DOM cannot run the sanitiser, so the policy is checked
// as written; a WebKit run of the same policy removed the input and kept the
// text around it.
test("should_forbid_every_form_control_in_model_markup_and_allow_the_drawn_box", () => {
  const allowedTags = SANITIZE_OPTIONS.ALLOWED_TAGS ?? [];
  const forbiddenTags = SANITIZE_OPTIONS.FORBID_TAGS ?? [];
  const allowedAttributes = SANITIZE_OPTIONS.ALLOWED_ATTR ?? [];

  for (const formControl of ["input", "button", "select", "textarea", "form"]) {
    assert.ok(forbiddenTags.includes(formControl), formControl);
    assert.ok(!allowedTags.includes(formControl), formControl);
  }
  assert.ok(allowedTags.includes("span") && allowedAttributes.includes("class") && allowedAttributes.includes("title"), "the drawn box survives");
  for (const formAttribute of ["type", "checked", "disabled", "value"]) {
    assert.ok(!allowedAttributes.includes(formAttribute), formAttribute);
  }
});

test("should_leave_text_in_square_brackets_alone_outside_a_list", async () => {
  const { body } = await renderedAnswer("See [x] and [ ] in the text.");

  assert.equal(body.querySelector(`.${TASK_BOX_CLASS}`), null);
  assert.match(body.textContent ?? "", /See \[x\] and \[ \] in the text\./);
});

test("should_export_the_drawn_boxes_as_tick_characters", async () => {
  const { view } = await renderedAnswer("- [x] Done\n- [ ] Todo");

  const exportedBody = cleanExportBody(view.renderedBodyClone("a") as HTMLElement, false);

  assert.deepEqual([...exportedBody.querySelectorAll("li")].map((item) => (item.textContent ?? "").replace(/\s+/g, " ").trim()), ["☑ Done", "☐ Todo"]);
});
