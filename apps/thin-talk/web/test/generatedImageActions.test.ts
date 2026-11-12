import assert from "node:assert/strict";
import { test } from "node:test";
import { assertNoElement, waitFor } from "./support/dom";
import { TrustBoundary } from "../src/TrustBoundary";
import { MathPipeline } from "../src/MathPipeline";
import { DiagramPipeline } from "../src/DiagramPipeline";
import { Renderer } from "../src/Renderer";
import { TranscriptView } from "../src/TranscriptView";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { TranscriptAction } from "../src/TranscriptAction";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";
import { ToolStep, ToolStepState } from "../src/ToolStep";

// A generated picture is made to be kept or sent to someone. Two buttons under
// it ask the host to save a copy or put it on the pasteboard; the page only
// ever names the stored image, because the host owns the files.

const STORED_IMAGE_NAME = "0b1d7f0e-3a52-4c8e-9d11-2f6a4c1e7b90.png";

function imageStep(generatedImageName: string | null): ToolStep {
  return new ToolStep({
    toolCallID: "call-draw",
    toolName: "generate_image",
    label: "Image generated",
    argumentsJSON: '{"prompt":"A crown"}',
    resultText: "Generated.",
    state: ToolStepState.SUCCEEDED,
    generatedImageName,
    generatedImageModelName: "flux2-klein-4b-4bit",
  });
}

function renderAnswerWith(toolSteps: ToolStep[]): { transcriptHost: HTMLElement; reportedActions: TranscriptAction[] } {
  const trust = new TrustBoundary();
  const renderer = new Renderer({ trust, math: new MathPipeline(trust), diagrams: new DiagramPipeline() });
  const transcriptHost = document.createElement("div");
  const reportedActions: TranscriptAction[] = [];
  const view = new TranscriptView({
    transcript: transcriptHost,
    emptyState: document.createElement("div"),
    renderer,
    onAction: (action: TranscriptAction) => reportedActions.push(action),
    onError: (detail: string) => detail,
  });
  view.render({
    messages: [new TranscriptMessage({ id: "answer-with-picture", role: ChatRole.ASSISTANT, markdown: "Here it is.", reasoning: "", state: ChatMessageState.COMPLETE, toolSteps })],
    notice: null,
  });
  return { transcriptHost, reportedActions };
}

function pictureButtons(transcriptHost: HTMLElement): HTMLButtonElement[] {
  return [...transcriptHost.querySelectorAll<HTMLButtonElement>(".generated-image__actions button")];
}

function click(button: HTMLButtonElement): void {
  button.dispatchEvent(new window.Event("click", { bubbles: true }));
}

test("should_offer_save_and_copy_under_a_generated_picture", async () => {
  const { transcriptHost } = renderAnswerWith([imageStep(STORED_IMAGE_NAME)]);
  await waitFor(() => pictureButtons(transcriptHost).length === 2, "the picture's buttons to render");

  assert.deepEqual(pictureButtons(transcriptHost).map((button) => button.textContent), ["Save…", "Copy picture"]);
});

test("should_ask_the_host_to_save_the_stored_picture_by_name_only", async () => {
  const { transcriptHost, reportedActions } = renderAnswerWith([imageStep(STORED_IMAGE_NAME)]);
  await waitFor(() => pictureButtons(transcriptHost).length === 2, "the picture's buttons to render");

  click(pictureButtons(transcriptHost)[0]!);

  assert.equal(reportedActions.length, 1);
  assert.equal(reportedActions[0]!.kind, "saveImage");
  assert.equal(reportedActions[0]!.detail, STORED_IMAGE_NAME);
  assert.equal(pictureButtons(transcriptHost)[0]!.textContent, "Save…", "the save panel is the confirmation");
});

test("should_ask_the_host_to_copy_the_picture_and_confirm_it_on_the_button", async () => {
  const { transcriptHost, reportedActions } = renderAnswerWith([imageStep(STORED_IMAGE_NAME)]);
  await waitFor(() => pictureButtons(transcriptHost).length === 2, "the picture's buttons to render");

  click(pictureButtons(transcriptHost)[1]!);

  assert.deepEqual(reportedActions.map((action) => [action.kind, action.detail]), [["copyImage", STORED_IMAGE_NAME]]);
  assert.equal(pictureButtons(transcriptHost)[1]!.textContent, "Copied");
});

test("should_offer_no_picture_buttons_for_a_step_without_a_stored_picture", async () => {
  // A name the store could not have produced is dropped when the step is
  // built, so neither a picture nor its buttons are drawn for it.
  for (const unusableImageName of [null, "../../etc/passwd", "picture.gif"]) {
    const { transcriptHost } = renderAnswerWith([imageStep(unusableImageName)]);
    await waitFor(() => transcriptHost.querySelector(".tool-step") !== null, "the step to render");

    assert.deepEqual(pictureButtons(transcriptHost), [], String(unusableImageName));
    assertNoElement(transcriptHost, ".generated-image", String(unusableImageName));
  }
});
