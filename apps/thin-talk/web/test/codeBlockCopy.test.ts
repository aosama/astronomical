import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { TrustBoundary } from "../src/TrustBoundary";
import { MathPipeline } from "../src/MathPipeline";
import { DiagramPipeline } from "../src/DiagramPipeline";
import { Renderer } from "../src/Renderer";
import { TranscriptView } from "../src/TranscriptView";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { TranscriptAction } from "../src/TranscriptAction";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";

// People copy a script far more often than a whole answer. Every fenced block
// carries its own Copy button, which copies the code and nothing around it,
// and neither that button nor the language label leaks into a copied answer.

const ANSWER_WITH_CODE = [
  "Here is the script:",
  "",
  "```python",
  "import random",
  "print(random.choice(excuses))",
  "```",
  "",
  "And how to run it:",
  "",
  "```",
  "python excuse.py",
  "```",
].join("\n");

function renderAnswer(markdown: string): { view: TranscriptView; transcriptHost: HTMLElement; reportedActions: TranscriptAction[] } {
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
    messages: [new TranscriptMessage({ id: "answer-with-code", role: ChatRole.ASSISTANT, markdown, reasoning: "", state: ChatMessageState.COMPLETE })],
    notice: null,
  });
  return { view, transcriptHost, reportedActions };
}

function copyButtons(transcriptHost: HTMLElement): HTMLButtonElement[] {
  return [...transcriptHost.querySelectorAll<HTMLButtonElement>("button.code-block__copy")];
}

test("should_give_every_code_block_a_copy_button_and_a_label_only_when_the_fence_names_a_language", async () => {
  const { transcriptHost } = renderAnswer(ANSWER_WITH_CODE);
  await waitFor(() => copyButtons(transcriptHost).length === 2, "both code blocks to get a Copy button");

  const languageLabels = [...transcriptHost.querySelectorAll(".code-block__label")].map((label) => label.textContent);
  assert.deepEqual(languageLabels, ["python"]);
});

test("should_copy_only_that_blocks_code_and_confirm_it_on_the_button", async () => {
  const { transcriptHost, reportedActions } = renderAnswer(ANSWER_WITH_CODE);
  await waitFor(() => copyButtons(transcriptHost).length === 2, "both code blocks to get a Copy button");

  const secondBlockCopy = copyButtons(transcriptHost)[1]!;
  secondBlockCopy.dispatchEvent(new window.Event("click", { bubbles: true }));

  assert.equal(reportedActions.length, 1);
  assert.equal(reportedActions[0]!.kind, "copy");
  assert.equal(reportedActions[0]!.detail.trim(), "python excuse.py");
  assert.equal(secondBlockCopy.textContent, "Copied");
  assert.equal(copyButtons(transcriptHost)[0]!.textContent, "Copy", "only the clicked block confirms");
});

test("should_leave_code_block_labels_and_buttons_out_of_a_copied_answer", async () => {
  const { view, transcriptHost } = renderAnswer(ANSWER_WITH_CODE);
  await waitFor(() => copyButtons(transcriptHost).length === 2, "both code blocks to get a Copy button");

  const copiedAnswer = view.renderedText("answer-with-code");

  assert.match(copiedAnswer, /import random/);
  assert.match(copiedAnswer, /python excuse\.py/);
  assert.ok(!copiedAnswer.includes("Copy"), "the Copy buttons are not part of the answer");
  assert.ok(!/excuses\)\)\s*python\s*And how/.test(copiedAnswer), "the language label is not pasted after the code");
  assert.equal(copyButtons(transcriptHost).length, 2, "copying reads a copy and leaves the page as it was");
});

test("should_add_no_copy_button_to_an_answer_without_code", async () => {
  const { view, transcriptHost } = renderAnswer("Mitosis has four phases: prophase, metaphase, anaphase and telophase.");
  await waitFor(() => view.renderedText("answer-with-code").includes("Mitosis"), "the answer to render");

  assert.deepEqual(copyButtons(transcriptHost), []);
});
