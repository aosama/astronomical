import assert from "node:assert/strict";
import { test } from "node:test";
import { TrustBoundary } from "../src/TrustBoundary";
import { MathPipeline } from "../src/MathPipeline";
import { DiagramPipeline } from "../src/DiagramPipeline";
import { Renderer, toolStepIdentifier } from "../src/Renderer";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";
import { ToolStep, ToolStepState } from "../src/ToolStep";

// Tool-step rows build their own markup from model and tool output, so every
// value they show must be escaped; nothing the model writes may become markup.

const trust = new TrustBoundary();
const renderer = new Renderer({ trust, math: new MathPipeline(trust), diagrams: new DiagramPipeline() });

function renderedSteps(toolSteps: ToolStep[]): HTMLElement {
  const container = document.createElement("div");
  container.innerHTML = renderer.messageHtml(
    new TranscriptMessage({ id: "answer-1", role: ChatRole.ASSISTANT, markdown: "", reasoning: "", state: ChatMessageState.COMPLETE, toolSteps }),
  );
  return container;
}

test("should_draw_one_collapsed_row_per_step_with_its_outcome", () => {
  const container = renderedSteps([
    new ToolStep({ toolCallID: "call-1", toolName: "get_current_time", label: "Current time", argumentsJSON: "{}", resultText: "2 PM", state: ToolStepState.SUCCEEDED }),
    new ToolStep({ toolCallID: "call-2", toolName: "get_current_time", label: "Current time", argumentsJSON: "{}", resultText: "", state: ToolStepState.RUNNING }),
  ]);
  const rows = [...container.querySelectorAll<HTMLDetailsElement>("details.tool-step")];
  assert.deepEqual(rows.map((row) => row.querySelector("summary")?.textContent), ["Current time: done", "Current time: running"]);
  assert.ok(rows.every((row) => !row.open));
  assert.equal(rows[0]?.id, toolStepIdentifier("answer-1", "call-1"));
  assert.match(rows[1]?.textContent ?? "", /Waiting for the result/);
});

test("should_escape_markup_in_every_value_a_row_shows", () => {
  const hostile = '<img src=x onerror="alert(1)">"';
  const container = renderedSteps([
    new ToolStep({ toolCallID: hostile, toolName: hostile, label: hostile, argumentsJSON: hostile, resultText: hostile, state: ToolStepState.FAILED }),
  ]);
  assert.equal(container.querySelector("img"), null, "no element is created from tool output");
  const row = container.querySelector<HTMLDetailsElement>("details.tool-step");
  assert.equal(row?.id, toolStepIdentifier("answer-1", hostile));
  assert.equal(row?.dataset.state, "failed");
  assert.ok((row?.textContent ?? "").includes(hostile));
});

test("should_draw_no_tool_rows_for_a_message_without_steps", () => {
  assert.equal(renderedSteps([]).querySelector(".tool-steps"), null);
});

test("should_show_a_generated_image_outside_the_rows_with_its_prompt_as_text", () => {
  const container = renderedSteps([
    new ToolStep({
      toolCallID: "call-9",
      toolName: "generate_image",
      label: "Image generation",
      argumentsJSON: JSON.stringify({ prompt: "A rose <b>at</b> dusk" }),
      resultText: "Generated.",
      state: ToolStepState.SUCCEEDED,
      generatedImageName: "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.png",
    }),
  ]);
  const image = container.querySelector<HTMLImageElement>("figure.generated-image img");
  assert.equal(image?.getAttribute("src"), "thintalk-asset://generated/0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.png");
  assert.equal(image?.getAttribute("alt"), "Generated image: A rose <b>at</b> dusk");
  assert.equal(image?.closest("details"), null, "the image is not hidden inside the collapsed row");
  assert.equal(container.querySelector("b"), null);
});

function imageStep(argumentsJSON: string, generatedImageName: string | null): ToolStep {
  return new ToolStep({
    toolCallID: "call-7", toolName: "generate_image", label: "Image generation",
    argumentsJSON, resultText: "Generated.", state: ToolStepState.SUCCEEDED, generatedImageName,
  });
}

test("should_fall_back_to_a_plain_description_and_shorten_a_long_prompt", () => {
  const jpegName = "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.jpg";
  for (const unusableArguments of ["{not json", "{}", JSON.stringify({ prompt: 42 }), JSON.stringify({ prompt: "  " })]) {
    const image = renderedSteps([imageStep(unusableArguments, jpegName)]).querySelector("img");
    assert.equal(image?.getAttribute("alt"), "Generated image", unusableArguments);
    assert.match(image?.getAttribute("src") ?? "", /\.jpg$/);
  }
  const longPrompt = "r".repeat(400);
  const longAlt = renderedSteps([imageStep(JSON.stringify({ prompt: longPrompt }), jpegName)]).querySelector("img")?.getAttribute("alt") ?? "";
  assert.equal(longAlt, `Generated image: ${"r".repeat(250)}…`);
});

test("should_show_no_image_for_a_step_without_one", () => {
  assert.equal(renderedSteps([imageStep('{"prompt":"A rose"}', null)]).querySelector("figure"), null);
});
