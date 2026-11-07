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
