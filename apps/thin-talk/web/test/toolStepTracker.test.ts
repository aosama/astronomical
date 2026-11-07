import assert from "node:assert/strict";
import { test } from "node:test";
import { ToolStepState, UNKNOWN_TOOL_LABEL } from "../src/ToolStep";
import { ToolStepTracker } from "../src/ToolStepTracker";

function tracker(): ToolStepTracker {
  return new ToolStepTracker((toolName) => (toolName === "get_current_time" ? "Current time" : toolName));
}

test("should_finish_a_step_with_the_arguments_it_started_with", () => {
  const toolSteps = tracker();
  assert.equal(toolSteps.started("call-1", "get_current_time", { zone: "local" }).state, ToolStepState.RUNNING);
  const finishedStep = toolSteps.finished("call-1", "get_current_time", { content: [{ type: "text", text: "2 PM" }] }, false);
  assert.equal(finishedStep.argumentsJSON, '{"zone":"local"}');
  assert.equal(finishedStep.resultText, "2 PM");
  assert.equal(finishedStep.state, ToolStepState.SUCCEEDED);
  assert.equal(finishedStep.label, "Current time");
});

test("should_number_rounds_from_zero_and_keep_a_calls_round_after_it_completes", () => {
  const toolSteps = tracker();
  toolSteps.started("call-1", "get_current_time", {});
  toolSteps.completeRound();
  assert.equal(toolSteps.finished("call-1", "get_current_time", {}, false).roundIndex, 0);
  assert.equal(toolSteps.started("call-2", "get_current_time", {}).roundIndex, 1);
  assert.equal(toolSteps.completedRounds, 1);
});

test("should_join_text_results_and_ignore_other_content", () => {
  const finishedStep = tracker().finished(
    "call-1",
    "get_current_time",
    { content: [{ type: "text", text: "a" }, { type: "image", data: "…" }, null, { type: "text", text: "b" }] },
    true,
  );
  assert.equal(finishedStep.resultText, "a\nb");
  assert.equal(finishedStep.state, ToolStepState.FAILED);
  assert.equal(finishedStep.argumentsJSON, "{}", "a call that never started records no arguments");
});

test("should_label_a_nameless_call_as_unknown_and_survive_unencodable_arguments", () => {
  const startedStep = tracker().started("call-1", "", { tick: BigInt(1) });
  assert.equal(startedStep.label, UNKNOWN_TOOL_LABEL);
  assert.equal(startedStep.argumentsJSON, "{}");
});

test("should_record_the_generated_image_name_it_is_given", () => {
  const storedName = "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.jpg";
  assert.equal(tracker().finished("call-1", "generate_image", { content: [] }, false, storedName).generatedImageName, storedName);
  assert.equal(tracker().finished("call-2", "generate_image", { content: [] }, false, "../x.png").generatedImageName, null);
  assert.equal(tracker().finished("call-3", "generate_image", { content: [] }, false).generatedImageName, null);
});
