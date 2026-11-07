import assert from "node:assert/strict";
import { test } from "node:test";
import { ToolStep, ToolStepState, UNFINISHED_TOOL_CALL_RESULT, UNKNOWN_TOOL_LABEL } from "../src/ToolStep";

function finishedStep(): ToolStep {
  return new ToolStep({
    toolCallID: "call-1",
    toolName: "get_current_time",
    label: "Current time",
    argumentsJSON: "{}",
    resultText: "Saturday, 2 PM",
    state: ToolStepState.SUCCEEDED,
  });
}

test("should_round_trip_a_finished_step_through_plain_json", () => {
  const restoredStep = ToolStep.fromPlain(JSON.parse(JSON.stringify(finishedStep().toPlain())));
  assert.deepEqual(restoredStep, finishedStep());
});

test("should_restore_a_step_saved_while_running_as_failed_with_the_unfinished_result", () => {
  const runningPlain = { ...finishedStep().toPlain(), state: ToolStepState.RUNNING, resultText: "" };
  const restoredStep = ToolStep.fromPlain(runningPlain);
  assert.equal(restoredStep?.state, ToolStepState.FAILED);
  assert.equal(restoredStep?.resultText, UNFINISHED_TOOL_CALL_RESULT);
});

test("should_label_a_step_with_its_tool_name_or_as_unknown_when_both_are_missing", () => {
  assert.equal(ToolStep.fromPlain({ ...finishedStep().toPlain(), label: "" })?.label, "get_current_time");
  const namelessStep = ToolStep.fromPlain({ ...finishedStep().toPlain(), label: "", toolName: "" });
  assert.equal(namelessStep?.label, UNKNOWN_TOOL_LABEL);
  assert.equal(namelessStep?.toolName, "");
});

test("should_keep_the_round_index_and_default_a_missing_or_bad_one_to_zero", () => {
  const laterRound = new ToolStep({ ...finishedStep(), roundIndex: 3 });
  assert.equal(ToolStep.fromPlain(laterRound.toPlain())?.roundIndex, 3);
  for (const badRoundIndex of [undefined, -1, 1.5, "2"]) {
    assert.equal(ToolStep.fromPlain({ ...finishedStep().toPlain(), roundIndex: badRoundIndex })?.roundIndex, 0, String(badRoundIndex));
  }
});

test("should_drop_a_step_without_a_call_id", () => {
  for (const damagedPlain of [null, "text", { toolName: "x" }, { toolCallID: "", toolName: "x" }, { toolCallID: 7 }]) {
    assert.equal(ToolStep.fromPlain(damagedPlain), null, JSON.stringify(damagedPlain));
  }
});

test("should_keep_a_stored_image_name_and_drop_one_that_is_not", () => {
  const storedName = "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.png";
  assert.equal(ToolStep.fromPlain({ ...finishedStep().toPlain(), generatedImageName: storedName })?.generatedImageName, storedName);
  for (const craftedName of ["../sessions/a.json", "x.png", `${storedName}" onerror="x`, 42]) {
    assert.equal(ToolStep.fromPlain({ ...finishedStep().toPlain(), generatedImageName: craftedName })?.generatedImageName, null, String(craftedName));
  }
});
