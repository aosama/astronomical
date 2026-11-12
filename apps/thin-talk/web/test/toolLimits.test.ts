import assert from "node:assert/strict";
import { test } from "node:test";
import type { ConversationTool } from "../src/ConversationTool";
import { MAXIMUM_TOOL_RESULT_CHARACTERS, capToolResult, executeWithinLimits } from "../src/ToolLimits";

/** A tool whose behaviour each test scripts: what it returns, whether it
 * throws, and whether it ever finishes. */
function scriptedTool(run: (signal: AbortSignal) => Promise<string>): ConversationTool {
  return {
    name: "scripted_tool",
    label: "Scripted tool",
    description: "A tool for tests.",
    parameters: { type: "object", properties: {} },
    execute: (_toolArguments, signal) => run(signal),
  };
}

test("should_return_a_result_within_the_cap_unchanged", async () => {
  const toolOutput = await executeWithinLimits(scriptedTool(async () => "Saturday"), {}, new AbortController().signal);
  assert.deepEqual(toolOutput, { text: "Saturday" });
});

test("should_keep_a_result_exactly_at_the_cap_and_cut_one_past_it", () => {
  const atCap = "x".repeat(MAXIMUM_TOOL_RESULT_CHARACTERS);
  assert.equal(capToolResult(atCap), atCap);
  const pastCap = capToolResult(`${atCap}y`);
  assert.ok(pastCap.startsWith(atCap));
  assert.match(pastCap, /result cut: 1 more characters not shown/);
});

test("should_report_a_tool_that_throws", async () => {
  await assert.rejects(
    executeWithinLimits(scriptedTool(async () => { throw new Error("clock unavailable"); }), {}, new AbortController().signal),
    /clock unavailable/,
  );
});

test("should_stop_a_tool_that_runs_past_its_timeout_even_if_it_ignores_the_signal", async () => {
  const neverFinishes = scriptedTool(() => new Promise<string>(() => undefined));
  await assert.rejects(executeWithinLimits(neverFinishes, {}, new AbortController().signal, undefined, { timeoutMilliseconds: 20 }), /took too long/);
});

test("should_stop_a_running_tool_when_the_turn_is_stopped", async () => {
  const stopController = new AbortController();
  let toolSawStop = false;
  const waitsForStop = scriptedTool((signal) => new Promise<string>((_resolve, reject) => {
    signal.addEventListener("abort", () => {
      toolSawStop = true;
      reject(new Error("stopped"));
    });
  }));
  const toolRun = executeWithinLimits(waitsForStop, {}, stopController.signal);
  stopController.abort();
  await assert.rejects(toolRun, /stopped/);
  assert.equal(toolSawStop, true);
});

test("should_not_start_a_tool_when_the_turn_was_already_stopped", async () => {
  const stopController = new AbortController();
  stopController.abort();
  let toolStarted = false;
  const quickTool = scriptedTool(async () => {
    toolStarted = true;
    return "ran anyway";
  });
  await assert.rejects(executeWithinLimits(quickTool, {}, stopController.signal), /was stopped/);
  assert.equal(toolStarted, false);
});

test("should_pass_an_image_name_through_and_cap_only_the_text", async () => {
  const imageTool = scriptedTool(async () => ({ text: "x".repeat(MAXIMUM_TOOL_RESULT_CHARACTERS + 5), generatedImageName: "image.png" }));
  const toolOutput = await executeWithinLimits(imageTool, {}, new AbortController().signal);
  assert.equal(toolOutput.generatedImageName, "image.png");
  assert.match(toolOutput.text, /result cut: 5 more characters/);
});

test("should_use_a_tools_own_timeout_when_it_sets_one", async () => {
  const slowButAllowed: ConversationTool = {
    ...scriptedTool(() => new Promise<string>((resolve) => setTimeout(() => resolve("done"), 40))),
    timeoutMilliseconds: 500,
  };
  assert.deepEqual(await executeWithinLimits(slowButAllowed, {}, new AbortController().signal), { text: "done" });
  const tooSlow: ConversationTool = { ...slowButAllowed, timeoutMilliseconds: 10 };
  await assert.rejects(executeWithinLimits(tooSlow, {}, new AbortController().signal), /took too long/);
});
