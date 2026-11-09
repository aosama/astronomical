import assert from "node:assert/strict";
import { test } from "node:test";
import { agentTool, generatedImageModelNameFromToolResult, generatedImageNameFromToolResult } from "../src/AgentToolAdapter";
import type { ConversationTool } from "../src/ConversationTool";

function toolReturning(toolResult: string | { text: string; generatedImageName?: string; generatedImageModelName?: string }): ConversationTool {
  return {
    name: "sample_tool",
    label: "Sample tool",
    description: "A tool for tests.",
    parameters: { type: "object", properties: {} },
    execute: async () => toolResult,
  };
}

test("should_carry_a_generated_image_name_in_details_and_read_it_back", async () => {
  const agentResult = await agentTool(toolReturning({ text: "Made it.", generatedImageName: "image.png" }), { conversationID: "c1", modelID: "m" }).execute("call-1", {}, new AbortController().signal);
  assert.deepEqual(agentResult.content, [{ type: "text", text: "Made it." }]);
  assert.equal(generatedImageNameFromToolResult(agentResult), "image.png");
});

test("should_carry_the_image_model_name_beside_the_image_name", async () => {
  const toolContext = { conversationID: "c1", modelID: "m" };
  const namedResult = await agentTool(
    toolReturning({ text: "Made it.", generatedImageName: "image.png", generatedImageModelName: "FLUX.2 Klein 4B 4-bit" }),
    toolContext,
  ).execute("call-1", {}, new AbortController().signal);
  assert.equal(generatedImageNameFromToolResult(namedResult), "image.png");
  assert.equal(generatedImageModelNameFromToolResult(namedResult), "FLUX.2 Klein 4B 4-bit");

  const unnamedResult = await agentTool(toolReturning({ text: "Made it.", generatedImageName: "image.png" }), toolContext)
    .execute("call-2", {}, new AbortController().signal);
  assert.deepEqual(unnamedResult.details, { generatedImageName: "image.png" });
  assert.equal(generatedImageModelNameFromToolResult(unnamedResult), null);

  const imagelessResult = await agentTool(toolReturning({ text: "No picture.", generatedImageModelName: "FLUX.2 Klein 4B full precision" }), toolContext)
    .execute("call-3", {}, new AbortController().signal);
  assert.equal(imagelessResult.details, undefined, "a model name without an image is not recorded");
});

test("should_leave_details_empty_for_a_text_only_tool", async () => {
  const agentResult = await agentTool(toolReturning("Saturday"), { conversationID: "c1", modelID: "m" }).execute("call-1", {}, new AbortController().signal);
  assert.equal(agentResult.details, undefined);
  assert.equal(generatedImageNameFromToolResult(agentResult), null);
});

test("should_read_no_image_name_from_malformed_results", () => {
  for (const malformedResult of [null, "text", { details: "x" }, { details: { generatedImageName: 7 } }]) {
    assert.equal(generatedImageNameFromToolResult(malformedResult), null, JSON.stringify(malformedResult));
  }
});

test("should_run_tools_that_ask_the_user_one_at_a_time", () => {
  const askingTool: ConversationTool = { ...toolReturning("ok"), needsUserAttention: true };
  assert.equal(agentTool(askingTool, { conversationID: "c1", modelID: "m" }).executionMode, "sequential");
  assert.equal(agentTool(toolReturning("ok"), { conversationID: "c1", modelID: "m" }).executionMode, undefined);
});
