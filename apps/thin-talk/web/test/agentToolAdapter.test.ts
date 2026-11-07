import assert from "node:assert/strict";
import { test } from "node:test";
import { agentTool, generatedImageNameFromToolResult } from "../src/AgentToolAdapter";
import type { ConversationTool } from "../src/ConversationTool";

function toolReturning(toolResult: string | { text: string; generatedImageName?: string }): ConversationTool {
  return {
    name: "sample_tool",
    label: "Sample tool",
    description: "A tool for tests.",
    parameters: { type: "object", properties: {} },
    execute: async () => toolResult,
  };
}

test("should_carry_a_generated_image_name_in_details_and_read_it_back", async () => {
  const agentResult = await agentTool(toolReturning({ text: "Made it.", generatedImageName: "image.png" })).execute("call-1", {}, new AbortController().signal);
  assert.deepEqual(agentResult.content, [{ type: "text", text: "Made it." }]);
  assert.equal(generatedImageNameFromToolResult(agentResult), "image.png");
});

test("should_leave_details_empty_for_a_text_only_tool", async () => {
  const agentResult = await agentTool(toolReturning("Saturday")).execute("call-1", {}, new AbortController().signal);
  assert.equal(agentResult.details, undefined);
  assert.equal(generatedImageNameFromToolResult(agentResult), null);
});

test("should_read_no_image_name_from_malformed_results", () => {
  for (const malformedResult of [null, "text", { details: "x" }, { details: { generatedImageName: 7 } }]) {
    assert.equal(generatedImageNameFromToolResult(malformedResult), null, JSON.stringify(malformedResult));
  }
});
