import assert from "node:assert/strict";
import { test } from "node:test";
import { skillMomentWords } from "../src/astro/SkillMomentWords";
import { ToolStepState } from "../src/ToolStep";

const KNOWN_TOOLS = ["generate_image", "describe_image", "read_file", "search_conversations", "get_current_time"];

test("should_tell_drawing_in_plain_words", () => {
  assert.equal(skillMomentWords("generate_image", "Generate image", ToolStepState.RUNNING), "Generating an image…");
  assert.equal(skillMomentWords("generate_image", "Generate image", ToolStepState.SUCCEEDED), "Image generated");
  assert.equal(skillMomentWords("generate_image", "Generate image", ToolStepState.FAILED), "Image generation failed");
});

test("should_have_distinct_words_for_every_state_of_every_known_tool", () => {
  for (const toolName of KNOWN_TOOLS) {
    const words = [ToolStepState.RUNNING, ToolStepState.SUCCEEDED, ToolStepState.FAILED].map((state) => skillMomentWords(toolName, "label", state));
    assert.equal(new Set(words).size, 3, toolName);
    assert.ok(words[0]!.endsWith("…"), `${toolName} running reads as ongoing`);
  }
});

test("should_fall_back_to_the_tool_label_for_a_tool_without_words", () => {
  assert.equal(skillMomentWords("make_a_poem", "Make a poem", ToolStepState.RUNNING), "Make a poem: working…");
  assert.equal(skillMomentWords("", "Unnamed tool", ToolStepState.FAILED), "Unnamed tool: didn't work");
});
