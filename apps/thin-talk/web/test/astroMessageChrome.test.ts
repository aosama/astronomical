import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatMessageState } from "../src/ChatMessageState";
import { ChatRole } from "../src/ChatRole";
import { ToolStep, ToolStepState } from "../src/ToolStep";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { TranscriptPersona } from "../src/TranscriptPersona";
import { AstroMessageChrome, AstroMood, astroMood } from "../src/astro/AstroMessageChrome";
import { AstroStyle } from "../src/astro/AstroStyle";

function escapeHtml(text: string): string {
  return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function toolStep(state: ToolStepState, toolName = "generate_image"): ToolStep {
  return new ToolStep({ toolCallID: "call-1", toolName, label: "Generate image", argumentsJSON: "{}", resultText: "", state, roundIndex: 0, generatedImageName: null, generatedImageModelName: null });
}

function answer(state: ChatMessageState, markdown: string, toolSteps: ToolStep[] = []): TranscriptMessage {
  return new TranscriptMessage({ id: "answer-1", role: ChatRole.ASSISTANT, markdown, reasoning: "", state, attachments: [], toolSteps });
}

const EXPLORER = new TranscriptPersona({ isAstroShown: true, style: AstroStyle.EXPLORER, nickname: "Maya" });
const PILOT = new TranscriptPersona({ isAstroShown: true, style: AstroStyle.PILOT, nickname: "Sam" });

test("should_pick_astros_mood_from_where_the_answer_stands", () => {
  assert.equal(astroMood(answer(ChatMessageState.STREAMING, "")), AstroMood.THINKING);
  assert.equal(astroMood(answer(ChatMessageState.STREAMING, "", [toolStep(ToolStepState.RUNNING)])), AstroMood.WORKING);
  assert.equal(astroMood(answer(ChatMessageState.STREAMING, "Here", [toolStep(ToolStepState.SUCCEEDED)])), AstroMood.TALKING);
  assert.equal(astroMood(answer(ChatMessageState.COMPLETE, "Done")), AstroMood.DONE);
  assert.equal(astroMood(answer(ChatMessageState.STOPPED, "Half")), AstroMood.STOPPED);
  assert.equal(astroMood(answer(ChatMessageState.FAILED, "")), AstroMood.STOPPED);
});

test("should_name_astro_and_tag_the_answer_with_its_skill", () => {
  const metaHtml = new AstroMessageChrome(EXPLORER, escapeHtml).assistantHeaderHtml(answer(ChatMessageState.COMPLETE, "Ta-da", [toolStep(ToolStepState.SUCCEEDED)]));
  assert.match(metaHtml, />Astro</);
  assert.match(metaHtml, /astro-skill-tag--pictures"[^>]*>Pictures</);
});

test("should_say_what_astro_is_doing_in_each_voice_and_keep_failed_plain", () => {
  const thinking = answer(ChatMessageState.STREAMING, "");
  assert.match(new AstroMessageChrome(EXPLORER, escapeHtml).assistantHeaderHtml(thinking), /<span class="message__state-words">is thinking…<\/span><span class="message__loading-words">is waking up its brain…<\/span><span class="message__streaming-caret"><\/span>/);
  assert.match(new AstroMessageChrome(PILOT, escapeHtml).assistantHeaderHtml(thinking), />thinking…</);
  assert.match(new AstroMessageChrome(EXPLORER, escapeHtml).assistantHeaderHtml(answer(ChatMessageState.FAILED, "")), />failed</);
  assert.match(new AstroMessageChrome(EXPLORER, escapeHtml).assistantHeaderHtml(thinking), /data-mood="thinking"/, "the face is drawn in the pose for the mood");
});

test("should_show_the_persons_initial_and_escaped_nickname_or_you_without_one", () => {
  const namedHtml = new AstroMessageChrome(new TranscriptPersona({ isAstroShown: true, style: AstroStyle.EXPLORER, nickname: "<b>bo" }), escapeHtml).userRoleHtml();
  assert.match(namedHtml, /astro-you-badge">&lt;</);
  assert.match(namedHtml, /&lt;b&gt;bo/);
  assert.doesNotMatch(namedHtml, /<b>bo/);
  assert.equal(new AstroMessageChrome(new TranscriptPersona({ isAstroShown: true, style: AstroStyle.EXPLORER, nickname: "" }), escapeHtml).userRoleHtml(), `<span class="message__role">You</span>`);
});

test("should_call_reasoning_how_i_worked_it_out_for_explorers_only", () => {
  assert.equal(new AstroMessageChrome(EXPLORER, escapeHtml).reasoningSummary(), "How I worked it out");
  assert.equal(new AstroMessageChrome(PILOT, escapeHtml).reasoningSummary(), "Reasoning");
});

test("should_show_the_image_model_on_a_moment_only_for_pilots", () => {
  const drawn = toolStep(ToolStepState.SUCCEEDED);
  assert.doesNotMatch(new AstroMessageChrome(EXPLORER, escapeHtml).toolStepSummaryHtml(drawn, " · FLUX"), /FLUX/);
  assert.match(new AstroMessageChrome(PILOT, escapeHtml).toolStepSummaryHtml(drawn, " · FLUX"), /Image generated · FLUX/);
});

test("should_colour_a_skill_bearing_step_and_leave_the_clock_uncoloured", () => {
  const chrome = new AstroMessageChrome(EXPLORER, escapeHtml);
  assert.equal(chrome.toolStepAttributes(toolStep(ToolStepState.SUCCEEDED)), ' data-skill="pictures"');
  assert.equal(chrome.toolStepAttributes(toolStep(ToolStepState.SUCCEEDED, "get_current_time")), "");
});
