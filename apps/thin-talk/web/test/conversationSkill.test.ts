import assert from "node:assert/strict";
import { test } from "node:test";
import { ConversationSkill, answerSkill, dominantSkill, parseConversationSkill, skillForToolName } from "../src/ConversationSkill";

test("should_map_each_skill_bearing_tool_and_leave_the_clock_as_chat", () => {
  assert.equal(skillForToolName("generate_image"), ConversationSkill.PICTURES);
  assert.equal(skillForToolName("describe_image"), ConversationSkill.LOOK);
  assert.equal(skillForToolName("read_file"), ConversationSkill.READ);
  assert.equal(skillForToolName("search_conversations"), ConversationSkill.SEARCH);
  assert.equal(skillForToolName("get_current_time"), null);
  assert.equal(skillForToolName(""), null);
});

test("should_tag_an_answer_by_its_last_skill_bearing_tool", () => {
  assert.equal(answerSkill("Here you go.", ["read_file", "generate_image", "get_current_time"]), ConversationSkill.PICTURES);
  assert.equal(answerSkill("It's 3 o'clock.", ["get_current_time"]), ConversationSkill.CHAT);
});

test("should_tag_formulas_as_maths_but_not_a_price", () => {
  assert.equal(answerSkill("So $$\\frac{1}{3} + \\frac{1}{4} = \\frac{7}{12}$$", []), ConversationSkill.MATHS);
  assert.equal(answerSkill("Inline \\(x^2\\) works too.", []), ConversationSkill.MATHS);
  assert.equal(answerSkill("Display \\[a+b\\] works too.", []), ConversationSkill.MATHS);
  assert.equal(answerSkill("A comic costs $5 and a book $12.", []), ConversationSkill.CHAT, "dollar signs in prices are not maths");
  assert.equal(answerSkill("", []), ConversationSkill.CHAT);
});

test("should_remember_a_chat_by_its_most_used_skill_even_among_many_questions", () => {
  const answers = [
    { markdown: "Blue light scatters.", toolNames: [] },
    { markdown: "Here it is.", toolNames: ["generate_image"] },
    { markdown: "Sure.", toolNames: [] },
  ];
  assert.equal(dominantSkill(answers), ConversationSkill.PICTURES);
});

test("should_break_a_tie_toward_the_skill_used_most_recently", () => {
  const answers = [
    { markdown: "", toolNames: ["read_file"] },
    { markdown: "", toolNames: ["describe_image"] },
  ];
  assert.equal(dominantSkill(answers), ConversationSkill.LOOK);
  assert.equal(dominantSkill([...answers, { markdown: "", toolNames: ["read_file"] }]), ConversationSkill.READ);
});

test("should_call_an_empty_or_plain_chat_a_chat", () => {
  assert.equal(dominantSkill([]), ConversationSkill.CHAT);
  assert.equal(dominantSkill([{ markdown: "Hello!", toolNames: [] }]), ConversationSkill.CHAT);
});

test("should_read_only_known_stored_skills", () => {
  assert.equal(parseConversationSkill("pictures"), ConversationSkill.PICTURES);
  assert.equal(parseConversationSkill("PICTURES"), ConversationSkill.CHAT);
  assert.equal(parseConversationSkill(undefined), ConversationSkill.CHAT);
  assert.equal(parseConversationSkill(7), ConversationSkill.CHAT);
});
