import assert from "node:assert/strict";
import { test } from "node:test";
import { hasSpeakableText, speakableTextOf } from "../src/speech/SpeakableText";

test("should_read_prose_headings_and_list_items_as_separate_sentences", () => {
  const markdown = "## The prologue\n\nTwo households, both alike in dignity.\n\n- A feud\n- A love\n";
  assert.equal(speakableTextOf(markdown), "The prologue.\n\nTwo households, both alike in dignity.\n\nA feud. A love.");
});

test("should_leave_out_code_tables_pictures_maths_and_pictographs", () => {
  const markdown = [
    "Before.",
    "```js\nconsole.log(1)\n```",
    "| a | b |\n|---|---|\n| 1 | 2 |",
    "![a diagram](thintalk-asset://x.png)",
    "The area is $\\pi r^2$ today 🎉.",
    "$$\\int_0^1 x\\,dx$$",
    "After.",
  ].join("\n\n");
  const spoken = speakableTextOf(markdown);
  assert.ok(!spoken.includes("console"), "code is not read");
  assert.ok(!spoken.includes("|"), "tables are not read");
  assert.ok(!spoken.includes("diagram"), "pictures are not read");
  assert.ok(!spoken.includes("pi"), "inline maths is not read");
  assert.ok(!spoken.includes("int"), "display maths is not read");
  assert.ok(!spoken.includes("🎉"), "pictographs are not read");
  assert.match(spoken, /Before\./);
  assert.match(spoken, /After\./);
});

test("should_read_a_links_words_and_not_its_address", () => {
  assert.equal(speakableTextOf("See [the Library](https://example.com/library) now."), "See the Library now.");
});

test("should_return_nothing_for_a_reply_with_no_words_to_hear", () => {
  assert.equal(speakableTextOf("```py\nprint(1)\n```"), "");
  assert.equal(speakableTextOf(""), "");
  assert.equal(speakableTextOf("![only a picture](thintalk-asset://x.png)"), "");
  assert.equal(speakableTextOf("🎉🎉"), "");
});

test("should_keep_prices_and_shell_variables_as_words", () => {
  assert.equal(speakableTextOf("It costs $5 a month or $50 a year."), "It costs $5 a month or $50 a year.");
  assert.match(speakableTextOf("Set `$HOME` and `$PATH` first."), /HOME and .*PATH first\./);
});

test("should_return_quotes_and_ampersands_as_plain_characters", () => {
  const spoken = speakableTextOf(`Tom & Jerry's "plan" works.`);
  assert.equal(spoken, `Tom & Jerry's "plan" works.`);
});

test("should_say_only_the_site_when_a_links_words_are_its_own_address", () => {
  assert.equal(speakableTextOf("Read https://www.example.com/a/very/long/path?x=1 now."), "Read example.com now.");
  assert.equal(speakableTextOf("Read [https://docs.example.org/x](https://docs.example.org/x) now."), "Read docs.example.org now.");
});

test("should_keep_the_words_of_a_link_that_merely_begin_with_an_address", () => {
  assert.equal(speakableTextOf("See [https://example.com is great](https://example.com) now."), "See https://example.com is great now.");
});

test("should_read_nested_lists_and_quotes_sentence_by_sentence", () => {
  const spoken = speakableTextOf("> A wise quote\n\n- Outer\n  - Inner one\n  - Inner two\n");
  assert.equal(spoken, "A wise quote\n\nOuter. Inner one. Inner two.");
});

test("should_not_add_a_full_stop_after_a_heading_that_already_ends_in_one", () => {
  assert.equal(speakableTextOf("## Why is the sky blue?\n\nBecause of scattering."), "Why is the sky blue?\n\nBecause of scattering.");
});

test("should_return_code_spans_with_their_angle_brackets_and_ampersands_plain", () => {
  assert.equal(speakableTextOf("Use `Map<K, V>` and `a && b` here."), "Use Map<K, V> and a && b here.");
});

test("should_offer_speech_only_for_a_reply_with_words_to_hear", () => {
  assert.equal(hasSpeakableText("Hello there."), true);
  assert.equal(hasSpeakableText("```py\nprint(1)\n```"), false);
});
