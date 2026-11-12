import assert from "node:assert/strict";
import { test } from "node:test";
import {
  OPENING_PIECE_MINIMUM_CHARACTERS,
  PIECE_MAXIMUM_CHARACTERS,
  SECOND_PIECE_MAXIMUM_CHARACTERS,
  speechPiecesOf,
} from "../src/speech/SpeechPieces";

const MERCY_SENTENCE = "The quality of mercy is not strained, it droppeth as the gentle rain from heaven.";

test("should_open_with_a_sentence_then_a_modest_second_piece_then_full_pieces", () => {
  const pieces = speechPiecesOf(Array(40).fill(MERCY_SENTENCE).join(" "));
  assert.ok(pieces.length > 3, "a long reply becomes several pieces");
  assert.ok(pieces[0].length >= OPENING_PIECE_MINIMUM_CHARACTERS, "the opening has about a sentence");
  assert.ok(pieces[1].length <= SECOND_PIECE_MAXIMUM_CHARACTERS, "the second piece stays small");
  for (const piece of pieces) {
    assert.ok(piece.length <= PIECE_MAXIMUM_CHARACTERS, "no piece exceeds the ceiling");
    assert.ok(piece.endsWith("heaven."), "every piece ends where a sentence ends");
  }
});

test("should_never_open_with_a_one_word_piece_in_front_of_a_long_sentence", () => {
  const longSentence = `${Array(40).fill("and so the story goes on").join(", ")}.`;
  const pieces = speechPiecesOf(`Sure! ${longSentence}`);
  assert.ok(pieces[0].length >= OPENING_PIECE_MINIMUM_CHARACTERS);
  assert.notEqual(pieces[0], "Sure!");
  assert.ok(pieces[0].startsWith("Sure! and so the story"));
});

test("should_keep_a_reply_shorter_than_an_opening_as_one_piece", () => {
  assert.deepEqual(speechPiecesOf("Hello from the fake daemon. Second sentence."), ["Hello from the fake daemon. Second sentence."]);
});

test("should_keep_decimals_and_domains_inside_one_sentence", () => {
  assert.deepEqual(speechPiecesOf("Version 1.5 is at example.com today."), ["Version 1.5 is at example.com today."]);
});

test("should_break_one_endless_sentence_at_a_comma_or_space_instead_of_overflowing", () => {
  const endlessSentence = `${Array(400).fill("word").join(", ")}.`;
  const pieces = speechPiecesOf(endlessSentence);
  assert.ok(pieces.length > 1);
  assert.ok(pieces.every((piece) => piece.length <= PIECE_MAXIMUM_CHARACTERS));
  assert.equal(pieces.join(" ").replace(/\s+/g, " "), endlessSentence);
});

test("should_make_no_pieces_from_empty_text", () => {
  assert.deepEqual(speechPiecesOf(""), []);
});

test("should_end_a_paragraph_that_has_no_final_stop_so_the_voice_pauses", () => {
  assert.deepEqual(speechPiecesOf("Here is the plan\n\nFirst step."), ["Here is the plan. First step."]);
});
