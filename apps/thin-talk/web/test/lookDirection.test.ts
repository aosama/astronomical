import assert from "node:assert/strict";
import { test } from "node:test";
import { lookFrameForPointer } from "../src/astro/LookDirection";

const LOOK_FRAME_COUNT = 9;

test("should_look_furthest_left_and_right_at_the_window_edges_and_ahead_in_the_middle", () => {
  assert.equal(lookFrameForPointer(0, LOOK_FRAME_COUNT), 0);
  assert.equal(lookFrameForPointer(0.5, LOOK_FRAME_COUNT), 4);
  assert.equal(lookFrameForPointer(1, LOOK_FRAME_COUNT), 8);
});

test("should_clamp_a_pointer_outside_the_window_to_the_nearest_edge", () => {
  assert.equal(lookFrameForPointer(-0.4, LOOK_FRAME_COUNT), 0);
  assert.equal(lookFrameForPointer(1.7, LOOK_FRAME_COUNT), 8);
});

test("should_look_ahead_when_the_position_is_not_a_number", () => {
  assert.equal(lookFrameForPointer(Number.NaN, LOOK_FRAME_COUNT), 4);
  assert.equal(lookFrameForPointer(Number.POSITIVE_INFINITY, LOOK_FRAME_COUNT), 4);
});

test("should_always_answer_frame_zero_with_one_frame_or_none", () => {
  assert.equal(lookFrameForPointer(0.9, 1), 0);
  assert.equal(lookFrameForPointer(0.9, 0), 0);
});

test("should_turn_steadily_without_skipping_a_frame_across_the_window", () => {
  const framesAcross = Array.from({ length: 101 }, (_, pointerPercent) => lookFrameForPointer(pointerPercent / 100, LOOK_FRAME_COUNT));
  for (let pointerPercent = 1; pointerPercent < framesAcross.length; pointerPercent += 1) {
    const step = framesAcross[pointerPercent]! - framesAcross[pointerPercent - 1]!;
    assert.ok(step === 0 || step === 1, `frame jumps by ${step} at ${pointerPercent}%`);
  }
});
