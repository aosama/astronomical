import assert from "node:assert/strict";
import { test } from "node:test";
import { PictureModelGuidance } from "../src/PictureModelGuidance";

test("should_start_in_classic_thin_talk_and_point_to_the_library", () => {
  const guidance = new PictureModelGuidance();
  assert.match(guidance.noPictureModelResult(), /the Models page in the Observatory/);
});

test("should_follow_the_focus_switch_both_ways", () => {
  const guidance = new PictureModelGuidance();
  guidance.setAstroShown(true);
  assert.match(guidance.noPictureModelResult(), /Pictures card in a new chat/);
  guidance.setAstroShown(false);
  assert.doesNotMatch(guidance.noPictureModelResult(), /Pictures card/);
});

test("should_read_as_a_plain_result_a_person_could_see_in_the_step", () => {
  for (const isAstroShown of [false, true]) {
    const guidance = new PictureModelGuidance();
    guidance.setAstroShown(isAstroShown);
    const result = guidance.noPictureModelResult();
    assert.match(result, /^No picture model is installed on this Mac, so nothing was drawn\./);
    assert.doesNotMatch(result, /System Settings|kindly/);
  }
});
