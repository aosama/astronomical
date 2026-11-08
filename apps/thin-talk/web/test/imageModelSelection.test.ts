import assert from "node:assert/strict";
import { test } from "node:test";
import { ImageModel } from "../src/ImageModel";
import { ImageModelSelection } from "../src/ImageModelSelection";

const PICTURES_PICK = new ImageModel({ id: "flux2-klein-4b-4bit", name: "FLUX.2 Klein 4B 4-bit", isPicturesPick: true });
const FULL_PRECISION = new ImageModel({ id: "FLUX.2-klein-4B", name: "FLUX.2 Klein 4B" });
const QWEN_IMAGE = new ImageModel({ id: "Qwen-Image-2.1-MLX-4bit", name: "Qwen-Image 2.1 4-bit" });

test("should_use_the_saved_choice_while_it_is_installed", () => {
  const selection = new ImageModelSelection("Qwen-Image-2.1-MLX-4bit");
  assert.equal(selection.resolve([PICTURES_PICK, FULL_PRECISION, QWEN_IMAGE]), QWEN_IMAGE);
});

test("should_fall_back_to_the_pictures_pick_when_the_choice_is_gone_or_never_made", () => {
  assert.equal(new ImageModelSelection("removed-model").resolve([FULL_PRECISION, PICTURES_PICK]), PICTURES_PICK);
  assert.equal(new ImageModelSelection().resolve([FULL_PRECISION, PICTURES_PICK]), PICTURES_PICK);
});

test("should_use_the_first_installed_model_when_the_library_names_no_pick", () => {
  assert.equal(new ImageModelSelection().resolve([QWEN_IMAGE, FULL_PRECISION]), QWEN_IMAGE);
});

test("should_resolve_to_nothing_when_no_image_model_is_installed", () => {
  assert.equal(new ImageModelSelection("FLUX.2-klein-4B").resolve([]), null);
});

test("should_follow_a_new_choice", () => {
  const selection = new ImageModelSelection();
  selection.prefer("FLUX.2-klein-4B");
  assert.equal(selection.preferredID, "FLUX.2-klein-4B");
  assert.equal(selection.resolve([PICTURES_PICK, FULL_PRECISION]), FULL_PRECISION);
});
