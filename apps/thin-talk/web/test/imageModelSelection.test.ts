import assert from "node:assert/strict";
import { test } from "node:test";
import { ImageModel } from "../src/ImageModel";
import { ImageModelSelection } from "../src/ImageModelSelection";

const IMAGE_GENERATION_PICK = new ImageModel({ id: "flux2-klein-4b-4bit", name: "FLUX.2 Klein 4B 4-bit", isImageGenerationPick: true });
const FULL_PRECISION = new ImageModel({ id: "FLUX.2-klein-4B", name: "FLUX.2 Klein 4B full precision" });
const QWEN_IMAGE = new ImageModel({ id: "Qwen-Image-2.1-MLX-4bit", name: "Qwen-Image 2.1 4-bit" });

function selectionOver(installedImageModels: ImageModel[], preferredImageModelID: string | null = null): ImageModelSelection {
  const selection = new ImageModelSelection(preferredImageModelID);
  selection.updateInstalled(installedImageModels);
  return selection;
}

test("should_use_the_saved_choice_while_it_is_installed", () => {
  assert.equal(selectionOver([IMAGE_GENERATION_PICK, FULL_PRECISION, QWEN_IMAGE], "Qwen-Image-2.1-MLX-4bit").current(), QWEN_IMAGE);
});

test("should_fall_back_to_the_image_generation_pick_when_the_choice_is_gone_or_never_made", () => {
  assert.equal(selectionOver([FULL_PRECISION, IMAGE_GENERATION_PICK], "removed-model").current(), IMAGE_GENERATION_PICK);
  assert.equal(selectionOver([FULL_PRECISION, IMAGE_GENERATION_PICK]).current(), IMAGE_GENERATION_PICK);
});

test("should_use_the_first_installed_model_when_the_catalog_names_no_pick", () => {
  assert.equal(selectionOver([QWEN_IMAGE, FULL_PRECISION]).current(), QWEN_IMAGE);
});

test("should_resolve_to_nothing_before_any_list_or_with_none_installed", () => {
  assert.equal(new ImageModelSelection("FLUX.2-klein-4B").current(), null);
  assert.equal(selectionOver([], "FLUX.2-klein-4B").current(), null);
});

test("should_follow_a_new_choice_and_tell_listeners_until_they_stop", () => {
  const selection = selectionOver([IMAGE_GENERATION_PICK, FULL_PRECISION]);
  const heardSelections: (string | undefined)[] = [];
  const stopListening = selection.onChange(() => heardSelections.push(selection.current()?.id));

  selection.prefer("FLUX.2-klein-4B");
  selection.updateInstalled([IMAGE_GENERATION_PICK]);
  stopListening();
  selection.prefer(null);

  assert.deepEqual(heardSelections, ["FLUX.2-klein-4B", "flux2-klein-4b-4bit"]);
  assert.deepEqual(selection.installed, [IMAGE_GENERATION_PICK]);
});
