import assert from "node:assert/strict";
import { test } from "node:test";
import { buildImageModels, parseLibraryImageEntries } from "../src/ImageModelCatalog";

const LIBRARY_CATALOG = {
  entries: [
    { role: "image_generation", get_started: false, requestable_model_id: "FLUX.2-klein-4B" },
    { role: "image_generation", get_started: true, requestable_model_id: "flux2-klein-4b-4bit" },
    { role: "image_generation", get_started: false, requestable_model_id: null },
    { role: "chat", get_started: true, requestable_model_id: "Ornith-1.5-9B" },
  ],
};

test("should_read_only_installed_image_generators_from_the_library", () => {
  const libraryImageEntries = parseLibraryImageEntries(JSON.stringify(LIBRARY_CATALOG));
  assert.deepEqual(libraryImageEntries, [
    { requestableModelID: "FLUX.2-klein-4B", isGetStartedPick: false },
    { requestableModelID: "flux2-klein-4b-4bit", isGetStartedPick: true },
  ]);
});

test("should_report_text_that_is_not_a_catalog", () => {
  for (const catalogBodyText of ["", "not json", "[]", "{}", '{"entries":{}}']) {
    assert.equal(parseLibraryImageEntries(catalogBodyText), null, catalogBodyText);
  }
});

test("should_skip_malformed_entries_of_a_real_catalog", () => {
  assert.deepEqual(parseLibraryImageEntries('{"entries":[42,null,{"role":"image_generation"}]}'), []);
});

test("should_put_the_image_generation_pick_first_and_keep_each_model_named_by_its_id", () => {
  const libraryImageEntries = parseLibraryImageEntries(JSON.stringify(LIBRARY_CATALOG)) ?? [];
  const imageModels = buildImageModels(["FLUX.2-klein-4B", "Qwen-Image-2.1-MLX-4bit", "flux2-klein-4b-4bit"], libraryImageEntries);
  assert.deepEqual(
    imageModels.map((imageModel) => [imageModel.id, imageModel.isImageGenerationPick]),
    [
      ["flux2-klein-4b-4bit", true],
      ["FLUX.2-klein-4B", false],
      ["Qwen-Image-2.1-MLX-4bit", false],
    ],
  );
});

test("should_keep_the_server_order_and_ids_without_a_library", () => {
  const imageModels = buildImageModels(["model-b", "model-a"], []);
  assert.deepEqual(imageModels.map((imageModel) => [imageModel.id, imageModel.isImageGenerationPick]), [
    ["model-b", false],
    ["model-a", false],
  ]);
  assert.deepEqual(buildImageModels([], parseLibraryImageEntries(JSON.stringify(LIBRARY_CATALOG)) ?? []), []);
});
