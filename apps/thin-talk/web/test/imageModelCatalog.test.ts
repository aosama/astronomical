import assert from "node:assert/strict";
import { test } from "node:test";
import { buildImageModels, parseLibraryImageEntries } from "../src/ImageModelCatalog";

const LIBRARY_CATALOG = {
  entries: [
    { display_name: "FLUX.2 Klein 4B", role: "image_generation", get_started: false, requestable_model_id: "FLUX.2-klein-4B" },
    { display_name: "FLUX.2 Klein 4B 4-bit", role: "image_generation", get_started: true, requestable_model_id: "flux2-klein-4b-4bit" },
    { display_name: "Qwen-Image 2.1 4-bit", role: "image_generation", get_started: false, requestable_model_id: null },
    { display_name: "Ornith 1.5 9B", role: "chat", get_started: true, requestable_model_id: "Ornith-1.5-9B" },
  ],
};

test("should_read_only_installed_image_generators_from_the_library", () => {
  const libraryImageEntries = parseLibraryImageEntries(JSON.stringify(LIBRARY_CATALOG));
  assert.deepEqual(libraryImageEntries, [
    { requestableModelID: "FLUX.2-klein-4B", displayName: "FLUX.2 Klein 4B", isGetStartedPick: false },
    { requestableModelID: "flux2-klein-4b-4bit", displayName: "FLUX.2 Klein 4B 4-bit", isGetStartedPick: true },
  ]);
});

test("should_read_nothing_from_a_catalog_that_is_missing_or_malformed", () => {
  for (const catalogBodyText of ["", "not json", "[]", "{}", '{"entries":[42,null,{"role":"image_generation"}]}']) {
    assert.deepEqual(parseLibraryImageEntries(catalogBodyText), [], catalogBodyText);
  }
});

test("should_put_the_pictures_pick_first_and_name_models_as_the_library_does", () => {
  const libraryImageEntries = parseLibraryImageEntries(JSON.stringify(LIBRARY_CATALOG));
  const imageModels = buildImageModels(["FLUX.2-klein-4B", "Qwen-Image-2.1-MLX-4bit", "flux2-klein-4b-4bit"], libraryImageEntries);
  assert.deepEqual(
    imageModels.map((imageModel) => [imageModel.id, imageModel.name, imageModel.isPicturesPick]),
    [
      ["flux2-klein-4b-4bit", "FLUX.2 Klein 4B 4-bit", true],
      ["FLUX.2-klein-4B", "FLUX.2 Klein 4B", false],
      ["Qwen-Image-2.1-MLX-4bit", "Qwen-Image-2.1-MLX-4bit", false],
    ],
  );
});

test("should_keep_the_server_order_and_ids_without_a_library", () => {
  const imageModels = buildImageModels(["model-b", "model-a"], []);
  assert.deepEqual(imageModels.map((imageModel) => [imageModel.id, imageModel.name, imageModel.isPicturesPick]), [
    ["model-b", "model-b", false],
    ["model-a", "model-a", false],
  ]);
  assert.deepEqual(buildImageModels([], parseLibraryImageEntries(JSON.stringify(LIBRARY_CATALOG))), []);
});
