import assert from "node:assert/strict";
import { test } from "node:test";
import { modelNameFromHuggingfaceID } from "../src/astro/HuggingfaceModelName";

test("should_drop_the_owner_from_a_model_id", () => {
  assert.equal(modelNameFromHuggingfaceID("example-org/Starter-9B-4bit"), "Starter-9B-4bit");
});

test("should_keep_an_id_that_has_no_owner", () => {
  assert.equal(modelNameFromHuggingfaceID("Starter-9B-4bit"), "Starter-9B-4bit");
  assert.equal(modelNameFromHuggingfaceID(""), "");
});

test("should_never_return_an_empty_name_for_an_id_that_ends_in_a_slash", () => {
  assert.equal(modelNameFromHuggingfaceID("example-org/"), "example-org/");
});

test("should_take_the_last_part_of_a_deeper_path", () => {
  assert.equal(modelNameFromHuggingfaceID("mirror/example-org/Starter"), "Starter");
});
