import assert from "node:assert/strict";
import { test } from "node:test";
import { ModelDescription, formatGigabytes, parseLibraryModelDescriptions } from "../src/ModelDescription";

const ONE_GIGABYTE = 1024 * 1024 * 1024;

function catalogText(entries: unknown[]): string {
  return JSON.stringify({ schema_version: 5, entries });
}

test("should_read_sizes_and_vision_for_installed_models_keyed_by_their_one_name", () => {
  const descriptions = parseLibraryModelDescriptions(catalogText([
    { requestable_model_id: "Ornith-35B-6bit", model_id: "Ornith-35B-6bit", approximate_size_bytes: 27.1 * ONE_GIGABYTE, capabilities: { supports_vision: true } },
  ]))!;
  const ornith = descriptions.get("Ornith-35B-6bit")!;
  assert.equal(ornith.detailLine(), "27.1 GB · sees images");
});

test("should_skip_models_that_are_not_installed", () => {
  const descriptions = parseLibraryModelDescriptions(catalogText([
    { model_id: "Not-installed" },
    { requestable_model_id: "", model_id: "Empty-id" },
    { requestable_model_id: 7 },
    "not an entry",
    null,
  ]))!;
  assert.equal(descriptions.size, 0);
});

/** A name is no longer needed to describe a model: the id is its name, so an
 * installed entry with nothing but its id still contributes its facts. */
test("should_describe_an_installed_model_that_carries_only_its_id", () => {
  const descriptions = parseLibraryModelDescriptions(catalogText([{ requestable_model_id: "bare-model" }]))!;
  assert.equal(descriptions.get("bare-model")?.detailLine(), "");
});

test("should_ignore_the_retired_names_if_an_old_catalog_still_sends_them", () => {
  const descriptions = parseLibraryModelDescriptions(catalogText([
    { requestable_model_id: "old-entry", short_name: "Old Entry", version_label: "6-bit", display_name: "Old Entry 6-bit", approximate_size_bytes: ONE_GIGABYTE },
  ]))!;
  assert.equal(descriptions.get("old-entry")?.detailLine(), "1.0 GB", "no version label comes back into the detail line");
});

test("should_return_null_for_text_that_is_not_a_catalog", () => {
  assert.equal(parseLibraryModelDescriptions("not json"), null);
  assert.equal(parseLibraryModelDescriptions(JSON.stringify({ models: [] })), null);
});

test("should_ignore_sizes_that_are_not_positive_finite_numbers", () => {
  for (const rawSize of [0, -5, "12", Number.NaN, null]) {
    const descriptions = parseLibraryModelDescriptions(catalogText([{ requestable_model_id: "m", approximate_size_bytes: rawSize }]))!;
    assert.equal(descriptions.get("m")?.sizeBytes, null, String(rawSize));
  }
});

test("should_build_an_empty_detail_line_when_nothing_is_known", () => {
  assert.equal(new ModelDescription({}).detailLine(), "");
});

test("should_print_sizes_as_gigabytes_to_one_decimal_like_the_models_page", () => {
  assert.equal(formatGigabytes(29_100_000_000), "27.1 GB");
  assert.equal(formatGigabytes(5_970_332_199), "5.6 GB");
  assert.equal(formatGigabytes(1), "0.0 GB");
});

test("should_print_nothing_for_an_empty_or_negative_size", () => {
  assert.equal(formatGigabytes(0), "0.0 GB");
  assert.equal(formatGigabytes(-5), "0.0 GB");
});
