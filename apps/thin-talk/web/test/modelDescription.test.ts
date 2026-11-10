import assert from "node:assert/strict";
import { test } from "node:test";
import { ModelDescription, displayNameFor, formatGigabytes, parseLibraryModelDescriptions } from "../src/ModelDescription";

const ONE_GIGABYTE = 1024 * 1024 * 1024;

function catalogText(entries: unknown[]): string {
  return JSON.stringify({ schema_version: 4, entries });
}

test("should_read_plain_names_versions_sizes_and_vision_for_installed_models", () => {
  const descriptions = parseLibraryModelDescriptions(catalogText([
    { requestable_model_id: "Ornith-35B-6bit", short_name: "Ornith 1.5 35B Vision", version_label: "6-bit", approximate_size_bytes: 27.1 * ONE_GIGABYTE, capabilities: { supports_vision: true } },
  ]))!;
  const ornith = descriptions.get("Ornith-35B-6bit")!;
  assert.equal(ornith.shortName, "Ornith 1.5 35B Vision");
  assert.equal(ornith.detailLine(), "6-bit · 27.1 GB · sees images");
});

test("should_skip_models_that_are_not_installed_or_have_no_name", () => {
  const descriptions = parseLibraryModelDescriptions(catalogText([
    { short_name: "Not installed" },
    { requestable_model_id: "", short_name: "Empty id" },
    { requestable_model_id: "nameless" },
    { requestable_model_id: "blank-name", short_name: "   " },
    "not an entry",
    null,
  ]))!;
  assert.equal(descriptions.size, 0);
});

test("should_fall_back_to_the_display_name_when_there_is_no_short_name", () => {
  const descriptions = parseLibraryModelDescriptions(catalogText([{ requestable_model_id: "older-entry", display_name: "Older Entry 4-bit" }]))!;
  assert.equal(descriptions.get("older-entry")?.shortName, "Older Entry 4-bit");
});

test("should_return_null_for_text_that_is_not_a_catalog", () => {
  assert.equal(parseLibraryModelDescriptions("not json"), null);
  assert.equal(parseLibraryModelDescriptions(JSON.stringify({ models: [] })), null);
});

test("should_ignore_sizes_that_are_not_positive_finite_numbers", () => {
  for (const rawSize of [0, -5, "12", Number.NaN, null]) {
    const descriptions = parseLibraryModelDescriptions(catalogText([{ requestable_model_id: "m", short_name: "M", approximate_size_bytes: rawSize }]))!;
    assert.equal(descriptions.get("m")?.sizeBytes, null, String(rawSize));
  }
});

test("should_build_an_empty_detail_line_when_nothing_is_known", () => {
  assert.equal(new ModelDescription({ shortName: "Bare" }).detailLine(), "");
});

test("should_print_sizes_as_gigabytes_to_one_decimal_like_the_models_page", () => {
  assert.equal(formatGigabytes(29_100_000_000), "27.1 GB");
  assert.equal(formatGigabytes(1), "0.0 GB");
});

test("should_use_the_id_when_a_model_has_no_description", () => {
  const descriptions = new Map([["known", new ModelDescription({ shortName: "Known" })]]);
  assert.equal(displayNameFor("known", descriptions), "Known");
  assert.equal(displayNameFor("unknown-folder-name", descriptions), "unknown-folder-name");
});
