// Option tests: what a plugins entry may say, and the message a wrong value
// produces.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { parseAstronomicalPluginOptions } from "../../src/plugin-options.ts";

describe("parseAstronomicalPluginOptions", () => {
  it("should_default_to_stable_only_when_no_options_are_given", () => {
    for (const rawOptions of [undefined, null, {}]) {
      assert.deepEqual(parseAstronomicalPluginOptions(rawOptions), {
        channelNames: ["stable"],
        baseUrl: undefined,
      });
    }
  });

  it("should_keep_the_named_channels_in_order_without_repeats", () => {
    assert.deepEqual(
      parseAstronomicalPluginOptions({
        channels: ["development", "stable", "development"],
      }).channelNames,
      ["development", "stable"],
    );
  });

  it("should_read_a_base_url_and_trim_it", () => {
    assert.equal(
      parseAstronomicalPluginOptions({ baseUrl: "  http://127.0.0.1:6732/v1 " })
        .baseUrl,
      "http://127.0.0.1:6732/v1",
    );
  });

  it("should_reject_an_unknown_or_misspelled_channel_by_name", () => {
    assert.throws(
      () => parseAstronomicalPluginOptions({ channels: ["dev"] }),
      /"channels" must be a list of "stable" or "development"; "dev" is not one/,
    );
  });

  it("should_reject_an_empty_or_non_list_channels_value", () => {
    for (const channels of [[], "stable", null, 6732]) {
      assert.throws(
        () => parseAstronomicalPluginOptions({ channels }),
        /"channels" must be a list/,
      );
    }
  });

  it("should_accept_an_https_address", () => {
    assert.equal(
      parseAstronomicalPluginOptions({ baseUrl: "https://models.example/v1" })
        .baseUrl,
      "https://models.example/v1",
    );
  });

  it("should_reject_a_base_url_that_is_not_an_http_address", () => {
    for (const baseUrl of [
      "",
      "   ",
      "localhost",
      "127.0.0.1:6732",
      "localhost:6732/v1",
      "ftp://127.0.0.1/v1",
      6732,
    ]) {
      assert.throws(
        () => parseAstronomicalPluginOptions({ baseUrl }),
        /"baseUrl" must be an http address/,
        `accepted ${JSON.stringify(baseUrl)}`,
      );
    }
  });

  it("should_reject_an_unknown_option_so_a_typo_is_caught", () => {
    assert.throws(
      () => parseAstronomicalPluginOptions({ channel: ["stable"] }),
      /unknown option "channel"/,
    );
  });

  it("should_reject_options_that_are_not_an_object", () => {
    assert.throws(
      () => parseAstronomicalPluginOptions(["stable"]),
      /"options" must be an object/,
    );
  });
});
