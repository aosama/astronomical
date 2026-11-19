// Channel tests: which addresses the parsed options turn into.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveAstronomicalChannels } from "../../src/channels.ts";
import { STABLE_CHANNEL, DEV_CHANNEL } from "../support/test-helpers.mjs";

function options(channelNames, baseUrl = undefined) {
  return { channelNames, baseUrl };
}

describe("resolveAstronomicalChannels", () => {
  it("should_give_each_named_channel_its_loopback_address_in_order", () => {
    assert.deepEqual(resolveAstronomicalChannels(options(["stable"])), [
      STABLE_CHANNEL,
    ]);
    assert.deepEqual(
      resolveAstronomicalChannels(options(["development", "stable"])),
      [DEV_CHANNEL, STABLE_CHANNEL],
    );
  });

  it("should_use_a_base_url_as_the_only_channel_and_ignore_the_named_ones", () => {
    assert.deepEqual(
      resolveAstronomicalChannels(
        options(["stable", "development"], "http://localhost:9999"),
      ),
      [{ instance: "custom", baseUrl: "http://localhost:9999/v1" }],
    );
  });

  it("should_normalize_a_base_url_with_v1_and_trailing_slashes", () => {
    for (const baseUrl of [
      "http://localhost:9999/v1",
      "http://localhost:9999/v1/",
      "http://localhost:9999//",
    ]) {
      assert.equal(
        resolveAstronomicalChannels(options(["stable"], baseUrl))[0].baseUrl,
        "http://localhost:9999/v1",
      );
    }
  });
});
