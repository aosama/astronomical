// Base URL resolution tests: which channel(s) to probe, from the environment.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveAstronomicalChannels } from "../../dist/astronomer-base-url.js";
import {
  STABLE,
  DEV,
  STABLE_CHANNEL,
  DEV_CHANNEL,
} from "../support/test-helpers.mjs";

describe("channel resolution", () => {
  it("should_probe_both_channels_stable_first_by_default", () => {
    assert.deepEqual(resolveAstronomicalChannels({}), [
      STABLE_CHANNEL,
      DEV_CHANNEL,
    ]);
  });

  it("should_honor_ASTRONOMICAL_INSTANCE_stable", () => {
    assert.deepEqual(
      resolveAstronomicalChannels({ ASTRONOMICAL_INSTANCE: "stable" }),
      [STABLE_CHANNEL],
    );
  });

  it("should_honor_ASTRONOMICAL_INSTANCE_dev", () => {
    assert.deepEqual(
      resolveAstronomicalChannels({ ASTRONOMICAL_INSTANCE: "dev" }),
      [DEV_CHANNEL],
    );
  });

  it("should_use_an_ASTRONOMICAL_BASE_URL_override_as_a_single_custom_channel", () => {
    assert.deepEqual(
      resolveAstronomicalChannels({
        ASTRONOMICAL_BASE_URL: "http://localhost:9999",
      }),
      [{ instance: "custom", baseUrl: "http://localhost:9999/v1" }],
    );
  });

  it("should_strip_a_trailing_slash_from_an_override", () => {
    assert.deepEqual(
      resolveAstronomicalChannels({
        ASTRONOMICAL_BASE_URL: "http://localhost:9999/v1/",
      }),
      [{ instance: "custom", baseUrl: "http://localhost:9999/v1" }],
    );
  });

  it("should_prefer_base_url_over_instance_when_both_are_set", () => {
    // One explicit endpoint wins: probing default ports beside a user-named
    // endpoint would register models from a daemon they did not ask for.
    assert.deepEqual(
      resolveAstronomicalChannels({
        ASTRONOMICAL_BASE_URL: "http://localhost:9999",
        ASTRONOMICAL_INSTANCE: "stable",
      }),
      [{ instance: "custom", baseUrl: "http://localhost:9999/v1" }],
    );
  });

  it("should_be_case_insensitive_for_the_instance_selector", () => {
    assert.deepEqual(
      resolveAstronomicalChannels({ ASTRONOMICAL_INSTANCE: "DEV" }),
      [DEV_CHANNEL],
    );
  });

  it("should_ignore_an_unknown_instance_selector_and_probe_both_channels", () => {
    // An unrecognized value degrades to the safe default rather than probing
    // nothing, which would silently unregister every model.
    assert.deepEqual(
      resolveAstronomicalChannels({ ASTRONOMICAL_INSTANCE: "beta" }),
      [STABLE_CHANNEL, DEV_CHANNEL],
    );
  });

  it("should_ignore_whitespace_only_overrides", () => {
    assert.deepEqual(
      resolveAstronomicalChannels({
        ASTRONOMICAL_BASE_URL: "   ",
        ASTRONOMICAL_INSTANCE: "  ",
      }),
      [STABLE_CHANNEL, DEV_CHANNEL],
    );
  });
});
