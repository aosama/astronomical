// Discovery tests: the state one pass produces for each situation.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { discoverProviderState } from "../../src/provider-discovery.ts";
import {
  STABLE,
  modelListFetch,
  representativeModels,
} from "../support/test-helpers.mjs";

const stableOnly = { channelNames: ["stable"], baseUrl: undefined };

describe("discoverProviderState", () => {
  it("should_be_ready_with_one_definition_per_answering_channel", async () => {
    const state = await discoverProviderState({
      options: stableOnly,
      fetchImpl: modelListFetch(representativeModels),
    });
    assert.equal(state.kind, "ready");
    assert.deepEqual(state.channels, [{ instance: "stable", modelCount: 2 }]);
  });

  it("should_be_empty_with_the_reason_when_a_channel_serves_no_chat_models", async () => {
    const state = await discoverProviderState({
      options: stableOnly,
      fetchImpl: modelListFetch(representativeModels.slice(2)),
    });
    assert.equal(state.kind, "empty");
    assert.match(
      state.reason,
      new RegExp(`No chat-capable models were returned by ${STABLE}`),
    );
  });

  it("should_be_unavailable_and_survive_a_diagnostic_observer_that_throws", async () => {
    const state = await discoverProviderState({
      options: stableOnly,
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      onDiagnostic: () => {
        throw new Error("observer failed");
      },
    });
    assert.equal(state.kind, "unavailable");
    assert.match(state.reason, /no Astronomical channel is reachable/);
  });

  it("should_be_failed_when_an_unexpected_error_escapes_the_probe", async () => {
    const state = await discoverProviderState({
      options: { channelNames: null, baseUrl: undefined },
    });
    assert.equal(state.kind, "failed");
  });
});
