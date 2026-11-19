// Discovery tests: the state one pass produces for each situation.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { discoverProviderState } from "../../dist/provider-discovery.js";
import {
  STABLE,
  catalogueFetch,
  representativeModels,
} from "../support/test-helpers.mjs";

const stableOnly = { ASTRONOMICAL_INSTANCE: "stable" };

describe("discoverProviderState", () => {
  it("should_be_ready_with_one_definition_per_answering_channel", async () => {
    const state = await discoverProviderState({
      environment: stableOnly,
      fetchImpl: catalogueFetch(representativeModels),
      readEnabled: async () => true,
    });
    assert.equal(state.kind, "ready");
    assert.deepEqual(state.channels, [{ instance: "stable", modelCount: 2 }]);
  });

  it("should_be_disabled_without_probing_when_the_toggle_is_off", async () => {
    let probeCount = 0;
    const state = await discoverProviderState({
      environment: stableOnly,
      fetchImpl: async () => {
        probeCount += 1;
        throw new Error("must not probe");
      },
      readEnabled: async () => false,
    });
    assert.deepEqual(state, { kind: "disabled" });
    assert.equal(probeCount, 0);
  });

  it("should_be_empty_with_the_reason_when_a_channel_serves_no_chat_models", async () => {
    const state = await discoverProviderState({
      environment: stableOnly,
      fetchImpl: catalogueFetch(representativeModels.slice(2)),
      readEnabled: async () => true,
    });
    assert.equal(state.kind, "empty");
    assert.match(
      state.reason,
      new RegExp(`No chat-capable models were returned by ${STABLE}`),
    );
  });

  it("should_be_unavailable_and_survive_a_diagnostic_observer_that_throws", async () => {
    const state = await discoverProviderState({
      environment: stableOnly,
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      readEnabled: async () => true,
      onDiagnostic: () => {
        throw new Error("observer failed");
      },
    });
    assert.equal(state.kind, "unavailable");
    assert.match(state.reason, /no Astronomical channel is reachable/);
  });

  it("should_be_failed_when_reading_the_toggle_throws", async () => {
    const state = await discoverProviderState({
      environment: stableOnly,
      readEnabled: async () => {
        throw new Error("state file locked");
      },
    });
    assert.deepEqual(state, { kind: "failed", reason: "state file locked" });
  });
});
