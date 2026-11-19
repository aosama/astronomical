// Refresh controller tests: when OpenCode is asked to reload, what is
// recorded, and how overlapping or late refreshes behave.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ProviderRefreshController } from "../../dist/provider-refresh-controller.js";

function readyState(modelIds) {
  return {
    kind: "ready",
    definitions: [{ info: { id: "astronomical" }, models: modelIds }],
    channels: [{ instance: "stable", modelCount: modelIds.length }],
  };
}

function stubContext() {
  const counts = { reloads: 0, addedDefinitions: 0 };
  let registeredTransform;
  return {
    counts,
    context: {
      provider: {
        async transform(transform) {
          registeredTransform = transform;
        },
        async reload() {
          counts.reloads += 1;
          registeredTransform({
            add: () => {
              counts.addedDefinitions += 1;
            },
          });
        },
      },
    },
  };
}

function recordingAudit() {
  const events = [];
  return {
    events,
    audit: async (event, detail) => events.push({ event, detail }),
  };
}

describe("ProviderRefreshController", () => {
  it("should_reload_only_when_the_definitions_change", async () => {
    const states = [
      readyState(["a"]),
      readyState(["a"]),
      readyState(["a", "b"]),
    ];
    const { counts, context } = stubContext();
    const { events, audit } = recordingAudit();
    const controller = new ProviderRefreshController(
      context,
      async () => states.shift(),
      audit,
    );

    await controller.refresh();
    await controller.registerTransform();
    await controller.refresh();
    assert.equal(counts.reloads, 0, "an unchanged list must not reload");
    await controller.refresh();
    assert.equal(counts.reloads, 1);
    assert.deepEqual(
      events.map((entry) => entry.event),
      ["provider-registered", "provider-refreshed"],
    );
  });

  it("should_record_removal_when_a_registered_provider_goes_away", async () => {
    const states = [readyState(["a"]), { kind: "unavailable", reason: "down" }];
    const { counts, context } = stubContext();
    const { events, audit } = recordingAudit();
    const controller = new ProviderRefreshController(
      context,
      async () => states.shift(),
      audit,
    );

    await controller.refresh();
    await controller.registerTransform();
    await controller.refresh();
    assert.equal(counts.reloads, 1);
    assert.deepEqual(events.at(-2), {
      event: "provider-removed",
      detail: { reason: "down" },
    });
  });

  it("should_ignore_a_refresh_after_dispose_and_one_that_overlaps_a_running_refresh", async () => {
    let discoverCount = 0;
    let releaseDiscovery;
    const { context } = stubContext();
    const controller = new ProviderRefreshController(
      context,
      () => {
        discoverCount += 1;
        return new Promise((resolve) => {
          releaseDiscovery = () => resolve(readyState(["a"]));
        });
      },
      async () => {},
    );

    const firstRefresh = controller.refresh();
    await controller.refresh();
    assert.equal(
      discoverCount,
      1,
      "an overlapping refresh must not start a second discovery",
    );
    releaseDiscovery();
    await firstRefresh;
    controller.dispose();
    await controller.refresh();
    assert.equal(
      discoverCount,
      1,
      "a disposed controller must not discover again",
    );
  });

  it("should_record_a_discovery_that_throws_without_letting_it_escape", async () => {
    const { context } = stubContext();
    const { events, audit } = recordingAudit();
    const controller = new ProviderRefreshController(
      context,
      async () => {
        throw new Error("boom");
      },
      audit,
    );

    await controller.refresh();
    assert.deepEqual(events, [
      {
        event: "diagnostic",
        detail: { stage: "provider-refresh", reason: "boom" },
      },
    ]);
  });
});
