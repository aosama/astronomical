// End-to-end provider-transform tests use a small fake domain editor to verify
// multi-channel discovery, fail-closed behavior, and refreshes without a
// running daemon.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createAstronomicalPlugin } from "../../dist/index.js";
import {
  STABLE,
  DEV,
  catalogueFetch,
  perUrlFetch,
  representativeModels,
  noOpAudit,
} from "./test-helpers.mjs";

// A dev-only model id so tests can prove the two channels' catalogues stay
// separate instead of one overwriting the other.
const devOnlyModel = { ...representativeModels[0], id: "dev-only-model" };

function createProviderDomain() {
  const providers = [];
  const transforms = [];
  let reloadCount = 0;

  const rebuild = () => {
    providers.length = 0;
    for (const transform of transforms) {
      transform({ add: (definition) => providers.push(definition) });
    }
  };

  return {
    context: {
      provider: {
        async transform(transform) {
          transforms.push(transform);
          rebuild();
          return { dispose: async () => {} };
        },
        async reload() {
          reloadCount += 1;
          rebuild();
        },
      },
    },
    providers,
    get reloadCount() {
      return reloadCount;
    },
  };
}

function bothChannelsFetch() {
  return perUrlFetch({
    [STABLE]: catalogueFetch(representativeModels),
    [DEV]: catalogueFetch([devOnlyModel]),
  });
}

describe("the V2 provider transform registers one provider per live channel", () => {
  it("should_register_stable_and_dev_providers_side_by_side_when_both_channels_answer", async () => {
    // The reported bug: dev models never appeared because the old catalogue
    // stopped at the first reachable channel. Both channels must now register.
    const plugin = createAstronomicalPlugin({
      env: {},
      fetchImpl: bothChannelsFetch(),
      readEnabled: async () => true,
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 2);
    const [stableDefinition, devDefinition] = domain.providers;

    assert.equal(stableDefinition.info.id, "astronomical");
    assert.equal(stableDefinition.info.name, "Astronomical");
    assert.equal(stableDefinition.info.settings.baseURL, STABLE);
    assert.deepEqual(
      stableDefinition.models.map((model) => model.id),
      ["text-only-model", "vision-model"],
    );

    assert.equal(devDefinition.info.id, "astronomical-development");
    assert.equal(devDefinition.info.name, "Astronomical Development");
    assert.equal(devDefinition.info.settings.baseURL, DEV);
    assert.deepEqual(
      devDefinition.models.map((model) => model.id),
      ["dev-only-model"],
    );

    cleanup();
  });

  it("should_register_only_dev_when_stable_is_unavailable", async () => {
    const plugin = createAstronomicalPlugin({
      env: {},
      fetchImpl: perUrlFetch({
        [STABLE]: () => {
          throw new Error("ECONNREFUSED");
        },
        [DEV]: catalogueFetch([devOnlyModel]),
      }),
      readEnabled: async () => true,
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 1);
    assert.equal(domain.providers[0].info.id, "astronomical-development");
    assert.equal(domain.providers[0].info.settings.baseURL, DEV);
    cleanup();
  });

  it("should_register_only_stable_when_dev_is_unavailable", async () => {
    const plugin = createAstronomicalPlugin({
      env: {},
      fetchImpl: perUrlFetch({
        [STABLE]: catalogueFetch(representativeModels),
        [DEV]: () => {
          throw new Error("ECONNREFUSED");
        },
      }),
      readEnabled: async () => true,
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 1);
    assert.equal(domain.providers[0].info.id, "astronomical");
    assert.equal(domain.providers[0].info.settings.baseURL, STABLE);
    cleanup();
  });

  it("should_honor_ASTRONOMICAL_INSTANCE_dev_by_probing_only_the_dev_channel", async () => {
    const plugin = createAstronomicalPlugin({
      env: { ASTRONOMICAL_INSTANCE: "dev" },
      fetchImpl: bothChannelsFetch(),
      readEnabled: async () => true,
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 1);
    assert.equal(domain.providers[0].info.id, "astronomical-development");
    cleanup();
  });
});

describe("the provider transform fails closed and follows the terminal toggle", () => {
  it("should_not_probe_or_register_when_disabled", async () => {
    let requestCount = 0;
    const plugin = createAstronomicalPlugin({
      env: {},
      fetchImpl: async () => {
        requestCount += 1;
        return new Response("{}", { status: 200 });
      },
      readEnabled: async () => false,
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 0);
    assert.equal(requestCount, 0);
    cleanup();
  });

  it("should_leave_providers_absent_and_record_a_diagnostic_when_all_channels_fail", async () => {
    const diagnostics = [];
    const auditEvents = [];
    const plugin = createAstronomicalPlugin({
      env: {},
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      readEnabled: async () => true,
      audit: async (event, detail) => auditEvents.push({ event, ...detail }),
      onDiagnostic: (message) => diagnostics.push(message),
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 0);
    assert.equal(diagnostics.length, 1);
    assert.match(diagnostics[0], /no Astronomical channel is reachable/);
    assert.equal(auditEvents[0].event, "setup-start");
    assert.equal(auditEvents[1].event, "setup-skipped");
    cleanup();
  });

  it("should_exclude_a_channel_that_answers_with_zero_chat_models", async () => {
    const onlyNonChatModels = representativeModels.filter(
      (model) => !model.supported_endpoints.includes("/v1/chat/completions"),
    );
    const plugin = createAstronomicalPlugin({
      env: {},
      fetchImpl: perUrlFetch({
        [STABLE]: catalogueFetch(onlyNonChatModels),
        [DEV]: catalogueFetch([devOnlyModel]),
      }),
      readEnabled: async () => true,
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 1);
    assert.equal(domain.providers[0].info.id, "astronomical-development");
    cleanup();
  });

  it("should_remove_restore_and_refresh_providers_as_shared_state_and_catalogues_change", async () => {
    let isEnabled = true;
    let stableModels = representativeModels;
    let scheduledRefresh;
    let isSchedulerStopped = false;
    const plugin = createAstronomicalPlugin({
      env: {},
      fetchImpl: perUrlFetch({
        [STABLE]: async () =>
          new Response(JSON.stringify({ data: stableModels }), { status: 200 }),
        [DEV]: catalogueFetch([devOnlyModel]),
      }),
      readEnabled: async () => isEnabled,
      audit: noOpAudit,
      scheduleRefresh: (refresh) => {
        scheduledRefresh = refresh;
        return () => {
          isSchedulerStopped = true;
        };
      },
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 2);
    assert.equal(domain.providers[0].models.length, 2);
    assert.equal(domain.providers[1].models.length, 1);

    isEnabled = false;
    await scheduledRefresh();
    assert.equal(domain.providers.length, 0);
    assert.equal(domain.reloadCount, 1);

    isEnabled = true;
    stableModels = [representativeModels[1]];
    await scheduledRefresh();
    assert.equal(domain.providers.length, 2);
    assert.deepEqual(
      domain.providers[0].models.map((model) => model.id),
      ["vision-model"],
    );
    assert.deepEqual(
      domain.providers[1].models.map((model) => model.id),
      ["dev-only-model"],
    );
    assert.equal(domain.reloadCount, 2);

    cleanup();
    assert.equal(isSchedulerStopped, true);
  });
});
