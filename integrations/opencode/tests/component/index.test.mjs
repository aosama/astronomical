// End-to-end provider-transform tests use a small fake domain editor to verify
// multi-channel discovery, fail-closed behavior, and refreshes without a
// running daemon.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createAstronomicalPlugin } from "../../src/index.ts";
import {
  STABLE,
  DEV,
  modelListFetch,
  perUrlFetch,
  representativeModels,
  noOpAudit,
} from "../support/test-helpers.mjs";

// A dev-only model id so tests can prove the two channels' model lists stay
// separate instead of one overwriting the other.
const devOnlyModel = { ...representativeModels[0], id: "dev-only-model" };

// Both instances, as a user who builds Astronomical from source configures it;
// tests of the default pass `null`, which the plugin reads as "no options".
const BOTH_CHANNELS = { channels: ["stable", "development"] };

function createProviderDomain(pluginOptions = BOTH_CHANNELS) {
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
      options: pluginOptions,
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
    [STABLE]: modelListFetch(representativeModels),
    [DEV]: modelListFetch([devOnlyModel]),
  });
}

describe("the provider transform registers one provider per live channel", () => {
  it("should_register_stable_and_dev_providers_side_by_side_when_both_channels_answer", async () => {
    // The reported bug: dev models never appeared because the old model list fetch
    // stopped at the first reachable channel. Both channels must now register.
    const plugin = createAstronomicalPlugin({
      fetchImpl: bothChannelsFetch(),
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
      fetchImpl: perUrlFetch({
        [STABLE]: () => {
          throw new Error("ECONNREFUSED");
        },
        [DEV]: modelListFetch([devOnlyModel]),
      }),
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
      fetchImpl: perUrlFetch({
        [STABLE]: modelListFetch(representativeModels),
        [DEV]: () => {
          throw new Error("ECONNREFUSED");
        },
      }),
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

  it("should_probe_only_the_development_channel_when_the_options_name_only_it", async () => {
    const plugin = createAstronomicalPlugin({
      fetchImpl: bothChannelsFetch(),
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain({ channels: ["development"] });
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 1);
    assert.equal(domain.providers[0].info.id, "astronomical-development");
    cleanup();
  });

  it("should_list_only_stable_when_no_options_are_given", async () => {
    const requestedUrls = [];
    const plugin = createAstronomicalPlugin({
      fetchImpl: async (url) => {
        requestedUrls.push(url);
        return bothChannelsFetch()(url);
      },
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain(null);
    const cleanup = await plugin.setup(domain.context);

    assert.deepEqual(
      domain.providers.map((provider) => provider.info.id),
      ["astronomical"],
    );
    assert.deepEqual(requestedUrls, [`${STABLE}/models`]);
    cleanup();
  });
});

describe("the provider transform fails closed", () => {
  it("should_stop_setup_with_a_message_naming_a_wrong_option", async () => {
    let requestCount = 0;
    const plugin = createAstronomicalPlugin({
      fetchImpl: async () => {
        requestCount += 1;
        return new Response("{}", { status: 200 });
      },
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain({ channels: ["beta"] });

    await assert.rejects(
      plugin.setup(domain.context),
      /"channels" must be a list/,
    );
    assert.equal(domain.providers.length, 0);
    assert.equal(requestCount, 0);
  });

  it("should_leave_providers_absent_and_record_a_diagnostic_when_all_channels_fail", async () => {
    const diagnostics = [];
    const auditEvents = [];
    const plugin = createAstronomicalPlugin({
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
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
      fetchImpl: perUrlFetch({
        [STABLE]: modelListFetch(onlyNonChatModels),
        [DEV]: modelListFetch([devOnlyModel]),
      }),
      audit: noOpAudit,
      scheduleRefresh: () => () => {},
    });
    const domain = createProviderDomain();
    const cleanup = await plugin.setup(domain.context);

    assert.equal(domain.providers.length, 1);
    assert.equal(domain.providers[0].info.id, "astronomical-development");
    cleanup();
  });

  it("should_remove_restore_and_refresh_providers_as_channels_and_models_change", async () => {
    let isStableUp = true;
    let stableModels = representativeModels;
    let scheduledRefresh;
    let isSchedulerStopped = false;
    const plugin = createAstronomicalPlugin({
      fetchImpl: perUrlFetch({
        [STABLE]: async () => {
          if (!isStableUp) throw new Error("ECONNREFUSED");
          return new Response(JSON.stringify({ data: stableModels }), {
            status: 200,
          });
        },
        [DEV]: modelListFetch([devOnlyModel]),
      }),
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

    isStableUp = false;
    await scheduledRefresh();
    assert.deepEqual(
      domain.providers.map((provider) => provider.info.id),
      ["astronomical-development"],
    );
    assert.equal(domain.reloadCount, 1);

    isStableUp = true;
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
