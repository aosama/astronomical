// Catalogue-loader tests: the chat-only capability filter, the fail-closed
// transport behavior, and the every-channel fan-in that replaced the old
// first-answer-wins probing.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { fetchChannelChatModels } from "../../dist/channel-chat-models.js";
import {
  STABLE,
  DEV,
  STABLE_CHANNEL,
  DEV_CHANNEL,
  catalogueFetch,
  perUrlFetch,
  representativeModels,
} from "../support/test-helpers.mjs";

describe("the catalogue keeps only chat-capable models", () => {
  it("should_keep_chat_and_vision_models_and_drop_image_generation_and_embeddings", async () => {
    const catalogue = await fetchChannelChatModels({
      channels: [STABLE_CHANNEL],
      fetchImpl: catalogueFetch(representativeModels),
    });

    assert.ok(catalogue.kind === "success");
    assert.deepEqual(
      catalogue.channels[0].models.map((model) => model.id),
      ["text-only-model", "vision-model"],
    );
  });

  it("should_serve_no_models_from_a_well_formed_response_that_is_not_the_expected_envelope", async () => {
    for (const body of [{}, { data: null }, { data: "text" }, []]) {
      const catalogue = await fetchChannelChatModels({
        channels: [STABLE_CHANNEL],
        fetchImpl: async () =>
          new Response(JSON.stringify(body), { status: 200 }),
      });

      assert.ok(catalogue.kind === "success");
      assert.equal(catalogue.channels.length, 1);
      assert.deepEqual(catalogue.channels[0].models, []);
    }
  });
});

describe("the catalogue fails closed on every transport and payload problem", () => {
  it("should_report_a_server_error_as_unreachable_never_as_a_silent_empty_list", async () => {
    const result = await fetchChannelChatModels({
      channels: [STABLE_CHANNEL],
      fetchImpl: async () => new Response("nope", { status: 500 }),
    });
    assert.equal(result.kind, "failure");
  });

  it("should_report_a_throwing_transport_as_unreachable", async () => {
    const result = await fetchChannelChatModels({
      channels: [STABLE_CHANNEL],
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
    });
    assert.equal(result.kind, "failure");
  });

  it("should_surface_a_diagnostic_for_each_failure_so_the_logs_explain_the_silence", async () => {
    const messages = [];
    const collect = (message) => messages.push(message);

    await fetchChannelChatModels({
      channels: [STABLE_CHANNEL],
      fetchImpl: async () => new Response("no", { status: 500 }),
      onDiagnostic: collect,
    });
    await fetchChannelChatModels({
      channels: [STABLE_CHANNEL],
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      onDiagnostic: collect,
    });

    assert.equal(messages.length, 2);
    // Each failed fetch produces exactly one summary diagnostic that folds the
    // channel and its reason together.
    assert.match(messages[0], /responded 500/);
    assert.match(messages[1], /ECONNREFUSED/);
  });
});

describe("the catalogue serves every answering channel, not just the first", () => {
  it("should_return_both_catalogues_when_both_channels_answer", async () => {
    // This is the regression test for the reported bug: with stable always
    // running, first-answer-wins probing made the dev catalogue permanently
    // invisible.
    const diagnostics = [];
    const catalogue = await fetchChannelChatModels({
      channels: [STABLE_CHANNEL, DEV_CHANNEL],
      fetchImpl: catalogueFetch(representativeModels),
      onDiagnostic: (message) => diagnostics.push(message),
    });

    assert.ok(catalogue.kind === "success");
    assert.deepEqual(
      catalogue.channels.map((channel) => channel.instance),
      ["stable", "dev"],
    );
    assert.deepEqual(
      catalogue.channels.map((channel) => channel.baseUrl),
      [STABLE, DEV],
    );
    assert.equal(diagnostics.length, 0);
  });

  it("should_serve_only_dev_when_stable_is_down_and_explain_the_skip", async () => {
    const diagnostics = [];
    const catalogue = await fetchChannelChatModels({
      channels: [STABLE_CHANNEL, DEV_CHANNEL],
      fetchImpl: perUrlFetch({
        [STABLE]: () => {
          throw new Error("ECONNREFUSED");
        },
        "*": catalogueFetch(representativeModels),
      }),
      onDiagnostic: (message) => diagnostics.push(message),
    });

    assert.ok(catalogue.kind === "success");
    assert.deepEqual(
      catalogue.channels.map((channel) => channel.instance),
      ["dev"],
    );
    assert.equal(diagnostics.length, 1);
    assert.match(diagnostics[0], /skipped unreachable/);
    assert.match(diagnostics[0], new RegExp(STABLE));
  });

  it("should_return_failure_and_a_diagnostic_when_no_channel_is_reachable", async () => {
    const diagnostics = [];
    const catalogue = await fetchChannelChatModels({
      channels: [STABLE_CHANNEL, DEV_CHANNEL],
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      onDiagnostic: (message) => diagnostics.push(message),
    });

    assert.equal(catalogue.kind, "failure");
    assert.equal(diagnostics.length, 1);
    assert.match(diagnostics[0], /no Astronomical channel is reachable/);
  });

  it("should_keep_a_channel_that_answers_with_zero_chat_models_distinguishable_from_a_down_one", async () => {
    const onlyNonChatModels = representativeModels.filter(
      (model) => !model.supported_endpoints.includes("/v1/chat/completions"),
    );
    const catalogue = await fetchChannelChatModels({
      channels: [STABLE_CHANNEL, DEV_CHANNEL],
      fetchImpl: perUrlFetch({
        [STABLE]: catalogueFetch(onlyNonChatModels),
        [DEV]: catalogueFetch(representativeModels),
      }),
    });

    assert.ok(catalogue.kind === "success");
    assert.equal(catalogue.channels.length, 2);
    assert.deepEqual(
      catalogue.channels[0].models.map((model) => model.id),
      [],
    );
    assert.equal(catalogue.channels[1].models.length, 2);
  });
});

describe("every channel is probed at the same time", () => {
  it("should_wait_for_the_slowest_channel_once_not_for_each_channel_in_turn", async () => {
    const answerAfter = (delayMs, response) => () =>
      new Promise((resolve) => setTimeout(() => resolve(response()), delayMs));
    const fetchImpl = perUrlFetch({
      [STABLE]: answerAfter(300, () => new Response("down", { status: 503 })),
      [DEV]: answerAfter(
        200,
        () =>
          new Response(JSON.stringify({ data: representativeModels }), {
            status: 200,
          }),
      ),
    });
    const startedAt = performance.now();
    const result = await fetchChannelChatModels({
      channels: [STABLE_CHANNEL, DEV_CHANNEL],
      fetchImpl,
    });
    const elapsedMs = performance.now() - startedAt;

    assert.ok(
      elapsedMs < 550,
      `probes ran one after another (${elapsedMs} ms)`,
    );
    assert.ok(result.kind === "success");
    assert.deepEqual(
      result.channels.map((channel) => channel.instance),
      ["dev"],
    );
  });
});
