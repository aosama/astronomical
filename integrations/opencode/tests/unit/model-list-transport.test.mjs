// Transport tests: one GET of /v1/models, with every failure reported as a
// reason rather than thrown.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { fetchModelListPayload } from "../../src/model-list-transport.ts";
import { STABLE } from "../support/test-helpers.mjs";

function fetchReturning(body, status = 200) {
  return async () => new Response(body, { status });
}

describe("fetchModelListPayload", () => {
  it("should_return_the_parsed_payload_from_the_models_route", async () => {
    let requestedUrl;
    const result = await fetchModelListPayload({
      baseUrl: STABLE,
      fetchImpl: async (url) => {
        requestedUrl = url;
        return new Response(JSON.stringify({ data: [] }), { status: 200 });
      },
      timeoutMs: 1_000,
    });
    assert.equal(requestedUrl, `${STABLE}/models`);
    assert.deepEqual(result, { kind: "success", payload: { data: [] } });
  });

  it("should_report_a_non_success_status_with_its_code", async () => {
    const result = await fetchModelListPayload({
      baseUrl: STABLE,
      fetchImpl: fetchReturning("nope", 503),
      timeoutMs: 1_000,
    });
    assert.deepEqual(result, { kind: "failure", reason: "responded 503" });
  });

  it("should_report_a_body_that_is_not_json", async () => {
    const result = await fetchModelListPayload({
      baseUrl: STABLE,
      fetchImpl: fetchReturning("<html>"),
      timeoutMs: 1_000,
    });
    assert.equal(result.kind, "failure");
  });

  it("should_report_a_transport_that_throws", async () => {
    const result = await fetchModelListPayload({
      baseUrl: STABLE,
      fetchImpl: async () => {
        throw new Error("connect ECONNREFUSED");
      },
      timeoutMs: 1_000,
    });
    assert.deepEqual(result, {
      kind: "failure",
      reason: "connect ECONNREFUSED",
    });
  });

  it("should_give_up_at_the_timeout_on_a_channel_that_never_answers", async () => {
    const neverAnswers = (_url, init) =>
      new Promise((_resolve, reject) => {
        init.signal.addEventListener("abort", () => reject(init.signal.reason));
      });
    const startedAt = performance.now();
    const result = await fetchModelListPayload({
      baseUrl: STABLE,
      fetchImpl: neverAnswers,
      timeoutMs: 50,
    });
    assert.equal(result.kind, "failure");
    assert.ok(performance.now() - startedAt < 1_000);
  });
});
