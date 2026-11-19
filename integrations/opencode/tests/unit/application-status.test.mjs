// Application status tests: the version is read when present and quietly
// unknown otherwise.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  fetchApplicationVersion,
  readApplicationVersion,
} from "../../dist/application-status.js";
import { STABLE } from "../support/test-helpers.mjs";

describe("readApplicationVersion", () => {
  it("should_read_application_version_from_a_status_payload", () => {
    assert.equal(
      readApplicationVersion({ application: { version: "0.2.118" } }),
      "0.2.118",
    );
  });

  it("should_return_undefined_for_a_missing_or_mistyped_version", () => {
    for (const payload of [
      null,
      [],
      {},
      { application: [] },
      { application: { version: 2 } },
    ]) {
      assert.equal(readApplicationVersion(payload), undefined);
    }
  });
});

describe("fetchApplicationVersion", () => {
  it("should_ask_the_status_route_of_the_channel", async () => {
    let requestedUrl;
    const version = await fetchApplicationVersion(STABLE, async (url) => {
      requestedUrl = url;
      return new Response(
        JSON.stringify({ application: { version: "0.2.118" } }),
      );
    });
    assert.equal(requestedUrl, `${STABLE}/status`);
    assert.equal(version, "0.2.118");
  });

  it("should_return_undefined_when_the_status_read_fails", async () => {
    assert.equal(
      await fetchApplicationVersion(
        STABLE,
        async () => new Response("x", { status: 500 }),
      ),
      undefined,
    );
    assert.equal(
      await fetchApplicationVersion(STABLE, async () => {
        throw new Error("ECONNREFUSED");
      }),
      undefined,
    );
  });
});
