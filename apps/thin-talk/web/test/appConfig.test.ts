import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";

function configForChannel(expectedChannel: string): AppConfig {
  return new AppConfig({
    daemonBaseURL: "http://127.0.0.1:6733",
    expectedChannel,
    expectedStateDirectory: "~/.astronomical-dev",
  });
}

test("should_name_development_in_the_application_title", () => {
  assert.equal(configForChannel("development").applicationTitle, "Astronomical Development");
});

test("should_keep_the_plain_application_title_for_stable", () => {
  assert.equal(configForChannel("stable").applicationTitle, "Astronomical");
});

test("should_not_claim_development_for_a_channel_it_does_not_recognize", () => {
  assert.equal(configForChannel("app-store").applicationTitle, "Astronomical");
  assert.equal(configForChannel("DEVELOPMENT").applicationTitle, "Astronomical");
});

function injectedConfig(titlebarInset: unknown): unknown {
  return {
    daemonBaseURL: "http://127.0.0.1:6733",
    expectedChannel: "stable",
    expectedStateDirectory: "~/.astronomical",
    titlebarInset,
  };
}

test("should_read_the_titlebar_inset_a_native_toolbar_host_injects", () => {
  assert.equal(AppConfig.fromWindow(injectedConfig(52)).titlebarInset, 52);
  assert.equal(AppConfig.fromWindow(injectedConfig(0)).titlebarInset, 0);
  assert.equal(AppConfig.fromWindow(injectedConfig(200)).titlebarInset, 200);
});

test("should_default_the_titlebar_inset_to_zero_for_a_plain_titlebar_host", () => {
  assert.equal(AppConfig.fromWindow(injectedConfig(undefined)).titlebarInset, 0);
});

test("should_refuse_a_titlebar_inset_that_would_hide_or_break_the_page", () => {
  for (const invalidInset of [-1, 201, Number.NaN, Number.POSITIVE_INFINITY, "52", null]) {
    assert.throws(() => AppConfig.fromWindow(injectedConfig(invalidInset)), /title bar inset/);
  }
});
