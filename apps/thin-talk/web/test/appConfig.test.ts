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
  assert.equal(configForChannel("development").applicationTitle, "Thin Talk Development");
});

test("should_keep_the_plain_application_title_for_stable", () => {
  assert.equal(configForChannel("stable").applicationTitle, "Thin Talk");
});

test("should_not_claim_development_for_a_channel_it_does_not_recognize", () => {
  assert.equal(configForChannel("app-store").applicationTitle, "Thin Talk");
  assert.equal(configForChannel("DEVELOPMENT").applicationTitle, "Thin Talk");
});
