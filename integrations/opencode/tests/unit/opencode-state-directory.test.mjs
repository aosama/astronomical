// State folder tests: the plugin's files sit in OpenCode's own state folder.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveOpenCodeStateDirectory } from "../../dist/opencode-state-directory.js";

describe("resolveOpenCodeStateDirectory", () => {
  it("should_use_XDG_STATE_HOME_when_it_is_set", () => {
    assert.equal(
      resolveOpenCodeStateDirectory(
        { XDG_STATE_HOME: "/xdg/state" },
        "/home/tester",
      ),
      "/xdg/state/opencode",
    );
  });

  it("should_fall_back_to_the_home_folder_when_XDG_STATE_HOME_is_unset_or_blank", () => {
    for (const environment of [{}, { XDG_STATE_HOME: "   " }]) {
      assert.equal(
        resolveOpenCodeStateDirectory(environment, "/home/tester"),
        "/home/tester/.local/state/opencode",
      );
    }
  });
});
