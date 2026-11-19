// The terminal plugin registers its status command, which reads the same
// options as the server part.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createAstronomicalTuiPlugin } from "../../src/tui.ts";
import {
  STABLE,
  DEV,
  perUrlFetch,
  representativeModels,
  noOpAudit,
} from "../support/test-helpers.mjs";

function createTuiContext(pluginOptions) {
  const commands = [];
  const toasts = [];

  return {
    context: {
      options: pluginOptions,
      keymap: {
        layer(factory) {
          const layer = factory();
          commands.push(...layer.commands);
        },
      },
      ui: {
        toast: {
          show(options) {
            toasts.push(options);
          },
        },
      },
    },
    commands,
    toasts,
  };
}

function bothChannelsStatusFetch() {
  return perUrlFetch({
    [STABLE]: async (url) => {
      if (url === `${STABLE}/models`) {
        return new Response(JSON.stringify({ data: representativeModels }), {
          status: 200,
        });
      }
      if (url === `${STABLE}/status`) {
        return new Response(
          JSON.stringify({ application: { version: "0.2.77" } }),
          { status: 200 },
        );
      }
      throw new Error(`Unexpected stable URL: ${url}`);
    },
    [DEV]: async (url) => {
      if (url === `${DEV}/models`) {
        return new Response(
          JSON.stringify({
            data: [{ ...representativeModels[0], id: "dev-model" }],
          }),
          { status: 200 },
        );
      }
      if (url === `${DEV}/status`) {
        return new Response(
          JSON.stringify({ application: { version: "0.3.0-nightly" } }),
          { status: 200 },
        );
      }
      throw new Error(`Unexpected dev URL: ${url}`);
    },
  });
}

describe("the terminal plugin", () => {
  it("should_register_the_status_command_and_report_every_reachable_channel", async () => {
    const tui = createTuiContext({ channels: ["stable", "development"] });
    const plugin = createAstronomicalTuiPlugin({
      fetchImpl: bothChannelsStatusFetch(),
      audit: noOpAudit,
    });

    await plugin.setup(tui.context);
    const statusCommand = tui.commands.find(
      (command) => command.id === "astronomical.status",
    );

    assert.ok(statusCommand);
    assert.deepEqual(statusCommand.slash, { name: "astronomical-status" });
    await statusCommand.run();
    assert.equal(tui.toasts.length, 1);
    assert.equal(tui.toasts[0].variant, "success");
    assert.match(tui.toasts[0].message, /Astronomical 0\.2\.77: 2 chat models/);
    assert.match(
      tui.toasts[0].message,
      /Astronomical Development 0\.3\.0-nightly: 1 chat models/,
    );
  });

  it("should_warn_when_no_channel_is_reachable", async () => {
    const tui = createTuiContext(undefined);
    const plugin = createAstronomicalTuiPlugin({
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      audit: noOpAudit,
    });

    await plugin.setup(tui.context);
    const statusCommand = tui.commands.find(
      (command) => command.id === "astronomical.status",
    );

    await statusCommand.run();
    assert.equal(tui.toasts[0].variant, "warning");
    assert.match(tui.toasts[0].message, /No Astronomical channel is reachable/);
  });

  it("should_register_only_the_status_command", async () => {
    const tui = createTuiContext(undefined);
    await createAstronomicalTuiPlugin({ audit: noOpAudit }).setup(tui.context);

    assert.deepEqual(
      tui.commands.map((command) => command.id),
      ["astronomical.status"],
    );
  });

  it("should_stop_setup_with_a_message_naming_a_wrong_option", async () => {
    const tui = createTuiContext({ channels: "stable" });
    await assert.rejects(
      createAstronomicalTuiPlugin({ audit: noOpAudit }).setup(tui.context),
      /"channels" must be a list/,
    );
  });
});
