// The V2 terminal plugin registers its status and provider-toggle commands.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createAstronomicalTuiPlugin } from "../../dist/tui.js";
import {
  STABLE,
  DEV,
  perUrlFetch,
  representativeModels,
  noOpAudit,
} from "../support/test-helpers.mjs";

function createTuiContext(choice) {
  const commands = [];
  const toasts = [];
  const selections = [];

  return {
    context: {
      keymap: {
        layer(factory) {
          const layer = factory();
          commands.push(...layer.commands);
        },
      },
      ui: {
        dialog: {
          async select(options) {
            selections.push(options);
            return choice;
          },
        },
        toast: {
          show(options) {
            toasts.push(options);
          },
        },
      },
    },
    commands,
    toasts,
    selections,
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

describe("the V2 terminal plugin", () => {
  it("should_register_the_status_command_and_report_every_reachable_channel", async () => {
    const tui = createTuiContext(undefined);
    const plugin = createAstronomicalTuiPlugin({
      env: {},
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
      env: {},
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

  it("should_persist_the_selected_provider_state_from_the_toggle_command", async () => {
    const tui = createTuiContext("disabled");
    const writtenStates = [];
    const plugin = createAstronomicalTuiPlugin({
      readEnabled: async () => true,
      writeEnabled: async (isEnabled) => writtenStates.push(isEnabled),
      audit: noOpAudit,
    });

    await plugin.setup(tui.context);
    const toggleCommand = tui.commands.find(
      (command) => command.id === "astronomical.toggle",
    );

    assert.ok(toggleCommand);
    assert.deepEqual(toggleCommand.slash, { name: "astronomical-toggle" });
    await toggleCommand.run();

    assert.equal(tui.selections.length, 1);
    assert.equal(tui.selections[0].current, "enabled");
    assert.deepEqual(writtenStates, [false]);
    assert.equal(tui.toasts[0].variant, "success");
    assert.match(tui.toasts[0].message, /disabled/);
  });

  it("should_leave_the_state_unchanged_when_the_selection_dialog_is_cancelled", async () => {
    const tui = createTuiContext(undefined);
    let writeCount = 0;
    const plugin = createAstronomicalTuiPlugin({
      readEnabled: async () => false,
      writeEnabled: async () => {
        writeCount += 1;
      },
      audit: noOpAudit,
    });

    await plugin.setup(tui.context);
    const toggleCommand = tui.commands.find(
      (command) => command.id === "astronomical.toggle",
    );
    assert.ok(toggleCommand);

    await toggleCommand.run();

    assert.equal(writeCount, 0);
    assert.equal(tui.toasts.length, 0);
  });

  it("should_show_an_error_and_record_it_when_the_toggle_cannot_be_saved", async () => {
    const tui = createTuiContext("disabled");
    const auditEvents = [];
    const plugin = createAstronomicalTuiPlugin({
      readEnabled: async () => true,
      writeEnabled: async () => {
        throw new Error("read-only file system");
      },
      audit: async (event, detail) => auditEvents.push({ event, detail }),
    });

    await plugin.setup(tui.context);
    const toggleCommand = tui.commands.find(
      (command) => command.id === "astronomical.toggle",
    );
    await toggleCommand.run();

    assert.equal(tui.toasts[0].variant, "error");
    assert.match(tui.toasts[0].message, /read-only file system/);
    assert.deepEqual(auditEvents.at(-1), {
      event: "diagnostic",
      detail: { stage: "toggle-command", reason: "read-only file system" },
    });
  });
});
