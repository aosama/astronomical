import assert from "node:assert/strict";
import { test } from "node:test";
import { HostCommand, parseHostCommand } from "../src/HostCommand";
import { HostCommandRouter } from "../src/HostCommandRouter";
import type { HostCommandHandlers } from "../src/HostCommandRouter";
import { lastAssistantMessageID } from "../src/HostCommandWiring";
import { ChatSurfaceState } from "../src/ChatSurfaceState";
import { WindowChrome } from "../src/WindowChrome";
import { waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";

// The native menu bar and toolbar drive the page through these commands, so a
// command that silently does nothing reads as a broken menu item.

function recordingHandlers(calledCommands: string[]): HostCommandHandlers {
  return {
    newChat: () => calledCommands.push("newChat"),
    toggleSidebar: () => calledCommands.push("toggleSidebar"),
    stopGenerating: () => calledCommands.push("stopGenerating"),
    copyLastAnswer: () => calledCommands.push("copyLastAnswer"),
    deleteConversation: () => calledCommands.push("deleteConversation"),
    exportConversation: () => calledCommands.push("exportConversation"),
    focusComposer: () => calledCommands.push("focusComposer"),
    openYourAstro: () => calledCommands.push("openYourAstro"),
    toggleAstro: () => calledCommands.push("toggleAstro"),
    setTitlebarInset: (titlebarInsetPixels) => calledCommands.push(`setTitlebarInset:${titlebarInsetPixels}`),
  };
}

function surfaceStateWithRoles(roles: string[]): ChatSurfaceState {
  const messages = roles.map((role, messagePosition) => ({ id: `message-${messagePosition}`, role }));
  return { transcript: { messages }, activeSessionID: null } as unknown as ChatSurfaceState;
}

test("should_parse_every_known_host_command_by_name", () => {
  for (const hostCommand of Object.values(HostCommand)) {
    assert.equal(parseHostCommand({ name: hostCommand }), hostCommand);
  }
});

test("should_refuse_payloads_that_are_not_a_known_command", () => {
  assert.equal(parseHostCommand(null), null);
  assert.equal(parseHostCommand("newChat"), null);
  assert.equal(parseHostCommand({}), null);
  assert.equal(parseHostCommand({ name: 7 }), null);
  assert.equal(parseHostCommand({ name: "formatDisk" }), null);
  assert.equal(parseHostCommand({ name: "NEWCHAT" }), null);
});

test("should_dispatch_each_command_to_its_own_handler_only", () => {
  for (const hostCommand of Object.values(HostCommand)) {
    if (hostCommand === HostCommand.SET_TITLEBAR_INSET) {
      continue;
    }
    const calledCommands: string[] = [];
    const wasUnderstood = new HostCommandRouter(recordingHandlers(calledCommands)).run({ name: hostCommand });
    assert.equal(wasUnderstood, true);
    assert.deepEqual(calledCommands, [hostCommand]);
  }
});

test("should_pass_a_valid_titlebar_inset_through_and_refuse_an_invalid_one", () => {
  const calledCommands: string[] = [];
  const router = new HostCommandRouter(recordingHandlers(calledCommands));
  assert.equal(router.run({ name: "setTitlebarInset", points: 104 }), true);
  assert.equal(router.run({ name: "setTitlebarInset", points: 0 }), true);
  for (const invalidPoints of [undefined, -1, 201, Number.NaN, "52"]) {
    assert.equal(router.run({ name: "setTitlebarInset", points: invalidPoints }), false);
  }
  assert.deepEqual(calledCommands, ["setTitlebarInset:104", "setTitlebarInset:0"]);
});

test("should_report_an_unknown_command_without_running_any_handler", () => {
  const calledCommands: string[] = [];
  const wasUnderstood = new HostCommandRouter(recordingHandlers(calledCommands)).run({ name: "unknown" });
  assert.equal(wasUnderstood, false);
  assert.deepEqual(calledCommands, []);
});

test("should_find_the_newest_assistant_message_and_nothing_before_any_answer", () => {
  assert.equal(lastAssistantMessageID(null), null);
  assert.equal(lastAssistantMessageID(surfaceStateWithRoles([])), null);
  assert.equal(lastAssistantMessageID(surfaceStateWithRoles(["user"])), null);
  assert.equal(
    lastAssistantMessageID(surfaceStateWithRoles(["user", "assistant", "user", "assistant", "user"])),
    "message-3",
  );
});

test("should_inset_the_page_under_a_native_titlebar_and_not_under_a_plain_one", () => {
  const appRoot = document.createElement("div");
  const chrome = new WindowChrome(appRoot);
  chrome.applyTitlebarInset(52);
  assert.equal(appRoot.style.getPropertyValue("--titlebar-inset"), "52px");
  assert.ok(appRoot.classList.contains("app--native-titlebar"));
  chrome.applyTitlebarInset(0);
  assert.equal(appRoot.style.getPropertyValue("--titlebar-inset"), "0px");
  assert.ok(!appRoot.classList.contains("app--native-titlebar"));
});

test("should_toggle_the_sidebar_back_and_forth", () => {
  const chrome = new WindowChrome(document.createElement("div"));
  assert.equal(chrome.isSidebarHidden(), false);
  chrome.toggleSidebar();
  assert.equal(chrome.isSidebarHidden(), true);
  chrome.toggleSidebar();
  assert.equal(chrome.isSidebarHidden(), false);
});

test("should_apply_the_injected_titlebar_inset_when_the_page_boots", async () => {
  bootJourney(undefined, { titlebarInset: 52 });
  await waitUntilReady();
  const appRoot = document.querySelector<HTMLElement>("#app");
  assert.ok(appRoot?.classList.contains("app--native-titlebar"));
  assert.equal(appRoot?.style.getPropertyValue("--titlebar-inset"), "52px");
}, { timeout: 30_000 });

test("should_copy_the_last_answer_and_delete_the_conversation_through_host_commands", async () => {
  const journey = bootJourney();
  await waitUntilReady();
  const commandEntry = window.__thintalkCommands;
  assert.ok(commandEntry != null, "the page installs the host command entry point");

  assert.equal(commandEntry.run({ name: "copyLastAnswer" }), true);
  assert.equal(journey.host.receivedActions.length, 0, "an empty conversation has nothing to copy");

  sendUserMessage("Summarize the prologue of Romeo and Juliet.");
  await waitFor(() => journey.host.fileCount() === 1, "the finished turn to be saved");
  commandEntry.run({ name: "copyLastAnswer" });
  const copyAction = journey.host.receivedActions.at(-1);
  assert.equal(copyAction?.["action"], "copy");
  assert.match(String(copyAction?.["detail"]), /Hello from the fake daemon\./);

  commandEntry.run({ name: "deleteConversation" });
  await waitFor(() => journey.host.fileCount() === 0, "the conversation file to be deleted");
  // Deleting the open conversation must clear it from the screen and leave a
  // fresh one that saves, not the deleted messages with nowhere to save to.
  await waitFor(() => window.__thintalk?.messageCount() === 0, "the deleted conversation to leave the screen");
  sendUserMessage("What does the prologue ask of the audience?");
  await waitFor(() => journey.host.fileCount() === 1, "the next turn to save to a new conversation");
}, { timeout: 30_000 });

test("should_do_nothing_when_deleting_before_any_conversation_exists", async () => {
  const journey = bootJourney();
  await waitUntilReady();
  assert.equal(window.__thintalkCommands?.run({ name: "deleteConversation" }), true);
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(journey.host.fileCount(), 0);
  assert.equal(window.__thintalk?.messageCount(), 0);
}, { timeout: 30_000 });

test("should_toggle_the_sidebar_focus_the_field_and_start_a_new_chat_through_host_commands", async () => {
  const journey = bootJourney(undefined, { titlebarInset: 52 });
  await waitUntilReady();
  const commandEntry = window.__thintalkCommands;
  assert.ok(commandEntry != null);
  const appRoot = document.querySelector<HTMLElement>("#app");

  commandEntry.run({ name: "toggleSidebar" });
  assert.ok(appRoot?.classList.contains("app--sidebar-hidden"));
  commandEntry.run({ name: "focusComposer" });
  assert.equal(document.activeElement?.id, "composer-input");
  commandEntry.run({ name: "setTitlebarInset", points: 26 });
  assert.equal(appRoot?.style.getPropertyValue("--titlebar-inset"), "26px");

  sendUserMessage("Summarize the prologue of Romeo and Juliet.");
  await waitFor(() => journey.host.fileCount() === 1, "the first conversation to save");
  commandEntry.run({ name: "newChat" });
  await waitFor(() => window.__thintalk?.messageCount() === 0, "the new chat to start empty");
  assert.equal(journey.host.fileCount(), 1, "a new chat keeps the earlier conversation saved");
}, { timeout: 30_000 });
