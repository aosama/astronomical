import assert from "node:assert/strict";
import { test } from "node:test";
import { bootChatClient } from "../src/boot";
import { assertNoElement, installConfig, installDom, waitFor } from "./support/dom";
import { FakeDaemon } from "./support/fakeDaemon";
import { FakeSessionHost } from "./support/fakeHost";
import { DAEMON_BASE_URL, waitUntilReady } from "./support/journey";

function seededUserLine(content: string): { id: string; role: string; content: string } {
  return { id: content, role: "user", content };
}

/** Boots the client over three saved chats, newest first: Hamlet, Macbeth, Lear. */
async function bootWithSavedChats(): Promise<FakeSessionHost> {
  installDom();
  installConfig(DAEMON_BASE_URL);
  const host = new FakeSessionHost(
    (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
    (callId, message) => window.__thintalkBridge?.reject(callId, message),
  );
  host.seedSession("hamlet", "Hamlet", "2026-03-03T00:00:00.000Z", [seededUserLine("To be?")]);
  host.seedSession("macbeth", "Macbeth", "2026-02-02T00:00:00.000Z", [seededUserLine("Is this a dagger?")]);
  host.seedSession("lear", "Lear", "2026-01-01T00:00:00.000Z", [seededUserLine("Nothing will come of nothing?")]);
  host.install();
  new FakeDaemon({ daemonBaseURL: DAEMON_BASE_URL, channel: "development", stateDirectory: "~/.astronomical-dev" }).install();
  bootChatClient();
  await waitUntilReady();
  await waitFor(() => sidebarTitles().length === 3, "the sidebar to list the saved chats");
  return host;
}

function sidebarTitles(): string[] {
  return [...document.querySelectorAll<HTMLElement>("#session-list .sidebar__session-title")].map((title) => title.textContent ?? "");
}

function rightClick(element: Element): MouseEvent {
  const contextMenuEvent = new window.MouseEvent("contextmenu", { bubbles: true, cancelable: true });
  element.dispatchEvent(contextMenuEvent);
  return contextMenuEvent;
}

function sessionRow(sessionID: string): HTMLElement {
  const row = document.querySelector<HTMLElement>(`#session-list [data-session-id="${sessionID}"]`);
  if (row == null) {
    throw new Error(`no sidebar row for ${sessionID}`);
  }
  return row;
}

async function openRenameField(host: FakeSessionHost, sessionID: string): Promise<HTMLInputElement> {
  host.conversationMenuChoice = "rename";
  rightClick(sessionRow(sessionID));
  await waitFor(() => document.querySelector('[data-testid="session-rename"]') !== null, "the rename field to appear");
  return document.querySelector<HTMLInputElement>('[data-testid="session-rename"]') as HTMLInputElement;
}

function pressKey(field: HTMLInputElement, key: string): void {
  field.dispatchEvent(new window.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
}

function renameRequests(host: FakeSessionHost): unknown[] {
  return host.receivedRequests.filter((request) => request.op === "rename").map((request) => request.payload);
}

test("should_rename_a_chat_in_place_without_moving_it", async () => {
  const host = await bootWithSavedChats();
  const renameField = await openRenameField(host, "macbeth");
  assert.equal(renameField.value, "Macbeth", "the field starts with the current name");
  renameField.value = "  The Scottish Play  ";
  pressKey(renameField, "Enter");
  await waitFor(() => sidebarTitles().includes("The Scottish Play"), "the new name to show");
  assert.deepEqual(sidebarTitles(), ["Hamlet", "The Scottish Play", "Lear"], "the renamed chat keeps its place");
  assert.deepEqual(renameRequests(host), [{ id: "macbeth", title: "The Scottish Play" }]);
}, { timeout: 30_000 });

test("should_keep_the_name_when_rename_is_cancelled_with_escape", async () => {
  const host = await bootWithSavedChats();
  const renameField = await openRenameField(host, "lear");
  renameField.value = "King Lear";
  pressKey(renameField, "Escape");
  await waitFor(() => document.querySelector('[data-testid="session-rename"]') === null, "the field to close");
  assert.deepEqual(sidebarTitles(), ["Hamlet", "Macbeth", "Lear"]);
  assert.equal(renameRequests(host).length, 0);
}, { timeout: 30_000 });

test("should_keep_the_name_when_the_new_one_is_blank", async () => {
  const host = await bootWithSavedChats();
  const renameField = await openRenameField(host, "hamlet");
  renameField.value = "   ";
  pressKey(renameField, "Enter");
  await waitFor(() => document.querySelector('[data-testid="session-rename"]') === null, "the field to close");
  assert.deepEqual(sidebarTitles(), ["Hamlet", "Macbeth", "Lear"]);
  assert.equal(renameRequests(host).length, 0);
}, { timeout: 30_000 });

test("should_delete_a_chat_the_user_confirmed_from_the_menu", async () => {
  const host = await bootWithSavedChats();
  host.conversationMenuChoice = "delete";
  rightClick(sessionRow("lear"));
  await waitFor(() => sidebarTitles().length === 2, "the deleted chat to leave the list");
  assert.deepEqual(sidebarTitles(), ["Hamlet", "Macbeth"]);
  assert.equal(host.fileCount(), 2);
}, { timeout: 30_000 });

test("should_change_nothing_when_the_menu_is_dismissed", async () => {
  const host = await bootWithSavedChats();
  host.conversationMenuChoice = "dismissed";
  const contextMenuEvent = rightClick(sessionRow("macbeth"));
  assert.equal(contextMenuEvent.defaultPrevented, true, "WebKit's text menu is replaced on a chat row");
  await waitFor(() => host.receivedRequests.some((request) => request.op === "showConversationMenu"), "the menu to be asked for");
  await new Promise((settle) => setTimeout(settle, 20));
  assert.deepEqual(sidebarTitles(), ["Hamlet", "Macbeth", "Lear"]);
  assert.equal(host.fileCount(), 3);
  assertNoElement(document, '[data-testid="session-rename"]');
}, { timeout: 30_000 });

test("should_leave_the_default_menu_outside_the_chat_rows", async () => {
  const host = await bootWithSavedChats();
  const contextMenuEvent = rightClick(document.querySelector("#session-list") as Element);
  assert.equal(contextMenuEvent.defaultPrevented, false);
  assert.equal(host.receivedRequests.some((request) => request.op === "showConversationMenu"), false);
}, { timeout: 30_000 });

test("should_open_the_chat_the_user_presses_while_renaming_another", async () => {
  const host = await bootWithSavedChats();
  const renameField = await openRenameField(host, "macbeth");
  renameField.value = "The Scottish Play";
  const learRow = sessionRow("lear");
  // Pressing another row blurs the field first, then the click lands on the row.
  renameField.dispatchEvent(new window.FocusEvent("blur"));
  learRow.click();
  await waitFor(() => document.querySelector('#session-list [data-session-id="lear"][aria-current="true"]') !== null, "the pressed chat to open");
  await waitFor(() => sidebarTitles().includes("The Scottish Play"), "the rename to land too");
}, { timeout: 30_000 });
