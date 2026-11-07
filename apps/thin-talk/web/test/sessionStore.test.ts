import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatMessage } from "../src/ChatMessage";
import { ChatRole } from "../src/ChatRole";
import { SessionBridge } from "../src/SessionBridge";
import { SessionStore } from "../src/SessionStore";
import type { SessionBridgeRequest } from "../src/SessionBridgeRequest";
import { scriptedBridge } from "./support/scriptedBridge";

test("should_keep_the_current_conversation_when_another_cannot_be_opened", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const store = new SessionStore(scriptedBridge({ load: null, save: null, list: [] }, sentRequests));
  const currentConversation = store.startNew();
  assert.equal(await store.open("missing-conversation"), null);
  assert.equal(store.activeSessionID(), currentConversation.id, "the open conversation stays active");
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Still here?" })]);
  await store.saveNow();
  const savedConversation = sentRequests.find((request) => request.op === "save")?.payload as { id: string } | undefined;
  assert.equal(savedConversation?.id, currentConversation.id, "later turns are still saved");
});

test("should_switch_to_a_conversation_that_opens", async () => {
  const savedConversation = { id: "saved-1", title: "Verona", createdAt: "a", updatedAt: "b", messages: [] };
  const store = new SessionStore(scriptedBridge({ load: savedConversation }));
  store.startNew();
  assert.equal((await store.open("saved-1"))?.id, "saved-1");
  assert.equal(store.activeSessionID(), "saved-1");
});

test("should_keep_the_current_conversation_when_the_host_fails", async () => {
  const store = new SessionStore(scriptedBridge({}));
  const currentConversation = store.startNew();
  assert.equal(await store.open("any"), null);
  assert.equal(store.activeSessionID(), currentConversation.id);
});

function savesIn(sentRequests: readonly SessionBridgeRequest[]): number {
  return sentRequests.filter((request) => request.op === "save").length;
}

test("should_not_save_a_conversation_that_was_only_opened_and_read", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const savedConversation = { id: "saved-1", title: "Verona", createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z", messages: [] };
  const store = new SessionStore(scriptedBridge({ load: savedConversation, save: null, list: [] }, sentRequests));
  await store.open("saved-1");
  await store.saveNow();
  assert.equal(savesIn(sentRequests), 0, "viewing a conversation must not move it to the top of the sidebar");
});

test("should_not_write_a_new_chat_that_never_got_a_message", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const store = new SessionStore(scriptedBridge({ save: null, list: [] }, sentRequests));
  store.startNew();
  await store.saveNow();
  assert.equal(savesIn(sentRequests), 0);
});

test("should_save_once_after_new_messages_and_not_again_without_more", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const store = new SessionStore(scriptedBridge({ save: null, list: [] }, sentRequests));
  store.startNew();
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "But soft, what light?" })]);
  await store.saveNow();
  await store.saveNow();
  assert.equal(savesIn(sentRequests), 1);
});

test("should_retry_the_save_after_a_failed_write", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const store = new SessionStore(scriptedBridge({ list: [] }, sentRequests));
  store.startNew();
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Wherefore art thou?" })]);
  await assert.rejects(store.saveNow());
  await assert.rejects(store.saveNow(), "the unsaved changes are still pending after the failure");
  assert.equal(savesIn(sentRequests), 2);
});

/** A bridge whose answers the test releases by hand, so a save can be held
 * open while the conversation changes underneath it. */
function heldBridge(sentRequests: SessionBridgeRequest[]): SessionStore {
  return new SessionStore(new SessionBridge((request) => {
    sentRequests.push(request);
    if (request.op === "list") {
      setTimeout(() => window.__thintalkBridge?.resolve(request.callId, []), 0);
    }
  }));
}

function answerSave(sentRequests: readonly SessionBridgeRequest[], saveIndex: number, failure?: string): void {
  const saveRequest = sentRequests.filter((request) => request.op === "save")[saveIndex];
  if (saveRequest == null) {
    throw new Error(`save ${saveIndex} was never sent`);
  }
  if (failure != null) {
    window.__thintalkBridge?.reject(saveRequest.callId, failure);
  } else {
    window.__thintalkBridge?.resolve(saveRequest.callId, null);
  }
}

test("should_save_again_when_messages_arrive_while_a_save_is_in_flight", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const store = heldBridge(sentRequests);
  store.startNew();
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Once more unto the breach" })]);
  const firstSave = store.saveNow();
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Once more unto the breach, dear friends" })]);
  answerSave(sentRequests, 0);
  await firstSave;
  const secondSave = store.saveNow();
  answerSave(sentRequests, 1);
  await secondSave;
  assert.equal(savesIn(sentRequests), 2);
});

test("should_not_mark_a_new_chat_when_the_previous_chat_fails_to_save", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const store = heldBridge(sentRequests);
  store.startNew();
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Out, damned spot!" })]);
  const failingSave = store.saveNow();
  store.startNew();
  answerSave(sentRequests, 0, "disk full");
  await assert.rejects(failingSave);
  await store.saveNow();
  assert.equal(savesIn(sentRequests), 1, "the untouched new chat is not written");
});

test("should_write_staged_messages_on_a_switch_even_without_a_scheduled_save", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const savedConversation = { id: "saved-1", title: "Verona", createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z", messages: [] };
  const store = new SessionStore(scriptedBridge({ load: savedConversation, save: null, list: [] }, sentRequests));
  const leftConversation = store.startNew();
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Good night, sweet prince" })]);
  await store.open("saved-1");
  const savedIDs = sentRequests.filter((request) => request.op === "save").map((request) => (request.payload as { id: string }).id);
  assert.deepEqual(savedIDs, [leftConversation.id]);
});

test("should_write_nothing_when_a_scheduled_save_fires_with_nothing_staged", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const savedConversation = { id: "saved-1", title: "Verona", createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z", messages: [] };
  const store = new SessionStore(scriptedBridge({ load: savedConversation, save: null, list: [] }, sentRequests));
  await store.open("saved-1");
  store.scheduleSave();
  await new Promise((settle) => setTimeout(settle, 600));
  assert.equal(savesIn(sentRequests), 0);
});

test("should_keep_a_new_name_for_the_open_chat_when_it_is_saved_again", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const savedConversation = { id: "saved-1", title: "Verona", createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z", messages: [] };
  const store = new SessionStore(scriptedBridge({ load: savedConversation, save: null, rename: null, list: [] }, sentRequests));
  await store.open("saved-1");
  await store.rename("saved-1", "Fair Verona");
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Two households, both alike in dignity" })]);
  await store.saveNow();
  const savedTitles = sentRequests.filter((request) => request.op === "save").map((request) => (request.payload as { title: string }).title);
  assert.deepEqual(savedTitles, ["Fair Verona"]);
});

test("should_keep_the_old_name_of_the_open_chat_when_renaming_fails", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const savedConversation = { id: "saved-1", title: "Verona", createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z", messages: [] };
  const store = new SessionStore(scriptedBridge({ load: savedConversation, save: null, list: [] }, sentRequests));
  await store.open("saved-1");
  await assert.rejects(store.rename("saved-1", "Fair Verona"));
  assert.equal(store.activeTitle(), "Verona");
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Two households, both alike in dignity" })]);
  await store.saveNow();
  const savedTitles = sentRequests.filter((request) => request.op === "save").map((request) => (request.payload as { title: string }).title);
  assert.deepEqual(savedTitles, ["Verona"]);
});
