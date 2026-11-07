import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatMessage } from "../src/ChatMessage";
import { ChatRole } from "../src/ChatRole";
import { ConversationSearch } from "../src/ConversationSearch";
import { SearchConversationsTool } from "../src/SearchConversationsTool";
import { SessionDocument } from "../src/SessionDocument";
import type { SessionBridgeRequest } from "../src/SessionBridgeRequest";
import { scriptedBridge } from "./support/scriptedBridge";

function savedConversation(id: string, title: string, contents: string[]): SessionDocument {
  return new SessionDocument({
    id, title, createdAt: "2026-11-01T10:00:00.000Z", updatedAt: "2026-11-02T10:00:00.000Z",
    messages: contents.map((content, messageIndex) => new ChatMessage({ id: `${id}-${messageIndex}`, role: ChatRole.USER, content })),
  });
}

const VERONA = savedConversation("verona", "Verona trip", ["Where is Juliet's balcony?", "It is on Via Cappello."]);
const ELSINORE = savedConversation("elsinore", "Hamlet notes", ["To be, or not to be"]);
const CURRENT = savedConversation("current", "Balcony question", ["Find the balcony chat"]);

function searchBridge(isApproved: boolean, sentRequests: SessionBridgeRequest[] = [], conversations = [VERONA, ELSINORE, CURRENT]): ReturnType<typeof scriptedBridge> {
  const conversationsByID = new Map(conversations.map((conversation) => [conversation.id, conversation.toPlain()]));
  const bridge = scriptedBridge({
    useHostTool: { outcome: isApproved ? "approved" : "refused" },
    list: conversations.map((conversation) => ({ id: conversation.id, title: conversation.title, updatedAt: conversation.updatedAt })),
  }, sentRequests);
  bridge.load = async (id: string) => {
    sentRequests.push({ kind: "sessionCall", callId: `load-${id}`, op: "load" as never, payload: { id } } as SessionBridgeRequest);
    const savedPlain = conversationsByID.get(id);
    if (savedPlain == null) {
      throw new Error("the host did not answer");
    }
    return SessionDocument.fromPlain(savedPlain);
  };
  return bridge;
}

test("should_find_the_phrase_in_other_conversations_with_context", async () => {
  const toolResult = await new SearchConversationsTool(searchBridge(true)).execute({ query: "balcony" }, new AbortController().signal, { conversationID: "current", modelID: "m" });
  assert.equal(toolResult, 'Conversations that mention "balcony":\n- Verona trip (2026-11-02): Where is Juliet\'s balcony?');
});

test("should_report_when_nothing_matches", async () => {
  const toolResult = await new SearchConversationsTool(searchBridge(true)).execute({ query: "Macbeth" }, new AbortController().signal, { conversationID: "current", modelID: "m" });
  assert.equal(toolResult, 'No saved conversation mentions "Macbeth".');
});

test("should_refuse_an_empty_query_before_asking_and_stop_when_not_allowed", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const tool = new SearchConversationsTool(searchBridge(false, sentRequests));
  await assert.rejects(tool.execute({ query: "  " }, new AbortController().signal, { conversationID: "current", modelID: "m" }), /needs a word or phrase/);
  assert.equal(sentRequests.length, 0);
  await assert.rejects(tool.execute({ query: "balcony" }, new AbortController().signal, { conversationID: "current", modelID: "m" }), /did not allow/);
  assert.deepEqual(sentRequests.map((request) => request.op), ["useHostTool"]);
});

test("should_match_titles_from_the_list_without_loading_them", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const toolResult = await new SearchConversationsTool(searchBridge(true, sentRequests)).execute({ query: "hamlet" }, new AbortController().signal, { conversationID: "current", modelID: "m" });
  assert.match(toolResult, /Hamlet notes \(2026-11-02\): Hamlet notes/);
  assert.equal(sentRequests.some((request) => request.op === "load" && (request.payload as { id: string }).id === "elsinore"), false);
});

test("should_stop_at_ten_matches", async () => {
  const manyConversations = Array.from({ length: 14 }, (_unused, conversationIndex) => savedConversation(`c${conversationIndex}`, "Notes", ["balcony scene"]));
  const toolResult = await new SearchConversationsTool(searchBridge(true, [], manyConversations)).execute({ query: "balcony" }, new AbortController().signal, { conversationID: "current", modelID: "m" });
  assert.equal(toolResult.split("\n").length - 1, 10);
});

test("should_end_the_search_at_the_first_conversation_that_cannot_be_loaded", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const bridge = searchBridge(true, sentRequests, [VERONA, ELSINORE]);
  const workingLoad = bridge.load;
  bridge.load = async (id: string) => (id === "verona" ? Promise.reject(new Error("timed out")) : workingLoad(id));
  const toolResult = await new SearchConversationsTool(bridge).execute({ query: "to be" }, new AbortController().signal, { conversationID: "current", modelID: "m" });
  assert.equal(toolResult, 'No saved conversation mentions "to be".');
});

test("should_stop_searching_when_the_turn_stops", async () => {
  const stopController = new AbortController();
  const bridge = searchBridge(true);
  const workingLoad = bridge.load;
  bridge.load = async (id: string) => {
    stopController.abort();
    return workingLoad(id);
  };
  await assert.rejects(new SearchConversationsTool(bridge).execute({ query: "balcony" }, stopController.signal, { conversationID: "current", modelID: "m" }));
});

test("should_find_message_matches_and_mark_cut_context", () => {
  assert.equal(ConversationSearch.firstMessageMatch(ELSINORE, "TO BE")?.snippet, "To be, or not to be");
  const longConversation = savedConversation("long", "Long", [`${"a".repeat(200)} needle ${"b".repeat(200)}`]);
  const snippet = ConversationSearch.firstMessageMatch(longConversation, "NEEDLE")?.snippet ?? "";
  assert.ok(snippet.startsWith("…") && snippet.endsWith("…"), snippet);
  assert.ok(snippet.includes("needle"));
  assert.equal(ConversationSearch.firstMessageMatch(longConversation, "absent"), null);
  assert.equal(ConversationSearch.snippetAround("", "x"), null);
});
