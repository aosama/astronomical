import assert from "node:assert/strict";
import { test } from "node:test";
import { installDom, installConfig, waitFor } from "./support/dom";
import { FakeSessionHost } from "./support/fakeHost";
import { FakeDaemon } from "./support/fakeDaemon";
import { bootChatClient } from "../src/boot";
import { DAEMON_BASE_URL, bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";

function renderedMessageTexts(): string[] {
  return [...document.querySelectorAll<HTMLElement>("#transcript article.message")].map(
    (element) => {
      const messageText: string | null = element.textContent;
      return (messageText != null ? messageText : "").trim();
    },
  );
}

function sidebarSessionCount(): number {
  return document.querySelectorAll<HTMLElement>("#session-list button").length;
}

test("should_boot_send_stream_and_persist_a_whole_chat", async () => {
  const { host, daemon } = bootJourney();

  // Boot settles: handshake, model catalog, and a fresh untitled session.
  await waitFor(
    () => daemon.requests.some((request) => request.url.endsWith("/v1/models")),
    "boot to reach the model catalog",
  );
  await waitUntilReady();

  // The user asks, the fake daemon streams the reply.
  sendUserMessage("Summarize the prologue of Romeo and Juliet.");
  await waitFor(
    () => renderedMessageTexts().some((text) => text.includes("Hello from the fake daemon.")),
    "the streamed reply to render",
  );
  await waitFor(() => host.fileCount() === 1, "the session document to be saved");
  await waitFor(() => sidebarSessionCount() >= 1, "the sidebar to list the saved session");

  // The saved document carries both turns.
  const savedRequest = host.receivedRequests.find((request) => request.op === "save");
  assert.ok(savedRequest, "a save call reached the host");
  const savedPayload = savedRequest.payload as { messages: { role: string; content: string }[] };
  assert.equal(savedPayload.messages[0].role, "user");
  assert.match(savedPayload.messages[0].content, /Romeo and Juliet/);
  assert.equal(savedPayload.messages[1].role, "assistant");
  assert.match(savedPayload.messages[1].content, /Hello from the fake daemon\./);

  // The chat request asked the daemon for a streaming completion.
  const chatBody = daemon.chatRequestBody();
  assert.ok(chatBody, "a chat completions request was made");
  assert.equal(chatBody["stream"], true);
  assert.equal(chatBody["model"], "fake-chat-model");
  // Only the question goes out: the assistant bubble opened for the answer
  // must never be sent back to the daemon as an empty reply.
  assert.equal((chatBody["messages"] as unknown[]).length, 1);
  assert.equal((chatBody["messages"] as { role: string }[])[0]?.role, "user");
}, { timeout: 30_000 });

test("should_restore_the_persisted_transcript_on_a_second_boot", async () => {
  // First life: have a conversation that persists.
  const first = bootJourney();
  await waitUntilReady();
  sendUserMessage("What is the prince's decree in the prologue?");
  await waitFor(() => first.host.fileCount() === 1, "the first life to save its session");

  // Second life: a fresh page, same host files. The transcript must return.
  bootJourney(first);
  await waitFor(
    () => renderedMessageTexts().some((text) => text.includes("What is the prince's decree")),
    "the restored transcript to render",
  );
}, { timeout: 30_000 });

test("should_show_the_failure_notice_instead_of_a_transcript_when_the_handshake_fails", async () => {
  installDom();
  installConfig(DAEMON_BASE_URL);
  const host = new FakeSessionHost(
    (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
    (callId, message) => window.__thintalkBridge?.reject(callId, message),
  );
  host.install();
  const daemon = new FakeDaemon({
    daemonBaseURL: DAEMON_BASE_URL,
    // A daemon from another channel must be refused by the handshake.
    channel: "stable",
    stateDirectory: "~/.astronomical-dev",
  });
  daemon.install();
  bootChatClient();

  // A load failure renders as the empty-state notice, not the composer banner
  // (the banner is reserved for failures while streaming).
  await waitFor(() => {
    const notice = document.querySelector<HTMLElement>("#empty-state-notice");
    const noticeText: string | null | undefined = notice?.textContent;
    return (noticeText != null ? noticeText : "").includes("Another copy of Astronomical is already running");
  }, "the failure notice to appear");
  assert.equal(
    document.querySelector("#empty-state-notice")?.classList.contains("empty-state__notice--asks"),
    true,
    "a notice the person must act on is drawn to be read",
  );
  assert.equal(host.fileCount(), 0, "no session is written when the daemon is unreachable");
}, { timeout: 30_000 });

function seededUserLine(content: string): { id: string; role: string; content: string } {
  return { id: content, role: "user", content };
}

function sidebarSessionIDs(): string[] {
  return [...document.querySelectorAll<HTMLElement>("#session-list [data-session-id]")].map((item) => item.dataset.sessionId ?? "");
}

test("should_keep_the_sidebar_order_while_clicking_between_past_chats", async () => {
  installDom();
  installConfig(DAEMON_BASE_URL);
  const host = new FakeSessionHost(
    (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
    (callId, message) => window.__thintalkBridge?.reject(callId, message),
  );
  host.seedSession("newest", "Hamlet", "2026-03-03T00:00:00.000Z", [seededUserLine("To be?")]);
  host.seedSession("middle", "Macbeth", "2026-02-02T00:00:00.000Z", [seededUserLine("Is this a dagger?")]);
  host.seedSession("oldest", "Lear", "2026-01-01T00:00:00.000Z", [seededUserLine("Nothing will come of nothing?")]);
  host.install();
  new FakeDaemon({ daemonBaseURL: DAEMON_BASE_URL, channel: "development", stateDirectory: "~/.astronomical-dev" }).install();
  bootChatClient();
  await waitUntilReady();
  await waitFor(() => sidebarSessionIDs().length === 3, "the sidebar to list the seeded chats");
  const initialOrder = sidebarSessionIDs();

  for (const sessionID of ["oldest", "middle", "newest", "oldest"]) {
    document.querySelector<HTMLElement>(`#session-list [data-session-id="${sessionID}"]`)?.click();
    await waitFor(
      () => document.querySelector(`#session-list [data-session-id="${sessionID}"][aria-current="true"]`) !== null,
      `the ${sessionID} chat to open`,
    );
  }

  assert.deepEqual(sidebarSessionIDs(), initialOrder, "reading a chat must not reorder the list");
  assert.equal(host.receivedRequests.filter((request) => request.op === "save").length, 0, "no chat was rewritten");
}, { timeout: 30_000 });
