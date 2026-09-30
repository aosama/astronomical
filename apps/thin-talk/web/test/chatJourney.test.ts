import assert from "node:assert/strict";
import { test } from "node:test";
import { installDom, installConfig, waitFor } from "./support/dom";
import { FakeSessionHost } from "./support/fakeHost";
import { FakeSupervisor } from "./support/fakeSupervisor";
import { bootChatClient } from "../src/boot";

const SUPERVISOR_BASE_URL = "http://127.0.0.1:6733";

interface Journey {
  host: FakeSessionHost;
  supervisor: FakeSupervisor;
}

/** Boots the real client wiring against the fake host and fake supervisor.
 * Pass a previous journey to keep its host files across a fresh page —
 * the same persistence a relaunch of the real app would see. */
function bootJourney(previous?: Journey): Journey {
  installDom();
  installConfig(SUPERVISOR_BASE_URL);
  const host = previous?.host ?? new FakeSessionHost(
    (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
    (callId, message) => window.__thintalkBridge?.reject(callId, message),
  );
  host.install();
  const supervisor = previous?.supervisor ?? new FakeSupervisor({
    supervisorBaseURL: SUPERVISOR_BASE_URL,
    channel: "development",
    stateDirectory: "~/.astronomical-dev",
  });
  supervisor.install();
  bootChatClient();
  return { host, supervisor };
}

function sendUserMessage(text: string): void {
  const field = document.querySelector<HTMLTextAreaElement>("#composer-input");
  assert.ok(field, "composer field exists");
  field.value = text;
  field.dispatchEvent(new window.Event("input", { bubbles: true }));
  const sendButton = document.querySelector<HTMLButtonElement>('[data-testid="composer-send"]');
  assert.ok(sendButton, "send button exists");
  sendButton.dispatchEvent(new window.Event("click", { bubbles: true }));
}

function renderedMessageTexts(): string[] {
  return [...document.querySelectorAll<HTMLElement>("#transcript article.message")].map(
    (element) => (element.textContent ?? "").trim(),
  );
}

function sidebarSessionCount(): number {
  return document.querySelectorAll<HTMLElement>("#session-list button").length;
}

/** The composer field enables once the load sequence has fully settled. */
async function waitUntilReady(): Promise<void> {
  await waitFor(() => {
    const field = document.querySelector<HTMLTextAreaElement>("#composer-input");
    return field !== null && !field.disabled;
  }, "the composer to accept input");
}

test("a full chat journey: boot, send, stream, persist", async () => {
  const { host, supervisor } = bootJourney();

  // Boot settles: handshake, model catalog, and a fresh untitled session.
  await waitFor(
    () => supervisor.requests.some((request) => request.url.endsWith("/v1/models")),
    "boot to reach the model catalog",
  );
  await waitUntilReady();

  // The user asks, the fake supervisor streams the reply.
  sendUserMessage("Summarize the prologue of Romeo and Juliet.");
  await waitFor(
    () => renderedMessageTexts().some((text) => text.includes("Hello from the fake supervisor.")),
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
  assert.match(savedPayload.messages[1].content, /Hello from the fake supervisor\./);

  // The chat request asked the supervisor for a streaming completion.
  const chatBody = supervisor.chatRequestBody();
  assert.ok(chatBody, "a chat completions request was made");
  assert.equal(chatBody["stream"], true);
  assert.equal(chatBody["model"], "fake-chat-model");
}, { timeout: 30_000 });

test("a second boot restores the persisted transcript from the host", async () => {
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

test("a failed handshake shows the failure notice instead of a transcript", async () => {
  installDom();
  installConfig(SUPERVISOR_BASE_URL);
  const host = new FakeSessionHost(
    (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
    (callId, message) => window.__thintalkBridge?.reject(callId, message),
  );
  host.install();
  const supervisor = new FakeSupervisor({
    supervisorBaseURL: SUPERVISOR_BASE_URL,
    // A supervisor from another channel must be refused by the handshake.
    channel: "stable",
    stateDirectory: "~/.astronomical-dev",
  });
  supervisor.install();
  bootChatClient();

  // A load failure renders as the empty-state notice, not the composer banner
  // (the banner is reserved for failures while streaming).
  await waitFor(() => {
    const notice = document.querySelector<HTMLElement>("#empty-state-notice");
    return (notice?.textContent ?? "").includes("runtime channel");
  }, "the failure notice to appear");
  assert.equal(host.fileCount(), 0, "no session is written when the supervisor is unreachable");
}, { timeout: 30_000 });
