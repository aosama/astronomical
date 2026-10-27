import { installConfig, installDom, waitFor } from "./dom";
import { FakeDaemon } from "./fakeDaemon";
import { FakeSessionHost } from "./fakeHost";
import { bootChatClient } from "../../src/boot";

export const DAEMON_BASE_URL = "http://127.0.0.1:6733";

export interface Journey {
  host: FakeSessionHost;
  daemon: FakeDaemon;
}

/** Boots the real client wiring against the fake host and fake daemon.
 * Pass a previous journey to keep its host files across a fresh page —
 * the same persistence a relaunch of the real app would see. Pass models
 * to control the daemon's model catalog, or seedPreferences to write durable
 * preferences before the client reads them. */
export function bootJourney(
  previous?: Journey,
  options?: { models?: unknown[]; seedPreferences?: unknown },
): Journey {
  installDom();
  installConfig(DAEMON_BASE_URL);
  const previousHost: FakeSessionHost | undefined = previous?.host;
  const host = previousHost != null
    ? previousHost
    : new FakeSessionHost(
      (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
      (callId, message) => window.__thintalkBridge?.reject(callId, message),
    );
  host.install();
  const seededPreferences: unknown = options?.seedPreferences;
  if (seededPreferences !== undefined) {
    host.seedPreferences(seededPreferences);
  }
  const previousDaemon: FakeDaemon | undefined = previous?.daemon;
  const daemon = previousDaemon != null
    ? previousDaemon
    : new FakeDaemon({
      daemonBaseURL: DAEMON_BASE_URL,
      channel: "development",
      stateDirectory: "~/.astronomical-dev",
      models: options?.models,
    });
  daemon.install();
  bootChatClient();
  return { host, daemon };
}

/** Types into the composer field and presses send, the way a user would. */
export function sendUserMessage(text: string): void {
  const field = document.querySelector<HTMLTextAreaElement>("#composer-input");
  if (!field) {
    throw new Error("composer field exists");
  }
  field.value = text;
  field.dispatchEvent(new window.Event("input", { bubbles: true }));
  const sendButton = document.querySelector<HTMLButtonElement>('[data-testid="composer-send"]');
  if (!sendButton) {
    throw new Error("send button exists");
  }
  sendButton.dispatchEvent(new window.Event("click", { bubbles: true }));
}

/** The composer field enables once the load sequence has fully settled. */
export async function waitUntilReady(): Promise<void> {
  await waitFor(() => {
    const field = document.querySelector<HTMLTextAreaElement>("#composer-input");
    return field !== null && !field.disabled;
  }, "the composer to accept input");
}
