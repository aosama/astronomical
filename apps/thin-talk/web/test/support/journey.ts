import { installConfig, installDom, waitFor } from "./dom";
import { FakeDaemon } from "./fakeDaemon";
import { FakeSessionHost } from "./fakeHost";
import { bootChatClient } from "../../src/boot";

export const DAEMON_BASE_URL = "http://127.0.0.1:6733";

/** A person who already finished Astro's get-started journey with Astro shown. */
export const MET_ASTRO_PROFILE = { nickname: "Maya", hasFinishedFirstFlight: true, isAstroShown: true };

/** Someone who met Astro and then hid it: classic Thin Talk, the default for
 * journeys about chat itself, so they test the plain look. */
export const CLASSIC_PROFILE = { ...MET_ASTRO_PROFILE, isAstroShown: false };

export interface Journey {
  host: FakeSessionHost;
  daemon: FakeDaemon;
}

/** Boots the real client wiring against the fake host and fake daemon.
 * Pass a previous journey to keep its host files across a fresh page —
 * the same persistence a relaunch of the real app would see. Pass models
 * to control the daemon's model catalog, or seedPreferences to write durable
 * preferences before the client reads them. */
/** The folder the test window expects its own engine to report. */
export const OWN_STATE_DIRECTORY = "~/.astronomical-dev";

export function bootJourney(
  previous?: Journey,
  options?: {
    models?: unknown[];
    seedPreferences?: unknown;
    titlebarInset?: number;
    chatReplySequence?: string[][];
    libraryCatalog?: unknown;
    libraryDownloadSequence?: unknown[];
    libraryDownloadRefusal?: { status: number; code: string };
    installsWhenDownloadFinishes?: { huggingfaceID: string; model: unknown };
    /** Astro's saved profile; omitted means someone who met Astro and hid
     * it, so journeys about chat neither click through the welcome nor see
     * Astro's look. */
    astroProfile?: unknown;
    /** Makes the speech endpoint refuse, as with no voice model installed. */
    speechRefusal?: { status: number; code: string };
    /** Whether the fake host says it can restart the engine. */
    canRestartEngine?: boolean;
    /** Where the engine says it keeps its files; another folder than the
     * window expects plays a second install holding the address. */
    engineStateDirectory?: string;
  },
): Journey {
  installDom();
  installConfig(DAEMON_BASE_URL, options?.titlebarInset, options?.canRestartEngine);
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
  if (previousHost == null) {
    host.astroProfile = options?.astroProfile !== undefined ? options.astroProfile : CLASSIC_PROFILE;
  }
  const previousDaemon: FakeDaemon | undefined = previous?.daemon;
  const daemon = previousDaemon != null
    ? previousDaemon
    : new FakeDaemon({
      daemonBaseURL: DAEMON_BASE_URL,
      channel: "development",
      stateDirectory: options?.engineStateDirectory ?? OWN_STATE_DIRECTORY,
      models: options?.models,
      chatReplySequence: options?.chatReplySequence,
      libraryCatalog: options?.libraryCatalog,
      libraryDownloadSequence: options?.libraryDownloadSequence,
      libraryDownloadRefusal: options?.libraryDownloadRefusal,
      installsWhenDownloadFinishes: options?.installsWhenDownloadFinishes,
      speechRefusal: options?.speechRefusal,
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
