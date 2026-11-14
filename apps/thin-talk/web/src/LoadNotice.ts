import { ChatFailure } from "./ChatFailure";
import { ChatFailureKind } from "./ChatFailureKind";
import { OTHER_COPY_RUNNING_NOTICE } from "./OtherCopyAnsweredError";

/** How the chat surface discovered its models at startup. */
export type LoadPhase = "idle" | "loading" | "ready" | "empty" | "failed";

/** What the window says while it waits for an engine that has not answered. */
export const WAITING_FOR_ENGINE_NOTICE = "Waiting for the engine to start…";

/** An engine that does not answer is being waited for, and the window says
 * so in its own words: the browser's text for a refused connection ("Load
 * failed") reads as a broken app. Any other failure is the engine's own
 * explanation and is shown as it is. */
export function describeLoadFailure(error: unknown): string {
  if (error instanceof ChatFailure && error.kind === ChatFailureKind.WORKER_UNAVAILABLE) {
    return WAITING_FOR_ENGINE_NOTICE;
  }
  return error instanceof Error ? error.message : String(error);
}

/** Whether the load notice tells the person to do something before the
 * window can work. Only another copy holding the engine's address does: Send
 * does nothing until they quit it, so the reason must not sit in small
 * print. A wait that ends by itself is not theirs to act on. */
export function loadNoticeAsksForAction(loadPhase: LoadPhase, loadFailureMessage: string): boolean {
  return loadPhase === "failed" && loadFailureMessage === OTHER_COPY_RUNNING_NOTICE;
}

/** The line the empty conversation shows for each load phase, or null once
 * a chat model is ready. */
export function loadNoticeText(loadPhase: LoadPhase, loadFailureMessage: string): string | null {
  switch (loadPhase) {
    case "empty":
      return "No chat model is available yet. Add one from the Models page in the Observatory.";
    case "idle":
    case "loading":
      return "Looking for a chat model on this Mac.";
    case "failed":
      return loadFailureMessage;
    case "ready":
      return null;
  }
}
