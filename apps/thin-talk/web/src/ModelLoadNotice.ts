import { ChatMessageState } from "./ChatMessageState";
import { ChatRole } from "./ChatRole";
import type { ChatSurfaceState } from "./ChatSurfaceState";

/** On the transcript while the engine is loading a model for the answer
 * being waited on; the stylesheet then swaps "thinking…" for the loading
 * words on that answer. */
export const MODEL_LOADING_CLASS = "transcript--model-loading";

/** Half a second: a warm load of a large model measured 2.5 seconds, and at
 * a slower pace the words showed for a second of it or not at all. The
 * status is a few kilobytes from this Mac, asked only while an answer's
 * first words are awaited. */
export const STATUS_POLL_MILLISECONDS = 500;

/** Whether the newest answer has been asked for and has shown nothing yet:
 * the only stretch in which a model can still be loading for it. */
export function isWaitingForFirstWords(state: ChatSurfaceState): boolean {
  const messages = state.transcript.messages;
  const newestMessage = messages[messages.length - 1];
  return (
    newestMessage !== undefined &&
    newestMessage.role === ChatRole.ASSISTANT &&
    newestMessage.state === ChatMessageState.STREAMING &&
    newestMessage.markdown.length === 0 &&
    newestMessage.reasoning.length === 0 &&
    newestMessage.toolSteps.length === 0
  );
}

/**
 * Says that a model is loading while someone waits for an answer's first
 * words.
 *
 * A large model takes tens of seconds to load, and the engine unloads idle
 * models, so the wait recurs. Without this the answer says "thinking…" the
 * whole time, which reads as a stuck app. The engine already reports a load
 * in its status; this asks for it only during that wait.
 */
export class ModelLoadNotice {
  private readonly isModelLoading: () => Promise<boolean>;
  private readonly transcriptElement: HTMLElement;
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  public constructor(init: { isModelLoading: () => Promise<boolean>; transcriptElement: HTMLElement }) {
    this.isModelLoading = init.isModelLoading;
    this.transcriptElement = init.transcriptElement;
  }

  /** Starts asking while an answer's first words are awaited, and stops
   * and clears the notice the moment they arrive or the turn ends. */
  public setWaitingForFirstWords(isWaiting: boolean): void {
    if (isWaiting && this.pollTimer === null) {
      this.pollTimer = setInterval(() => void this.refresh(), STATUS_POLL_MILLISECONDS);
      void this.refresh();
    } else if (!isWaiting && this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
      this.transcriptElement.classList.remove(MODEL_LOADING_CLASS);
    }
  }

  /** A status that cannot be read is treated as no load: the notice is a
   * courtesy, and the chat's own failure handling reports a dead engine. */
  private async refresh(): Promise<void> {
    const isLoading = await this.isModelLoading().catch(() => false);
    if (this.pollTimer !== null) {
      this.transcriptElement.classList.toggle(MODEL_LOADING_CLASS, isLoading);
    }
  }
}
