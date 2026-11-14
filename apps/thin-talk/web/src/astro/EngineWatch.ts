import type { AstroLibrary } from "./AstroLibrary";

/**
 * How long Astro waits on its engine. Real values suit a person; the tests
 * shorten them so a silent engine can be shown in milliseconds.
 */
export const ENGINE_PATIENCE = {
  /** Between two asks while the engine has not answered yet. */
  askEveryMilliseconds: 1_000,
  /** How long the "starting up" screen stays before it says the engine did not start.
   * The engine normally answers within a few seconds of the app starting. */
  silentAfterMilliseconds: 15_000,
};

type AnswerListener = (hasChatModel: boolean) => void;

/**
 * Asks the engine whether a chat model is installed, and keeps asking while
 * anyone is waiting for an answer.
 *
 * On a first launch the window can open before the engine has finished
 * starting. Reading that silence as "has a chat model" made a new Mac skip
 * the download, with nothing asking again. Every ask is fresh, so a model
 * downloaded since is never missed; asks made at the same moment share one
 * request; and the timer runs only while a listener waits, so a silent
 * engine is not asked forever after nobody needs the answer.
 */
export class EngineWatch {
  private readonly library: AstroLibrary;
  private readonly listeners = new Set<AnswerListener>();
  private askInFlight: Promise<boolean | null> | null = null;
  private askTimer: ReturnType<typeof setTimeout> | null = null;

  public constructor(library: AstroLibrary) {
    this.library = library;
  }

  /** Whether the engine that last answered belongs to another install. */
  public get otherCopyIsRunning(): boolean {
    return this.library.otherCopyIsRunning;
  }

  /** True or false when the engine answers; null when it cannot be reached. */
  public async ask(): Promise<boolean | null> {
    if (this.askInFlight !== null) {
      return this.askInFlight;
    }
    this.clearTimer();
    this.askInFlight = this.library.hasChatModel();
    const answer = await this.askInFlight;
    this.askInFlight = null;
    if (answer === null) {
      this.scheduleAskWhileWaited();
      return null;
    }
    for (const listener of Array.from(this.listeners)) {
      listener(answer);
    }
    this.listeners.clear();
    return answer;
  }

  /** Calls `listener` once, at the engine's next answer, asking on a timer
   * until then. Returns a function that stops waiting. */
  public onceAnswered(listener: AnswerListener): () => void {
    this.listeners.add(listener);
    this.scheduleAskWhileWaited();
    return () => {
      this.listeners.delete(listener);
      if (this.listeners.size === 0) {
        this.clearTimer();
      }
    };
  }

  private scheduleAskWhileWaited(): void {
    if (this.listeners.size === 0 || this.askTimer !== null || this.askInFlight !== null) {
      return;
    }
    this.askTimer = setTimeout(() => {
      this.askTimer = null;
      void this.ask();
    }, ENGINE_PATIENCE.askEveryMilliseconds);
  }

  private clearTimer(): void {
    if (this.askTimer !== null) {
      clearTimeout(this.askTimer);
      this.askTimer = null;
    }
  }
}
