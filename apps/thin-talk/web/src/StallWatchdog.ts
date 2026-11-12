/** How often the watchdog polls for activity. A stall timeout is tens of
 * seconds, so noticing one within a second is prompt enough. */
const POLL_INTERVAL_MILLISECONDS = 1_000;

/**
 * Detects a stalled generation: no stream delta within the timeout. A stall has
 * no backend marker, so the chat surface owns this watchdog. Detection is
 * suspended while paused (a tool is running), and the poller fires `onStall`
 * at most once per run.
 */
export class StallWatchdog {
  private readonly stallTimeoutMilliseconds: number;
  private readonly onStall: () => void;
  private readonly pollIntervalMilliseconds: number;
  private pollTimer: number | null = null;
  private lastActivityAt = 0;
  private hasFired = false;
  private isPaused = false;

  public constructor(stallTimeoutMilliseconds: number, onStall: () => void, pollIntervalMilliseconds: number = POLL_INTERVAL_MILLISECONDS) {
    this.stallTimeoutMilliseconds = stallTimeoutMilliseconds;
    this.onStall = onStall;
    this.pollIntervalMilliseconds = pollIntervalMilliseconds;
  }

  public start(): void {
    this.stop();
    this.hasFired = false;
    this.isPaused = false;
    this.lastActivityAt = Date.now();
    this.pollTimer = window.setInterval(() => this.poll(), this.pollIntervalMilliseconds);
  }

  public stop(): void {
    if (this.pollTimer !== null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  /** Suspends stall detection while a tool runs: a tool sends nothing until
   * it finishes, and its own timeout already bounds how long it may take. */
  public pause(): void {
    this.isPaused = true;
  }

  /** Resumes stall detection with a fresh deadline, extended by
   * `graceMilliseconds`: the request after a tool may first have to load the
   * chat model back (image generation swaps it out), which is slow but not a
   * stall. */
  public resume(graceMilliseconds = 0): void {
    this.isPaused = false;
    this.lastActivityAt = Date.now() + graceMilliseconds;
  }

  /** Records stream activity, pushing the stall deadline out. */
  public activity(): void {
    this.lastActivityAt = Date.now();
  }

  private poll(): void {
    if (this.hasFired || this.isPaused) {
      return;
    }
    const elapsed = Date.now() - this.lastActivityAt;
    if (elapsed >= this.stallTimeoutMilliseconds) {
      this.hasFired = true;
      this.onStall();
    }
  }
}
