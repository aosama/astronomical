/** How often the watchdog polls for activity. */
const POLL_INTERVAL_MILLISECONDS = 1_000;

/**
 * Detects a stalled generation: no stream delta within the timeout. A stall has
 * no backend marker, so the chat surface owns this watchdog. The poller fires
 * `onStall` at most once per run.
 */
export class StallWatchdog {
  private readonly stallTimeoutMilliseconds: number;
  private readonly onStall: () => void;
  private pollTimer: number | null = null;
  private lastActivityAt = 0;
  private hasFired = false;

  public constructor(stallTimeoutMilliseconds: number, onStall: () => void) {
    this.stallTimeoutMilliseconds = stallTimeoutMilliseconds;
    this.onStall = onStall;
  }

  public start(): void {
    this.stop();
    this.hasFired = false;
    this.lastActivityAt = Date.now();
    this.pollTimer = window.setInterval(() => this.poll(), POLL_INTERVAL_MILLISECONDS);
  }

  public stop(): void {
    if (this.pollTimer !== null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  /** Records stream activity, pushing the stall deadline out. */
  public activity(): void {
    this.lastActivityAt = Date.now();
  }

  private poll(): void {
    if (this.hasFired) {
      return;
    }
    const elapsed = Date.now() - this.lastActivityAt;
    if (elapsed >= this.stallTimeoutMilliseconds) {
      this.hasFired = true;
      this.onStall();
    }
  }
}
