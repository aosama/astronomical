/** How far back the download speed is averaged. Long enough to smooth over a
 * file boundary or a brief Wi-Fi stall, short enough to notice a real change
 * in speed within a few polls. */
const SPEED_WINDOW_MILLISECONDS = 15_000;
/** Below this much history the speed is a guess, so no time is promised. */
const MINIMUM_SPEED_HISTORY_MILLISECONDS = 2_000;

interface ProgressSample {
  readonly atMilliseconds: number;
  readonly bytesCompleted: number;
}

/**
 * Turns a stream of byte counts into an honest "about N minutes left".
 *
 * The speed is measured over a sliding window rather than since the start, so
 * a resumed download (whose first poll already reports gigabytes done) never
 * shows a wildly fast or slow estimate.
 */
export class DownloadTimeEstimate {
  private readonly samples: ProgressSample[] = [];

  public record(atMilliseconds: number, bytesCompleted: number): void {
    const newestSample = this.samples[this.samples.length - 1];
    if (newestSample !== undefined && bytesCompleted < newestSample.bytesCompleted) {
      // The count went backwards (a retried file): older samples no longer
      // describe this download's speed.
      this.samples.length = 0;
    }
    this.samples.push({ atMilliseconds, bytesCompleted });
    while (this.samples.length > 2 && atMilliseconds - this.samples[0]!.atMilliseconds > SPEED_WINDOW_MILLISECONDS) {
      this.samples.shift();
    }
  }

  /** Seconds left at the recent speed, or null while the speed is unknown. */
  public secondsLeft(bytesTotal: number): number | null {
    const oldestSample = this.samples[0];
    const newestSample = this.samples[this.samples.length - 1];
    if (oldestSample === undefined || newestSample === undefined || bytesTotal <= 0) {
      return null;
    }
    const elapsedMilliseconds = newestSample.atMilliseconds - oldestSample.atMilliseconds;
    const bytesInWindow = newestSample.bytesCompleted - oldestSample.bytesCompleted;
    if (elapsedMilliseconds < MINIMUM_SPEED_HISTORY_MILLISECONDS || bytesInWindow <= 0) {
      return null;
    }
    const bytesPerSecond = bytesInWindow / (elapsedMilliseconds / 1000);
    return Math.max(0, (bytesTotal - newestSample.bytesCompleted) / bytesPerSecond);
  }
}

/** Plain words for a wait: never a bare number of seconds, never "0 min". */
export function describeTimeLeft(secondsLeft: number | null): string {
  if (secondsLeft === null) {
    return "working out the time…";
  }
  if (secondsLeft < 60) {
    return "less than a minute left";
  }
  const minutesLeft = Math.round(secondsLeft / 60);
  if (minutesLeft < 90) {
    return minutesLeft === 1 ? "about 1 minute left" : `about ${minutesLeft} minutes left`;
  }
  const hoursLeft = Math.round(minutesLeft / 60);
  return hoursLeft === 1 ? "about an hour left" : `about ${hoursLeft} hours left`;
}
