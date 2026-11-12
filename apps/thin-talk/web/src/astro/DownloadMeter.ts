import { makeElement } from "./AstroDom";
import { describeGigabytes, describeTimeLeft } from "./DownloadTimeEstimate";
import { ModelDownloadPhase } from "./ModelDownload";
import type { ModelDownloadProgress } from "./ModelDownload";

/**
 * A progress bar with an honest size and time underneath: "2.4 of
 * 6.0 GB" on the left, "about 4 minutes left" on the right. Before any bytes
 * move it shows the one-time size, so the person knows what they agreed to.
 */
export class DownloadMeter {
  public readonly element: HTMLElement;
  private readonly fill: HTMLElement;
  private readonly sizeLabel: HTMLElement;
  private readonly timeLabel: HTMLElement;
  private readonly approximateSizeBytes: number;

  public constructor(document: Document, idPrefix: string, approximateSizeBytes: number) {
    this.approximateSizeBytes = approximateSizeBytes;
    this.element = makeElement(document, "div", "download-meter");
    const track = makeElement(document, "div", "download-meter__track");
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    this.fill = makeElement(document, "div", "download-meter__fill");
    track.append(this.fill);
    const labels = makeElement(document, "div", "download-meter__labels");
    this.sizeLabel = makeElement(document, "span", "download-meter__size", `${describeGigabytes(approximateSizeBytes)}, one time`);
    this.sizeLabel.dataset.testid = `${idPrefix}-download-size`;
    this.timeLabel = makeElement(document, "span", "download-meter__time", "");
    this.timeLabel.dataset.testid = `${idPrefix}-download-time`;
    labels.append(this.sizeLabel, this.timeLabel);
    this.element.append(track, labels);
  }

  public show(progress: ModelDownloadProgress): void {
    const totalBytes = progress.bytesTotal > 0 ? progress.bytesTotal : this.approximateSizeBytes;
    const percentDone = progress.phase === ModelDownloadPhase.READY ? 100 : Math.min(100, (progress.bytesCompleted / Math.max(1, totalBytes)) * 100);
    this.fill.style.width = `${percentDone.toFixed(1)}%`;
    this.fill.parentElement?.setAttribute("aria-valuenow", percentDone.toFixed(0));
    this.sizeLabel.textContent = this.describeSize(progress, totalBytes);
    this.timeLabel.textContent = DownloadMeter.describeTime(progress);
  }

  private describeSize(progress: ModelDownloadProgress, totalBytes: number): string {
    if (progress.phase === ModelDownloadPhase.DOWNLOADING || progress.phase === ModelDownloadPhase.FINISHING) {
      return `${describeGigabytes(progress.bytesCompleted).replace(" GB", "")} of ${describeGigabytes(totalBytes)}`;
    }
    return `${describeGigabytes(this.approximateSizeBytes)}, one time`;
  }

  private static describeTime(progress: ModelDownloadProgress): string {
    switch (progress.phase) {
      case ModelDownloadPhase.STARTING:
        return "getting ready…";
      case ModelDownloadPhase.DOWNLOADING:
        return describeTimeLeft(progress.secondsLeft);
      case ModelDownloadPhase.FINISHING:
        return "almost there: unpacking…";
      case ModelDownloadPhase.READY:
        return "done";
      case ModelDownloadPhase.FAILED:
      case ModelDownloadPhase.BUSY:
        return "";
    }
  }
}
