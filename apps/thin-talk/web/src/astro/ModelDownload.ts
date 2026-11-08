import { LIBRARY_BUSY_ERROR_CODE } from "../LibraryDownloadStatus";
import type { LibraryDownloadStatus } from "../LibraryDownloadStatus";
import { DownloadTimeEstimate } from "./DownloadTimeEstimate";

/** Where one model's download stands, in the terms Astro talks about. */
export enum ModelDownloadPhase {
  STARTING = "starting",
  DOWNLOADING = "downloading",
  /** Checking and installing the files; no bytes move, so no time is shown. */
  FINISHING = "finishing",
  READY = "ready",
  FAILED = "failed",
  /** Another model holds the Library's one download slot. */
  BUSY = "busy",
}

export interface ModelDownloadProgress {
  readonly phase: ModelDownloadPhase;
  readonly bytesCompleted: number;
  readonly bytesTotal: number;
  readonly secondsLeft: number | null;
}

/** What following a download needs from the outside world, injected so tests
 * can run the whole loop with a fake daemon and a fake clock. */
export interface ModelDownloadDependencies {
  startDownload(huggingfaceID: string): Promise<LibraryDownloadStatus>;
  /** Continues the Library's paused download. */
  resumeDownload(): Promise<void>;
  readStatus(): Promise<LibraryDownloadStatus>;
  isInstalled(huggingfaceID: string): Promise<boolean>;
  nowMilliseconds(): number;
  schedule(callback: () => void, delayMilliseconds: number): void;
}

/** The Observatory's Library page polls once a second too; faster would only
 * add load while the bar cannot visibly move any quicker. */
export const DOWNLOAD_POLL_MILLISECONDS = 1_000;
/** After the job disappears, the model may take a moment to show as installed;
 * this many idle polls without it means the download did not land. */
const IDLE_POLLS_BEFORE_GIVING_UP = 10;
/** A download the daemon paused (it parks interrupted downloads as paused
 * when it restarts) is resumed once; if it is still paused this many polls
 * later, the person is offered "Try again" instead of a bar that never moves. */
const PAUSED_POLLS_BEFORE_GIVING_UP = 5;
/** The daemon reports a model that is already installed as a failed job with
 * this code, not as a refusal of the request. */
const MODEL_ALREADY_PRESENT_ERROR_CODE = "model_already_present";
const ACTIVE_STATES: ReadonlySet<string> = new Set(["checking_disk", "fetching_manifest", "downloading", "paused", "resuming", "verifying", "publishing"]);
const FINISHING_STATES: ReadonlySet<string> = new Set(["verifying", "publishing"]);

/**
 * Starts one Library download (or joins it, when that model is already
 * downloading) and reports progress until the model is installed.
 *
 * The daemon owns the download: `stop` only stops listening, so closing Astro's
 * screen never cancels a download a person asked for.
 */
export class ModelDownload {
  private readonly huggingfaceID: string;
  private readonly dependencies: ModelDownloadDependencies;
  private readonly onProgress: (progress: ModelDownloadProgress) => void;
  private readonly estimate = new DownloadTimeEstimate();
  private isListening = false;
  private idlePollCount = 0;
  private pausedPollCount = 0;

  public constructor(huggingfaceID: string, dependencies: ModelDownloadDependencies, onProgress: (progress: ModelDownloadProgress) => void) {
    this.huggingfaceID = huggingfaceID;
    this.dependencies = dependencies;
    this.onProgress = onProgress;
  }

  public async start(): Promise<void> {
    this.isListening = true;
    this.report(ModelDownloadPhase.STARTING, 0, 0);
    const currentStatus = await this.dependencies.readStatus().catch(() => null);
    if (currentStatus !== null && ACTIVE_STATES.has(currentStatus.state)) {
      if (currentStatus.huggingfaceID !== this.huggingfaceID) {
        this.finish(ModelDownloadPhase.BUSY);
        return;
      }
      if (currentStatus.state === "paused") {
        await this.resume();
      }
      this.scheduleNextPoll();
      return;
    }
    await this.requestDownload();
  }

  public stop(): void {
    this.isListening = false;
  }

  private async requestDownload(): Promise<void> {
    try {
      await this.dependencies.startDownload(this.huggingfaceID);
    } catch (refusal) {
      const refusalCode = (refusal as { refusalCode?: unknown }).refusalCode;
      this.finish(refusalCode === LIBRARY_BUSY_ERROR_CODE ? ModelDownloadPhase.BUSY : ModelDownloadPhase.FAILED);
      return;
    }
    this.scheduleNextPoll();
  }

  private scheduleNextPoll(): void {
    if (this.isListening) {
      this.dependencies.schedule(() => void this.poll(), DOWNLOAD_POLL_MILLISECONDS);
    }
  }

  private async poll(): Promise<void> {
    if (!this.isListening) {
      return;
    }
    let status: LibraryDownloadStatus;
    try {
      status = await this.dependencies.readStatus();
    } catch {
      // A missed poll is not a failed download; the next one catches up.
      this.scheduleNextPoll();
      return;
    }
    if (status.huggingfaceID === this.huggingfaceID && status.state !== "idle") {
      await this.followOwnJob(status);
      return;
    }
    await this.checkWhetherInstalled();
  }

  private async followOwnJob(status: LibraryDownloadStatus): Promise<void> {
    if (status.state === "failed") {
      this.finish(status.errorCode === MODEL_ALREADY_PRESENT_ERROR_CODE ? ModelDownloadPhase.READY : ModelDownloadPhase.FAILED);
      return;
    }
    if (status.state === "paused") {
      await this.followPausedJob();
      return;
    }
    this.pausedPollCount = 0;
    this.idlePollCount = 0;
    this.estimate.record(this.dependencies.nowMilliseconds(), status.bytesCompleted);
    const phase = FINISHING_STATES.has(status.state) ? ModelDownloadPhase.FINISHING : ModelDownloadPhase.DOWNLOADING;
    this.report(phase, status.bytesCompleted, status.bytesTotal);
    this.scheduleNextPoll();
  }

  /** Resumes on the first paused poll; a job that stays paused ends as failed,
   * so "Try again" (which resumes once more) replaces a stuck bar. */
  private async followPausedJob(): Promise<void> {
    this.pausedPollCount += 1;
    if (this.pausedPollCount === 1) {
      await this.resume();
    }
    if (this.pausedPollCount >= PAUSED_POLLS_BEFORE_GIVING_UP) {
      this.finish(ModelDownloadPhase.FAILED);
      return;
    }
    this.scheduleNextPoll();
  }

  private async resume(): Promise<void> {
    await this.dependencies.resumeDownload().catch(() => undefined);
  }

  private async checkWhetherInstalled(): Promise<void> {
    const isInstalled = await this.dependencies.isInstalled(this.huggingfaceID).catch(() => false);
    if (isInstalled) {
      this.finish(ModelDownloadPhase.READY);
      return;
    }
    this.idlePollCount += 1;
    if (this.idlePollCount >= IDLE_POLLS_BEFORE_GIVING_UP) {
      this.finish(ModelDownloadPhase.FAILED);
      return;
    }
    this.scheduleNextPoll();
  }

  private finish(phase: ModelDownloadPhase): void {
    this.report(phase, 0, 0);
    this.isListening = false;
  }

  private report(phase: ModelDownloadPhase, bytesCompleted: number, bytesTotal: number): void {
    if (!this.isListening) {
      return;
    }
    const secondsLeft = phase === ModelDownloadPhase.DOWNLOADING ? this.estimate.secondsLeft(bytesTotal) : null;
    this.onProgress({ phase, bytesCompleted, bytesTotal, secondsLeft });
  }
}
