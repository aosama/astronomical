import type { DaemonClient } from "../DaemonClient";
import { OtherCopyAnsweredError } from "../OtherCopyAnsweredError";
import { isLibraryModelReady, parseLibraryStarterPicks } from "../LibraryStarterPicks";
import type { LibraryStarterPicks } from "../LibraryStarterPicks";
import { ModelDownload, ModelDownloadPhase } from "./ModelDownload";
import type { ModelDownloadDependencies, ModelDownloadProgress } from "./ModelDownload";

type ProgressListener = (progress: ModelDownloadProgress) => void;

/** Runs a callback after a delay; the app uses the page's timer. */
type DelayScheduler = (callback: () => void, delayMilliseconds: number) => void;

function scheduleWithPageTimer(callback: () => void, delayMilliseconds: number): void {
  setTimeout(callback, delayMilliseconds);
}

/** One model's download and everyone currently watching it. */
interface DownloadFollower {
  readonly listeners: Set<ProgressListener>;
  lastProgress: ModelDownloadProgress | null;
}

const FINISHED_PHASES: ReadonlySet<ModelDownloadPhase> = new Set([ModelDownloadPhase.READY, ModelDownloadPhase.FAILED, ModelDownloadPhase.BUSY]);

/**
 * Everything Astro asks the daemon: whether a chat model or an image model
 * is installed, which models the Library recommends, and the downloads that install
 * them. Answers are best effort: an unreachable daemon reads as "unknown",
 * never as an error the person has to deal with.
 *
 * Each model has at most one download being followed, however many screens
 * show it: a screen that goes away only stops listening, the daemon keeps
 * downloading, and `onModelInstalled` still fires when the model lands, so
 * the chat window learns about a new chat model even after the welcome closed.
 */
export class AstroLibrary {
  private readonly client: DaemonClient;
  private readonly schedule: DelayScheduler;
  private readonly followersByModelID = new Map<string, DownloadFollower>();
  private readonly installListeners = new Set<() => void>();
  private otherCopyAnswered = false;

  /** `schedule` is what waits between download polls. Tests pass one that
   * does not wait, so a download's whole life runs in milliseconds. */
  public constructor(client: DaemonClient, schedule: DelayScheduler = scheduleWithPageTimer) {
    this.client = client;
    this.schedule = schedule;
  }

  /** True with any chat model installed, false with none, null when this
   * app's own daemon cannot be asked (then Astro skips the download screen,
   * because a download could not start either).
   *
   * The engine's identity is checked first: with two installs on one Mac the
   * other copy's engine answers on the same address, and reading its models
   * as this app's would skip a newcomer's download or send it to the other
   * copy. */
  public async hasChatModel(): Promise<boolean | null> {
    try {
      await this.client.handshake();
      this.otherCopyAnswered = false;
      return (await this.client.fetchModels()).length > 0;
    } catch (error) {
      this.otherCopyAnswered = error instanceof OtherCopyAnsweredError;
      return null;
    }
  }

  /** Whether the last ask was answered by another install's engine, so a
   * waiting screen can say that and not that the engine failed to start. */
  public get otherCopyIsRunning(): boolean {
    return this.otherCopyAnswered;
  }

  public async hasImageModel(): Promise<boolean> {
    try {
      return (await this.client.fetchImageModelIDs()).length > 0;
    } catch {
      return false;
    }
  }

  public async starterPicks(): Promise<LibraryStarterPicks | null> {
    try {
      return parseLibraryStarterPicks(await this.client.fetchLibraryCatalogText());
    } catch {
      return null;
    }
  }

  /** Whether this model's download is already under way, so a screen opened
   * mid-download can show it without the person pressing anything. */
  public async isDownloading(huggingfaceID: string): Promise<boolean> {
    try {
      const status = await this.client.fetchLibraryDownloadStatus();
      return status.huggingfaceID === huggingfaceID && status.state !== "idle" && status.state !== "failed";
    } catch {
      return false;
    }
  }

  /** Calls `listener` each time a download Astro follows lands. */
  public onModelInstalled(listener: () => void): void {
    this.installListeners.add(listener);
  }

  /**
   * Starts (or joins) a model's download and reports its progress to
   * `onProgress` until it lands. Returns the call that stops this listener;
   * a listener joining late gets the latest progress at once.
   */
  public followDownload(huggingfaceID: string, onProgress: ProgressListener): () => void {
    let follower = this.followersByModelID.get(huggingfaceID);
    if (follower === undefined) {
      follower = this.startFollowing(huggingfaceID);
    }
    follower.listeners.add(onProgress);
    if (follower.lastProgress !== null) {
      onProgress(follower.lastProgress);
    }
    const followedBy = follower;
    return () => {
      followedBy.listeners.delete(onProgress);
    };
  }

  private startFollowing(huggingfaceID: string): DownloadFollower {
    const follower: DownloadFollower = { listeners: new Set(), lastProgress: null };
    this.followersByModelID.set(huggingfaceID, follower);
    const modelDownload = new ModelDownload(huggingfaceID, this.downloadDependencies(), (progress) => this.deliver(huggingfaceID, follower, progress));
    void modelDownload.start();
    return follower;
  }

  private deliver(huggingfaceID: string, follower: DownloadFollower, progress: ModelDownloadProgress): void {
    follower.lastProgress = progress;
    for (const listener of follower.listeners) {
      listener(progress);
    }
    if (FINISHED_PHASES.has(progress.phase)) {
      // A finished follower is forgotten, so "Try again" starts a fresh one.
      this.followersByModelID.delete(huggingfaceID);
    }
    if (progress.phase === ModelDownloadPhase.READY) {
      for (const installListener of this.installListeners) {
        installListener();
      }
    }
  }

  private downloadDependencies(): ModelDownloadDependencies {
    return {
      startDownload: (huggingfaceID) => this.client.startLibraryDownload(huggingfaceID),
      resumeDownload: () => this.client.resumeLibraryDownload(),
      readStatus: () => this.client.fetchLibraryDownloadStatus(),
      isInstalled: async (huggingfaceID) => isLibraryModelReady(await this.client.fetchLibraryCatalogText(), huggingfaceID),
      nowMilliseconds: () => Date.now(),
      schedule: this.schedule,
    };
  }
}
