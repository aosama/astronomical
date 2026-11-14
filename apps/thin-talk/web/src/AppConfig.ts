import { LETTER_PAPER, paperSizeFromHost } from "./export/PaperSize";
import type { PaperSize } from "./export/PaperSize";
import { validTitlebarInset } from "./WindowChrome";

/**
 * The runtime configuration the Swift host injects before the bundle runs: where
 * the daemon listens and which instance this window belongs to. The page
 * cannot discover these itself (the daemon port is channel-specific), so the
 * host is the single source of truth.
 */
export class AppConfig {
  public readonly daemonBaseURL: string;
  public readonly expectedChannel: string;
  public readonly expectedStateDirectory: string;
  /** Points of native title bar and toolbar the host draws over the top of the
   * page. Content starts below it, because the native strip takes the clicks
   * that land there (it is what drags the window). Zero for a plain title bar. */
  public readonly titlebarInset: number;
  /** Whether the host restarts the local engine when asked. Only a host that
   * says so gets a restart button, so the button always does something. */
  public readonly canRestartEngine: boolean;
  /** The paper the Mac prints on, which an export is laid out for. Letter
   * when the host does not say. */
  public readonly printedPaper: PaperSize;

  public constructor(init: {
    daemonBaseURL: string;
    expectedChannel: string;
    expectedStateDirectory: string;
    titlebarInset?: number;
    canRestartEngine?: boolean;
    printedPaper?: PaperSize;
  }) {
    this.daemonBaseURL = init.daemonBaseURL;
    this.expectedChannel = init.expectedChannel;
    this.expectedStateDirectory = init.expectedStateDirectory;
    this.titlebarInset = init.titlebarInset ?? 0;
    this.canRestartEngine = init.canRestartEngine === true;
    this.printedPaper = init.printedPaper ?? LETTER_PAPER;
  }

  /** The Models page (the Observatory), served by the same engine the window
   * talks to. */
  public get modelsPageURL(): string {
    return `${this.daemonBaseURL}/download-models`;
  }

  /** What this window calls itself. Mirrors ThinTalkApplicationIdentity.windowTitle,
   * so the sidebar agrees with the title bar about which instance this is. */
  public get applicationTitle(): string {
    return this.expectedChannel === "development" ? "Astronomical Development" : "Astronomical";
  }

  /** Reads and validates the host-injected config, throwing when the host did
   * not inject one or the endpoint is not the loopback daemon. */
  public static fromWindow(config: unknown): AppConfig {
    if (typeof config !== "object" || config === null) {
      throw new Error("the host did not inject a runtime configuration");
    }
    const record = config as Record<string, unknown>;
    const daemonBaseURL = record["daemonBaseURL"];
    const expectedChannel = record["expectedChannel"];
    const expectedStateDirectory = record["expectedStateDirectory"];
    if (typeof daemonBaseURL !== "string" || !AppConfig.isLoopbackHTTP(daemonBaseURL)) {
      throw new Error("the injected daemon endpoint is not a loopback HTTP URL");
    }
    if (typeof expectedChannel !== "string" || expectedChannel.length === 0) {
      throw new Error("the injected configuration has no runtime channel");
    }
    if (typeof expectedStateDirectory !== "string" || expectedStateDirectory.length === 0) {
      throw new Error("the injected configuration has no state directory");
    }
    const titlebarInset = AppConfig.titlebarInsetFrom(record["titlebarInset"]);
    const canRestartEngine = record["canRestartEngine"] === true;
    const printedPaper = paperSizeFromHost(record["printedPaper"]);
    return new AppConfig({ daemonBaseURL, expectedChannel, expectedStateDirectory, titlebarInset, canRestartEngine, printedPaper });
  }

  /** An absent inset means a plain title bar (zero); a present one must pass
   * the same check the host's later inset updates do. */
  private static titlebarInsetFrom(candidate: unknown): number {
    if (candidate === undefined) {
      return 0;
    }
    const titlebarInset = validTitlebarInset(candidate);
    if (titlebarInset === null) {
      throw new Error("the injected title bar inset is not a valid number of points");
    }
    return titlebarInset;
  }

  private static isLoopbackHTTP(candidate: string): boolean {
    try {
      const parsed = new URL(candidate);
      return parsed.protocol === "http:" && (parsed.hostname === "127.0.0.1" || parsed.hostname === "localhost");
    } catch {
      return false;
    }
  }
}
