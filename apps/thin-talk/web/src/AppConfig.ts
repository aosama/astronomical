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

  public constructor(init: {
    daemonBaseURL: string;
    expectedChannel: string;
    expectedStateDirectory: string;
  }) {
    this.daemonBaseURL = init.daemonBaseURL;
    this.expectedChannel = init.expectedChannel;
    this.expectedStateDirectory = init.expectedStateDirectory;
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
    return new AppConfig({ daemonBaseURL, expectedChannel, expectedStateDirectory });
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
