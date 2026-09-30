/**
 * The runtime configuration the Swift host injects before the bundle runs: where
 * the supervisor listens and which instance this window belongs to. The page
 * cannot discover these itself (the supervisor port is channel-specific), so the
 * host is the single source of truth.
 */
export class AppConfig {
  public readonly supervisorBaseURL: string;
  public readonly expectedChannel: string;
  public readonly expectedStateDirectory: string;

  public constructor(init: {
    supervisorBaseURL: string;
    expectedChannel: string;
    expectedStateDirectory: string;
  }) {
    this.supervisorBaseURL = init.supervisorBaseURL;
    this.expectedChannel = init.expectedChannel;
    this.expectedStateDirectory = init.expectedStateDirectory;
  }

  /** Reads and validates the host-injected config, throwing when the host did
   * not inject one or the endpoint is not the loopback supervisor. */
  public static fromWindow(config: unknown): AppConfig {
    if (typeof config !== "object" || config === null) {
      throw new Error("the host did not inject a runtime configuration");
    }
    const record = config as Record<string, unknown>;
    const supervisorBaseURL = record["supervisorBaseURL"];
    const expectedChannel = record["expectedChannel"];
    const expectedStateDirectory = record["expectedStateDirectory"];
    if (typeof supervisorBaseURL !== "string" || !AppConfig.isLoopbackHTTP(supervisorBaseURL)) {
      throw new Error("the injected supervisor endpoint is not a loopback HTTP URL");
    }
    if (typeof expectedChannel !== "string" || expectedChannel.length === 0) {
      throw new Error("the injected configuration has no runtime channel");
    }
    if (typeof expectedStateDirectory !== "string" || expectedStateDirectory.length === 0) {
      throw new Error("the injected configuration has no state directory");
    }
    return new AppConfig({ supervisorBaseURL, expectedChannel, expectedStateDirectory });
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
