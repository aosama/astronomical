/**
 * The decoded `/v1/status` document used for the readiness handshake. Both keys
 * are optional in the type because the supervisor may answer with a partial
 * document; the handshake treats a missing key as a mismatch.
 */
export class SupervisorStatus {
  public readonly channel: string | null;
  public readonly stateDirectory: string | null;
  public readonly status: string | null;

  public constructor(init: {
    channel: string | null;
    stateDirectory: string | null;
    status: string | null;
  }) {
    this.channel = init.channel;
    this.stateDirectory = init.stateDirectory;
    this.status = init.status;
  }

  public static fromPlain(plain: unknown): SupervisorStatus {
    if (typeof plain !== "object" || plain === null) {
      return new SupervisorStatus({ channel: null, stateDirectory: null, status: null });
    }
    const record = plain as Record<string, unknown>;
    const application = record["application"];
    const applicationRecord =
      typeof application === "object" && application !== null ? (application as Record<string, unknown>) : {};
    const channel = typeof applicationRecord["channel"] === "string" ? applicationRecord["channel"] : null;
    const stateDirectory =
      typeof applicationRecord["state_directory"] === "string" ? applicationRecord["state_directory"] : null;
    const status = typeof record["status"] === "string" ? record["status"] : null;
    return new SupervisorStatus({ channel, stateDirectory, status });
  }
}
