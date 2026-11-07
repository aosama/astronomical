import type { ConversationTool, ToolParametersSchema } from "./ConversationTool";

/**
 * Tells the model the local date, time and time zone. Models have no clock,
 * so without this a question like "what day is it" gets a guess from the
 * training data. It reads nothing personal and changes nothing, so it runs
 * without asking.
 */
export class CurrentTimeTool implements ConversationTool {
  public readonly name = "get_current_time";
  public readonly label = "Current time";
  public readonly description =
    "Returns the current local date, time and time zone on this Mac. Use it whenever the answer depends on today's date or the time.";
  // Extra arguments are accepted and ignored. A small model sometimes sends
  // the schema itself as the arguments; rejecting that only produces a failed
  // step and a retry for a tool that reads no arguments anyway.
  public readonly parameters: ToolParametersSchema = { type: "object", properties: {} };
  private readonly readClock: () => Date;

  /** The clock is injectable so tests can pin the time. */
  public constructor(readClock: () => Date = () => new Date()) {
    this.readClock = readClock;
  }

  public async execute(): Promise<string> {
    const now = this.readClock();
    const timeZoneName = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const readableDate = now.toLocaleString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    });
    return `${readableDate} (time zone ${timeZoneName}; ISO 8601 ${now.toISOString()})`;
  }
}
