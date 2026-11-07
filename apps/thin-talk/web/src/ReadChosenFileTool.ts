import type { ConversationTool, ToolCallContext, ToolParametersSchema } from "./ConversationTool";
import { runHostToolUse } from "./HostToolUseRunner";
import { SessionBridge } from "./SessionBridge";

/** The tool waits while the user answers the sheet and picks a file, which
 * takes as long as they take; this bounds a use nobody ever answers. */
const USER_CHOICE_TOOL_TIMEOUT_MILLISECONDS = 10 * 60_000;

/**
 * Reads one text file the user chooses. The host shows the approval sheet and
 * then an open panel as one step, so the model never names a path and only
 * sees a file the user handed over.
 */
export class ReadChosenFileTool implements ConversationTool {
  public readonly name = "read_file";
  public readonly label = "Read a file";
  public readonly description =
    "Asks the user to choose a text file on this Mac and returns its contents. Use it when the user wants you to read, summarise or answer questions about one of their files.";
  public readonly parameters: ToolParametersSchema = { type: "object", properties: {} };
  public readonly timeoutMilliseconds = USER_CHOICE_TOOL_TIMEOUT_MILLISECONDS;
  public readonly needsUserAttention = true;
  private readonly bridge: SessionBridge;

  public constructor(bridge: SessionBridge) {
    this.bridge = bridge;
  }

  public async execute(_toolArguments: Record<string, unknown>, signal: AbortSignal, context: ToolCallContext): Promise<string> {
    const hostToolUse = await runHostToolUse(this.bridge, this.name, this.label, context, signal);
    if (hostToolUse.outcome !== "chosenText") {
      throw new Error("The host did not return a text file.");
    }
    return `Contents of ${hostToolUse.fileName}:\n\n${hostToolUse.text}`;
  }
}
