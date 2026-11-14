import type { ConversationTool, ToolCallContext, ToolParametersSchema } from "./ConversationTool";
import { DaemonClient } from "./DaemonClient";
import { runHostToolUse } from "./HostToolUseRunner";
import { SessionBridge } from "./SessionBridge";

/** The user answers the sheet and picks the image, then a vision model (which
 * may have to load first) answers, so a use needs well over the default. */
const IMAGE_DESCRIPTION_TOOL_TIMEOUT_MILLISECONDS = 10 * 60_000;
const DEFAULT_IMAGE_QUESTION = "Describe this image in detail.";

/**
 * Lets the model look at one image the user chooses. The image goes to the
 * chat model answering the turn when it can see images, otherwise to the first
 * installed vision model, and that model's description becomes the result. A
 * vision model is looked for before the user is asked anything, so they are
 * never shown a sheet for a tool that cannot work.
 */
export class DescribeChosenImageTool implements ConversationTool {
  public readonly name = "describe_image";
  public readonly label = "Describe an image";
  // The wording decides whether the picker opens at all. With "Asks the user
  // to choose an image…" alone, the starter model (Ornith 9B, 4-bit) answered
  // "please choose an image" in words and never called the tool in 8 of 16
  // tries; told when to use it and to call it straight away, it called the
  // tool in 15 of 16. The engine cannot force a tool call, so the description
  // is the only lever.
  public readonly description =
    "Opens a picker so the user can choose an image (PNG or JPEG) on this Mac, and returns a description of it, answering the question if one is given. Use it whenever the user wants you to look at, describe or answer questions about a photo or picture of theirs. Call it straight away: only this tool can open the picker, so never ask the user in words to choose or send the image first.";
  public readonly parameters: ToolParametersSchema = {
    type: "object",
    properties: { question: { type: "string", description: "What to find out about the image; a general description when omitted." } },
  };
  public readonly timeoutMilliseconds = IMAGE_DESCRIPTION_TOOL_TIMEOUT_MILLISECONDS;
  public readonly needsUserAttention = true;
  private readonly client: DaemonClient;
  private readonly bridge: SessionBridge;

  public constructor(client: DaemonClient, bridge: SessionBridge) {
    this.client = client;
    this.bridge = bridge;
  }

  public async execute(toolArguments: Record<string, unknown>, signal: AbortSignal, context: ToolCallContext): Promise<string> {
    const visionModelIDs = await this.client.fetchVisionModelIDs(signal);
    const visionModelID = visionModelIDs.includes(context.modelID) ? context.modelID : visionModelIDs[0];
    if (visionModelID == null) {
      throw new Error("No model on this Mac can look at images. The user can add a vision model from the Models page in the Observatory.");
    }
    const hostToolUse = await runHostToolUse(this.bridge, this.name, this.label, context, signal);
    if (hostToolUse.outcome !== "chosenImage") {
      throw new Error("The host did not return an image.");
    }
    const askedQuestion = typeof toolArguments["question"] === "string" ? toolArguments["question"].trim() : "";
    const description = await this.client.describeImage(
      {
        modelID: visionModelID,
        question: askedQuestion.length > 0 ? askedQuestion : DEFAULT_IMAGE_QUESTION,
        mimeType: hostToolUse.mimeType,
        base64: hostToolUse.base64,
      },
      signal,
    );
    return `What ${visionModelID} sees in ${hostToolUse.fileName}:\n\n${description}`;
  }
}
