import type { ConversationTool, ToolOutput, ToolParametersSchema } from "./ConversationTool";
import { DaemonClient } from "./DaemonClient";
import type { ImageGenerationRequestBody } from "./ImageGenerationRequestBody";
import { SessionBridge } from "./SessionBridge";

/** Image shapes the model may ask for, as width and height in pixels. All are
 * within the daemon's 1024-pixel limit and multiples of 16. */
const IMAGE_SIZES_BY_SHAPE: Record<string, { width: number; height: number }> = {
  square: { width: 1024, height: 1024 },
  landscape: { width: 1024, height: 768 },
  portrait: { width: 768, height: 1024 },
};

/** Loading an image model and generating one image can take well over the
 * default tool timeout, especially on the first image after a chat. */
const IMAGE_GENERATION_TIMEOUT_MILLISECONDS = 5 * 60_000;

/**
 * Makes an image with the first installed image model and shows it in the
 * answer. The image is kept by the app (so the conversation still shows it
 * after a relaunch); the model is told it was shown but cannot see it, so it
 * does not invent a description. Running it only calls the local daemon, so
 * it runs without asking.
 */
export class GenerateImageTool implements ConversationTool {
  public readonly name = "generate_image";
  public readonly label = "Image generation";
  public readonly description =
    "Creates an image from a text description and shows it to the user. Use it when the user asks you to draw, paint or create a picture.";
  public readonly parameters: ToolParametersSchema = {
    type: "object",
    properties: {
      prompt: { type: "string", description: "A detailed description of the image to create." },
      shape: { type: "string", enum: Object.keys(IMAGE_SIZES_BY_SHAPE), description: "The image shape; square when omitted." },
    },
    required: ["prompt"],
  };
  public readonly timeoutMilliseconds = IMAGE_GENERATION_TIMEOUT_MILLISECONDS;
  private readonly client: DaemonClient;
  private readonly bridge: SessionBridge;

  public constructor(client: DaemonClient, bridge: SessionBridge) {
    this.client = client;
    this.bridge = bridge;
  }

  public async execute(toolArguments: Record<string, unknown>, signal: AbortSignal): Promise<ToolOutput> {
    const prompt = typeof toolArguments["prompt"] === "string" ? toolArguments["prompt"].trim() : "";
    if (prompt.length === 0) {
      throw new Error("The image needs a description.");
    }
    const imageModelIDs = await this.client.fetchImageModelIDs(signal);
    const imageModelID = imageModelIDs[0];
    if (imageModelID == null) {
      throw new Error("No image model is installed. The user can add one from the Library.");
    }
    const requestedShape = typeof toolArguments["shape"] === "string" ? toolArguments["shape"] : "square";
    const imageSize = IMAGE_SIZES_BY_SHAPE[requestedShape] ?? IMAGE_SIZES_BY_SHAPE["square"]!;
    const requestBody: ImageGenerationRequestBody = {
      model: imageModelID,
      prompt,
      width: imageSize.width,
      height: imageSize.height,
      response_format: "b64_json",
      n: 1,
    };
    const encodedImage = await this.client.generateImage(requestBody, signal);
    const generatedImageName = await this.bridge.storeGeneratedImage(encodedImage);
    return {
      text: `Generated a ${imageSize.width}×${imageSize.height} image with ${imageModelID} and showed it to the user. You cannot see the image, so do not describe what it shows.`,
      generatedImageName,
    };
  }
}
