import type { ConversationTool, ToolCallContext, ToolOutput, ToolParametersSchema } from "./ConversationTool";
import { DaemonClient } from "./DaemonClient";
import { GeneratedImageReview, imageMimeType } from "./GeneratedImageReview";
import { ImageModelSelection } from "./ImageModelSelection";
import type { ImageGenerationRequestBody } from "./ImageGenerationRequestBody";
import { SessionBridge } from "./SessionBridge";

/** Image shapes the model may ask for, as width and height in pixels. All are
 * within the daemon's 1024-pixel limit and multiples of 16. */
const IMAGE_SIZES_BY_SHAPE: Record<string, { width: number; height: number }> = {
  square: { width: 1024, height: 1024 },
  landscape: { width: 1024, height: 768 },
  portrait: { width: 768, height: 1024 },
};

/** Loading an image model, generating one image, then loading a vision model
 * to look at it can take well over the default tool timeout, especially on
 * the first image after a chat, when both models have to be loaded. */
const IMAGE_GENERATION_TIMEOUT_MILLISECONDS = 10 * 60_000;

/**
 * Makes an image with the image model the person chose (see
 * `ImageModelSelection`) and shows it in the answer. The image is kept by the app (so the conversation still shows it
 * after a relaunch). Running it only calls the local daemon, so it runs
 * without asking.
 *
 * The model then gets to see what it made, so it can notice a defect (an
 * extra limb, garbled lettering) and offer to try again. A chat model that
 * can see images is handed the image itself, attached to the result. A
 * text-only chat model cannot take pixels, so another installed vision model
 * looks for it and its account goes into the result instead. With no vision
 * model at all, the model is told it cannot see the image, so it does not
 * invent one.
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
  private readonly imageReview: GeneratedImageReview;
  private readonly imageModelSelection: ImageModelSelection;

  public constructor(client: DaemonClient, bridge: SessionBridge, imageModelSelection: ImageModelSelection) {
    this.client = client;
    this.bridge = bridge;
    this.imageModelSelection = imageModelSelection;
    this.imageReview = new GeneratedImageReview(client);
  }

  public async execute(toolArguments: Record<string, unknown>, signal: AbortSignal, context: ToolCallContext): Promise<ToolOutput> {
    const prompt = typeof toolArguments["prompt"] === "string" ? toolArguments["prompt"].trim() : "";
    if (prompt.length === 0) {
      throw new Error("The image needs a description.");
    }
    // The installed models are read afresh for each picture, so a model
    // added or removed since Thin Talk opened is honoured.
    const imageModel = this.imageModelSelection.resolve(await this.client.fetchImageModels(signal));
    if (imageModel == null) {
      throw new Error("No image model is installed. The user can add one from the Library.");
    }
    const requestedShape = typeof toolArguments["shape"] === "string" ? toolArguments["shape"] : "square";
    const imageSize = IMAGE_SIZES_BY_SHAPE[requestedShape] ?? IMAGE_SIZES_BY_SHAPE["square"]!;
    const requestBody: ImageGenerationRequestBody = {
      model: imageModel.id,
      prompt,
      width: imageSize.width,
      height: imageSize.height,
      response_format: "b64_json",
      n: 1,
    };
    const encodedImage = await this.client.generateImage(requestBody, signal);
    const generatedImageName = await this.bridge.storeGeneratedImage(encodedImage);
    const generationReport = `Generated a ${imageSize.width}×${imageSize.height} image with ${imageModel.name} and showed it to the user.`;
    const visionModelIDs = await this.visionModelIDsOrNone(signal);
    if (visionModelIDs.includes(context.modelID)) {
      return {
        text: `${generationReport} The image is attached right after this result: look at it yourself before you talk about it, `
          + "and check it against what was asked for.",
        generatedImageName,
        generatedImageModelName: imageModel.name,
        imageForModel: { base64: encodedImage, mimeType: imageMimeType(encodedImage) },
      };
    }
    const imageAccount = await this.imageReview.accountForTextOnlyModel(prompt, encodedImage, visionModelIDs[0] ?? null, signal);
    return { text: `${generationReport}\n\n${imageAccount}`, generatedImageName, generatedImageModelName: imageModel.name };
  }

  /** The installed vision models, or none when the list cannot be read: the
   * image is already stored and on screen, so failing here would lose it from
   * the answer. Only a stop or timeout propagates. */
  private async visionModelIDsOrNone(signal: AbortSignal): Promise<string[]> {
    try {
      return await this.client.fetchVisionModelIDs(signal);
    } catch (listFailure) {
      if (signal.aborted) {
        throw listFailure;
      }
      return [];
    }
  }
}
