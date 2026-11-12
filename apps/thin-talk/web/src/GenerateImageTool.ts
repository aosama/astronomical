import type { ConversationTool, ToolCallContext, ToolOutput, ToolParametersSchema } from "./ConversationTool";
import { DaemonClient } from "./DaemonClient";
import { GeneratedImageReview, imageMimeType } from "./GeneratedImageReview";
import type { ImageModel } from "./ImageModel";
import { ImageModelDirectory } from "./ImageModelDirectory";
import { ImageModelSelection } from "./ImageModelSelection";
import type { ImageGenerationRequestBody } from "./ImageGenerationRequestBody";
import { SessionBridge } from "./SessionBridge";
import { PictureModelGuidance } from "./PictureModelGuidance";

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

/** How long to wait before reading the model list a second time when the
 * first read lists no image model. A list read just as the daemon refreshes
 * its models (a download landing, a restart) can come back short; one second
 * is enough for that to settle and short enough not to be noticed. */
const IMAGE_MODEL_RECHECK_DELAY_MILLISECONDS = 1_000;


/**
 * Makes an image and shows it in the answer. The model is the one
 * `ImageModelSelection` names: the person's choice while it is installed, else
 * the Models page Image Generation pick, else the first installed image model. The image
 * is kept by the app, so the conversation still shows it after a relaunch.
 * Running it only calls the local daemon, so it runs without asking.
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
  private readonly imageModelDirectory: ImageModelDirectory;
  private readonly imageModelSelection: ImageModelSelection;
  private readonly pictureModelGuidance: PictureModelGuidance;

  public constructor(
    client: DaemonClient,
    bridge: SessionBridge,
    imageModelDirectory: ImageModelDirectory,
    imageModelSelection: ImageModelSelection,
    pictureModelGuidance: PictureModelGuidance = new PictureModelGuidance(),
  ) {
    this.client = client;
    this.bridge = bridge;
    this.imageModelDirectory = imageModelDirectory;
    this.imageModelSelection = imageModelSelection;
    this.pictureModelGuidance = pictureModelGuidance;
    this.imageReview = new GeneratedImageReview(client);
  }

  /**
   * The model to draw with, read afresh for each picture so a model added or
   * removed since Thin Talk opened is honoured; writing the list back to the
   * shared selection updates the composer's model panel to match what draws. An
   * empty list is read once more before giving up, since telling someone
   * "I can't draw" when an image model is installed is the worse mistake.
   */
  private async installedImageModel(signal: AbortSignal): Promise<ImageModel | null> {
    for (let readNumber = 1; readNumber <= 2; readNumber += 1) {
      if (readNumber > 1) {
        await GenerateImageTool.pause(IMAGE_MODEL_RECHECK_DELAY_MILLISECONDS, signal);
      }
      this.imageModelSelection.updateInstalled(await this.imageModelDirectory.load(signal));
      const imageModel = this.imageModelSelection.current();
      if (imageModel != null) {
        return imageModel;
      }
    }
    return null;
  }

  private static pause(delayMilliseconds: number, signal: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      if (signal.aborted) {
        reject(signal.reason);
        return;
      }
      const stopPause = (): void => {
        clearTimeout(pauseTimer);
        reject(signal.reason);
      };
      const pauseTimer = setTimeout(() => {
        signal.removeEventListener("abort", stopPause);
        resolve();
      }, delayMilliseconds);
      signal.addEventListener("abort", stopPause, { once: true });
    });
  }

  public async execute(toolArguments: Record<string, unknown>, signal: AbortSignal, context: ToolCallContext): Promise<ToolOutput> {
    const prompt = typeof toolArguments["prompt"] === "string" ? toolArguments["prompt"].trim() : "";
    if (prompt.length === 0) {
      throw new Error("The image needs a description.");
    }
    const imageModel = await this.installedImageModel(signal);
    if (imageModel == null) {
      // A model left to guess where pictures come from invents settings
      // screens that do not exist, so it is told the one real place.
      throw new Error(this.pictureModelGuidance.noPictureModelResult());
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
    const generationReport = `Generated a ${imageSize.width}×${imageSize.height} image with ${imageModel.id} and showed it to the user.`;
    const visionModelIDs = await this.visionModelIDsOrNone(signal);
    if (visionModelIDs.includes(context.modelID)) {
      return {
        text: `${generationReport} The image is attached right after this result: look at it yourself before you talk about it, `
          + "and check it against what was asked for.",
        generatedImageName,
        generatedImageModelName: imageModel.id,
        imageForModel: { base64: encodedImage, mimeType: imageMimeType(encodedImage) },
      };
    }
    const imageAccount = await this.imageReview.accountForTextOnlyModel(prompt, encodedImage, visionModelIDs[0] ?? null, signal);
    return { text: `${generationReport}\n\n${imageAccount}`, generatedImageName, generatedImageModelName: imageModel.id };
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
