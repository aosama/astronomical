import type { DaemonClient } from "./DaemonClient";

/**
 * Looking at a generated image for a chat model that cannot take images: an
 * installed vision model describes what the image really shows, and its
 * account goes into the image tool's result. Kept apart from the image tool,
 * which makes and stores images; this only reviews one.
 */
export class GeneratedImageReview {
  private readonly client: DaemonClient;

  public constructor(client: DaemonClient) {
    this.client = client;
  }

  /** What `visionModelID` sees in the new image, or why nobody could look. A
   * failed look must not lose the image the user already has on screen, so
   * only a stop or timeout propagates; any other failure becomes a note. */
  public async accountForTextOnlyModel(prompt: string, encodedImage: string, visionModelID: string | null, signal: AbortSignal): Promise<string> {
    if (visionModelID == null) {
      return "No model on this Mac can look at images, so you cannot see it; do not describe what it shows.";
    }
    try {
      const imageDescription = await this.client.describeImage(
        { modelID: visionModelID, question: generatedImageQuestion(prompt), mimeType: imageMimeType(encodedImage), base64: encodedImage },
        signal,
      );
      return `You cannot take images, so ${visionModelID} looked at it for you and reports what it actually shows:\n\n`
        + `${imageDescription}\n\nTrust this report over the prompt when you talk about the image.`;
    } catch (lookFailure) {
      if (signal.aborted) {
        throw lookFailure;
      }
      const failureReason = lookFailure instanceof Error ? lookFailure.message : String(lookFailure);
      return `Looking at the image failed (${failureReason}), so you cannot see it; do not describe what it shows.`;
    }
  }
}

/** The question the vision model answers about a generated image: what is
 * really there, and where it falls short of what was asked for. */
export function generatedImageQuestion(prompt: string): string {
  return "This image was just generated from the prompt below. Describe what it actually shows, concretely: "
    + "the subjects, composition, colours and any text. Then list any visible defects (extra or missing limbs or fingers, "
    + "distorted faces, garbled lettering) and any ways it differs from the prompt.\n\nPrompt: " + prompt;
}

/** The image's type from its first bytes. The daemon returns PNG today; JPEG
 * is recognised too, since the app's image store accepts both. */
export function imageMimeType(encodedImage: string): string {
  return encodedImage.startsWith("/9j/") ? "image/jpeg" : "image/png";
}
