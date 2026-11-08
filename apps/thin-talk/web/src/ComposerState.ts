import { ThinkingEffort } from "./ThinkingEffort";
import { ChatModel } from "./ChatModel";
import { ImageModel } from "./ImageModel";

/** The failure banner's content, or null when no failure is showing. */
export interface ComposerFailure {
  message: string;
  nextAction: string;
}

/**
 * The state the composer, the model pill, the effort pill, and the failure
 * banner render. The controller produces it; the composer only draws it and
 * reports intents back.
 */
export class ComposerState {
  public readonly isStreaming: boolean;
  public readonly acceptsInput: boolean;
  public readonly isReady: boolean;
  public readonly effort: ThinkingEffort;
  public readonly models: readonly ChatModel[];
  public readonly selectedModelID: string | null;
  /** The Tools switch as the user set it. */
  public readonly toolsEnabled: boolean;
  /** Whether the chosen model can call tools at all; the switch is shown
   * disabled when it cannot, so it never promises tools that will not run. */
  public readonly toolsAvailable: boolean;
  public readonly failure: ComposerFailure | null;
  /** The installed image models, in the order the picture pill lists them. */
  public readonly imageModels: readonly ImageModel[];
  /** The image model that draws the next picture, or null with none installed. */
  public readonly selectedImageModelID: string | null;

  public constructor(init: {
    isStreaming: boolean;
    acceptsInput: boolean;
    isReady: boolean;
    effort: ThinkingEffort;
    models: readonly ChatModel[];
    selectedModelID: string | null;
    toolsEnabled?: boolean;
    toolsAvailable?: boolean;
    failure: ComposerFailure | null;
    imageModels?: readonly ImageModel[];
    selectedImageModelID?: string | null;
  }) {
    this.isStreaming = init.isStreaming;
    this.acceptsInput = init.acceptsInput;
    this.isReady = init.isReady;
    this.effort = init.effort;
    this.models = init.models;
    this.selectedModelID = init.selectedModelID;
    this.toolsEnabled = init.toolsEnabled !== false;
    this.toolsAvailable = init.toolsAvailable === true;
    this.failure = init.failure;
    this.imageModels = init.imageModels != null ? init.imageModels : [];
    const configuredImageModelID: string | null | undefined = init.selectedImageModelID;
    this.selectedImageModelID = configuredImageModelID != null ? configuredImageModelID : null;
  }
}
