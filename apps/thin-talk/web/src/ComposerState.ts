import { ThinkingEffort } from "./ThinkingEffort";
import { ChatModel } from "./ChatModel";

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
  public readonly failure: ComposerFailure | null;

  public constructor(init: {
    isStreaming: boolean;
    acceptsInput: boolean;
    isReady: boolean;
    effort: ThinkingEffort;
    models: readonly ChatModel[];
    selectedModelID: string | null;
    failure: ComposerFailure | null;
  }) {
    this.isStreaming = init.isStreaming;
    this.acceptsInput = init.acceptsInput;
    this.isReady = init.isReady;
    this.effort = init.effort;
    this.models = init.models;
    this.selectedModelID = init.selectedModelID;
    this.failure = init.failure;
  }
}
