/**
 * A discovered chat-capable Library model the user can send messages to. The
 * daemon models an OpenAI-compatible model list; vision models advertise an
 * "image" input modality, so attachment can be offered only when the model
 * supports it.
 */
export class ChatModel {
  public readonly id: string;
  public readonly name: string;
  public readonly inputModalities: readonly string[];
  /** Whether the daemon says this model can call tools. */
  public readonly supportsToolCalls: boolean;
  /** The model's context window in tokens, or null when unknown. */
  public readonly contextWindowTokens: number | null;

  public constructor(init: {
    id: string;
    name: string;
    inputModalities: readonly string[];
    supportsToolCalls?: boolean;
    contextWindowTokens?: number | null;
  }) {
    this.id = init.id;
    this.name = init.name;
    this.inputModalities = init.inputModalities;
    this.supportsToolCalls = init.supportsToolCalls === true;
    const configuredContextWindow: number | null | undefined = init.contextWindowTokens;
    this.contextWindowTokens = configuredContextWindow != null ? configuredContextWindow : null;
  }

  /** Whether the loaded model can accept images. */
  public get supportsVision(): boolean {
    return this.inputModalities.includes("image");
  }
}
