/**
 * A discovered chat-capable Library model the user can send messages to. The
 * supervisor models an OpenAI-compatible model list; vision models advertise an
 * "image" input modality, so attachment can be offered only when the model
 * supports it.
 */
export class ChatModel {
  public readonly id: string;
  public readonly name: string;
  public readonly inputModalities: readonly string[];

  public constructor(init: { id: string; name: string; inputModalities: readonly string[] }) {
    this.id = init.id;
    this.name = init.name;
    this.inputModalities = init.inputModalities;
  }

  /** Whether the loaded model can accept images. */
  public get supportsVision(): boolean {
    return this.inputModalities.includes("image");
  }
}
