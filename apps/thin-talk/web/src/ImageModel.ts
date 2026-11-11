/**
 * An installed model that makes images. Like every model it has one name, the
 * id image requests send, which is also what the picker shows.
 */
export class ImageModel {
  /** The id image requests send. */
  public readonly id: string;
  /** Whether the Models page recommends it for Image Generation on its Get started page. */
  public readonly isImageGenerationPick: boolean;

  public constructor(init: { id: string; isImageGenerationPick?: boolean }) {
    this.id = init.id;
    this.isImageGenerationPick = init.isImageGenerationPick === true;
  }
}
