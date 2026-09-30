/**
 * One attachment rendered beside a message. `assetURL` is an opaque canvas URL
 * produced by the asset registry, never a file path.
 */
export class TranscriptAttachment {
  public readonly id: string;
  public readonly assetURL: string;
  public readonly label: string;

  public constructor(init: { id: string; assetURL: string; label: string }) {
    this.id = init.id;
    this.assetURL = init.assetURL;
    this.label = init.label;
  }
}
