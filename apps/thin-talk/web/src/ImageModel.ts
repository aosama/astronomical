/**
 * An installed model that makes images, named the way the Library names it so
 * the person sees "FLUX.2 Klein 4B 4-bit" rather than a repository id.
 */
export class ImageModel {
  /** The id image requests send. */
  public readonly id: string;
  /** The Library's display name, or the id when the Library does not list it. */
  public readonly name: string;
  /** Whether the Library recommends it for Pictures on its Get started page. */
  public readonly isPicturesPick: boolean;

  public constructor(init: { id: string; name: string; isPicturesPick?: boolean }) {
    this.id = init.id;
    this.name = init.name;
    this.isPicturesPick = init.isPicturesPick === true;
  }
}
