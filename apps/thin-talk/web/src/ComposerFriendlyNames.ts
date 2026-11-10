/**
 * Plain words for the composer's brain control, used while Astro talks to a
 * younger person (the Explorer words live in astro/AstroPresentation.ts). The
 * panel behind the control keeps the real names, and the control's tooltip
 * still names the model, so a grown-up can always tell which one is in use.
 */
export interface ComposerFriendlyNames {
  readonly brain: string;
}
