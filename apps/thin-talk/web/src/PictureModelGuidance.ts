/**
 * Where a person can get a picture model, in the words that match what is on
 * their screen: the Pictures card while Astro is shown, the Observatory's
 * Download Models page in classic Thin Talk. The drawing tool reads it when no picture
 * model is installed, so the chat model points to a place that exists instead
 * of inventing one, and the step's result reads sensibly to a person too.
 */
export class PictureModelGuidance {
  private isAstroShown = false;

  public setAstroShown(isAstroShown: boolean): void {
    this.isAstroShown = isAstroShown;
  }

  public noPictureModelResult(): string {
    const whereToGetOne = this.isAstroShown
      ? "The person can download one by choosing the Pictures card in a new chat."
      : "The person can download one from the Models page in the Observatory.";
    return `No picture model is installed on this Mac, so nothing was drawn. ${whereToGetOne} Suggest only that.`;
  }
}
