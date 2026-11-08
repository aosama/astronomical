/**
 * How Astro talks and how big the controls are. An 8-year-old and a 17-year-old
 * use the same app, but not the same voice: Explorer is warmer, chattier and
 * larger, Pilot is quicker and quieter. Ages are guidance, never asked for.
 */
export enum AstroStyle {
  EXPLORER = "explorer",
  PILOT = "pilot",
}

/** Reads a stored style; anything unknown falls back to Explorer, the gentler
 * of the two, so a damaged profile never makes Astro curt with a young child. */
export function parseAstroStyle(storedStyle: unknown): AstroStyle {
  return storedStyle === AstroStyle.PILOT ? AstroStyle.PILOT : AstroStyle.EXPLORER;
}
