/**
 * Which mermaid theme a diagram is drawn with.
 *
 * Mermaid bakes its theme into each SVG it renders, so the theme cannot come
 * from the stylesheet's tokens the way every other colour does; it has to be
 * chosen in script from the Mac's appearance. A diagram already on screen keeps
 * the colours it was drawn with if the appearance changes afterwards.
 */
export type MermaidThemeName = "dark" | "default";

/** The part of `window` this module reads; absent in hosts without media queries. */
export interface AppearanceQueryHost {
  matchMedia?: (mediaQuery: string) => { matches: boolean };
}

/** A host that cannot answer is treated as light, the stylesheet's own default. */
export function prefersDarkAppearance(appearanceQueryHost: AppearanceQueryHost): boolean {
  if (typeof appearanceQueryHost.matchMedia !== "function") {
    return false;
  }
  return appearanceQueryHost.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function mermaidThemeFor(appearanceQueryHost: AppearanceQueryHost): MermaidThemeName {
  return prefersDarkAppearance(appearanceQueryHost) ? "dark" : "default";
}
