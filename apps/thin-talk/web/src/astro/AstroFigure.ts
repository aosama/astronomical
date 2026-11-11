import { astroConstellationHtml } from "./AstroConstellation";

/**
 * Astro as drawn on screen: the constellation (AstroConstellation.ts), and
 * the small reactions around it (a hop when something happens, its lines
 * drawing in turn while it charges up a download).
 *
 * It is also where a skill card is dropped onto Astro, so it keeps the size
 * the character had. With Reduce Motion on, Astro never hops or draws.
 */
export class AstroFigure {
  public readonly element: HTMLElement;

  public constructor(document: Document, extraClassName: string) {
    this.element = document.createElement("div");
    this.element.className = `astro-figure ${extraClassName}`;
    this.element.setAttribute("aria-hidden", "true");
    const body = document.createElement("div");
    body.className = "astro-figure__body";
    body.innerHTML = astroConstellationHtml();
    this.element.append(body);
  }

  /** Restarting the animation needs a style flush between removing and adding
   * the class; reading offsetWidth forces that flush. */
  public hop(): void {
    this.element.classList.remove("astro-figure--hop");
    void this.element.offsetWidth;
    this.element.classList.add("astro-figure--hop");
  }

  public setCharging(isCharging: boolean): void {
    this.element.classList.toggle("astro-figure--charging", isCharging);
  }

  public containsPoint(clientX: number, clientY: number): boolean {
    const figureBounds = this.element.getBoundingClientRect();
    return clientX >= figureBounds.left && clientX <= figureBounds.right && clientY >= figureBounds.top && clientY <= figureBounds.bottom;
  }
}
