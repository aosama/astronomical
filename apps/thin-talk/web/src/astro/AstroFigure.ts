import { astroGalaxyHtml } from "./AstroGalaxy";

/**
 * Astro as drawn on screen: the galaxy (AstroGalaxy.ts), and the small
 * reactions around it (a hop when something happens, its arms turning while
 * it charges up a download).
 *
 * It is also where a skill card is dropped onto Astro, so it keeps the size
 * the character had. With Reduce Motion on, Astro never hops or turns.
 */
export class AstroFigure {
  public readonly element: HTMLElement;

  public constructor(document: Document, extraClassName: string) {
    this.element = document.createElement("div");
    this.element.className = `astro-figure ${extraClassName}`;
    this.element.setAttribute("aria-hidden", "true");
    const body = document.createElement("div");
    body.className = "astro-figure__body";
    body.innerHTML = astroGalaxyHtml();
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
