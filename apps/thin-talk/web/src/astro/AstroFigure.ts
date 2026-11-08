import { lookFrameForPointer } from "./LookDirection";

/** How many pre-rendered turns of Astro the page has (astro-look-0 to -8.webp).
 * They are renders of Astro's 3D model, one camera and nine turns, made with
 * Blender from the mascot's own art and scripts, which are kept with the
 * mascot rather than in this repository; the WebP files are the shipped form. */
export const LOOK_FRAME_COUNT = 9;
const RESTING_LOOK_FRAME = Math.floor(LOOK_FRAME_COUNT / 2);
const STREAK_COUNT = 4;

/** Relative to the page, so the same path works in the app's private scheme
 * and in the browser preview. */
export function lookFrameURL(lookFrame: number): string {
  return `astro/astro-look-${lookFrame}.webp`;
}

/**
 * Astro as drawn on screen: a pre-rendered turn of the 3D model that faces the
 * pointer, the pastel space trail beneath it, and the small reactions (a hop
 * when something happens, a glow while charging).
 *
 * Every frame is preloaded once, so turning never flickers while an image
 * loads. With Reduce Motion on, Astro still turns (a still change of pose) but
 * never hops or bobs.
 */
export class AstroFigure {
  public readonly element: HTMLElement;
  private readonly body: HTMLImageElement;
  private shownLookFrame = RESTING_LOOK_FRAME;

  public constructor(document: Document, extraClassName: string) {
    this.element = document.createElement("div");
    this.element.className = `astro-figure ${extraClassName}`;
    this.element.setAttribute("aria-hidden", "true");
    this.element.append(AstroFigure.makeTrail(document));
    this.body = document.createElement("img");
    this.body.className = "astro-figure__body";
    this.body.alt = "";
    this.body.draggable = false;
    this.body.src = lookFrameURL(RESTING_LOOK_FRAME);
    this.element.append(this.body);
    AstroFigure.preloadLookFrames(document);
  }

  /** Turns Astro toward a pointer `pointerAcrossFraction` of the way across. */
  public lookToward(pointerAcrossFraction: number): void {
    const lookFrame = lookFrameForPointer(pointerAcrossFraction, LOOK_FRAME_COUNT);
    if (lookFrame !== this.shownLookFrame) {
      this.shownLookFrame = lookFrame;
      this.body.src = lookFrameURL(lookFrame);
    }
  }

  public lookAhead(): void {
    this.lookToward(0.5);
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

  /** Makes Astro watch the pointer anywhere inside `watchedArea`. */
  public followPointerWithin(watchedArea: HTMLElement): void {
    watchedArea.addEventListener("pointermove", (pointerEvent) => {
      // A hidden Astro (Focus switch, a conversation under way) has nothing to turn.
      if (this.element.closest("[hidden]") !== null) {
        return;
      }
      const areaBounds = watchedArea.getBoundingClientRect();
      if (areaBounds.width > 0) {
        this.lookToward((pointerEvent.clientX - areaBounds.left) / areaBounds.width);
      }
    });
    watchedArea.addEventListener("pointerleave", () => this.lookAhead());
  }

  public containsPoint(clientX: number, clientY: number): boolean {
    const figureBounds = this.element.getBoundingClientRect();
    return clientX >= figureBounds.left && clientX <= figureBounds.right && clientY >= figureBounds.top && clientY <= figureBounds.bottom;
  }

  private static makeTrail(document: Document): HTMLElement {
    const trail = document.createElement("div");
    trail.className = "astro-figure__trail";
    for (let streakNumber = 0; streakNumber < STREAK_COUNT; streakNumber += 1) {
      trail.append(document.createElement("span"));
    }
    return trail;
  }

  private static preloadLookFrames(document: Document): void {
    for (let lookFrame = 0; lookFrame < LOOK_FRAME_COUNT; lookFrame += 1) {
      const preloadedFrame = document.createElement("img");
      preloadedFrame.src = lookFrameURL(lookFrame);
    }
  }
}
