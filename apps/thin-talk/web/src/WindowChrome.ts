const TITLEBAR_INSET_VARIABLE = "--titlebar-inset";
const NATIVE_TITLEBAR_CLASS = "app--native-titlebar";
const SIDEBAR_HIDDEN_CLASS = "app--sidebar-hidden";

/** The largest inset a real title bar plus toolbar reaches, at the smallest
 * page zoom, stays far below this; anything beyond it is a host bug that would
 * hide the conversation, so it is refused instead of applied. */
const MAXIMUM_TITLEBAR_INSET_PIXELS = 200;

/** A usable inset in CSS pixels, or null for anything that is not a finite,
 * non-negative number within the bound above. */
export function validTitlebarInset(candidate: unknown): number | null {
  if (
    typeof candidate !== "number" ||
    !Number.isFinite(candidate) ||
    candidate < 0 ||
    candidate > MAXIMUM_TITLEBAR_INSET_PIXELS
  ) {
    return null;
  }
  return candidate;
}

/**
 * The page's side of the native window frame: where the host's title bar and
 * toolbar sit over the page, and whether the session sidebar is showing.
 *
 * The inset is in CSS pixels. Page zoom scales CSS pixels, so the host re-sends
 * the inset divided by the zoom whenever the zoom changes; the strip itself is
 * native and never zooms.
 *
 * Under a native toolbar the page's own "Thin Talk / New chat" header repeats
 * what the toolbar and menu bar already offer, so the stylesheet hides it when
 * the inset is non-zero; a plain-title-bar host keeps the header.
 */
export class WindowChrome {
  private readonly appRoot: HTMLElement;

  public constructor(appRoot: HTMLElement) {
    this.appRoot = appRoot;
  }

  public applyTitlebarInset(titlebarInsetPixels: number): void {
    this.appRoot.style.setProperty(TITLEBAR_INSET_VARIABLE, `${titlebarInsetPixels}px`);
    this.appRoot.classList.toggle(NATIVE_TITLEBAR_CLASS, titlebarInsetPixels > 0);
  }

  public toggleSidebar(): void {
    this.appRoot.classList.toggle(SIDEBAR_HIDDEN_CLASS);
  }

  public isSidebarHidden(): boolean {
    return this.appRoot.classList.contains(SIDEBAR_HIDDEN_CLASS);
  }
}
