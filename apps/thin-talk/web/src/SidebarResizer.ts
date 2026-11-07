import { SessionPreferences } from "./SessionPreferences";

const SIDEBAR_WIDTH_VARIABLE = "--sidebar-width";

/** The conversation keeps at least this much room beside the sidebar, however
 * far the edge is dragged, so a narrow window never squeezes the transcript
 * away. */
const MINIMUM_CONVERSATION_WIDTH_PIXELS = 360;

/**
 * The drag handle on the sidebar's right edge, and the one place the
 * sidebar's width is written to the page.
 *
 * While the pointer is down the width follows it directly, so the drag stays
 * smooth without a controller round trip per pixel, and a saved width
 * published meanwhile (a streaming reply publishes on every chunk) does not
 * pull the edge back. The final width is reported once, on release, and only
 * when the edge actually moved. A double-click puts the sidebar back to its
 * default width, as in Finder and Mail.
 */
export class SidebarResizer {
  private readonly sidebarElement: HTMLElement;
  private readonly onResizeEnd: (widthPixels: number | null) => void;
  private readonly resizeHandle: HTMLElement;
  private draggedWidthPixels: number | null = null;
  private hasMovedDuringDrag = false;

  public constructor(init: { sidebarElement: HTMLElement; onResizeEnd: (widthPixels: number | null) => void }) {
    this.sidebarElement = init.sidebarElement;
    this.onResizeEnd = init.onResizeEnd;
    this.resizeHandle = document.createElement("div");
    this.resizeHandle.className = "sidebar__resize-handle";
    this.resizeHandle.dataset.testid = "sidebar-resize-handle";
    this.resizeHandle.setAttribute("role", "separator");
    this.resizeHandle.setAttribute("aria-orientation", "vertical");
    this.resizeHandle.setAttribute("aria-label", "Resize the conversation list");
    this.sidebarElement.appendChild(this.resizeHandle);
    this.resizeHandle.addEventListener("pointerdown", (event) => this.beginDrag(event));
    this.resizeHandle.addEventListener("pointermove", (event) => this.followPointer(event));
    this.resizeHandle.addEventListener("pointerup", () => this.endDrag());
    this.resizeHandle.addEventListener("pointercancel", () => this.endDrag());
    this.resizeHandle.addEventListener("lostpointercapture", () => this.endDrag());
    this.resizeHandle.addEventListener("dblclick", () => this.onResizeEnd(null));
  }

  /** Shows the saved width, or the stylesheet default for null. Ignored while
   * a drag is under way: the pointer owns the width until it is released. */
  public applyWidth(widthPixels: number | null): void {
    if (this.draggedWidthPixels != null) {
      return;
    }
    if (widthPixels != null) {
      document.documentElement.style.setProperty(SIDEBAR_WIDTH_VARIABLE, `${widthPixels}px`);
    } else {
      document.documentElement.style.removeProperty(SIDEBAR_WIDTH_VARIABLE);
    }
  }

  private beginDrag(event: PointerEvent): void {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    // Captured so the drag keeps following the pointer when it moves faster
    // than the thin handle and leaves it.
    this.resizeHandle.setPointerCapture?.(event.pointerId);
    this.draggedWidthPixels = this.sidebarElement.getBoundingClientRect().width;
    this.hasMovedDuringDrag = false;
    document.documentElement.classList.add("is-resizing-sidebar");
  }

  private followPointer(event: PointerEvent): void {
    if (this.draggedWidthPixels == null) {
      return;
    }
    const sidebarLeftEdge = this.sidebarElement.getBoundingClientRect().left;
    const widestForWindow = window.innerWidth - MINIMUM_CONVERSATION_WIDTH_PIXELS;
    const requestedWidth = Math.min(event.clientX - sidebarLeftEdge, widestForWindow);
    this.draggedWidthPixels = SessionPreferences.clampSidebarWidth(requestedWidth);
    this.hasMovedDuringDrag = true;
    document.documentElement.style.setProperty(SIDEBAR_WIDTH_VARIABLE, `${this.draggedWidthPixels}px`);
  }

  private endDrag(): void {
    const finalWidthPixels = this.draggedWidthPixels;
    if (finalWidthPixels == null) {
      return;
    }
    this.draggedWidthPixels = null;
    document.documentElement.classList.remove("is-resizing-sidebar");
    // A click on the edge (or the first half of a double-click) moves
    // nothing, so it must not pin the current width as a saved choice.
    if (this.hasMovedDuringDrag) {
      this.onResizeEnd(finalWidthPixels);
    }
  }
}
