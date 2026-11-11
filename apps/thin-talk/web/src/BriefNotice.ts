/** Long enough to read a file name, short enough not to linger. */
export const BRIEF_NOTICE_MILLISECONDS = 4000;

/**
 * Says one short thing for a moment above the message field: that a file
 * was saved (the save panel closes on its own, so nothing else shows the
 * export worked), or why a command had nothing to act on.
 */
export class BriefNotice {
  private readonly noticeElement: HTMLElement;
  private hideTimer: ReturnType<typeof setTimeout> | null = null;

  public constructor(noticeElement: HTMLElement) {
    this.noticeElement = noticeElement;
  }

  public show(noticeText: string): void {
    this.noticeElement.textContent = noticeText;
    this.noticeElement.hidden = false;
    if (this.hideTimer !== null) {
      clearTimeout(this.hideTimer);
    }
    this.hideTimer = setTimeout(() => {
      this.noticeElement.hidden = true;
      this.hideTimer = null;
    }, BRIEF_NOTICE_MILLISECONDS);
  }
}
