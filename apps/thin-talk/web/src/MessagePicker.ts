import type { TranscriptView } from "./TranscriptView";

/** The class on the app root while messages are being picked; the stylesheet
 * hides the composer and shows the pick bar under it. */
export const PICKING_CLASS = "app--picking";

/** The elements the picker draws in, all from index.html. */
export interface MessagePickerElements {
  readonly appRoot: HTMLElement;
  readonly transcriptElement: HTMLElement;
  readonly pickBar: HTMLElement;
  readonly pickCount: HTMLElement;
  readonly doneButton: HTMLButtonElement;
  readonly cancelButton: HTMLButtonElement;
}

/**
 * Lets the person tick messages in the conversation itself, for exporting
 * part of a chat.
 *
 * The marks are attributes on the messages' own articles (`data-picked`,
 * `aria-checked`): the transcript replaces an article's children when it
 * redraws, but never its attributes, so a mark survives a redraw.
 */
export class MessagePicker {
  private readonly elements: MessagePickerElements;
  private readonly transcript: TranscriptView;
  private pickableMessageIDs: readonly string[] = [];
  private readonly pickedMessageIDs = new Set<string>();
  private finishPicking: ((pickedMessageIDs: readonly string[] | null) => void) | null = null;

  public constructor(elements: MessagePickerElements, transcript: TranscriptView) {
    this.elements = elements;
    this.transcript = transcript;
    // Capture runs before the transcript's own handler, so while picking a
    // click on a message ticks it instead of pressing Copy or opening a link.
    elements.transcriptElement.addEventListener("click", (event) => this.handleClick(event), true);
    elements.transcriptElement.addEventListener("keydown", (event) => this.handleKey(event), true);
    // Esc is heard on the whole page: when picking starts the keyboard focus
    // is wherever the export sheet left it, not on a message, and Esc pressed
    // there did nothing.
    document.addEventListener("keydown", (event) => this.handleEscape(event), true);
    elements.doneButton.addEventListener("click", () => this.finish([...this.pickedInConversationOrder()]));
    elements.cancelButton.addEventListener("click", () => this.finish(null));
  }

  public get isPicking(): boolean {
    return this.finishPicking !== null;
  }

  /** Starts picking among `pickableMessageIDs`, with `alreadyPickedIDs`
   * ticked; resolves with the ticked ids, or null when the person cancels. */
  public pick(pickableMessageIDs: readonly string[], alreadyPickedIDs: readonly string[]): Promise<readonly string[] | null> {
    this.finish(null);
    this.pickableMessageIDs = pickableMessageIDs;
    this.pickedMessageIDs.clear();
    alreadyPickedIDs.filter((messageID) => pickableMessageIDs.includes(messageID)).forEach((messageID) => this.pickedMessageIDs.add(messageID));
    this.elements.appRoot.classList.add(PICKING_CLASS);
    this.elements.pickBar.hidden = false;
    for (const messageID of pickableMessageIDs) {
      const article = this.transcript.messageArticle(messageID);
      article?.setAttribute("role", "checkbox");
      article?.setAttribute("tabindex", "0");
      this.markArticle(messageID);
    }
    this.renderCount();
    return new Promise((resolve) => {
      this.finishPicking = resolve;
    });
  }

  private pickedInConversationOrder(): string[] {
    return this.pickableMessageIDs.filter((messageID) => this.pickedMessageIDs.has(messageID));
  }

  private finish(pickedMessageIDs: readonly string[] | null): void {
    const finishPicking = this.finishPicking;
    if (finishPicking === null) {
      return;
    }
    this.finishPicking = null;
    for (const messageID of this.pickableMessageIDs) {
      const article = this.transcript.messageArticle(messageID);
      for (const attributeName of ["role", "tabindex", "aria-checked", "data-picked"]) {
        article?.removeAttribute(attributeName);
      }
    }
    this.elements.appRoot.classList.remove(PICKING_CLASS);
    this.elements.pickBar.hidden = true;
    finishPicking(pickedMessageIDs);
  }

  private pickableMessageIDAt(target: EventTarget | null): string | null {
    const article = target instanceof Element ? target.closest<HTMLElement>("[data-message-id]") : null;
    const messageID = article?.dataset.messageId ?? null;
    return messageID !== null && this.pickableMessageIDs.includes(messageID) ? messageID : null;
  }

  private handleClick(event: MouseEvent): void {
    if (!this.isPicking) {
      return;
    }
    // Nothing in the conversation acts while picking: a stray click on a
    // link or a disclosure would otherwise leave the picker behind.
    event.preventDefault();
    event.stopPropagation();
    const messageID = this.pickableMessageIDAt(event.target);
    if (messageID !== null) {
      this.togglePicked(messageID);
    }
  }

  /** Esc cancels, as on a sheet. */
  private handleEscape(event: KeyboardEvent): void {
    if (this.isPicking && event.key === "Escape") {
      event.preventDefault();
      this.finish(null);
    }
  }

  /** Space or Return ticks the focused message. */
  private handleKey(event: KeyboardEvent): void {
    if (!this.isPicking) {
      return;
    }
    const messageID = event.key === " " || event.key === "Enter" ? this.pickableMessageIDAt(event.target) : null;
    if (messageID !== null) {
      event.preventDefault();
      this.togglePicked(messageID);
    }
  }

  private togglePicked(messageID: string): void {
    if (this.pickedMessageIDs.has(messageID)) {
      this.pickedMessageIDs.delete(messageID);
    } else {
      this.pickedMessageIDs.add(messageID);
    }
    this.markArticle(messageID);
    this.renderCount();
  }

  private markArticle(messageID: string): void {
    const article = this.transcript.messageArticle(messageID);
    const isPicked = this.pickedMessageIDs.has(messageID);
    article?.setAttribute("aria-checked", String(isPicked));
    article?.toggleAttribute("data-picked", isPicked);
  }

  private renderCount(): void {
    const pickedCount = this.pickedMessageIDs.size;
    this.elements.pickCount.textContent = pickedCount === 0 ? "Click the messages to export" : pickedCount === 1 ? "1 message picked" : `${pickedCount} messages picked`;
    this.elements.doneButton.disabled = pickedCount === 0;
  }
}
