import { makeElement } from "./astro/AstroDom";
import type { ProblemReportClient } from "./ProblemReportClient";

/**
 * Asks, once, "May this Mac be counted?", from the second day of use.
 *
 * The card shows the exact line this Mac would send, with its real values,
 * because a line someone can read is more convincing than the word
 * "anonymous". Either answer is saved and the card never returns; closing
 * the window without answering asks again another day.
 */
export class DailyCountCard {
  private readonly client: ProblemReportClient;
  private readonly cardElement: HTMLElement;

  public constructor(init: { client: ProblemReportClient; document: Document; placeAfter: HTMLElement }) {
    this.client = init.client;
    this.cardElement = makeElement(init.document, "aside", "problem-report");
    this.cardElement.dataset.testid = "daily-count-question";
    this.cardElement.hidden = true;
    init.placeAfter.after(this.cardElement);
  }

  public get element(): HTMLElement {
    return this.cardElement;
  }

  /** Shows the question when the engine says it is time; hides it otherwise. */
  public async refresh(): Promise<void> {
    const dailyCount = await this.client.fetchDailyCount();
    if (dailyCount === null || !dailyCount.shouldAsk) {
      this.hide();
      return;
    }
    if (this.cardElement.hidden) {
      this.show(dailyCount.line);
    }
  }

  private show(line: string): void {
    const document = this.cardElement.ownerDocument;
    const heading = makeElement(document, "p", "problem-report__heading", "May this Mac be counted?");
    const explanation = makeElement(document, "p", "problem-report__explanation", "Astronomical can't see who uses it. Once a day, this Mac would send exactly this line:");
    const lineText = makeElement(document, "pre", "problem-report__lines", line);
    lineText.dataset.testid = "daily-count-line";
    const promise = makeElement(document, "p", "problem-report__promise", "No name, no account, no chats, and nothing that tells one day's line from another's. You can stop any time in Settings.");
    const outcome = makeElement(document, "p", "problem-report__outcome");
    outcome.dataset.testid = "daily-count-outcome";
    const declineButton = this.button("No thanks", "daily-count-decline", false);
    const agreeButton = this.button("Count this Mac", "daily-count-agree", true);
    declineButton.addEventListener("click", () => void this.answer(false, outcome));
    agreeButton.addEventListener("click", () => void this.answer(true, outcome));
    const actions = makeElement(document, "div", "problem-report__actions");
    actions.append(declineButton, agreeButton);
    this.cardElement.replaceChildren(heading, explanation, lineText, promise, actions, outcome);
    this.cardElement.hidden = false;
  }

  private button(label: string, testID: string, isPrimary: boolean): HTMLButtonElement {
    const button = makeElement(this.cardElement.ownerDocument, "button", isPrimary ? "problem-report__send" : "problem-report__decline", label);
    button.type = "button";
    button.dataset.testid = testID;
    return button;
  }

  /** Saves the answer and hides the card; an answer that could not be saved
   * keeps the card, since an unsaved "No thanks" would be asked again. */
  private async answer(isCounted: boolean, outcome: HTMLElement): Promise<void> {
    if (await this.client.saveDailyCountAnswer(isCounted)) {
      this.hide();
      return;
    }
    outcome.textContent = "That couldn't be saved. Try again.";
  }

  private hide(): void {
    this.cardElement.hidden = true;
    this.cardElement.replaceChildren();
  }
}
