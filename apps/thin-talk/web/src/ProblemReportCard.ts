import { makeElement } from "./astro/AstroDom";
import type { ProblemReportClient, WaitingProblemReport } from "./ProblemReportClient";

/** The promise shown on every report, word for word what the engine keeps. */
export const NEVER_INCLUDED_TEXT = "Never included: your chats, files, images or folder names.";

/** The longest note the engine keeps; the field stops at the same length. */
export const MAXIMUM_NOTE_CHARACTERS = 500;

/** The optional box for the person's own words. What they type is sent with
 * the report and is the only free text in it. */
export function makeNoteField(document: Document, testID: string): HTMLTextAreaElement {
  const noteField = makeElement(document, "textarea", "problem-report__note");
  noteField.dataset.testid = testID;
  noteField.rows = 2;
  noteField.maxLength = MAXIMUM_NOTE_CHARACTERS;
  noteField.placeholder = "What were you doing? (optional, sent with the report)";
  noteField.setAttribute("aria-label", "What were you doing? Optional, sent with the report.");
  return noteField;
}

/**
 * Shows the newest report waiting for the person, in full, right under the
 * failure banner, with Send report and Don't send.
 *
 * The person decides while looking at exactly what would be sent, which is
 * the whole point: nothing is sent from a promise alone. It appears only
 * when the person chose to be asked; a report sent without asking or a
 * choice of never shows nothing here.
 */
export class ProblemReportCard {
  private readonly client: ProblemReportClient;
  private readonly cardElement: HTMLElement;
  private shownReportNumber: number | null = null;

  public constructor(init: { client: ProblemReportClient; document: Document; placeAfter: HTMLElement }) {
    this.client = init.client;
    this.cardElement = makeElement(init.document, "aside", "problem-report");
    this.cardElement.dataset.testid = "problem-report";
    this.cardElement.hidden = true;
    init.placeAfter.after(this.cardElement);
  }

  public get element(): HTMLElement {
    return this.cardElement;
  }

  /** Asks the engine for waiting reports and shows the newest. A failure to
   * ask hides the card: offering a report is a courtesy, never a blocker. */
  public async refresh(): Promise<void> {
    try {
      const problemReports = await this.client.fetchProblemReports();
      const newestReport = problemReports.waiting[problemReports.waiting.length - 1];
      if (problemReports.choice !== "ask" || newestReport === undefined) {
        this.hide();
        return;
      }
      if (newestReport.reportNumber !== this.shownReportNumber) {
        this.show(newestReport);
      }
    } catch {
      this.hide();
    }
  }

  private show(waitingReport: WaitingProblemReport): void {
    const document = this.cardElement.ownerDocument;
    this.shownReportNumber = waitingReport.reportNumber;
    const heading = makeElement(document, "p", "problem-report__heading", "Send a report about this?");
    const explanation = makeElement(document, "p", "problem-report__explanation", "This is the whole report. Nothing else is sent.");
    const reportText = makeElement(document, "pre", "problem-report__lines", waitingReport.readableLines.join("\n"));
    reportText.dataset.testid = "problem-report-lines";
    const promise = makeElement(document, "p", "problem-report__promise", NEVER_INCLUDED_TEXT);
    const noteField = makeNoteField(document, "problem-report-note");
    const alwaysLabel = makeElement(document, "label", "problem-report__always");
    const alwaysBox = makeElement(document, "input", "");
    alwaysBox.type = "checkbox";
    alwaysBox.dataset.testid = "problem-report-always";
    alwaysLabel.append(alwaysBox, " Send reports like this without asking");
    const outcome = makeElement(document, "p", "problem-report__outcome");
    outcome.dataset.testid = "problem-report-outcome";
    const declineButton = this.button("Don't send", "problem-report-decline", false);
    const sendButton = this.button("Send report", "problem-report-send", true);
    declineButton.addEventListener("click", () => void this.answer(waitingReport.reportNumber, false, alwaysBox.checked, "", outcome));
    sendButton.addEventListener("click", () => void this.answer(waitingReport.reportNumber, true, alwaysBox.checked, noteField.value, outcome));
    const actions = makeElement(document, "div", "problem-report__actions");
    actions.append(declineButton, sendButton);
    this.cardElement.replaceChildren(heading, explanation, reportText, promise, noteField, alwaysLabel, actions, outcome);
    this.cardElement.hidden = false;
  }

  private button(label: string, testID: string, isPrimary: boolean): HTMLButtonElement {
    const button = makeElement(this.cardElement.ownerDocument, "button", isPrimary ? "problem-report__send" : "problem-report__decline", label);
    button.type = "button";
    button.dataset.testid = testID;
    return button;
  }

  /** Sends or declines, saving "always" first when the box is ticked and
   * the person pressed Send. A report that did not go stays on screen with
   * the engine's reason, so the person can try again. */
  private async answer(reportNumber: number, isSending: boolean, isAlwaysChosen: boolean, typedNote: string, outcome: HTMLElement): Promise<void> {
    if (isSending && isAlwaysChosen) {
      await this.client.saveChoice("always").catch(() => undefined);
    }
    const reportAnswer = isSending ? await this.client.sendReport(reportNumber, typedNote) : await this.client.declineReport(reportNumber);
    if (reportAnswer.isDone) {
      this.hide();
      return;
    }
    outcome.textContent = reportAnswer.message;
  }

  private hide(): void {
    this.cardElement.hidden = true;
    this.cardElement.replaceChildren();
    this.shownReportNumber = null;
  }
}
