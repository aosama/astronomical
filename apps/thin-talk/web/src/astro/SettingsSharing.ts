import { makeElement } from "./AstroDom";
import type { ProblemReportChoice, ProblemReportClient, ProblemReports, SentProblemReport } from "../ProblemReportClient";
import { NEVER_INCLUDED_TEXT, makeNoteField } from "../ProblemReportCard";
import { ProblemReportsNotOfferedError } from "../ProblemReportClient";
import type { DailyCount, WaitingProblemReport } from "../ProblemReportClient";
import { makeRow, makeSection, rowButton } from "./SettingsRows";

/** The picker's choices, in the words the person reads. */
const PROBLEM_REPORT_CHOICES: ReadonlyArray<{ choice: ProblemReportChoice; label: string }> = [
  { choice: "ask", label: "Ask each time" },
  { choice: "always", label: "Always send" },
  { choice: "never", label: "Never" },
];

/** What each choice means, shown beside the picker so the words always
 * match the stored choice. */
const PROBLEM_REPORT_CHOICE_DETAILS: Readonly<Record<ProblemReportChoice, string>> = {
  ask: "When something fails, you see the whole report and choose whether to send it.",
  always: "When something fails, a report is sent without asking. Each one is listed below.",
  never: "No report is offered or sent.",
};

/**
 * The page's "What leaves this Mac" group: everything Astronomical can send
 * from this Mac, with the one row the person can change, and every report
 * that has already gone, so anyone can check our word.
 */
export class SettingsSharing {
  public readonly element: HTMLElement;
  private readonly document: Document;
  private readonly client: ProblemReportClient;
  private isSentListOpen = false;
  private dailyCount: DailyCount | null = null;
  /** The report the person is writing after pressing Report a problem. */
  private personReport: WaitingProblemReport | null = null;
  private personReportOutcome = "";

  public constructor(document: Document, client: ProblemReportClient) {
    this.document = document;
    this.client = client;
    this.element = makeSection(document, "What leaves this Mac", "settings-sharing");
  }

  public async refresh(): Promise<void> {
    this.dailyCount = await this.client.fetchDailyCount();
    try {
      this.showRows(await this.client.fetchProblemReports());
    } catch (error) {
      this.showRows(null, error instanceof ProblemReportsNotOfferedError);
    }
  }

  /** Draws the rows; `null` when the engine could not be asked, in which
   * case the fixed rows still show and the changeable ones say why not. */
  private showRows(problemReports: ProblemReports | null, isNotOffered = false): void {
    const document = this.document;
    const rows: HTMLElement[] = [
      makeRow(document, "Chats, files and images", "Never", null),
      makeRow(document, "Model downloads", "Only the ones you choose", null),
    ];
    if (problemReports === null && isNotOffered) {
      rows.push(makeRow(document, "Problem reports", "Never. This version doesn't send them.", null), makeRow(document, "Daily count", "Never", null));
      this.element.replaceChildren(this.element.firstElementChild ?? "", ...rows);
      return;
    }
    if (problemReports === null) {
      rows.push(makeRow(document, null, makeElement(document, "span", "settings__row-value--muted", "The engine isn't answering, so these choices can't be shown right now."), null));
      this.element.replaceChildren(this.element.firstElementChild ?? "", ...rows);
      return;
    }
    rows.push(
      makeRow(document, "Problem reports", this.reportDetail(problemReports), this.choicePicker(problemReports.choice)),
      this.dailyCountRow(),
      this.sentRow(problemReports.sent),
    );
    if (this.isSentListOpen) {
      rows.push(this.sentList(problemReports.sent));
    }
    rows.push(this.personReport === null ? this.reportProblemRow() : this.personReportForm(this.personReport));
    this.element.replaceChildren(this.element.firstElementChild ?? "", ...rows);
  }

  private reportDetail(problemReports: ProblemReports): HTMLElement {
    const detailText = !problemReports.isSendingSetUp
      ? "When something fails, you see the whole report. Sending isn't set up in this version."
      : PROBLEM_REPORT_CHOICE_DETAILS[problemReports.choice];
    return makeElement(this.document, "span", "settings__row-value--muted", detailText);
  }

  private choicePicker(currentChoice: ProblemReportChoice): HTMLSelectElement {
    const picker = makeElement(this.document, "select", "settings__picker");
    picker.dataset.testid = "settings-problem-reports";
    picker.setAttribute("aria-label", "Problem reports");
    for (const { choice, label } of PROBLEM_REPORT_CHOICES) {
      const option = makeElement(this.document, "option", "", label);
      option.value = choice;
      option.selected = choice === currentChoice;
      picker.append(option);
    }
    picker.addEventListener("change", () => void this.saveChoice(picker.value as ProblemReportChoice));
    return picker;
  }

  /** Saves the new choice and redraws from the engine, so the page shows
   * what was stored rather than what was picked. */
  private async saveChoice(choice: ProblemReportChoice): Promise<void> {
    await this.client.saveChoice(choice).catch(() => undefined);
    await this.refresh();
  }

  private sentRow(sentReports: SentProblemReport[]): HTMLElement {
    const sentText = sentReports.length === 0 ? "Nothing yet" : sentReports.length === 1 ? "1 report" : `${sentReports.length} reports`;
    const toggle = sentReports.length === 0
      ? null
      : rowButton(this.document, "settings-sent-reports", this.isSentListOpen ? "Hide" : "Show", () => {
          this.isSentListOpen = !this.isSentListOpen;
          void this.refresh();
        });
    return makeRow(this.document, "Problem reports sent", sentText, toggle);
  }

  /** Every sent report, newest first, as the exact lines that were sent. */
  private sentList(sentReports: SentProblemReport[]): HTMLElement {
    const list = makeElement(this.document, "div", "settings__sent-reports");
    list.dataset.testid = "settings-sent-report-list";
    for (const sentReport of [...sentReports].reverse()) {
      const sentAt = new Date(sentReport.sentAtUnixSeconds * 1000).toLocaleString();
      list.append(
        makeElement(this.document, "p", "settings__sent-report-time", `Sent ${sentAt}`),
        makeElement(this.document, "pre", "settings__sent-report-lines", sentReport.readableLines.join("\n")),
      );
    }
    return list;
  }

  /** The daily count: the exact line, whether it is on, when it last went,
   * and the button that turns it on or off. */
  private dailyCountRow(): HTMLElement {
    const dailyCount = this.dailyCount;
    if (dailyCount === null) {
      return makeRow(this.document, "Daily count", makeElement(this.document, "span", "settings__row-value--muted", "Not sent"), null);
    }
    const isOn = dailyCount.answer === true;
    const lastSent = dailyCount.lastSentUnixDay === null ? "" : ` Last sent ${new Date(dailyCount.lastSentUnixDay * 86_400_000).toLocaleDateString(undefined, { timeZone: "UTC" })}.`;
    const detailText = isOn ? `On. Once a day: ${dailyCount.line}.${lastSent}` : `Off. If on, once a day: ${dailyCount.line}`;
    const detail = makeElement(this.document, "span", "settings__row-value--muted", detailText);
    detail.dataset.testid = "settings-daily-count-detail";
    return makeRow(this.document, "Daily count", detail, rowButton(this.document, "settings-daily-count", isOn ? "Turn off" : "Turn on", () => void this.setDailyCount(!isOn)));
  }

  private async setDailyCount(isCounted: boolean): Promise<void> {
    await this.client.saveDailyCountAnswer(isCounted);
    await this.refresh();
  }

  private reportProblemRow(): HTMLElement {
    const detail = makeElement(this.document, "span", "settings__row-value--muted", this.personReportOutcome || "Something slow, wrong or confusing? Tell the developers.");
    detail.dataset.testid = "settings-report-problem-detail";
    return makeRow(this.document, "Report a problem", detail, rowButton(this.document, "settings-report-problem", "Write a report", () => void this.startPersonReport()));
  }

  private async startPersonReport(): Promise<void> {
    this.personReport = await this.client.startPersonReport().catch(() => null);
    this.personReportOutcome = this.personReport === null ? "A report can't be started right now." : "";
    await this.refresh();
  }

  /** The report being written: every line that will be sent, the person's
   * own words, and Send report / Cancel. */
  private personReportForm(personReport: WaitingProblemReport): HTMLElement {
    const form = makeElement(this.document, "div", "settings__person-report");
    form.dataset.testid = "settings-person-report";
    const noteField = makeNoteField(this.document, "settings-person-report-note");
    noteField.placeholder = "What happened? This is sent with the report.";
    const outcome = makeElement(this.document, "p", "settings__sent-report-time", this.personReportOutcome);
    const cancelButton = rowButton(this.document, "settings-person-report-cancel", "Cancel", () => void this.finishPersonReport(personReport, null));
    const sendButton = rowButton(this.document, "settings-person-report-send", "Send report", () => void this.finishPersonReport(personReport, noteField.value));
    const actions = makeElement(this.document, "div", "settings__person-report-actions");
    actions.append(cancelButton, sendButton);
    form.append(
      makeElement(this.document, "p", "settings__sent-report-time", "This is the whole report, plus what you write below."),
      makeElement(this.document, "pre", "settings__sent-report-lines", personReport.readableLines.join("\n")),
      makeElement(this.document, "p", "settings__sent-report-time", NEVER_INCLUDED_TEXT),
      noteField,
      actions,
      outcome,
    );
    return form;
  }

  /** Sends with the note, or cancels when the note is null. A report that
   * did not go stays open with the engine's reason. */
  private async finishPersonReport(personReport: WaitingProblemReport, typedNote: string | null): Promise<void> {
    const answer = typedNote === null ? await this.client.declineReport(personReport.reportNumber) : await this.client.sendReport(personReport.reportNumber, typedNote);
    if (answer.isDone || typedNote === null) {
      this.personReport = null;
      this.personReportOutcome = typedNote === null ? "" : answer.message;
    } else {
      this.personReportOutcome = answer.message;
    }
    await this.refresh();
  }
}
