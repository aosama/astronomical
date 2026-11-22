/** How the person chose to handle problem reports. */
export type ProblemReportChoice = "ask" | "always" | "never";

/** One report waiting for the person, as the lines they read. */
export interface WaitingProblemReport {
  reportNumber: number;
  readableLines: string[];
}

/** One report already sent from this Mac. */
export interface SentProblemReport {
  sentAtUnixSeconds: number;
  readableLines: string[];
}

export interface ProblemReports {
  choice: ProblemReportChoice;
  isSendingSetUp: boolean;
  waiting: WaitingProblemReport[];
  sent: SentProblemReport[];
}

/** The daily count as the engine describes it for this Mac. */
export interface DailyCount {
  /** The exact line this Mac sends, with its real values. */
  line: string;
  /** True or false once answered; null while the person has not been asked. */
  answer: boolean | null;
  shouldAsk: boolean;
  /** Days since 1970-01-01 (UTC) of the last count sent, or null. */
  lastSentUnixDay: number | null;
}

/** What the engine said after Send report or Don't send. */
export interface ProblemReportAnswer {
  isDone: boolean;
  message: string;
}

/** The engine has no problem reports at all, as in the Mac App Store
 * version, which sends none. Different from an engine that is not answering. */
export class ProblemReportsNotOfferedError extends Error {
  public constructor() {
    super("problem reports are not offered by this version");
    this.name = "ProblemReportsNotOfferedError";
  }
}

/** The fetch signature, so tests can stand in for the engine. */
export type FetchFunction = (input: string, init?: RequestInit) => Promise<Response>;

/**
 * Talks to the engine's problem report endpoints on this Mac. It never talks
 * to the internet: sending a report is the engine's job, after the person
 * presses Send report.
 */
export class ProblemReportClient {
  private readonly daemonBaseURL: string;
  private readonly fetchFunction: FetchFunction;

  public constructor(init: { daemonBaseURL: string; fetchFunction?: FetchFunction }) {
    this.daemonBaseURL = init.daemonBaseURL;
    this.fetchFunction = init.fetchFunction ?? ((input, requestInit) => fetch(input, requestInit));
  }

  public async fetchProblemReports(): Promise<ProblemReports> {
    const response = await this.fetchFunction(`${this.daemonBaseURL}/v1/problem-reports`);
    if (response.status === 404) {
      throw new ProblemReportsNotOfferedError();
    }
    if (!response.ok) {
      throw new Error(`problem reports answered ${response.status}`);
    }
    return ProblemReportClient.reportsFromPlain(await response.json());
  }

  public async sendReport(reportNumber: number, typedNote = ""): Promise<ProblemReportAnswer> {
    return this.answerReport(reportNumber, "send", typedNote);
  }

  /** Starts a report the person wants to write themselves, and returns it
   * as the lines they will see; null when the engine cannot start one. */
  public async startPersonReport(): Promise<WaitingProblemReport | null> {
    const response = await this.fetchFunction(`${this.daemonBaseURL}/v1/problem-reports`, { method: "POST" });
    if (!response.ok) {
      return null;
    }
    const started = ProblemReportClient.reportsFromPlain({ waiting: [await response.json()] }).waiting[0];
    return started ?? null;
  }

  public async declineReport(reportNumber: number): Promise<ProblemReportAnswer> {
    return this.answerReport(reportNumber, "decline");
  }

  public async saveChoice(choice: ProblemReportChoice): Promise<void> {
    const response = await this.fetchFunction(`${this.daemonBaseURL}/v1/config/sharing`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ problem_reports: choice }),
    });
    if (!response.ok) {
      throw new Error(`saving the sharing choice answered ${response.status}`);
    }
  }

  /** The daily count for this Mac, or null when this version has none (the
   * Mac App Store version) or the engine cannot be asked. */
  public async fetchDailyCount(): Promise<DailyCount | null> {
    try {
      const response = await this.fetchFunction(`${this.daemonBaseURL}/v1/daily-count`);
      if (!response.ok) {
        return null;
      }
      const plain = (await response.json()) as Record<string, unknown>;
      if (typeof plain["line"] !== "string") {
        return null;
      }
      return {
        line: plain["line"],
        answer: typeof plain["answer"] === "boolean" ? plain["answer"] : null,
        shouldAsk: plain["should_ask"] === true,
        lastSentUnixDay: typeof plain["last_sent_unix_day"] === "number" ? plain["last_sent_unix_day"] : null,
      };
    } catch {
      return null;
    }
  }

  /** Saves the person's answer to "May this Mac be counted?". */
  public async saveDailyCountAnswer(isCounted: boolean): Promise<boolean> {
    const response = await this.fetchFunction(`${this.daemonBaseURL}/v1/config/sharing`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ daily_count: isCounted }),
    }).catch(() => null);
    return response !== null && response.ok;
  }

  private async answerReport(reportNumber: number, answer: "send" | "decline", typedNote = ""): Promise<ProblemReportAnswer> {
    const requestInit: RequestInit = { method: "POST" };
    if (typedNote.trim().length > 0) {
      requestInit.headers = { "Content-Type": "application/json" };
      requestInit.body = JSON.stringify({ note: typedNote });
    }
    const response = await this.fetchFunction(`${this.daemonBaseURL}/v1/problem-reports/${reportNumber}/${answer}`, requestInit);
    const answerBody: unknown = await response.json().catch(() => null);
    const message = typeof answerBody === "object" && answerBody !== null ? (answerBody as Record<string, unknown>)["message"] : null;
    return { isDone: response.ok, message: typeof message === "string" ? message : "The engine did not answer." };
  }

  /** Reads the engine's answer, keeping only well-formed reports. */
  public static reportsFromPlain(plain: unknown): ProblemReports {
    const document = (typeof plain === "object" && plain !== null ? plain : {}) as Record<string, unknown>;
    const choice = document["choice"] === "always" || document["choice"] === "never" ? document["choice"] : "ask";
    const waiting: WaitingProblemReport[] = [];
    for (const entry of Array.isArray(document["waiting"]) ? document["waiting"] : []) {
      const lines = readableLinesOf(entry);
      const reportNumber = (entry as Record<string, unknown>)["report_number"];
      if (lines !== null && typeof reportNumber === "number") {
        waiting.push({ reportNumber, readableLines: lines });
      }
    }
    const sent: SentProblemReport[] = [];
    for (const entry of Array.isArray(document["sent"]) ? document["sent"] : []) {
      const lines = readableLinesOf(entry);
      const sentAt = (entry as Record<string, unknown>)["sent_at_unix_seconds"];
      if (lines !== null && typeof sentAt === "number") {
        sent.push({ sentAtUnixSeconds: sentAt, readableLines: lines });
      }
    }
    return { choice, isSendingSetUp: document["is_sending_set_up"] === true, waiting, sent };
  }
}

function readableLinesOf(entry: unknown): string[] | null {
  if (typeof entry !== "object" || entry === null) {
    return null;
  }
  const lines = (entry as Record<string, unknown>)["readable_lines"];
  if (!Array.isArray(lines) || !lines.every((line) => typeof line === "string")) {
    return null;
  }
  return lines as string[];
}
