import assert from "node:assert/strict";
import { test } from "node:test";
import { NEVER_INCLUDED_TEXT, ProblemReportCard } from "../src/ProblemReportCard";
import { ProblemReportClient, type FetchFunction } from "../src/ProblemReportClient";

const REPORT_LINES = ["App: Astronomical 0.2.126", "Mac: Apple M5 Pro, 48 GB, macOS 27.2", "Problem: model load failed"];

/** A stand-in engine: answers the list with `listBody` and records every
 * other call, answering sends with `sendStatus`. */
function standInEngine(listBody: unknown, sendStatus = 200): { fetchFunction: FetchFunction; calls: string[] } {
  const calls: string[] = [];
  const fetchFunction: FetchFunction = async (input, init) => {
    const method = init?.method ?? "GET";
    calls.push(`${method} ${input.replace("http://engine", "")}${init?.body ? ` ${String(init.body)}` : ""}`);
    if (input.endsWith("/v1/problem-reports")) {
      return new Response(JSON.stringify(listBody), { status: 200 });
    }
    const status = input.endsWith("/send") ? sendStatus : 200;
    return new Response(JSON.stringify({ status: "x", message: status === 200 ? "Report sent. Thank you." : "Sending reports is not set up in this version yet. Nothing was sent." }), { status });
  };
  return { fetchFunction, calls };
}

function cardFor(listBody: unknown, sendStatus = 200): { card: ProblemReportCard; calls: string[] } {
  const banner = document.createElement("aside");
  document.body.replaceChildren(banner);
  const engine = standInEngine(listBody, sendStatus);
  const client = new ProblemReportClient({ daemonBaseURL: "http://engine", fetchFunction: engine.fetchFunction });
  return { card: new ProblemReportCard({ client, document, placeAfter: banner }), calls: engine.calls };
}

const ONE_WAITING = { choice: "ask", is_sending_set_up: true, waiting: [{ report_number: 4, readable_lines: REPORT_LINES }], sent: [] };

function click(card: ProblemReportCard, testID: string): void {
  (card.element.querySelector(`[data-testid="${testID}"]`) as HTMLElement).click();
}

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test("should_show_the_whole_report_and_the_promise_when_the_person_chose_to_be_asked", async () => {
  const { card } = cardFor(ONE_WAITING);
  await card.refresh();
  assert.equal(card.element.hidden, false);
  assert.equal(card.element.querySelector('[data-testid="problem-report-lines"]')?.textContent, REPORT_LINES.join("\n"));
  assert.ok(card.element.textContent?.includes(NEVER_INCLUDED_TEXT));
});

test("should_show_nothing_when_no_report_waits_or_the_person_chose_always_or_never", async () => {
  for (const listBody of [
    { ...ONE_WAITING, waiting: [] },
    { ...ONE_WAITING, choice: "always" },
    { ...ONE_WAITING, choice: "never" },
    "not an answer",
  ]) {
    const { card } = cardFor(listBody);
    await card.refresh();
    assert.equal(card.element.hidden, true, JSON.stringify(listBody));
  }
});

test("should_send_only_after_the_press_and_hide_once_sent", async () => {
  const { card, calls } = cardFor(ONE_WAITING);
  await card.refresh();
  assert.deepEqual(calls, ["GET /v1/problem-reports"], "showing a report must not send it");
  click(card, "problem-report-send");
  await settle();
  assert.deepEqual(calls.slice(1), ["POST /v1/problem-reports/4/send"]);
  assert.equal(card.element.hidden, true);
});

test("should_save_always_before_sending_when_the_box_is_ticked", async () => {
  const { card, calls } = cardFor(ONE_WAITING);
  await card.refresh();
  (card.element.querySelector('[data-testid="problem-report-always"]') as HTMLInputElement).checked = true;
  click(card, "problem-report-send");
  await settle();
  assert.deepEqual(calls.slice(1), ['PUT /v1/config/sharing {"problem_reports":"always"}', "POST /v1/problem-reports/4/send"]);
});

test("should_decline_without_saving_always_even_when_the_box_is_ticked", async () => {
  const { card, calls } = cardFor(ONE_WAITING);
  await card.refresh();
  (card.element.querySelector('[data-testid="problem-report-always"]') as HTMLInputElement).checked = true;
  click(card, "problem-report-decline");
  await settle();
  assert.deepEqual(calls.slice(1), ["POST /v1/problem-reports/4/decline"]);
  assert.equal(card.element.hidden, true);
});

test("should_keep_the_report_on_screen_with_the_reason_when_it_did_not_go", async () => {
  const { card } = cardFor(ONE_WAITING, 503);
  await card.refresh();
  click(card, "problem-report-send");
  await settle();
  assert.equal(card.element.hidden, false);
  assert.match(card.element.querySelector('[data-testid="problem-report-outcome"]')?.textContent ?? "", /not set up/);
});

test("should_keep_only_well_formed_reports_from_the_engine", () => {
  const problemReports = ProblemReportClient.reportsFromPlain({
    choice: "sometimes",
    is_sending_set_up: "yes",
    waiting: [{ report_number: 1, readable_lines: ["ok"] }, { report_number: "2", readable_lines: ["bad"] }, { readable_lines: [3] }],
    sent: [{ sent_at_unix_seconds: 1700000000, readable_lines: ["sent"] }, { readable_lines: ["no time"] }],
  });
  assert.equal(problemReports.choice, "ask");
  assert.equal(problemReports.isSendingSetUp, false);
  assert.deepEqual(problemReports.waiting, [{ reportNumber: 1, readableLines: ["ok"] }]);
  assert.deepEqual(problemReports.sent, [{ sentAtUnixSeconds: 1700000000, readableLines: ["sent"] }]);
});

test("should_send_what_the_person_typed_with_the_report_and_nothing_when_left_empty", async () => {
  const { card, calls } = cardFor(ONE_WAITING);
  await card.refresh();
  (card.element.querySelector('[data-testid="problem-report-note"]') as HTMLTextAreaElement).value = "the reply stopped halfway";
  click(card, "problem-report-send");
  await settle();
  assert.deepEqual(calls.slice(1), ['POST /v1/problem-reports/4/send {"note":"the reply stopped halfway"}']);
  const empty = cardFor(ONE_WAITING);
  await empty.card.refresh();
  click(empty.card, "problem-report-send");
  await settle();
  assert.deepEqual(empty.calls.slice(1), ["POST /v1/problem-reports/4/send"]);
});

test("should_show_no_card_when_the_engine_offers_no_problem_reports", async () => {
  const banner = document.createElement("aside");
  document.body.replaceChildren(banner);
  const client = new ProblemReportClient({ daemonBaseURL: "http://engine", fetchFunction: async () => new Response("problem reports are unavailable", { status: 404 }) });
  const card = new ProblemReportCard({ client, document, placeAfter: banner });
  await card.refresh();
  assert.equal(card.element.hidden, true);
});
