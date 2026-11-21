import assert from "node:assert/strict";
import { test } from "node:test";
import { SettingsSharing } from "../src/astro/SettingsSharing";
import { ProblemReportClient, type FetchFunction } from "../src/ProblemReportClient";

function sharingFor(listBody: unknown | null, listStatus = 200, dailyCountBody: unknown = null): { sharing: SettingsSharing; calls: string[] } {
  const calls: string[] = [];
  const fetchFunction: FetchFunction = async (input, init) => {
    calls.push(`${init?.method ?? "GET"} ${input.replace("http://engine", "")}${init?.body ? ` ${String(init.body)}` : ""}`);
    if (listBody === null) {
      throw new TypeError("engine not answering");
    }
    if (input.endsWith("/v1/daily-count")) {
      return new Response(JSON.stringify(dailyCountBody), { status: dailyCountBody === null ? 404 : 200 });
    }
    if (init?.method === "PUT") {
      return new Response("{}", { status: 200 });
    }
    if (init?.method === "POST" && input.endsWith("/v1/problem-reports")) {
      return new Response(JSON.stringify({ report_number: 9, readable_lines: ["Problem: reported by you"] }), { status: 200 });
    }
    if (init?.method === "POST") {
      return new Response(JSON.stringify({ status: "sent", message: "Report sent. Thank you." }), { status: 200 });
    }
    return new Response(JSON.stringify(listBody), { status: listStatus });
  };
  const client = new ProblemReportClient({ daemonBaseURL: "http://engine", fetchFunction });
  return { sharing: new SettingsSharing(document, client), calls };
}

const TWO_SENT = {
  choice: "ask",
  is_sending_set_up: true,
  waiting: [],
  sent: [
    { sent_at_unix_seconds: 1700000000, readable_lines: ["Problem: model load failed"] },
    { sent_at_unix_seconds: 1700000600, readable_lines: ["Problem: reply stopped, graphics memory ran out"] },
  ],
};

test("should_say_chats_never_leave_and_show_the_stored_choice_and_sent_count", async () => {
  const { sharing } = sharingFor(TWO_SENT);
  await sharing.refresh();
  const pageText = sharing.element.textContent ?? "";
  assert.match(pageText, /Chats, files and imagesNever/);
  assert.match(pageText, /Sent so far2 reports/);
  const picker = sharing.element.querySelector('[data-testid="settings-problem-reports"]') as HTMLSelectElement;
  assert.equal(picker.value, "ask");
});

test("should_show_every_sent_report_newest_first_when_opened", async () => {
  const { sharing } = sharingFor(TWO_SENT);
  await sharing.refresh();
  (sharing.element.querySelector('[data-testid="settings-sent-reports"]') as HTMLButtonElement).click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const sentLines = [...sharing.element.querySelectorAll(".settings__sent-report-lines")].map((element) => element.textContent);
  assert.deepEqual(sentLines, ["Problem: reply stopped, graphics memory ran out", "Problem: model load failed"]);
});

test("should_save_a_new_choice_and_redraw_from_the_engine", async () => {
  const { sharing, calls } = sharingFor({ ...TWO_SENT, sent: [] });
  await sharing.refresh();
  const picker = sharing.element.querySelector('[data-testid="settings-problem-reports"]') as HTMLSelectElement;
  picker.value = "never";
  picker.dispatchEvent(new Event("change"));
  await new Promise((resolve) => setTimeout(resolve, 0));
  const callsAfterFirstDraw = calls.slice(2);
  assert.equal(callsAfterFirstDraw[0], 'PUT /v1/config/sharing {"problem_reports":"never"}');
  assert.equal(callsAfterFirstDraw[callsAfterFirstDraw.length - 1], "GET /v1/problem-reports", "the page redraws from the engine after saving");
});

test("should_say_nothing_was_sent_and_offer_no_list_when_the_log_is_empty", async () => {
  const { sharing } = sharingFor({ ...TWO_SENT, sent: [] });
  await sharing.refresh();
  assert.match(sharing.element.textContent ?? "", /Sent so farNothing yet/);
  assert.equal(sharing.element.querySelector('[data-testid="settings-sent-reports"]'), null);
});

test("should_keep_the_fixed_rows_and_explain_when_the_engine_is_not_answering", async () => {
  const { sharing } = sharingFor(null);
  await sharing.refresh();
  const pageText = sharing.element.textContent ?? "";
  assert.match(pageText, /Chats, files and imagesNever/);
  assert.match(pageText, /isn't answering/);
  assert.equal(sharing.element.querySelector('[data-testid="settings-problem-reports"]'), null);
});

test("should_let_the_person_write_and_send_a_report_of_their_own", async () => {
  const { sharing, calls } = sharingFor({ ...TWO_SENT, sent: [] });
  await sharing.refresh();
  (sharing.element.querySelector('[data-testid="settings-report-problem"]') as HTMLButtonElement).click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const form = sharing.element.querySelector('[data-testid="settings-person-report"]') as HTMLElement;
  assert.match(form.textContent ?? "", /Problem: reported by you/);
  (form.querySelector('[data-testid="settings-person-report-note"]') as HTMLTextAreaElement).value = "replies are slow since the update";
  (form.querySelector('[data-testid="settings-person-report-send"]') as HTMLButtonElement).click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.ok(calls.includes('POST /v1/problem-reports/9/send {"note":"replies are slow since the update"}'));
  assert.equal(sharing.element.querySelector('[data-testid="settings-person-report"]'), null);
  assert.match(sharing.element.textContent ?? "", /Report sent\. Thank you\./);
});

test("should_cancel_a_report_of_their_own_without_sending_it", async () => {
  const { sharing, calls } = sharingFor({ ...TWO_SENT, sent: [] });
  await sharing.refresh();
  (sharing.element.querySelector('[data-testid="settings-report-problem"]') as HTMLButtonElement).click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  (sharing.element.querySelector('[data-testid="settings-person-report-cancel"]') as HTMLButtonElement).click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.ok(calls.includes("POST /v1/problem-reports/9/decline"));
  assert.ok(!calls.some((call) => call.includes("/send")));
});

test("should_say_this_version_never_sends_reports_when_the_engine_offers_none", async () => {
  const { sharing } = sharingFor("problem reports are unavailable", 404);
  await sharing.refresh();
  const pageText = sharing.element.textContent ?? "";
  assert.match(pageText, /Problem reportsNever\. This version doesn't send them\./);
  assert.match(pageText, /Daily countNever/);
  assert.doesNotMatch(pageText, /isn't answering/);
  assert.equal(sharing.element.querySelector('[data-testid="settings-problem-reports"]'), null);
  assert.equal(sharing.element.querySelector('[data-testid="settings-report-problem"]'), null);
});

const COUNT_LINE = "Astronomical 0.2.127 · macOS 27.2 · Apple M5 Pro · 48 GB";

test("should_show_the_daily_count_line_and_turn_it_on_from_settings", async () => {
  const { sharing, calls } = sharingFor({ ...TWO_SENT, sent: [] }, 200, { line: COUNT_LINE, answer: null, should_ask: false, last_sent_unix_day: null });
  await sharing.refresh();
  assert.match(sharing.element.querySelector('[data-testid="settings-daily-count-detail"]')?.textContent ?? "", new RegExp(`^Off\\. If on, once a day: ${COUNT_LINE}`));
  const turnOn = sharing.element.querySelector('[data-testid="settings-daily-count"]') as HTMLButtonElement;
  assert.equal(turnOn.textContent, "Turn on");
  turnOn.click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.ok(calls.includes('PUT /v1/config/sharing {"daily_count":true}'));
});

test("should_say_the_count_is_on_with_its_last_sent_day_and_offer_to_turn_it_off", async () => {
  const { sharing } = sharingFor({ ...TWO_SENT, sent: [] }, 200, { line: COUNT_LINE, answer: true, should_ask: false, last_sent_unix_day: 20778 });
  await sharing.refresh();
  const detailText = sharing.element.querySelector('[data-testid="settings-daily-count-detail"]')?.textContent ?? "";
  assert.match(detailText, /^On\. Once a day: /);
  assert.match(detailText, /Last sent /);
  assert.equal(sharing.element.querySelector('[data-testid="settings-daily-count"]')?.textContent, "Turn off");
});
