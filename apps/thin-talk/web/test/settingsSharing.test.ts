import assert from "node:assert/strict";
import { test } from "node:test";
import { SettingsSharing } from "../src/astro/SettingsSharing";
import { ProblemReportClient, type FetchFunction } from "../src/ProblemReportClient";

function sharingFor(listBody: unknown | null): { sharing: SettingsSharing; calls: string[] } {
  const calls: string[] = [];
  const fetchFunction: FetchFunction = async (input, init) => {
    calls.push(`${init?.method ?? "GET"} ${input.replace("http://engine", "")}${init?.body ? ` ${String(init.body)}` : ""}`);
    if (listBody === null) {
      throw new TypeError("engine not answering");
    }
    return new Response(JSON.stringify(listBody), { status: 200 });
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
  assert.deepEqual(calls.slice(1), ['PUT /v1/config/sharing {"problem_reports":"never"}', "GET /v1/problem-reports"]);
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
