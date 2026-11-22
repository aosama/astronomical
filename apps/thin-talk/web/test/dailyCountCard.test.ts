import assert from "node:assert/strict";
import { test } from "node:test";
import { DailyCountCard } from "../src/DailyCountCard";
import { ProblemReportClient, type FetchFunction } from "../src/ProblemReportClient";

const LINE = "Astronomical 0.2.127 · macOS 27.2 · Apple M5 Pro · 48 GB";

function cardFor(dailyCountBody: unknown, countStatus = 200, saveStatus = 200): { card: DailyCountCard; calls: string[] } {
  const calls: string[] = [];
  const fetchFunction: FetchFunction = async (input, init) => {
    calls.push(`${init?.method ?? "GET"} ${input.replace("http://engine", "")}${init?.body ? ` ${String(init.body)}` : ""}`);
    if (init?.method === "PUT") {
      return new Response("{}", { status: saveStatus });
    }
    return new Response(JSON.stringify(dailyCountBody), { status: countStatus });
  };
  const anchor = document.createElement("aside");
  document.body.replaceChildren(anchor);
  const client = new ProblemReportClient({ daemonBaseURL: "http://engine", fetchFunction });
  return { card: new DailyCountCard({ client, document, placeAfter: anchor }), calls };
}

const ASK_NOW = { line: LINE, answer: null, should_ask: true, last_sent_unix_day: null };

async function press(card: DailyCountCard, testID: string): Promise<void> {
  (card.element.querySelector(`[data-testid="${testID}"]`) as HTMLElement).click();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test("should_ask_with_the_exact_line_when_the_engine_says_it_is_time", async () => {
  const { card, calls } = cardFor(ASK_NOW);
  await card.refresh();
  assert.equal(card.element.hidden, false);
  assert.equal(card.element.querySelector('[data-testid="daily-count-line"]')?.textContent, LINE);
  assert.deepEqual(calls, ["GET /v1/daily-count"], "asking must not save or send anything");
});

test("should_not_ask_on_the_first_day_after_an_answer_or_when_unavailable", async () => {
  for (const [body, status] of [
    [{ ...ASK_NOW, should_ask: false }, 200],
    [{ ...ASK_NOW, answer: true, should_ask: false }, 200],
    [{ ...ASK_NOW, answer: false, should_ask: false }, 200],
    ["the daily count is unavailable", 404],
    [{ should_ask: true }, 200],
  ] as Array<[unknown, number]>) {
    const { card } = cardFor(body, status);
    await card.refresh();
    assert.equal(card.element.hidden, true, JSON.stringify(body));
  }
});

test("should_save_yes_and_hide_when_the_person_agrees", async () => {
  const { card, calls } = cardFor(ASK_NOW);
  await card.refresh();
  await press(card, "daily-count-agree");
  assert.deepEqual(calls.slice(1), ['PUT /v1/config/sharing {"daily_count":true}']);
  assert.equal(card.element.hidden, true);
});

test("should_save_no_and_hide_when_the_person_declines", async () => {
  const { card, calls } = cardFor(ASK_NOW);
  await card.refresh();
  await press(card, "daily-count-decline");
  assert.deepEqual(calls.slice(1), ['PUT /v1/config/sharing {"daily_count":false}']);
  assert.equal(card.element.hidden, true);
});

test("should_keep_the_question_and_say_so_when_the_answer_could_not_be_saved", async () => {
  const { card } = cardFor(ASK_NOW, 200, 500);
  await card.refresh();
  await press(card, "daily-count-decline");
  assert.equal(card.element.hidden, false);
  assert.match(card.element.querySelector('[data-testid="daily-count-outcome"]')?.textContent ?? "", /couldn't be saved/);
});
