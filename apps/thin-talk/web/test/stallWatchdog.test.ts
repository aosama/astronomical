import assert from "node:assert/strict";
import { test } from "node:test";
import { StallWatchdog } from "../src/StallWatchdog";
import { waitUntil } from "./support/scriptedChatFetch";

const SHORT_STALL_TIMEOUT_MILLISECONDS = 30;
/** The watchdog polls once a second, so a stall shows within about that. */
const STALL_REPORT_WAIT_MILLISECONDS = 3_000;

test("should_report_a_stall_once_after_the_timeout_without_activity", async () => {
  let stallCount = 0;
  const watchdog = new StallWatchdog(SHORT_STALL_TIMEOUT_MILLISECONDS, () => { stallCount += 1; });
  watchdog.start();
  await waitUntil(() => stallCount === 1, "the stall to be reported", STALL_REPORT_WAIT_MILLISECONDS);
  await new Promise((resolve) => setTimeout(resolve, 1_100));
  watchdog.stop();
  assert.equal(stallCount, 1);
});

test("should_not_report_a_stall_while_paused_and_restart_the_deadline_on_resume", async () => {
  let stallCount = 0;
  const watchdog = new StallWatchdog(SHORT_STALL_TIMEOUT_MILLISECONDS, () => { stallCount += 1; });
  watchdog.start();
  watchdog.pause();
  await new Promise((resolve) => setTimeout(resolve, 1_100));
  assert.equal(stallCount, 0, "a running tool is not a stall");
  watchdog.resume();
  await waitUntil(() => stallCount === 1, "the stall after resuming", STALL_REPORT_WAIT_MILLISECONDS);
  watchdog.stop();
});

test("should_clear_a_pause_when_a_new_run_starts", async () => {
  let stallCount = 0;
  const watchdog = new StallWatchdog(SHORT_STALL_TIMEOUT_MILLISECONDS, () => { stallCount += 1; });
  watchdog.start();
  watchdog.pause();
  watchdog.start();
  await waitUntil(() => stallCount === 1, "the stall of the new run", STALL_REPORT_WAIT_MILLISECONDS);
  watchdog.stop();
});

test("should_hold_off_a_stall_for_the_grace_given_on_resume", async () => {
  let stallCount = 0;
  const watchdog = new StallWatchdog(SHORT_STALL_TIMEOUT_MILLISECONDS, () => { stallCount += 1; });
  watchdog.start();
  watchdog.pause();
  watchdog.resume(60_000);
  await new Promise((resolve) => setTimeout(resolve, 1_100));
  watchdog.stop();
  assert.equal(stallCount, 0);
});
